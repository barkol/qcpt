#include "race.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>

#include "certify.hpp"
#include "pool.hpp"

namespace qcpt {
namespace {

using Clock = std::chrono::steady_clock;
double since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

struct Member {
    std::string name, kind; int offset = 0;
    BeamParams beam; std::vector<std::string> lib_names; Library lib;       // beam members: the library is built once and reused by every round
    TemplateParams tpl; Library tlib;                                          // template members: library with the extra sign patterns
    std::vector<uint8_t> db;
};

struct Winner { bool set = false; std::string name; int cost = 0; std::vector<std::string> lines; };

std::string gate_line(const Library& lib, int gi) { const Gate& g = lib[gi]; std::string s = "G " + g.name; for (int i = 0; i < g.nargs; ++i) s += " " + std::to_string(g.args[i]); return s; }

}  // namespace

int run_race(std::istream& in) {
    const auto T0 = Clock::now(); int start, best, nmem; double limit, alpha, gamma; in >> start >> best >> limit >> nmem >> alpha >> gamma;
    std::vector<Member> M(nmem); const std::string dbpath = getenv_q("QCPT_DB") ? getenv_q("QCPT_DB") : "";
    std::vector<uint8_t> db; bool db_loaded = false;
    for (auto& m : M) {
        std::string tag; in >> tag >> m.name >> m.kind >> m.offset;
        if (m.kind == "beam") { read_beam(in, m.beam, m.lib_names); m.lib = select_library(m.beam.n, m.lib_names, m.beam.allow); m.beam.nthreads = 2; }
        else { read_template(in, m.tpl); m.tlib = make_library(m.tpl.n, true); m.tpl.nproc = getenv_q("QCPT_TPL_NPROC") ? std::atoi(getenv_q("QCPT_TPL_NPROC")) : 2; if (!db_loaded && !dbpath.empty()) { db = load_db(dbpath); db_loaded = true; } }
    }
    // work estimate of a member (element operations) and its start delay: alpha * (work of all cheaper members) / throughput
    constexpr double kRate = 3e9; std::vector<double> est(nmem, 0.0), delay(nmem, 0.0), weight(nmem, 1.0);
    for (int i = 0; i < nmem; ++i) {
        const Member& m = M[i];
        if (m.kind == "beam") est[i] = (double)m.beam.beam * m.lib.size() * m.beam.inputs.size() * 2.0 * m.beam.depth_side;
        else {
            double fr = m.tpl.n + m.tpl.n * (m.tpl.n - 1.0), s1 = 1, s2 = 1, p = 1; for (int k = 0; k < m.tpl.f1; ++k) { p *= fr; s1 += p; } p = 1; for (int k = 0; k < m.tpl.f2; ++k) { p *= fr; s2 += p; }
            double ga = 0; for (const Gate& g : m.tlib) if (std::find(m.tpl.ga.begin(), m.tpl.ga.end(), g.name) != m.tpl.ga.end()) ga += 1;
            est[i] = (s1 + s2) * ga * m.tpl.inputs.size() * (1.0 + m.tpl.n * (m.tpl.n - 1.0) * (m.tpl.n - 2.0) / 6.0);
        }
    }
    for (int i = 0; i < nmem; ++i) { double cheaper = 0; for (int j = 0; j < nmem; ++j) if (est[j] < est[i]) cheaper += est[j]; delay[i] = alpha * cheaper / kRate; }
    { const char* e = getenv_q("QCPT_DMAX"); const double dmax = e ? std::atof(e) : 2.0; for (double& d : delay) d = std::min(d, dmax); }     // cap: a late start must not push a member past the round limit
    int cur = start; Winner last; double t_best = 0; int steps = 0;
    for (;;) {
        const int target = cur - 1; std::atomic<bool> cancel{false}; std::mutex mu; Winner w; std::atomic<int> running{nmem}; const auto tr = Clock::now();
        std::vector<std::thread> th;
        for (int mi = 0; mi < nmem; ++mi) th.emplace_back([&, mi] {
            Member& mm = M[mi]; current_weight() = weight[mi]; current_cancel() = &cancel; current_low() = mm.kind != "beam";   // searches that can run in the background (templates) must not delay the beams
            while (since(tr) < delay[mi] && !cancel.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            if (cancel.load()) { --running; return; } std::vector<std::string> lines; int cost = -1;
            if (mm.kind == "beam") {
                BeamParams P = mm.beam; P.max_cost = target - mm.offset; P.cancel = &cancel; P.verbose = getenv_q("QCPT_VERBOSE") != nullptr; const BeamResult r = bidir_search(P, mm.lib);
                if (r.found && !cancel.load()) { cost = r.cost + mm.offset; for (int gi : r.gates) lines.push_back(gate_line(mm.lib, gi)); }
            } else {
                TemplateParams P = mm.tpl; P.max_cost = target - mm.offset; P.cancel = &cancel; const TemplateResult r = template_search(P, mm.tlib, db);
                if (r.found && !cancel.load()) {
                    cost = r.cost + mm.offset;
                    for (const Step& s : r.steps) {
                        if (s.gate >= 0) lines.push_back(gate_line(mm.tlib, s.gate));
                        else { std::ostringstream o; o << "DBMID " << s.db.q[0] << " " << s.db.q[1] << " " << s.db.q[2]; for (int v : s.db.perm) o << " " << v; lines.push_back(o.str()); }
                    }
                }
            }
            if (cost >= 0 && cost <= target) { std::lock_guard<std::mutex> l(mu); if (!w.set) { w.set = true; w.name = mm.name; w.cost = cost; w.lines = lines; cancel.store(true); } }
            --running;
        });
        while (running.load() > 0 && !cancel.load() && since(tr) < limit) std::this_thread::sleep_for(std::chrono::milliseconds(2));
        cancel.store(true); for (auto& t : th) t.join();
        if (!w.set) break;
        for (int i = 0; i < nmem; ++i) if (M[i].name == w.name) weight[i] *= gamma;
        cur = w.cost; last = w; t_best = since(T0); ++steps;
        std::printf("STEP %s %d %.3f %.3f\n", w.name.c_str(), cur, since(tr), t_best); std::fflush(stdout);
        if (best > 0 && cur <= best) break;
    }
    std::printf("RESULT %d %.3f %.3f\n", cur, t_best, since(T0));
    if (last.set) { std::printf("WINNER %s\n", last.name.c_str()); for (auto& l : last.lines) std::printf("%s\n", l.c_str()); }
    std::printf("END\n"); return 0;
}

}  // namespace qcpt

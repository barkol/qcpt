// qcpt: command line front end of the search core. Reads a problem on stdin, writes the result on stdout.
//
//   qcpt --dump-lib N [extra]          print the gate library (checked against the Python reference by tests/check_library.sh)
//   qcpt template DB_FILE|-            template search (see race.hpp for the problem format)
//   qcpt beam                          bidirectional beam search
//   qcpt race                          all members on one thread pool, first success lowers the target for the next round (QCPT_DB = database file)
//   results: "NONE", or "COST c" followed by one line per step ("G name args..." or "DBMID q0 q1 q2 p0..p7") and "END"
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "env.hpp"
#include "certify.hpp"
#include "race.hpp"

using namespace qcpt;

static int dump_lib(int n, bool extra) {
    for (const Gate& g : make_library(n, extra)) {
        std::printf("%s", g.name.c_str()); for (int i = 0; i < g.nargs; ++i) std::printf(" %d", g.args[i]);
        std::printf(" | %d |", g.cost); for (uint16_t v : g.p) std::printf(" %d", v);
        std::printf(" |"); for (uint8_t v : g.f) std::printf(" %d", v);
        std::printf("\n");
    }
    return 0;
}

static int run_template(const char* db_path) {
    TemplateParams P; read_template(std::cin, P); P.verbose = getenv_q("QCPT_VERBOSE") != nullptr;
    const Library lib = make_library(P.n, true); const auto db = std::strcmp(db_path, "-") ? load_db(db_path) : std::vector<uint8_t>();
    const TemplateResult r = template_search(P, lib, db);
    if (!r.found) { std::printf("NONE\n"); return 0; }
    std::printf("COST %d\n", r.cost);
    for (const Step& s : r.steps) {
        if (s.gate >= 0) { const Gate& g = lib[s.gate]; std::printf("G %s", g.name.c_str()); for (int i = 0; i < g.nargs; ++i) std::printf(" %d", g.args[i]); std::printf("\n"); }
        else { std::printf("DBMID %d %d %d", s.db.q[0], s.db.q[1], s.db.q[2]); for (int v : s.db.perm) std::printf(" %d", v); std::printf("\n"); }
    }
    std::printf("END\n"); return 0;
}

static int run_beam() {
    BeamParams P; std::vector<std::string> names; read_beam(std::cin, P, names); P.verbose = getenv_q("QCPT_VERBOSE") != nullptr;
    const Library lib = select_library(P.n, names, P.allow); const BeamResult r = bidir_search(P, lib);
    if (!r.found) { std::printf("NONE\n"); return 0; }
    std::printf("COST %d\n", r.cost);
    for (int gi : r.gates) { const Gate& g = lib[gi]; std::printf("G %s", g.name.c_str()); for (int i = 0; i < g.nargs; ++i) std::printf(" %d", g.args[i]); std::printf("\n"); }
    std::printf("CERT %s\nEND\n", certify(lib, r.gates, P.inputs, P.target, P.target_phase) ? "ok" : "FAILED"); return 0;
}

int run_certify(std::istream& in);   // columns.cpp
int run_certify_mc(std::istream& in);

int main(int argc, char** argv) {
    if (argc >= 2 && !std::strcmp(argv[1], "beam")) return run_beam();
    if (argc >= 2 && !std::strcmp(argv[1], "race")) return run_race(std::cin);
    if (argc >= 2 && !std::strcmp(argv[1], "certify")) return run_certify(std::cin);
    if (argc >= 2 && !std::strcmp(argv[1], "certify_mc")) return run_certify_mc(std::cin);
    if (argc >= 3 && !std::strcmp(argv[1], "--dump-lib")) return dump_lib(std::atoi(argv[2]), argc > 3 && !std::strcmp(argv[3], "extra"));
    if (argc >= 3 && !std::strcmp(argv[1], "template")) return run_template(argv[2]);
    std::fprintf(stderr, "usage: qcpt --dump-lib N [extra] | qcpt template DB_FILE | qcpt beam | qcpt race | qcpt certify\n"); return 2;
}

#include "template_search.hpp"
#include "pool.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <map>
#include <numeric>
#include <thread>
#include <tuple>
#include <unordered_map>

namespace qcpt {
namespace {

using Clock = std::chrono::steady_clock;
double since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

struct Seq { std::vector<int> g; int cost; };

// gate sequences (as permutation + Z8 phase tables over all N states), deduplicated up to a global phase, cheapest first
struct Tab {
    std::vector<Seq> seq;
    std::vector<uint8_t> P, F;      // seq.size() x N
};

std::vector<std::vector<int>> product_seqs(const std::vector<int>& free, int k) {
    std::vector<std::vector<int>> out{{}}, prev{{}};
    for (int L = 1; L <= k; ++L) {
        std::vector<std::vector<int>> nxt;
        for (auto& s : prev) for (int f : free) { auto t = s; t.push_back(f); nxt.push_back(t); }
        out.insert(out.end(), nxt.begin(), nxt.end()); prev.swap(nxt);
    }
    return out;
}

Tab make_tab(const Library& lib, const std::vector<std::vector<int>>& sqs, bool inverse, int N) {
    const size_t K = sqs.size(); std::vector<uint8_t> P(K * N), F(K * N); std::vector<int> cost(K, 0);
    std::vector<int> cur(N), ph(N);
    for (size_t k = 0; k < K; ++k) {
        std::iota(cur.begin(), cur.end(), 0); std::fill(ph.begin(), ph.end(), 0);
        auto step = [&](int gi) {
            const Gate& g = lib[gi]; const auto& pt = inverse ? g.ip : g.p; const auto& ft = inverse ? g.inf : g.f;
            for (int x = 0; x < N; ++x) { ph[x] += ft[cur[x]]; cur[x] = pt[cur[x]]; }
            cost[k] += g.cost;
        };
        if (inverse) for (size_t i = sqs[k].size(); i-- > 0;) step(sqs[k][i]); else for (int gi : sqs[k]) step(gi);
        for (int x = 0; x < N; ++x) { P[k * N + x] = (uint8_t)cur[x]; F[k * N + x] = (uint8_t)(ph[x] & 7); }
    }
    std::unordered_map<std::string, size_t> best;
    for (size_t k = 0; k < K; ++k) {
        std::string key((const char*)&P[k * N], N);
        for (int x = 0; x < N; ++x) key.push_back((char)((F[k * N + x] - F[k * N]) & 7));
        auto it = best.find(key);
        if (it == best.end()) best.emplace(std::move(key), k); else if (cost[k] < cost[it->second]) it->second = k;
    }
    std::vector<size_t> keep; keep.reserve(best.size()); for (auto& kv : best) keep.push_back(kv.second);
    std::sort(keep.begin(), keep.end());
    Tab T; T.P.resize(keep.size() * N); T.F.resize(keep.size() * N);
    for (size_t i = 0; i < keep.size(); ++i) {
        T.seq.push_back({sqs[keep[i]], cost[keep[i]]});
        std::copy_n(&P[keep[i] * N], N, &T.P[i * N]); std::copy_n(&F[keep[i] * N], N, &T.F[i * N]);
    }
    return T;
}

uint64_t splitmix(uint64_t& s) { uint64_t z = (s += 0x9e3779b97f4a7c15ULL); z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL; z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL; return z ^ (z >> 31); }

// hash of a state row modulo the global phase (phases relative to the first input); `mask` clears the output bits that a database middle will complete
inline uint64_t row_hash(const uint8_t* Y, const uint8_t* S, const uint64_t* R1, const uint64_t* R2, int m, uint8_t mask) {
    uint64_t h = 0; const uint8_t s0 = S[0];
    for (int i = 0; i < m; ++i) h += (uint64_t)(Y[i] & mask) * R1[i] + (uint64_t)((uint8_t)(S[i] - s0) & 7) * R2[i];
    return h;
}

struct Best { int tot = 1 << 30; size_t fi = 0, bj = 0; int mi = 0; int q[3] = {0, 0, 0}; std::array<int, 8> perm{}; };
bool better(const Best& a, const Best& b) { return std::tie(a.tot, a.fi, a.bj, a.mi) < std::tie(b.tot, b.fi, b.bj, b.mi); }


struct Completion { int cost; std::array<int, 8> perm; };       // cost < 0: none

// cheapest database completion of a partial three-qubit permutation O (O[v] = -1 where undefined)
Completion complete(const int8_t* O, const std::vector<uint8_t>& db) {
    std::vector<int> miss_in, miss_out; bool used[8] = {false};
    for (int v = 0; v < 8; ++v) { if (O[v] < 0) miss_in.push_back(v); else used[O[v]] = true; }
    for (int v = 0; v < 8; ++v) if (!used[v]) miss_out.push_back(v);
    Completion best{-1, {}};
    if (miss_in.size() > 5) return best;
    do {
        std::array<int, 8> cand; uint32_t key = 0; size_t k = 0; bool ident = true;
        for (int v = 0; v < 8; ++v) { cand[v] = O[v] >= 0 ? O[v] : miss_out[k++]; key |= (uint32_t)cand[v] << (3 * v); ident &= (cand[v] == v); }
        if (ident) continue;
        int c = db[key]; if (c == 255) continue;
        if (best.cost < 0 || c < best.cost) { best.cost = c; best.perm = cand; }
    } while (std::next_permutation(miss_out.begin(), miss_out.end()));
    return best;
}

}  // namespace

std::vector<uint8_t> load_db(const std::string& path) {
    std::vector<uint8_t> db(1 << 24, 255); std::ifstream in(path); int p[8], c;
    if (!in) return {};
    while (in >> p[0] >> p[1] >> p[2] >> p[3] >> p[4] >> p[5] >> p[6] >> p[7] >> c) { uint32_t key = 0; for (int v = 0; v < 8; ++v) key |= (uint32_t)p[v] << (3 * v); db[key] = (uint8_t)c; }
    return db;
}

TemplateResult template_search(const TemplateParams& P, const Library& lib, const std::vector<uint8_t>& db) {
    current_cancel() = P.cancel; const auto t0 = Clock::now(); const int n = P.n, N = 1 << n, m = (int)P.inputs.size(); TemplateResult out;
    std::vector<uint8_t> X(P.inputs.begin(), P.inputs.end()), Yt(P.target.begin(), P.target.end());
    uint64_t sm = P.seed * 0x1234567ULL + 99; std::vector<uint64_t> R1(m), R2(m); for (int i = 0; i < m; ++i) { R1[i] = splitmix(sm) | 1; R2[i] = splitmix(sm) | 1; }
    auto in_set = [](const std::vector<std::string>& s, const std::string& nm) { return std::find(s.begin(), s.end(), nm) != s.end(); };
    std::vector<int> free, GA, GB, single;
    for (int i = 0; i < (int)lib.size(); ++i) {
        const auto& nm = lib[i].name; if (nm == "cx" || nm == "x") free.push_back(i);
        if (in_set(P.ga, nm)) GA.push_back(i);
        if (in_set(P.gb.empty() ? P.ga : P.gb, nm)) GB.push_back(i);
        if (in_set(P.middles, nm)) single.push_back(i);
    }
    Tab T1 = make_tab(lib, product_seqs(free, P.f1), false, N), T2 = make_tab(lib, product_seqs(free, P.f2), true, N);
    auto msq = product_seqs(free, P.fm); for (int i : single) msq.push_back({i});
    Tab TM = make_tab(lib, msq, false, N);
    if (P.verbose) std::fprintf(stderr, "F1 %zu, Ga %zu, middles %zu, Gb %zu, F2 %zu\n", T1.seq.size(), GA.size(), TM.seq.size(), GB.size(), T2.seq.size());

    const size_t NF = T1.seq.size() * GA.size(), NB = T2.seq.size() * GB.size();
    std::vector<uint8_t> FY(NF * m), FS(NF * m), BY(NB * m), BS(NB * m); std::vector<int> cF(NF), cB(NB);
    for (size_t s = 0; s < T1.seq.size(); ++s) for (size_t a = 0; a < GA.size(); ++a) {
        const Gate& g = lib[GA[a]]; const size_t r = s * GA.size() + a; if ((r & 16383) == 0 && cancelled()) return out; cF[r] = T1.seq[s].cost + g.cost;
        for (int i = 0; i < m; ++i) { int y1 = T1.P[s * N + X[i]], s1 = T1.F[s * N + X[i]]; FY[r * m + i] = (uint8_t)g.p[y1]; FS[r * m + i] = (uint8_t)((s1 + g.f[y1]) & 7); }
    }
    for (size_t s = 0; s < T2.seq.size(); ++s) for (size_t b = 0; b < GB.size(); ++b) {
        const Gate& g = lib[GB[b]]; const size_t r = s * GB.size() + b; if ((r & 16383) == 0 && cancelled()) return out; cB[r] = T2.seq[s].cost + g.cost;
        for (int i = 0; i < m; ++i) { int y1 = T2.P[s * N + Yt[i]], s1 = T2.F[s * N + Yt[i]]; BY[r * m + i] = (uint8_t)g.ip[y1]; BS[r * m + i] = (uint8_t)((s1 + g.inf[y1]) & 7); }
    }
    if (NF == 0 || NB == 0) return out;
    const int minB = *std::min_element(cB.begin(), cB.end());
    if (P.verbose) std::fprintf(stderr, "forward %zu, backward %zu states (%.1f s)\n", NF, NB, since(t0));

    // ---- plain middles: hash of the transformed forward state against the sorted backward hashes
    std::vector<uint64_t> HB(NB); parallel_for(P.nproc, (NB + 4095) / 4096, [&](size_t c) { for (size_t j = c * 4096; j < std::min(NB, (c + 1) * 4096); ++j) HB[j] = row_hash(&BY[j * m], &BS[j * m], R1.data(), R2.data(), m, 0xFF); });
    std::vector<size_t> BORD(NB); std::iota(BORD.begin(), BORD.end(), 0);
    std::stable_sort(BORD.begin(), BORD.end(), [&](size_t a, size_t b) { return HB[a] < HB[b]; });
    std::vector<uint64_t> HBs(NB); for (size_t k = 0; k < NB; ++k) HBs[k] = HB[BORD[k]];
    const size_t CH = 1 << 14, nch = (NF + CH - 1) / CH, nmid = TM.seq.size();
    std::vector<Best> mid_best(nmid * nch);
    parallel_for(P.nproc, nmid * nch, [&](size_t task) {
        const size_t mi = task / nch, c = task % nch; const int cM = TM.seq[mi].cost; Best& bst = mid_best[task];
        std::vector<uint8_t> Y2(m), S2(m);
        for (size_t fi = c * CH; fi < std::min(NF, (c + 1) * CH); ++fi) {
            if (((fi - c * CH) & 2047) == 0 && cancelled()) break;
            if (cF[fi] + cM + minB > P.max_cost) continue;
            for (int i = 0; i < m; ++i) { const uint8_t y = FY[fi * m + i]; Y2[i] = TM.P[mi * N + y]; S2[i] = (uint8_t)((FS[fi * m + i] + TM.F[mi * N + y]) & 7); }
            const uint64_t h = row_hash(Y2.data(), S2.data(), R1.data(), R2.data(), m, 0xFF);
            auto it = std::lower_bound(HBs.begin(), HBs.end(), h); if (it == HBs.end() || *it != h) continue;
            const size_t j = BORD[it - HBs.begin()]; Best cand; cand.tot = cF[fi] + cM + cB[j]; cand.fi = fi; cand.bj = j; cand.mi = (int)mi;
            if (cand.tot <= P.max_cost && better(cand, bst)) bst = cand;
        }
    });
    if (cancelled()) return out;
    Best res; for (auto& b : mid_best) if (better(b, res)) res = b;
    if (P.verbose) std::fprintf(stderr, "plain middles done (%.1f s), best %d\n", since(t0), res.tot);

    // ---- database middles on every triple of qubits
    Best dbres;
    if (!db.empty() && n >= 3) {
        std::vector<std::array<int, 3>> triples;
        for (int a = 0; a < n; ++a) for (int b = a + 1; b < n; ++b) for (int c = b + 1; c < n; ++c) triples.push_back({a, b, c});
        std::vector<Best> tb(triples.size());
        parallel_for(P.nproc, triples.size(), [&](size_t ti) {
            if (cancelled()) return;
            const auto Q = triples[ti]; const uint8_t mask = (uint8_t)((1 << Q[0]) | (1 << Q[1]) | (1 << Q[2])), invm = (uint8_t)(0xFF ^ mask);
            std::vector<uint64_t> hb(NB); for (size_t j = 0; j < NB; ++j) hb[j] = row_hash(&BY[j * m], &BS[j * m], R1.data(), R2.data(), m, invm);
            std::vector<size_t> ord(NB); std::iota(ord.begin(), ord.end(), 0);
            std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b) { return std::tie(hb[a], cB[a], a) < std::tie(hb[b], cB[b], b); });
            std::vector<uint64_t> hs(NB); for (size_t k = 0; k < NB; ++k) hs[k] = hb[ord[k]];
            std::unordered_map<uint32_t, Completion> memo; Best best;
            for (size_t fi = 0; fi < NF; ++fi) {
                if ((fi & 1023) == 0 && cancelled()) break;
                if (cF[fi] + minB + 1 > P.max_cost) continue;
                const uint64_t h = row_hash(&FY[fi * m], &FS[fi * m], R1.data(), R2.data(), m, invm);
                auto lo = std::lower_bound(hs.begin(), hs.end(), h);
                for (size_t k = lo - hs.begin(); k < NB && hs[k] == h; ++k) {
                    const size_t j = ord[k]; const int t0c = cF[fi] + cB[j]; const int lim = best.tot < (1 << 30) ? best.tot - 1 : P.max_cost;
                    if (t0c + 1 > lim) break;                    // sorted by cost inside a hash class
                    int8_t O[8]; std::fill(O, O + 8, -1); bool bad = false;
                    for (int i = 0; i < m && !bad; ++i) {
                        const int fy = FY[fi * m + i], by = BY[j * m + i];
                        const int v = ((fy >> Q[0]) & 1) | (((fy >> Q[1]) & 1) << 1) | (((fy >> Q[2]) & 1) << 2), w = ((by >> Q[0]) & 1) | (((by >> Q[1]) & 1) << 1) | (((by >> Q[2]) & 1) << 2);
                        if (O[v] < 0) O[v] = (int8_t)w; else if (O[v] != w) bad = true;
                    }
                    for (int v = 0; v < 8 && !bad; ++v) for (int w = v + 1; w < 8; ++w) if (O[v] >= 0 && O[v] == O[w]) { bad = true; break; }
                    if (bad) continue;
                    uint32_t key = 0; for (int v = 0; v < 8; ++v) key |= (uint32_t)(O[v] + 1) << (4 * v);
                    auto it = memo.find(key); if (it == memo.end()) it = memo.emplace(key, complete(O, db)).first;
                    if (it->second.cost < 0) continue;
                    const int tot = t0c + it->second.cost;
                    if (tot <= P.max_cost && tot < best.tot) { best.tot = tot; best.fi = fi; best.bj = j; std::copy(Q.begin(), Q.end(), best.q); best.perm = it->second.perm; }
                }
            }
            tb[ti] = best;
        });
        for (auto& b : tb) if (b.tot < dbres.tot) dbres = b;
        if (P.verbose) std::fprintf(stderr, "database middles done (%.1f s), best %d\n", since(t0), dbres.tot);
    }

    if (cancelled()) return out;
    auto seq_steps = [&](const Seq& s) { std::vector<Step> v; for (int g : s.g) { Step st; st.gate = g; v.push_back(st); } return v; };
    auto append = [&](std::vector<Step>& dst, const std::vector<Step>& src) { dst.insert(dst.end(), src.begin(), src.end()); };
    const bool use_db = dbres.tot < (1 << 30) && (res.tot >= (1 << 30) || dbres.tot < res.tot);
    const Best& w = use_db ? dbres : res; if (w.tot >= (1 << 30)) return out;
    const size_t i1 = w.fi / GA.size(), ia = w.fi % GA.size(), i2 = w.bj / GB.size(), ib = w.bj % GB.size();
    append(out.steps, seq_steps(T1.seq[i1])); { Step s; s.gate = GA[ia]; out.steps.push_back(s); }
    if (use_db) { Step s; for (int k = 0; k < 3; ++k) s.db.q[k] = w.q[k]; s.db.perm = w.perm; out.steps.push_back(s); }
    else append(out.steps, seq_steps(TM.seq[w.mi]));
    { Step s; s.gate = GB[ib]; out.steps.push_back(s); }
    append(out.steps, seq_steps(T2.seq[i2]));
    out.found = true; out.cost = w.tot; return out;
}

}  // namespace qcpt

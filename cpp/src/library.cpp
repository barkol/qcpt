#include "library.hpp"

#include <cstring>

namespace qcpt {
namespace {

struct Table {
    const char* name;
    int k, cost;
    std::array<int, 16> perm, ph;
};

const std::vector<Table>& tables() {
    static const std::vector<Table> T = {
#include "gadget_tables.inc"
    };
    return T;
}

bool is_parity(const Table& t) { return std::strlen(t.name) == 5 && t.name[0] == 'm' && t.name[1] == 'g' && std::strchr("pqrstuvw", t.name[2]); }
bool is_extra_sign(const Table& t) { return is_parity(t) && std::strchr("tuvw", t.name[2]); }

void add(Library& L, int N, const std::string& name, const std::vector<int>& args, int cost, const std::vector<int>& perm, const std::vector<int>& ph) {
    Gate g; g.name = name; g.nargs = (int)args.size(); for (size_t i = 0; i < args.size(); ++i) g.args[i] = args[i]; g.cost = cost;
    g.p.resize(N); g.ip.resize(N); g.f.resize(N); g.inf.resize(N);
    for (int x = 0; x < N; ++x) { g.p[x] = (uint16_t)perm[x]; g.f[x] = (uint8_t)(ph[x] & 7); }
    for (int x = 0; x < N; ++x) { g.ip[perm[x]] = (uint16_t)x; g.inf[perm[x]] = (uint8_t)((8 - (ph[x] & 7)) & 7); }
    L.push_back(std::move(g));
}

// places a gadget on the qubits qs (local qubit i -> qs[i]) of an n-qubit register
void add_on(Library& L, int n, const Table& t, const std::vector<int>& qs) {
    const int N = 1 << n; std::vector<int> perm(N), ph(N); int mask = 0;
    for (int q : qs) mask |= 1 << q;
    for (int x = 0; x < N; ++x) {
        int loc = 0; for (size_t i = 0; i < qs.size(); ++i) loc |= ((x >> qs[i]) & 1) << i;
        int o = t.perm[loc], y = x & ~mask;
        for (size_t i = 0; i < qs.size(); ++i) y |= ((o >> i) & 1) << qs[i];
        perm[x] = y; ph[x] = t.ph[loc];
    }
    add(L, N, t.name, qs, t.cost, perm, ph);
}

// all subsets of size r of `from`, in lexicographic order (itertools.combinations)
void combos(const std::vector<int>& from, int r, std::vector<std::vector<int>>& out) {
    std::vector<int> idx(r); out.clear(); const int m = (int)from.size(); if (r > m) return;
    for (int i = 0; i < r; ++i) idx[i] = i;
    for (;;) {
        std::vector<int> c(r); for (int i = 0; i < r; ++i) c[i] = from[idx[i]]; out.push_back(c);
        int i = r - 1; while (i >= 0 && idx[i] == m - r + i) --i;
        if (i < 0) break;
        ++idx[i]; for (int j = i + 1; j < r; ++j) idx[j] = idx[j - 1] + 1;
    }
}

}  // namespace

Library make_library(int n, bool extra_signs) {
    const int N = 1 << n; Library L; std::vector<int> zero(N, 0), id(N); for (int x = 0; x < N; ++x) id[x] = x;
    for (int t = 0; t < n; ++t) { std::vector<int> p(N); for (int x = 0; x < N; ++x) p[x] = x ^ (1 << t); add(L, N, "x", {t}, 0, p, zero); }
    static const struct { const char* nm; int e; } diag[] = {{"t", 1}, {"tdg", 7}, {"s", 2}, {"sdg", 6}, {"z", 4}};
    for (int q = 0; q < n; ++q) for (auto& d : diag) { std::vector<int> ph(N); for (int x = 0; x < N; ++x) ph[x] = d.e * ((x >> q) & 1); add(L, N, d.nm, {q}, 0, id, ph); }
    for (int c = 0; c < n; ++c) for (int t = 0; t < n; ++t) if (c != t) { std::vector<int> p(N); for (int x = 0; x < N; ++x) p[x] = x ^ (((x >> c) & 1) << t); add(L, N, "cx", {c, t}, 1, p, zero); }
    for (int a = 0; a < n; ++a) for (int b = a + 1; b < n; ++b) { std::vector<int> ph(N); for (int x = 0; x < N; ++x) ph[x] = 4 * ((x >> a) & 1) * ((x >> b) & 1); add(L, N, "cz", {a, b}, 1, id, ph); }
    std::vector<std::vector<int>> cs;
    for (int t = 0; t < n; ++t) {
        std::vector<int> rest; for (int q = 0; q < n; ++q) if (q != t) rest.push_back(q);
        combos(rest, 2, cs);
        for (auto& ab : cs) for (auto& tb : tables()) if (!is_parity(tb)) add_on(L, n, tb, {ab[0], ab[1], t});
    }
    for (auto& tb : tables()) {
        if (!is_parity(tb) || (!extra_signs && is_extra_sign(tb))) continue;
        const int na = tb.name[3] - '0', nb = tb.name[4] - '0';
        for (int t = 0; t < n; ++t) {
            std::vector<int> rest; for (int q = 0; q < n; ++q) if (q != t) rest.push_back(q);
            std::vector<std::vector<int>> As; combos(rest, na, As);
            for (auto& A : As) {
                std::vector<int> rem; for (int q : rest) { bool in = false; for (int a : A) in |= (a == q); if (!in) rem.push_back(q); }
                std::vector<std::vector<int>> Bs; combos(rem, nb, Bs);
                for (auto& B : Bs) { std::vector<int> qs = A; qs.insert(qs.end(), B.begin(), B.end()); qs.push_back(t); add_on(L, n, tb, qs); }
            }
        }
    }
    return L;
}

}  // namespace qcpt

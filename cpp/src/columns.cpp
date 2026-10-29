// qcpt certify: whole-circuit certificate on the input subspace, computed column by column (cache-resident state vectors, threads).
// Input (stdin, text):
//   n C                      number of qubits, number of input columns
//   c_0 ... c_{C-1}          basis states (columns of the unitary) to propagate
//   M                        number of distinct gate matrices
//   M times:  k  then 2^k*2^k pairs (re im), row-major, local bit r = qubit q_r of the gate
//   A: ng gp   then ng lines  "matrix_id q_0 ... q_{k-1}"      (circuit A, global phase gp)
//   B: ng gp   then ng lines
// Output: "FRO <||A-e^{i th}B||_F over the columns> MAX <max entry> RE <cos th> IM <sin th>" with th from the first column.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>
#include "env.hpp"

namespace {
using cd = std::complex<double>;

struct Mat {
    int k = 0, dim = 0;
    std::vector<cd> m;
    bool mono = false;            // every column has a single non-zero entry
    std::vector<int> perm;        // mono: row of the non-zero entry of column l
    std::vector<cd> ph;           // mono: its value
};
struct Gate { int mat; int q[6]; };
struct Circ { std::vector<Gate> g; double gp = 0; };

static void apply(const std::vector<Mat>& mats, const Gate& g, int n, std::vector<cd>& v) {
    const Mat& M = mats[g.mat];
    const size_t N = size_t(1) << n;
    const int k = M.k;
    size_t mask = 0;
    for (int r = 0; r < k; ++r) mask |= size_t(1) << g.q[r];
    size_t off[64];
    for (int l = 0; l < M.dim; ++l) { size_t o = 0; for (int r = 0; r < k; ++r) if ((l >> r) & 1) o |= size_t(1) << g.q[r]; off[l] = o; }
    if (k == 1) {
        const size_t b = size_t(1) << g.q[0];
        if (M.mono && M.perm[0] == 0) {                     // diagonal
            const cd d0 = M.ph[0], d1 = M.ph[1];
            for (size_t i = 0; i < N; ++i) v[i] *= (i & b) ? d1 : d0;
        } else if (M.mono) {                                // X-like
            const cd p0 = M.ph[0], p1 = M.ph[1];            // column 0 -> row 1 with p0, column 1 -> row 0 with p1
            for (size_t i = 0; i < N; ++i) if (!(i & b)) { cd a = v[i], c = v[i | b]; v[i] = p1 * c; v[i | b] = p0 * a; }
        } else {
            const cd m00 = M.m[0], m01 = M.m[1], m10 = M.m[2], m11 = M.m[3];
            for (size_t i = 0; i < N; ++i) if (!(i & b)) { cd a = v[i], c = v[i | b]; v[i] = m00 * a + m01 * c; v[i | b] = m10 * a + m11 * c; }
        }
        return;
    }
    cd in[64], out[64];
    for (size_t i = 0; i < N; ++i) {
        if (i & mask) continue;
        for (int l = 0; l < M.dim; ++l) in[l] = v[i | off[l]];
        if (M.mono) {
            for (int l = 0; l < M.dim; ++l) out[M.perm[l]] = M.ph[l] * in[l];
        } else {
            for (int r = 0; r < M.dim; ++r) { cd s = 0; const cd* row = &M.m[size_t(r) * M.dim]; for (int l = 0; l < M.dim; ++l) s += row[l] * in[l]; out[r] = s; }
        }
        for (int l = 0; l < M.dim; ++l) v[i | off[l]] = out[l];
    }
}

static void column(const std::vector<Mat>& mats, const Circ& c, int n, size_t col, std::vector<cd>& v) {
    std::fill(v.begin(), v.end(), cd(0));
    v[col] = 1.0;
    for (const Gate& g : c.g) apply(mats, g, n, v);
    if (c.gp != 0.0) { const cd f = std::polar(1.0, c.gp); for (auto& x : v) x *= f; }
}

static bool read_circ(std::istream& in, const std::vector<Mat>& mats, Circ& c) {
    size_t ng; if (!(in >> ng >> c.gp)) return false;
    c.g.resize(ng);
    for (size_t i = 0; i < ng; ++i) {
        in >> c.g[i].mat; if (c.g[i].mat < 0 || size_t(c.g[i].mat) >= mats.size()) return false;
        for (int r = 0; r < mats[c.g[i].mat].k; ++r) in >> c.g[i].q[r];
    }
    return bool(in);
}
}  // namespace

int run_certify(std::istream& in) {
    int n; size_t C;
    if (!(in >> n >> C) || n < 1 || n > 24) { std::puts("ERROR header"); return 1; }
    std::vector<size_t> cols(C);
    for (auto& c : cols) in >> c;
    size_t M; in >> M;
    std::vector<Mat> mats(M);
    for (auto& m : mats) {
        in >> m.k; if (m.k < 1 || m.k > 6) { std::puts("ERROR k"); return 1; }
        m.dim = 1 << m.k; m.m.resize(size_t(m.dim) * m.dim);
        for (auto& x : m.m) { double re, im; in >> re >> im; x = cd(re, im); }
        m.perm.assign(m.dim, -1); m.ph.assign(m.dim, cd(0)); m.mono = true;
        for (int l = 0; l < m.dim && m.mono; ++l) {
            int cnt = 0;
            for (int r = 0; r < m.dim; ++r) if (std::abs(m.m[size_t(r) * m.dim + l]) > 1e-14) { ++cnt; m.perm[l] = r; m.ph[l] = m.m[size_t(r) * m.dim + l]; }
            if (cnt != 1) m.mono = false;
        }
    }
    Circ A, B;
    if (!read_circ(in, mats, A) || !read_circ(in, mats, B)) { std::puts("ERROR circuits"); return 1; }
    const size_t N = size_t(1) << n;
    std::vector<cd> a(N), b(N);
    column(mats, A, n, cols[0], a); column(mats, B, n, cols[0], b);
    cd ov = 0; for (size_t i = 0; i < N; ++i) ov += std::conj(a[i]) * b[i];
    const cd ph = std::abs(ov) > 1e-15 ? ov / std::abs(ov) : cd(1);                 // B_0 ~ ph * A_0
    unsigned T = 0;
    if (const char* e = qcpt::getenv_q("QCPT_NPROC")) T = unsigned(std::atoi(e));
    if (T == 0) T = std::min(32u, std::max(1u, std::thread::hardware_concurrency()));
    T = std::min<size_t>(T, C);
    const double njobs = double(C) * double(A.g.size() + B.g.size()) * double(N);          // amplitude updates
    T = unsigned(std::max<double>(1.0, std::min<double>(T, njobs / 4e6)));                  // small jobs: few threads (start-up cost)
    std::vector<double> fro2(T, 0.0), mx(T, 0.0);
    auto work = [&](unsigned t) {
        std::vector<cd> x(N), y(N);
        for (size_t j = t; j < C; j += T) {
            column(mats, A, n, cols[j], x); column(mats, B, n, cols[j], y);
            for (size_t i = 0; i < N; ++i) { double d = std::abs(y[i] - ph * x[i]); fro2[t] += d * d; if (d > mx[t]) mx[t] = d; }
        }
    };
    std::vector<std::thread> th;
    for (unsigned t = 1; t < T; ++t) th.emplace_back(work, t);
    work(0);
    for (auto& x : th) x.join();
    double f2 = 0, m = 0; for (unsigned t = 0; t < T; ++t) { f2 += fro2[t]; m = std::max(m, mx[t]); }
    std::printf("FRO %.17g MAX %.17g RE %.17g IM %.17g\n", std::sqrt(f2), m, ph.real(), ph.imag());
    return 0;
}


// qcpt certify_mc: probabilistic certificate with random vectors (Monte Carlo).
// Input (stdin, text):  n K seed S   (S input columns, 0 = all 2^n)  then the S columns, then M, the matrices, circuit A, circuit B
// exactly as for ``certify``. K vectors g_i with independent N(0,1)+iN(0,1)-over-sqrt(2) components on the support are propagated
// through A and B; with ph from g_0 the output is  "MC K RE IM" and K lines "T_i NORM2_i" where T_i = ||(B - ph A) g_i||^2.
#include <random>
namespace {
static void propagate(const std::vector<Mat>& mats, const Circ& c, int n, std::vector<cd>& v) {
    for (const Gate& g : c.g) apply(mats, g, n, v);
    if (c.gp != 0.0) { const cd f = std::polar(1.0, c.gp); for (auto& x : v) x *= f; }
}
}  // namespace

int run_certify_mc(std::istream& in) {
    int n; size_t K, S; unsigned long long seed;
    if (!(in >> n >> K >> seed >> S) || n < 1 || n > 26 || K < 2) { std::puts("ERROR header"); return 1; }
    const size_t N = size_t(1) << n;
    std::vector<size_t> cols(S);
    for (auto& c : cols) in >> c;
    size_t M; in >> M;
    std::vector<Mat> mats(M);
    for (auto& m : mats) {
        in >> m.k; if (m.k < 1 || m.k > 6) { std::puts("ERROR k"); return 1; }
        m.dim = 1 << m.k; m.m.resize(size_t(m.dim) * m.dim);
        for (auto& x : m.m) { double re, im; in >> re >> im; x = cd(re, im); }
        m.perm.assign(m.dim, -1); m.ph.assign(m.dim, cd(0)); m.mono = true;
        for (int l = 0; l < m.dim && m.mono; ++l) {
            int cnt = 0;
            for (int r = 0; r < m.dim; ++r) if (std::abs(m.m[size_t(r) * m.dim + l]) > 1e-14) { ++cnt; m.perm[l] = r; m.ph[l] = m.m[size_t(r) * m.dim + l]; }
            if (cnt != 1) m.mono = false;
        }
    }
    Circ A, B;
    if (!read_circ(in, mats, A) || !read_circ(in, mats, B)) { std::puts("ERROR circuits"); return 1; }
    const size_t supp = S ? S : N;
    auto draw = [&](size_t i, std::vector<cd>& v, double& norm2) {
        std::mt19937_64 rng(seed * 1000003ULL + i);
        std::normal_distribution<double> nd(0.0, std::sqrt(0.5));
        std::fill(v.begin(), v.end(), cd(0));
        norm2 = 0;
        for (size_t j = 0; j < supp; ++j) {
            const size_t idx = S ? cols[j] : j;
            const double re = nd(rng), im = nd(rng);
            v[idx] = cd(re, im); norm2 += re * re + im * im;
        }
    };
    std::vector<double> T(K, 0.0), NR(K, 0.0);
    std::vector<cd> a(N), b(N), g(N);
    draw(0, g, NR[0]); a = g; b = g;
    propagate(mats, A, n, a); propagate(mats, B, n, b);
    cd ov = 0; for (size_t i = 0; i < N; ++i) ov += std::conj(a[i]) * b[i];
    const cd ph = std::abs(ov) > 1e-15 ? ov / std::abs(ov) : cd(1);
    { double t = 0; for (size_t i = 0; i < N; ++i) { double d = std::abs(b[i] - ph * a[i]); t += d * d; } T[0] = t; }
    unsigned Th = 0;
    if (const char* e = qcpt::getenv_q("QCPT_NPROC")) Th = unsigned(std::atoi(e));
    if (Th == 0) Th = std::min(32u, std::max(1u, std::thread::hardware_concurrency()));
    Th = std::min<size_t>(Th, K - 1);
    auto work = [&](unsigned t) {
        std::vector<cd> x(N), y(N), v(N);
        for (size_t i = 1 + t; i < K; i += Th) {
            double nr; draw(i, v, nr); x = v; y = v;
            propagate(mats, A, n, x); propagate(mats, B, n, y);
            double s = 0; for (size_t q = 0; q < N; ++q) { double d = std::abs(y[q] - ph * x[q]); s += d * d; }
            T[i] = s; NR[i] = nr;
        }
    };
    std::vector<std::thread> th;
    for (unsigned t = 1; t < Th; ++t) th.emplace_back(work, t);
    work(0);
    for (auto& x : th) x.join();
    std::printf("MC %zu %.17g %.17g\n", K, ph.real(), ph.imag());
    for (size_t i = 0; i < K; ++i) std::printf("%.17g %.17g\n", T[i], NR[i]);
    return 0;
}

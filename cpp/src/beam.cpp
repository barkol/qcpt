#include "beam.hpp"
#include "pool.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <map>
#include <numeric>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace qcpt {
namespace {

using Clock = std::chrono::steady_clock;
double since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }


uint64_t mix(uint64_t z) { z += 0x9e3779b97f4a7c15ULL; z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL; z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL; return z ^ (z >> 31); }

// counter-based standard normal: depends only on (seed, gate, parent), so the result does not depend on threads or engine
double gauss(uint64_t seed, uint32_t gate, uint32_t parent) {
    const uint64_t a = mix(seed ^ (uint64_t)gate * 0xd1342543de82ef95ULL ^ ((uint64_t)parent << 20)), b = mix(a ^ 0x5851f42d4c957f2dULL);
    const double u1 = ((a >> 11) + 0.5) * (1.0 / 9007199254740992.0), u2 = (b >> 11) * (1.0 / 9007199254740992.0);
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
}

constexpr int kCd[8] = {0, 1, 2, 3, 4, 3, 2, 1};

// A library gate acts on at most four qubits, so its permutation and phase are tables of 16 entries over the local index; with n <= 8 and AVX2 a
// byte shuffle (vpshufb) applies them to 32 inputs at once.
struct LocalGate {
    int k = 0, sh[4] = {0, 0, 0, 0}; uint8_t keep = 0xFF; uint16_t keep16 = 0xFFFF; alignas(32) uint8_t tdep[32] = {0}, tph[32] = {0}, tdep_hi[32] = {0};
};

LocalGate make_local(const std::vector<uint16_t>& P, const std::vector<uint8_t>& F, size_t off, const std::array<int, 4>& q, int k, int N) {
    LocalGate g; if (k > 4) { g.k = -1; return g; } g.k = k; int mask = 0; for (int j = 0; j < k; ++j) { g.sh[j] = q[j] - j; mask |= 1 << q[j]; }
    g.keep = (uint8_t)(~mask & 0xFF); g.keep16 = (uint16_t)(~mask & 0xFFFF);
    for (int l = 0; l < (1 << k); ++l) { int y = 0; for (int j = 0; j < k; ++j) y |= ((l >> j) & 1) << q[j]; g.tdep[l] = g.tdep[16 + l] = (uint8_t)(P[off + y] & 0xFF); g.tdep_hi[l] = g.tdep_hi[16 + l] = (uint8_t)(P[off + y] >> 8); g.tph[l] = g.tph[16 + l] = F[off + y]; }
    for (int y = 0; y < N; ++y) {                       // locality check: the full table must agree with the local one
        int l = 0; for (int j = 0; j < k; ++j) l |= ((y >> q[j]) & 1) << j;
        if (P[off + y] != ((y & ~mask) | g.tdep[l] | (g.tdep_hi[l] << 8)) || F[off + y] != g.tph[l]) { g.k = -1; break; }
    }
    return g;
}

#ifdef __AVX2__
// h = sum_i popcount(out_i xor goal_i) + sum_i cd(phase_i - phase_0) for one (gate, parent), 32 inputs per step
inline int h_avx2(const LocalGate& g, const uint8_t* Y, const uint8_t* S, const uint8_t* goal, int mp, __m256i lastmask, int base) {
    const __m256i keep = _mm256_set1_epi8((char)g.keep), low = _mm256_set1_epi8(0x0f), c8 = _mm256_set1_epi8(8), c7 = _mm256_set1_epi8(7), vb = _mm256_set1_epi8((char)base);
    const __m256i lut = _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
    const __m256i tdep = _mm256_load_si256((const __m256i*)g.tdep), tph = _mm256_load_si256((const __m256i*)g.tph), zero = _mm256_setzero_si256();
    __m256i acc = zero;
    for (int b = 0; b < mp; b += 32) {
        const __m256i y = _mm256_loadu_si256((const __m256i*)(Y + b)); __m256i idx = zero;
        for (int j = 0; j < g.k; ++j) {
            const __m256i t = g.sh[j] >= 0 ? _mm256_srl_epi16(y, _mm_cvtsi32_si128(g.sh[j])) : _mm256_sll_epi16(y, _mm_cvtsi32_si128(-g.sh[j]));
            idx = _mm256_or_si256(idx, _mm256_and_si256(t, _mm256_set1_epi8((char)(1 << j))));
        }
        const __m256i ny = _mm256_or_si256(_mm256_and_si256(y, keep), _mm256_shuffle_epi8(tdep, idx)), ph = _mm256_shuffle_epi8(tph, idx);
        const __m256i x = _mm256_xor_si256(ny, _mm256_loadu_si256((const __m256i*)(goal + b)));
        const __m256i pc = _mm256_add_epi8(_mm256_shuffle_epi8(lut, _mm256_and_si256(x, low)), _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(x, 4), low)));
        const __m256i r = _mm256_and_si256(_mm256_sub_epi8(_mm256_add_epi8(_mm256_loadu_si256((const __m256i*)(S + b)), ph), vb), c7);
        __m256i v = _mm256_add_epi8(pc, _mm256_min_epu8(r, _mm256_sub_epi8(c8, r)));
        if (b + 32 >= mp) v = _mm256_and_si256(v, lastmask);
        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(v, zero));
    }
    alignas(32) uint64_t t[4]; _mm256_store_si256((__m256i*)t, acc); return (int)(t[0] + t[1] + t[2] + t[3]);
}
#endif

#ifdef __AVX2__
// the same heuristic for n = 9, 10: 16 inputs per vector, the permutation table split into a low and a high byte table
inline int h_avx2_16(const LocalGate& g, const uint16_t* Y, const uint8_t* S, const uint16_t* goal, int mp, __m256i lastmask, int base) {
    const __m256i keep = _mm256_set1_epi16((short)g.keep16), low = _mm256_set1_epi8(0x0f), c8 = _mm256_set1_epi16(8), c7 = _mm256_set1_epi16(7), vb = _mm256_set1_epi16((short)base), ff = _mm256_set1_epi16(0x00FF), ones = _mm256_set1_epi8(1);
    const __m256i lut = _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
    const __m256i tlo = _mm256_load_si256((const __m256i*)g.tdep), thi = _mm256_load_si256((const __m256i*)g.tdep_hi), tph = _mm256_load_si256((const __m256i*)g.tph);
    __m256i acc = _mm256_setzero_si256();
    for (int b = 0; b < mp; b += 16) {
        const __m256i y = _mm256_loadu_si256((const __m256i*)(Y + b)); __m256i idx = _mm256_setzero_si256();
        for (int j = 0; j < g.k; ++j) {
            const __m256i t = g.sh[j] >= 0 ? _mm256_srl_epi16(y, _mm_cvtsi32_si128(g.sh[j])) : _mm256_sll_epi16(y, _mm_cvtsi32_si128(-g.sh[j]));
            idx = _mm256_or_si256(idx, _mm256_and_si256(t, _mm256_set1_epi16((short)(1 << j))));
        }
        const __m256i dep = _mm256_or_si256(_mm256_and_si256(_mm256_shuffle_epi8(tlo, idx), ff), _mm256_slli_epi16(_mm256_shuffle_epi8(thi, idx), 8));
        const __m256i ny = _mm256_or_si256(_mm256_and_si256(y, keep), dep), ph = _mm256_and_si256(_mm256_shuffle_epi8(tph, idx), ff);
        const __m256i x = _mm256_xor_si256(ny, _mm256_loadu_si256((const __m256i*)(goal + b)));
        const __m256i pc8 = _mm256_add_epi8(_mm256_shuffle_epi8(lut, _mm256_and_si256(x, low)), _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(x, 4), low)));
        const __m256i pc = _mm256_maddubs_epi16(pc8, ones);
        const __m256i s = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(S + b)));
        const __m256i r = _mm256_and_si256(_mm256_sub_epi16(_mm256_add_epi16(s, ph), vb), c7);
        __m256i v = _mm256_add_epi16(pc, _mm256_min_epu16(r, _mm256_sub_epi16(c8, r)));
        if (b + 16 >= mp) v = _mm256_and_si256(v, lastmask);
        acc = _mm256_add_epi16(acc, v);
    }
    const __m256i s32 = _mm256_madd_epi16(acc, _mm256_set1_epi16(1)); alignas(32) int32_t t[8]; _mm256_store_si256((__m256i*)t, s32);
    return t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
}
#endif

// sort [first,last) on the shared pool: sorted chunks, then pairwise merges (flat divide and conquer, no recursion needed)
template <class It, class Cmp> void psort(int nt, It first, It last, Cmp cmp) {
    const size_t n = (size_t)(last - first), CH = 16384;
    if (nt <= 1 || n <= 2 * CH) { std::sort(first, last, cmp); return; }
    parallel_for(nt, (n + CH - 1) / CH, [&](size_t c) { std::sort(first + c * CH, first + std::min(n, (c + 1) * CH), cmp); });
    for (size_t w = CH; w < n; w *= 2)
        parallel_for(nt, (n + 2 * w - 1) / (2 * w), [&](size_t q) { const size_t a = q * 2 * w, m = std::min(n, a + w), b = std::min(n, a + 2 * w); if (m < b) std::inplace_merge(first + a, first + m, first + b, cmp); });
}

class CpuEngine : public Engine {
public:
    explicit CpuEngine(int nthreads) : nt_(std::max(1, nthreads)) {}

    void init(int n, int m, const std::vector<int>& cost, const std::vector<uint64_t>& R1, const std::vector<uint64_t>& R2) override {
        n_ = n; N_ = 1 << n; m_ = m; mp_ = (m + 31) / 32 * 32; Gn_ = (int)cost.size(); cost_ = cost; R1_ = R1; R2_ = R2;
#ifdef __AVX2__
        fast_ = N_ <= 1024 && !getenv_q("QCPT_NO_SIMD"); wide_ = N_ > 256;
#endif
    }
    void reset_side(int s, const SideSpec& sp) override {
        Side& d = sd_[s]; d.perm = sp.perm; d.phase = sp.phase; d.goal = sp.goal; d.Bn = 1;
        d.Y.assign(mp_, 0); d.S.assign(mp_, 0); std::copy(sp.start.begin(), sp.start.end(), d.Y.begin()); std::copy(sp.start_phase.begin(), sp.start_phase.end(), d.S.begin());
        d.Y8.assign(mp_, 0); d.goal8.assign(mp_, 0); d.goal16.assign(mp_, 0); for (int i = 0; i < m_; ++i) d.goal16[i] = d.goal[i]; for (int i = 0; i < m_; ++i) { d.Y8[i] = (uint8_t)d.Y[i]; d.goal8[i] = (uint8_t)d.goal[i]; }
        d.lg.clear();
        if (fast_) for (int g = 0; g < Gn_ && fast_; ++g) { d.lg.push_back(make_local(*d.perm, *d.phase, (size_t)g * N_, (*sp.support)[g], (*sp.nsupport)[g], N_)); if (d.lg.back().k < 0) fast_ = false; }
    }
    int size(int s) const override { return sd_[s].Bn; }
    uint64_t root_hash(int s) const override { return hash_row(sd_[s].Y.data(), sd_[s].S.data()); }

    size_t candidates(int s, const std::vector<int32_t>& g, int lim, int beam, int L, double noise, uint64_t nseed) override {
        const Side& d = sd_[s]; const int Bn = d.Bn, ngr = (Gn_ + kGroup - 1) / kGroup; const size_t k = std::min<size_t>((size_t)beam * 2, (size_t)kGroup * Bn - 1);
        std::vector<std::vector<uint64_t>> keys(ngr);
#ifdef __AVX2__
        alignas(32) uint8_t lm[32]; for (int j = 0; j < 32; ++j) lm[j] = (uint8_t)((mp_ - 32 + j) < m_ ? 0xFF : 0); const __m256i lastmask = _mm256_load_si256((const __m256i*)lm);
        alignas(32) uint16_t lm16[16]; for (int j = 0; j < 16; ++j) lm16[j] = (uint16_t)((mp_ - 16 + j) < m_ ? 0xFFFF : 0); const __m256i lastmask16 = _mm256_load_si256((const __m256i*)lm16);
#endif
        // phase 1: score every (gate, parent) pair of a batch of groups (tasks = group x chunk of parents) into a dense int32 array per group, INT32_MAX = pruned
        const size_t GB = (size_t)kGroup * Bn, CP = 1024, nchunk = (Bn + CP - 1) / CP; const int hw = (int)std::max(1u, std::thread::hardware_concurrency()); const int bgroups = nt_ <= 1 ? 2 : (int)std::max<size_t>(4, std::min<size_t>((size_t)2 * hw, ((size_t)256 << 20) / (GB * 4)));   // batch of groups: all cores, at most 256 MB of scores
        std::vector<std::vector<int32_t>> F(bgroups);
        for (int gb0 = 0; gb0 < ngr; gb0 += bgroups) {
            const int nb = std::min(bgroups, ngr - gb0);
            parallel_for(nt_, (size_t)nb, [&](size_t b) { F[b].assign(GB, INT32_MAX); });
            parallel_for(nt_, (size_t)nb * nchunk, [&](size_t task) {
                const int b = (int)(task / nchunk); const size_t c = task % nchunk; const int g0 = (gb0 + b) * kGroup, g1 = std::min(Gn_, g0 + kGroup); int32_t* Fo = F[b].data();
                for (int gate = g0; gate < g1; ++gate) {
                    const uint16_t* P = &(*d.perm)[(size_t)gate * N_]; const uint8_t* Fp = &(*d.phase)[(size_t)gate * N_];
                    for (size_t p = c * CP; p < std::min<size_t>(Bn, (c + 1) * CP); ++p) {
                        const int gc = g[p] + cost_[gate]; if (gc > lim) continue;
                        const uint16_t* Y = &d.Y[p * mp_]; const uint8_t* S = &d.S[p * mp_];
                        const int base = (S[0] + Fp[Y[0]]) & 7; int h = 0;
#ifdef __AVX2__
                        if (fast_) h = wide_ ? h_avx2_16(d.lg[gate], Y, S, d.goal16.data(), mp_, lastmask16, base) : h_avx2(d.lg[gate], &d.Y8[p * mp_], S, d.goal8.data(), mp_, lastmask, base); else
#endif
                        for (int i = 0; i < m_; ++i) { const int y = Y[i]; h += __builtin_popcount(P[y] ^ d.goal[i]) + kCd[(S[i] + Fp[y] - base) & 7]; }
                        long f = (long)gc * 1000 + (long)L * h;
                        if (noise > 0) f += std::lround(noise * 1000.0 * gauss(nseed, gate, (uint32_t)p));
                        Fo[(size_t)(gate - g0) * Bn + p] = (int32_t)std::max(f, 0L);
                    }
                }
            });
            // phase 2: per group the k best by (f, local index): threshold from a sample, exact selection among the entries below it
            parallel_for(nt_, nb, [&](size_t b) {
                const int32_t* Fo = F[b].data(); auto& K = keys[gb0 + b]; size_t valid = 0; for (size_t i = 0; i < GB; ++i) valid += Fo[i] != INT32_MAX;
                int32_t cut = INT32_MAX;
                if (valid > k) {
                    const size_t stride = std::max<size_t>(1, GB / 16384); std::vector<int32_t> smp; for (size_t i = 0; i < GB; i += stride) smp.push_back(Fo[i]);
                    const size_t r = std::min(smp.size() - 1, (size_t)((double)k / GB * smp.size() * 1.5) + 16); std::nth_element(smp.begin(), smp.begin() + r, smp.end()); cut = smp[r];
                }
                for (int pass = 0; pass < 2; ++pass) {
                    K.clear(); for (size_t i = 0; i < GB; ++i) if (Fo[i] != INT32_MAX && Fo[i] <= cut) K.push_back(((uint64_t)Fo[i] << 32) | i);
                    if (K.size() >= std::min(k, valid) || cut == INT32_MAX) break; cut = INT32_MAX;
                }
                if (K.size() > k) { std::nth_element(K.begin(), K.begin() + k, K.end()); K.resize(k); }
            });
        }
        size_t T = 0; for (auto& K : keys) T += K.size();
        all_.assign(T, E{}); std::vector<size_t> off(ngr + 1, 0); for (int gr = 0; gr < ngr; ++gr) off[gr + 1] = off[gr] + keys[gr].size();
        parallel_for(nt_, ngr, [&](size_t gr) { size_t o = off[gr]; for (uint64_t key : keys[gr]) { const size_t loc = key & 0xffffffffu; all_[o++] = {(int32_t)(key >> 32), (int32_t)(gr * kGroup + loc / Bn), (int32_t)(loc % Bn)}; } });
        g_ = g; side_ = s; sorted_ = 0; cache_ = Pool{}; return T;
    }

    Pool take(size_t count) override {
        const Side& d = sd_[side_]; count = std::min(count, all_.size());
        auto less = [](const E& a, const E& b) { return std::tie(a.f, a.gate, a.parent) < std::tie(b.f, b.gate, b.parent); };
        if (count > sorted_) {                                     // the next smallest entries; everything before them stays where it is
            const size_t lo = sorted_;
            if (count < all_.size()) std::nth_element(all_.begin() + lo, all_.begin() + count, all_.end(), less);
            psort(nt_, all_.begin() + lo, all_.begin() + count, less);
            cache_.f.resize(count); cache_.g.resize(count); cache_.gate.resize(count); cache_.parent.resize(count); cache_.hash.resize(count);
            parallel_for(nt_, (count - lo + 4095) / 4096, [&](size_t c) {
                std::vector<uint16_t> Yc(m_); std::vector<uint8_t> Sc(m_);
                for (size_t t = lo + c * 4096; t < std::min(count, lo + (c + 1) * 4096); ++t) {
                    const E& e = all_[t]; cache_.f[t] = e.f; cache_.gate[t] = e.gate; cache_.parent[t] = e.parent; cache_.g[t] = g_[e.parent] + cost_[e.gate];
                    child(d, e.gate, e.parent, Yc.data(), Sc.data()); cache_.hash[t] = hash_row(Yc.data(), Sc.data());
                }
            });
            sorted_ = count;
        }
        Pool out; out.f.assign(cache_.f.begin(), cache_.f.begin() + count); out.g.assign(cache_.g.begin(), cache_.g.begin() + count); out.gate.assign(cache_.gate.begin(), cache_.gate.begin() + count);
        out.parent.assign(cache_.parent.begin(), cache_.parent.begin() + count); out.hash.assign(cache_.hash.begin(), cache_.hash.begin() + count); return out;
    }

    void advance(int s, const std::vector<int32_t>& gate, const std::vector<int32_t>& parent) override {
        Side& d = sd_[s]; const size_t B = gate.size(); std::vector<uint16_t> Y(B * mp_, 0); std::vector<uint8_t> S(B * mp_, 0), Y8(B * mp_, 0);
        parallel_for(nt_, (B + 1023) / 1024, [&](size_t c) { for (size_t t = c * 1024; t < std::min(B, (c + 1) * 1024); ++t) { child(d, gate[t], parent[t], &Y[t * mp_], &S[t * mp_]); for (int i = 0; i < m_; ++i) Y8[t * mp_ + i] = (uint8_t)Y[t * mp_ + i]; } });
        d.Y.swap(Y); d.S.swap(S); d.Y8.swap(Y8); d.Bn = (int)B;
    }

private:
    struct Side { const std::vector<uint16_t>* perm = nullptr; const std::vector<uint8_t>* phase = nullptr; std::vector<uint16_t> goal, Y; std::vector<uint8_t> S, Y8, goal8; std::vector<uint16_t> goal16; std::vector<LocalGate> lg; int Bn = 0; };
    struct E { int32_t f, gate, parent; }; std::vector<E> all_; Pool cache_; size_t sorted_ = 0; std::vector<int32_t> g_; int side_ = 0;
    bool fast_ = false, wide_ = false; int nt_, n_ = 0, N_ = 0, m_ = 0, mp_ = 0, Gn_ = 0; std::vector<int> cost_; std::vector<uint64_t> R1_, R2_; Side sd_[2];

    uint64_t hash_row(const uint16_t* Y, const uint8_t* S) const { uint64_t h = 0; const int s0 = S[0]; for (int i = 0; i < m_; ++i) h += (uint64_t)Y[i] * R1_[i] + (uint64_t)((S[i] - s0) & 7) * R2_[i]; return h; }
    void child(const Side& d, int gate, int p, uint16_t* Yo, uint8_t* So) const {
        const uint16_t* P = &(*d.perm)[(size_t)gate * N_]; const uint8_t* F = &(*d.phase)[(size_t)gate * N_]; const uint16_t* Y = &d.Y[(size_t)p * mp_]; const uint8_t* S = &d.S[(size_t)p * mp_];
        for (int i = 0; i < m_; ++i) { Yo[i] = P[Y[i]]; So[i] = (uint8_t)((S[i] + F[Y[i]]) & 7); }
    }
};

struct HostSide {
    std::vector<int32_t> g; std::vector<std::vector<int32_t>> lpar, lgate;
    std::unordered_map<uint64_t, int> seen; std::unordered_map<uint64_t, std::tuple<int, int, int>> tab;
};

}  // namespace

std::unique_ptr<Engine> make_cpu_engine(int nthreads) { return std::make_unique<CpuEngine>(nthreads); }

BeamResult bidir_search(const BeamParams& P, const Library& lib) {
    current_cancel() = P.cancel; const auto t0 = Clock::now(); BeamResult res; const int n = P.n, N = 1 << n, m = (int)P.inputs.size(), Gn = (int)lib.size();
    std::vector<uint16_t> fp((size_t)Gn * N), ip((size_t)Gn * N); std::vector<uint8_t> ff((size_t)Gn * N), iff((size_t)Gn * N); std::vector<int> cost(Gn);
    for (int g = 0; g < Gn; ++g) { cost[g] = lib[g].cost; std::copy(lib[g].p.begin(), lib[g].p.end(), fp.begin() + (size_t)g * N); std::copy(lib[g].ip.begin(), lib[g].ip.end(), ip.begin() + (size_t)g * N); std::copy(lib[g].f.begin(), lib[g].f.end(), ff.begin() + (size_t)g * N); std::copy(lib[g].inf.begin(), lib[g].inf.end(), iff.begin() + (size_t)g * N); }
    uint64_t sm = P.seed * 0x9e3779b97f4a7c15ULL + 12345; std::vector<uint64_t> R1(m), R2(m); for (int i = 0; i < m; ++i) { R1[i] = mix(sm++) | 1; R2[i] = mix(sm++) | 1; }
    std::unique_ptr<Engine> eng;
#ifdef QCPT_CUDA
    if (P.gpu) eng = make_gpu_engine();
#endif
    if (!eng) eng = make_cpu_engine(P.nthreads);
    eng->init(n, m, cost, R1, R2);
    std::vector<std::array<int, 4>> sup(Gn); std::vector<int> nsup(Gn);
    for (int g = 0; g < Gn; ++g) { nsup[g] = std::min(4, lib[g].nargs); sup[g] = {0, 0, 0, 0}; for (int j = 0; j < nsup[g]; ++j) sup[g][j] = lib[g].args[j]; if (lib[g].nargs > 4) nsup[g] = 9; }
    SideSpec sf, sb; sf.support = sb.support = &sup; sf.nsupport = sb.nsupport = &nsup; sf.perm = &fp; sf.phase = &ff; sb.perm = &ip; sb.phase = &iff;
    for (int i = 0; i < m; ++i) { sf.start.push_back((uint16_t)P.inputs[i]); sf.goal.push_back((uint16_t)P.target[i]); sb.start.push_back((uint16_t)P.target[i]); sb.goal.push_back((uint16_t)P.inputs[i]); }
    sf.start_phase.assign(m, 0); sb.start_phase.assign(m, 0); if (!P.target_phase.empty()) for (int i = 0; i < m; ++i) sb.start_phase[i] = (uint8_t)(P.target_phase[i] & 7);
    eng->reset_side(0, sf); eng->reset_side(1, sb);
    const uint64_t hF = eng->root_hash(0), hB = eng->root_hash(1); if (hF == hB) { res.found = true; return res; }
    HostSide H[2]; for (int s = 0; s < 2; ++s) { H[s].g = {0}; H[s].lpar.push_back({-1}); H[s].lgate.push_back({-1}); const uint64_t h = s ? hB : hF; H[s].tab[h] = {0, 0, 0}; H[s].seen[h] = 0; }
    int best = P.max_cost + 1; int bsd = -1, bdepth = 0, bi = 0, bod = 0, boi = 0;
    const int L = (int)std::lround(P.lam * 1000), beam = P.beam; const double div = P.diversity;

    auto expand = [&](int s, int depth) -> bool {
        HostSide& hs = H[s]; const int lim = best - 1; const auto tc0 = Clock::now();
        const size_t avail = eng->candidates(s, hs.g, lim, beam, L, P.noise, P.seed * 1000003ULL + depth * 131ULL + (s ? 7 : 0));
        if (avail == 0) return false; const double t_cand = since(tc0); const auto td0 = Clock::now();
        size_t total = avail; if (div > 0) total = std::min<size_t>(total, (size_t)(beam * (4 + 4 * div)));
        const size_t blk = std::max<size_t>(4 * (size_t)beam, 4096); Pool pool;
        std::vector<size_t> uniq; std::unordered_set<uint64_t> seen; size_t pos = 0, want = div > 0 ? total : std::min(total, blk);
        while (pos < total) {
            pool = eng->take(want);                                   // prefix of the sorted pool; a larger prefix extends the same order
            std::vector<uint8_t> dup(want - pos);                      // already reached in an earlier layer of this side: independent per entry, so checked in parallel
            parallel_for(P.nthreads, (want - pos + 8191) / 8192, [&](size_t c) {
                for (size_t i = pos + c * 8192; i < std::min(want, pos + (c + 1) * 8192); ++i) { auto it = hs.seen.find(pool.hash[i]); dup[i - pos] = it != hs.seen.end() && it->second <= pool.g[i]; }
            });
            for (size_t i = pos; i < want; ++i) {
                if (dup[i - pos]) continue; const uint64_t h = pool.hash[i]; if (seen.count(h)) continue;
                seen.insert(h); uniq.push_back(i);
            }
            pos = want; if (div == 0 && (int)uniq.size() >= beam) break;
            want = std::min(total, want * 2);
        }
        const size_t n1 = div > 0 ? (size_t)(beam - (int)(div * beam)) : (size_t)beam;
        std::vector<size_t> keep(uniq.begin(), uniq.begin() + std::min(uniq.size(), n1));
        if (div > 0 && uniq.size() > n1) {
            std::map<int, std::vector<size_t>> buckets; for (size_t k = n1; k < uniq.size(); ++k) buckets[pool.g[uniq[k]]].push_back(uniq[k]);
            std::vector<int> levels; for (auto& kv : buckets) levels.push_back(kv.first); std::map<int, size_t> ptr; for (int lv : levels) ptr[lv] = 0;
            while ((int)keep.size() < beam && !levels.empty()) {
                for (size_t li = 0; li < levels.size();) {
                    const int lv = levels[li];
                    if (ptr[lv] < buckets[lv].size()) { keep.push_back(buckets[lv][ptr[lv]++]); ++li; } else levels.erase(levels.begin() + li);
                    if ((int)keep.size() >= beam) break;
                }
            }
        }
        if (keep.empty()) return false; const double t_dedupe = since(td0); const auto ta0 = Clock::now();
        std::vector<int32_t> gate, parent, g; for (size_t k : keep) { gate.push_back(pool.gate[k]); parent.push_back(pool.parent[k]); g.push_back(pool.g[k]); }
        eng->advance(s, gate, parent); hs.g = g; hs.lpar.push_back(parent); hs.lgate.push_back(gate);
        hs.seen.reserve(hs.seen.size() + keep.size()); hs.tab.reserve(hs.tab.size() + keep.size());
        for (size_t t = 0; t < keep.size(); ++t) { const uint64_t h = pool.hash[keep[t]]; auto it = hs.seen.find(h); if (it == hs.seen.end() || g[t] < it->second) hs.seen[h] = g[t]; }
        const HostSide& ot = H[1 - s];
        std::vector<std::array<int, 3>> mt(keep.size(), std::array<int, 3>{INT_MAX, 0, 0});          // meeting with the other side: read-only lookups, in parallel
        parallel_for(P.nthreads, (keep.size() + 8191) / 8192, [&](size_t c) {
            for (size_t t = c * 8192; t < std::min(keep.size(), (c + 1) * 8192); ++t) { auto it = ot.tab.find(pool.hash[keep[t]]); if (it != ot.tab.end()) mt[t] = {g[t] + std::get<2>(it->second), std::get<0>(it->second), std::get<1>(it->second)}; }
        });
        for (size_t t = 0; t < keep.size(); ++t) if (mt[t][0] < best) { best = mt[t][0]; bsd = s; bdepth = depth; bi = (int)t; bod = mt[t][1]; boi = mt[t][2]; }
        for (size_t t = 0; t < keep.size(); ++t) hs.tab.emplace(pool.hash[keep[t]], std::make_tuple(depth, (int)t, g[t]));
        if (P.verbose) std::fprintf(stderr, "    side %d: candidates %.3f s (pool %zu), dedupe %.3f s, advance+meet %.3f s\n", s, t_cand, avail, t_dedupe, since(ta0));
        return true;
    };

    for (int d = 1; d <= P.depth_side; ++d) {
        if (cancelled()) return res;
        const bool okF = expand(0, d), okB = expand(1, d);
        if (cancelled()) return res;
        if (P.verbose) std::fprintf(stderr, "  level %d: meeting cost %d, beams %d / %d (%.2f s)\n", d, bsd >= 0 ? best : -1, eng->size(0), eng->size(1), since(t0));
        if (!(okF || okB)) break;
    }
    if (bsd < 0) return res;
    auto path = [&](int s, int depth, int idx) { std::vector<int> pth; while (depth > 0) { pth.push_back(H[s].lgate[depth][idx]); idx = H[s].lpar[depth][idx]; --depth; } std::reverse(pth.begin(), pth.end()); return pth; };
    const int other = 1 - bsd; auto pa = path(bsd, bdepth, bi), pb = path(other, bod, boi);
    auto& fpath = bsd == 0 ? pa : pb; auto& bpath = bsd == 0 ? pb : pa;
    res.found = true; res.cost = best; res.gates = fpath; res.gates.insert(res.gates.end(), bpath.rbegin(), bpath.rend());
    return res;
}

}  // namespace qcpt

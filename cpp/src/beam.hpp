// Bidirectional beam search in the Z8-monomial state space (port of resynth.py::bidir_search).
//
// The host (beam.cpp) owns the search logic: levels, deduplication, exploration slots, meeting of the two sides, path reconstruction.
// An Engine owns the states (rows of m outputs and m phases per beam member) and does the arithmetic: candidate scoring, hashing, materialising survivors.
// Engines: CPU (threads) and, when built with CUDA=1, GPU.
#pragma once
#include "env.hpp"
#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "library.hpp"

namespace qcpt {

struct BeamParams {
    std::vector<int> allow;                                // indices (in the order of make_library(n, false)) of the gates the search may use; empty = all (coupler filter)
    int n = 0, max_cost = 0, beam = 6000, depth_side = 8, nthreads = 1;
    double lam = 0.6, diversity = 0.0, noise = 0.0;
    uint64_t seed = 0;
    bool gpu = false, verbose = false;
    std::atomic<bool>* cancel = nullptr;                  // set by another thread to stop the search (result: not found)
    std::vector<int> inputs, target, target_phase;       // target_phase empty = zeros
};

struct BeamResult {
    bool found = false;
    int cost = 0;
    std::vector<int> gates;                               // indices into the library, in circuit order
};

BeamResult bidir_search(const BeamParams& P, const Library& lib);

// ---- engine interface (internal, but shared with the CUDA backend)

struct Pool {                                             // scored candidates, sorted by (f, gate, parent)
    std::vector<int32_t> f, g, gate, parent;
    std::vector<uint64_t> hash;                           // canonical hash of the child state
};

struct SideSpec {                                         // one side of the search
    const std::vector<uint16_t>* perm;                    // Gn x N, gate applied to a state before it
    const std::vector<uint8_t>* phase;                    // Gn x N
    std::vector<uint16_t> start, goal;                    // m values each
    std::vector<uint8_t> start_phase;                     // m values
    const std::vector<std::array<int, 4>>* support;       // qubits each gate acts on (local qubit j -> support[g][j])
    const std::vector<int>* nsupport;                     // number of local qubits of each gate (<= 4)
};

class Engine {
public:
    virtual ~Engine() = default;
    virtual void init(int n, int m, const std::vector<int>& cost, const std::vector<uint64_t>& R1, const std::vector<uint64_t>& R2) = 0;
    virtual void reset_side(int side, const SideSpec& s) = 0;                       // beam of one member: the start state
    virtual int size(int side) const = 0;
    virtual uint64_t root_hash(int side) const = 0;
    // scores the children of the current beam of `side` (top 2*beam per group of `kGroup` gates by f = 1000*g + L*h (+ noise), g <= lim)
    // and keeps them; returns how many there are
    virtual size_t candidates(int side, const std::vector<int32_t>& g, int lim, int beam, int L, double noise, uint64_t noise_seed) = 0;
    // the `count` best of them, sorted by (f, gate, parent), with the hash of every child state
    virtual Pool take(size_t count) = 0;
    // replaces the beam of `side` by the given children
    virtual void advance(int side, const std::vector<int32_t>& gate, const std::vector<int32_t>& parent) = 0;
};

std::unique_ptr<Engine> make_cpu_engine(int nthreads);
#ifdef QCPT_CUDA
std::unique_ptr<Engine> make_gpu_engine();
#endif

inline constexpr int kGroup = 32;                         // library gates per candidate group (the pool is the union of per-group top lists)

}  // namespace qcpt

# QCPT search core: design

Status: design. The Python implementation is kept in `toffoli_optimizer/`; the C++ template
search (`cpp/`) already replaces its counterpart and is the first component of the new core.

## 1. Goals and constraints

1. Same results as the archived implementation: identical costs on the 33 circuits, every returned circuit certified exactly (column check on the valid inputs).
2. Fastest time to the best circuit; energy follows from time, so no separate target.
3. Minimal dependencies: C++17 standard library and pthreads for the CPU engine; CUDA (nvcc, no torch, no cuBLAS) only for the optional GPU backend.
   Python stays a thin layer (Qiskit I/O, stage 1, certificates, plots) behind a C ABI called through `ctypes` (no pybind11, no Cython).
4. One process, one thread pool, one scheduler. No process groups, no `fork` pools, no per-round interpreter start.

## 2. Where the time goes today (measured, controlled_adder_3b, beam 7500, 1900 gates, 128 inputs)

| part | share of a beam run |
|---|---|
| GPU candidate kernel (`_gpu_cands`, 14 calls, 0.58 s each) | 79 % |
| Python deduplication of candidates (`expand`) | 11 % |
| hashing (`canon`) | 6 % |
| rest | 4 % |

One expansion evaluates Bn x Gn x m = 1.8e9 table lookups in 0.58 s, i.e. 3e9 per second, about 150 GB/s of memory traffic on a card with 448 GB/s peak.
The kernel is built from torch ops (gather, add, xor, two table lookups, two reductions), each of which writes a 4 or 8 byte temporary of size Bn x Gn x m to memory.
On the CPU path (windows, noisy members) 86 % of the time is the same computation in numpy.
Per race the portfolio also starts eight Python processes, each importing torch and rebuilding the library, in every round.

## 3. Architecture

```
cpp/
  include/qcpt/   monomial.hpp  library.hpp  hash.hpp  beam.hpp  template.hpp  scheduler.hpp  certify.hpp  capi.h
  src/           library.cpp  beam_cpu.cpp  template.cpp  scheduler.cpp  certify.cpp  capi.cpp  main.cpp
  cuda/          beam_kernel.cu  (optional, built with `make CUDA=1`)
  data/          db3.tab  (generated)      tools/  export_tables.py  (generated tables)
  tests/         golden tests against the archived Python results
python/qcpt/      stage1.py (selector, certificates)  emit.py (gates -> Qiskit)  engine.py (ctypes wrapper)
```

### 3.1 Data model
- State of one search node: `m` bytes of output (values < 2^n, n <= 8; `uint16` for n <= 16) and `m` bytes of phase (Z8), `m` = number of valid inputs.
  Rows are stored structure-of-arrays in one arena per side, no per-node allocation.
- Gate tables: `perm[Gn][N]` (uint8) and `phase[Gn][N]` (uint8), built once per (n, library kind) and shared read-only by all members.
- Canonical hash modulo the global phase, 64 bit, computed in the same pass that produces the child state.

### 3.2 Beam kernel (the main gain)
Fused: for a block of parents and a block of gates, compute for every (gate, parent) pair the child f = g + cost + lambda * h(child) without writing the child:
`h = sum_i popcount(P[g][Y_i] xor goal_i) + sum_i cd[(S_i + F[g][Y_i] - S_0 - F[g][Y_0]) & 7]`.
- Gate tables live in shared memory (256 B + 256 B per gate), parent rows stream from L2, the heuristic accumulates in registers; **no intermediate arrays**.
  Memory traffic drops from about 50 B to about 1 B per (gate, parent, input) triple; the kernel becomes compute-bound.
  Expected 8 to 15x over the torch version (3e9 -> 3e10 lookups/s); measured before it is accepted.
- Selection without a global sort: each block keeps its best 2 x beam candidates with a threshold (running k-th value, then a radix select on the survivors);
  only the survivors (a few times beam) leave the device, as (f, g, parent, gate).
- Only the survivors are materialised and hashed (one more small kernel), then deduplicated on the device with a sort on (hash, f) and a unique pass (CUB).
  This removes the Python dedupe loop (11 %) and the `canon` cost (6 %); the dedupe semantics (first `beam` unique candidates in order of f, no cap, optional exploration slots) are reproduced exactly.
- CPU backend of the same kernel: threads over gate blocks, 16-lane `pshufb` or scalar gathers, same selection code. It serves windows (n <= 4), noisy members and machines without a GPU, so the GPU is optional and the results are the same.
- Noise (stochastic ranking) is applied inside the kernel with a counter-based generator (Philox), so noisy members no longer force the slow path.

### 3.3 Template search
Already in C++ (`cpp/src/template_search.cpp`), 10 to 100x faster than numpy. Remaining: forward/backward state generation and hashing into the same arenas, and
sharing the forward table between the A and C configurations of one circuit.

### 3.4 Scheduler (replaces `portfolio.py`)
- One process; members are tasks on a thread pool plus one CUDA stream per GPU member; cancellation by an atomic flag checked between kernels and every few thousand nodes.
- All members start at once. Core budget: GPU members need one thread each, CPU members share the rest; a finished member releases its threads immediately.
  This removes the process oversubscription (up to 64 workers on 28 cores) and the second wave with its delay.
- Library and tables are built once per circuit, not once per member and round; no interpreter or torch start per round.
- First success lowers the target to cost - 1 for all running members without restarting them (a member only needs the new bound for pruning);
  members that cannot use the new bound (finished or hopeless) are dropped. The confirming round is optional (`--confirm`) and is not counted in time to best.
- Windowed resynthesis: all windows of a pass go into one batch (one launch per search depth), not one process each.

### 3.5 Certificates
The classical-with-phases check (permutation equality and Z8 phase equality up to one global phase on the valid inputs) is a table comparison; implement it in C++ (`certify.cpp`)
so that every circuit leaves the engine already checked. The Qiskit column check stays in Python as the independent second check used by the paper.

### 3.6 Interfaces
`capi.h`: `qcpt_library`, `qcpt_beam`, `qcpt_template`, `qcpt_race`, `qcpt_certify` with plain C structs; `python/qcpt/engine.py` wraps them with `ctypes`.
`qcpt` the executable offers the same functions on stdin/stdout for scripts and tests.

## 4. Test and acceptance
1. Golden tests: library tables equal to the archived Python library for n = 3..8 (done for 4, 5, 7); template results equal; beam costs equal on the 33 circuits at the archived settings.
2. Every returned circuit passes `certify` and the Qiskit column check (deviation < 1e-12).
3. Performance: beam expansion >= 8x faster than `_gpu_cands` on the same tensors; time to best on the nine circuits (mean 69 s, median 17.8 s today) reduced; reported with the same metering (`race_energy`).
4. Reproducibility: fixed seeds give identical results across thread counts (ties broken by (f, hash, index), not by thread order).

## 5. Phases
| phase | content | exit criterion |
|---|---|---|
| 0 | archive (done), template search in C++ (done) | nine-circuit race 69 s mean to best |
| 1 | `certify`, arenas, hash in C++; CPU beam (windows, noisy members) | windows and B4 without Python/numpy; costs equal |
| 2 | CUDA beam kernel + device dedupe | >= 8x on the expansion; costs equal on 33 circuits |
| 3 | scheduler in C++, C API, Python wrapper | race without Python processes; torch no longer a dependency |
| 4 | clean-up: legacy modules out of the package, `SearchConfig` in place of environment variables | package: stage 1 + wrapper only |

## 6. Risks
- Dedupe and tie-breaking must reproduce the archived order or results may differ by a gate on some circuits; mitigated by the golden tests, differences are reported, not hidden.
- The fused kernel is compute-bound only if tables fit shared memory: N = 256 states x 2 tables x 1 B x 32 gates per block = 16 KB, fine up to n = 8; larger n falls back to global memory.
- Selection by threshold needs a second pass if too few candidates survive; handled with a bounded retry (the first pass uses a generous bound).

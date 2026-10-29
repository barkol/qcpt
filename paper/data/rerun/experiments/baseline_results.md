# Table II baselines — error-budget comparison

Every method takes the same CCX/MCX benchmark network (12 circuits from `benchmarks.benchmark_suite()`), lowers it to a two-qubit basis (CX + 1q), and is scored with the IDENTICAL metric used by our gated selector: QCPT's `HardwareErrorModel` (p2q=0.01, p1q=0.001) for two-qubit infidelity / two-qubit count, and `ExactEquivalenceVerifier` (up to ancilla + output permutation) for correctness. Infidelity and 2q-count are SUMMED over the suite; depth is reported as total and mean; ancilla is the total extra qubits added.

## Per-method totals

| Method | 2q-infidelity (sum) | 2q-count (sum) | Depth (total) | Depth (mean) | 2q-depth (total) | Ancilla added | Verified |
|---|---|---|---|---|---|---|---|
| Qiskit opt-3 | 2.6904 | 281 | 439 | 36.58 | 247 | 0 | 12/12 |
| tket (FullPeephole) | 2.6602 | 279 | 448 | 37.33 | 246 | 0 | 12/12 |
| Exact-only (ours, all-exact) | 2.7437 | 281 | 477 | 39.75 | 247 | 0 | 12/12 |
| **Ours (gated)** | 1.9722 | 194 | 362 | 30.17 | 177 | 0 | 12/12 |
| QContext (arXiv:2302.02003) | — | — | — | — | — | — | not run |

> **QContext** has no usable public implementation. We cite its reported approach: it optimizes two-qubit **gate count** via context-aware synthesis but provides neither a hardware-fidelity (error-budget) model nor an exact equivalence certificate, so it is not directly comparable on the infidelity / verified columns. We therefore leave its row not-run rather than fabricate numbers.

## Relative reduction of Ours (gated) vs each baseline

| vs Method | their 2q-infid | our 2q-infid | infid reduction | their 2q | our 2q | 2q reduction |
|---|---|---|---|---|---|---|
| Qiskit opt-3 | 2.6904 | 1.9722 | 26.7% | 281 | 194 | 31.0% |
| tket (FullPeephole) | 2.6602 | 1.9722 | 25.9% | 279 | 194 | 30.5% |
| Exact-only (ours, all-exact) | 2.7437 | 1.9722 | 28.1% | 281 | 194 | 31.0% |

## Per-circuit detail (2q-count / verified)

| Circuit | Qiskit opt-3 | tket (FullPeephole) | Exact-only (ours, all-exact) | **Ours (gated)** |
|---|---|---|---|---|
| ripple_carry_adder_2b | 33 (ok) | 31 (ok) | 33 (ok) | 21 (ok) |
| controlled_adder_2b | 48 (ok) | 48 (ok) | 48 (ok) | 30 (ok) |
| grover_oracle_mcx3 | 25 (ok) | 25 (ok) | 25 (ok) | 13 (ok) |
| grover_oracle_mcx4 | 37 (ok) | 37 (ok) | 37 (ok) | 19 (ok) |
| grover_native_mcx4 | 36 (ok) | 36 (ok) | 36 (ok) | 36 (ok) |
| compute_uncompute_pair | 13 (ok) | 13 (ok) | 13 (ok) | 7 (ok) |
| nested_compute_uncompute | 25 (ok) | 25 (ok) | 25 (ok) | 13 (ok) |
| live_and_chain | 12 (ok) | 12 (ok) | 12 (ok) | 12 (ok) |
| phase_sensitive_oracle | 6 (ok) | 6 (ok) | 6 (ok) | 6 (ok) |
| single_live_toffoli | 6 (ok) | 6 (ok) | 6 (ok) | 6 (ok) |
| half_uncomputed | 19 (ok) | 19 (ok) | 19 (ok) | 10 (ok) |
| adder_live_carry | 21 (ok) | 21 (ok) | 21 (ok) | 21 (ok) |


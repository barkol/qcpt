# Scale evaluation — error-budget Toffoli decomposition past the exhaustive-verification limit

Large CCX/MCX benchmarks (12-24 qubits, `large_benchmarks.large_suite()`), decomposed by the certified `ScalableErrorBudgetSelector` (structural pairs (C), certified windows (W), mirrored gadgets under (R) with clean ancillas pinned to |0>) and scored with QCPT's `HardwareErrorModel` (p2q=0.01, p1q=0.001). Every output is certified against the all-exact decomposition on the input subspace (subroutine equivalence): densely when that is cheap, otherwise with QCEC (pinned qubits declared as ancillas). An output that cannot be certified is replaced by the all-exact circuit and reported as such.

## Verification regimes (kept strictly separate)

- **exhaustively verified** — `ExactEquivalenceVerifier.verify` ran (ground truth). Feasible only up to ~12-13 qubits here.
- **QCEC-verified on the input subspace (up to global phase)** — decision-diagram proof at widths where exhaustive is infeasible.
- **program mode (condition U)** — relative-phase substitutions whose phase is only *unobservable* at the final measurement are NOT subroutine-equivalent and are not used here; see `scale_program.py`.

## Per-circuit results

| Circuit | n | family | relphase_safe | sites (appl/found) | 2q exact | 2q ours | 2q reduction | 2q qiskit-O3 | infid exact | infid ours | cert method | cert | cert time (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ripple_adder_5b | 12 | adder/cu | True | 1/1 | 81 | 51 | +37.0% | 81 | 0.5951 | 0.4245 | qcec | equiv | 0.0209 |
| grover_oracle_6c | 12 | grover/cu | True | 5/5 | 61 | 31 | +49.2% | 61 | 0.5050 | 0.2964 | qcec | equiv | 0.0205 |
| cla_adder_4b | 13 | adder/cla | True | 4/4 | 85 | 43 | +49.4% | 85 | 0.6248 | 0.3863 | qcec | equiv | 0.0888 |
| mod_incr_6b | 12 | modular/cu | True | 0/0 | 66 | 36 | +45.5% | 56 | 0.5292 | 0.3309 | dense_subspace | equiv | 29.0570 |
| ripple_adder_7b | 16 | adder/cu | True | 1/1 | 113 | 77 | +31.9% | 113 | 0.7168 | 0.5683 | qcec_ancilla | equiv | 0.2123 |
| grover_oracle_8c | 16 | grover/cu | True | 7/7 | 85 | 43 | +49.4% | 85 | 0.6248 | 0.3863 | qcec | equiv | 0.0240 |
| cla_adder_5b | 16 | adder/cla | True | 5/5 | 109 | 55 | +49.5% | 109 | 0.7157 | 0.4646 | qcec_ancilla | equiv | 0.0722 |
| mod_incr_8b | 16 | modular/cu | True | 0/0 | 92 | 50 | +45.7% | 78 | 0.6503 | 0.4280 | qcec_ancilla | equiv | 0.0280 |
| ripple_adder_10b | 22 | adder/cu | True | 1/1 | 161 | 125 | +22.4% | 161 | 0.8344 | 0.7475 | qcec_ancilla | equiv | 43.9475 |
| grover_oracle_11c | 22 | grover/cu | True | 10/10 | 121 | 61 | +49.6% | 121 | 0.7525 | 0.5000 | qcec | equiv | 0.0587 |
| cla_adder_7b | 22 | adder/cla | True | 7/7 | 157 | 79 | +49.7% | 157 | 0.8367 | 0.5926 | qcec_ancilla | equiv | 0.7554 |
| mod_incr_11b | 22 | modular/cu | True | 0/0 | 131 | 71 | +45.8% | 111 | 0.7761 | 0.5478 | qcec_ancilla | equiv | 0.0665 |
| array_mult_3b | 12 | multiplier/live | False | 0/0 | 54 | 54 | +0.0% | 54 | 0.4641 | 0.4641 | qcec | equiv | 0.0135 |
| ripple_live_carry_6b | 13 | adder/live | False | 0/0 | 42 | 42 | +0.0% | 42 | 0.3788 | 0.3788 | qcec | equiv | 0.0147 |
| half_uncomputed_6c | 13 | grover/mixed | False | 5/5 | 67 | 37 | +44.8% | 67 | 0.5381 | 0.3435 | qcec | equiv | 0.0162 |
| array_mult_4b | 16 | multiplier/live | False | 0/0 | 96 | 96 | +0.0% | 96 | 0.6701 | 0.6701 | qcec | equiv | 0.0157 |
| ripple_live_carry_8b | 17 | adder/live | False | 0/0 | 56 | 56 | +0.0% | 56 | 0.4700 | 0.4700 | qcec | equiv | 0.0163 |
| half_uncomputed_8c | 17 | grover/mixed | False | 7/7 | 91 | 49 | +46.2% | 91 | 0.6499 | 0.4274 | qcec | equiv | 0.0243 |
| array_mult_5b | 20 | multiplier/live | False | 0/0 | 150 | 150 | +0.0% | 150 | 0.8232 | 0.8232 | qcec | equiv | 0.0187 |
| array_mult_6b | 24 | multiplier/live | False | 0/0 | 216 | 216 | +0.0% | 216 | 0.9175 | 0.9175 | qcec | equiv | 0.0196 |

## Aggregate reductions

| Subset | #circ | 2q exact | 2q ours | 2q red vs exact | 2q red vs qiskit-O3 | infid red vs exact |
|---|---|---|---|---|---|---|
| ALL | 20 | 2034 | 1422 | 30.1% | 28.5% | 22.2% |
| compute/uncompute (relphase-safe) | 12 | 1262 | 722 | 42.8% | 40.7% | 30.5% |
| LIVE (relphase-unsafe) | 8 | 772 | 700 | 9.3% | 9.3% | 8.5% |
| width > 10 (QCEC-certified only) | 20 | 2034 | 1422 | 30.1% | 28.5% | 22.2% |

> On this suite the reduction over exact-only is **30.1% 2q / 22.2% infidelity** across all 20 circuits, and **42.8% 2q** on the compute/uncompute-structured subset. Mean 2q reduction by family: grover ~48%, modular ~46%, adder ~30%, multiplier ~0%. The size tracks the share of Toffolis that are uncomputed or write into clean ancillas; Toffolis accumulating into undeclared output registers (array multipliers, live-carry adders) stay exact.

## Phase-aware regime (QCEC-uncertifiable; analysis-sound)

The phase-AWARE standalone path (a lone relative-phase Toffoli whose phase is merely *unobservable*) is NOT unitary-equivalent to exact CCX, so QCEC reports `not_equivalent` for it by construction. Its admissibility is decided by the width-independent static analysis below. The certified selector evaluated above does NOT use this path, so these numbers do not inflate the reductions; they count the Toffolis that program mode could additionally touch.

| Circuit | n | standalone CCX | phase-unobservable (analysis) |
|---|---|---|---|
| ripple_adder_5b | 12 | 8 | 8 |
| grover_oracle_6c | 12 | 0 | 0 |
| cla_adder_4b | 13 | 6 | 6 |
| mod_incr_6b | 12 | 10 | 10 |
| ripple_adder_7b | 16 | 12 | 12 |
| grover_oracle_8c | 16 | 0 | 0 |
| cla_adder_5b | 16 | 8 | 8 |
| mod_incr_8b | 16 | 14 | 14 |
| ripple_adder_10b | 22 | 18 | 18 |
| grover_oracle_11c | 22 | 0 | 0 |
| cla_adder_7b | 22 | 12 | 12 |
| mod_incr_11b | 22 | 20 | 20 |
| array_mult_3b | 12 | 9 | 9 |
| ripple_live_carry_6b | 13 | 6 | 6 |
| half_uncomputed_6c | 13 | 1 | 1 |
| array_mult_4b | 16 | 16 | 16 |
| ripple_live_carry_8b | 17 | 8 | 8 |
| half_uncomputed_8c | 17 | 1 | 1 |
| array_mult_5b | 20 | 25 | 25 |
| array_mult_6b | 24 | 36 | 36 |

### Exhaustive ground-truth spot-check (tiny instances)

The stock exhaustive phase-aware `ErrorBudgetSelector` run on TINY (<=10q) instances of each family, where its O(2^n) internal gate is fast. This is the ground-truth backstop: it certifies on the reachable basis (phase-insensitive, ancilla clean) EXHAUSTIVELY at these widths.

| Circuit | n | relphase_safe | verified | pair sites | phase-aware admitted | certification | time (s) |
|---|---|---|---|---|---|---|---|
| grover_oracle_3c | 6 | True | True | 2 | 0 | exact_unitary | 0.012 |
| cla_adder_2b | 7 | True | True | 2 | 0 | exact_unitary | 0.045 |
| mod_incr_3b | 6 | True | True | 0 | 0 | exact_unitary | 0.048 |
| array_mult_2b | 8 | False | True | 0 | 0 | exact_unitary | 0.086 |
| half_uncomputed_3c | 7 | False | True | 2 | 0 | exact_unitary | 0.045 |

> Where a standalone phase is provably unobservable (`is_phase_unobservable`) AND agrees with exact CCX on the reachable basis (`reachable_overapprox`, a SOUND superset), the substitution is analysis-sound. The over-approximation can only OVER-reject, never wrongly accept. The `live`/`mixed` tiny instances correctly admit nothing unsafe (the selector keeps live Toffolis exact and stays `verified`).

## Verification crossover — exhaustive vs QCEC wall-clock vs width

Same equivalence (a width-w ripple adder's exact lowering vs the same with every verified compute/uncompute pair substituted), timed both ways as width grows. This is the paper's scalability boundary, demonstrated.

| n (qubits) | exhaustive time (s) | exhaustive verdict | QCEC time (s) | QCEC verdict | exhaustive feasible? |
|---|---|---|---|---|---|
| 6 | 0.0053 | equiv | 0.0268 | equiv | yes |
| 8 | 0.0653 | equiv | 0.0158 | equiv | yes |
| 10 | 1.8885 | equiv | 0.0201 | equiv | yes |
| 12 | 38.0074 | equiv | 0.0188 | equiv | yes |
| 14 | — | — | 0.0167 | equiv | NO (infeasible) |
| 16 | — | — | 0.0167 | equiv | NO (infeasible) |
| 18 | — | — | 0.0196 | equiv | NO (infeasible) |
| 20 | — | — | 0.0235 | equiv | NO (infeasible) |
| 22 | — | — | 0.0246 | equiv | NO (infeasible) |

> Exhaustive verification time grows ~2^n: it is sub-second up to ~n=10, then explodes -- the dense unitary is a 2^n x 2^n matrix built gate-by-gate (at the largest width we even attempt, **n=12**, exhaustive takes **38.0 s** vs QCEC's **0.019 s**, a **~2026x** gap). QCEC stays under **0.03 s** across the WHOLE 6-22 qubit range. The crossover -- where exhaustive stops being practical and QCEC takes over -- is therefore around **n = 11-12 qubits**.


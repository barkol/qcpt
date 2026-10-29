# Safety experiment: verified vs. count-greedy relative-phase substitution

Count-greedy / pattern-matched relative-phase (Margolus / RCCX) Toffoli substitution -- the kind a template optimizer performs when it trusts its rewrite library -- is applied two ways over the benchmark suite:

* **naive**: replace every CCX with a relative-phase Toffoli, no checking.
* **gated**: same per-gate substitution, kept only if a per-window exact verification (`ExactEquivalenceVerifier`) proves the circuit still computes the original function.


## Per-circuit results

| circuit | qubits | CCX | relphase-safe (truth) | naive correct? | gated correct? | relphase applied (gated) |
|---|---:|---:|:--:|:--:|:--:|---:|
| `ripple_carry_adder_2b` | 6 | 4 | safe | yes | yes | 4 |
| `controlled_adder_2b` | 7 | 8 | UNSAFE | **NO** | yes | 0 |
| `grover_oracle_mcx3` | 6 | 4 | safe | yes | yes | 4 |
| `grover_oracle_mcx4` | 8 | 6 | safe | yes | yes | 6 |
| `grover_native_mcx4` | 5 | 0 | safe | yes | yes | 0 |
| `compute_uncompute_pair` | 4 | 2 | safe | yes | yes | 2 |
| `nested_compute_uncompute` | 6 | 4 | safe | yes | yes | 4 |
| `live_and_chain` | 5 | 2 | UNSAFE | **NO** | yes | 0 |
| `phase_sensitive_oracle` | 3 | 1 | UNSAFE | **NO** | yes | 0 |
| `single_live_toffoli` | 3 | 1 | UNSAFE | **NO** | yes | 0 |
| `half_uncomputed` | 5 | 3 | UNSAFE | **NO** | yes | 2 |
| `adder_live_carry` | 5 | 3 | UNSAFE | **NO** | yes | 0 |

## Aggregate (headline)

**Naive substitution silently corrupts 6 of 12 circuits; the verification gate yields 0 errors while still applying 22 valid relative-phase decompositions.**


Every naive failure is a circuit that the count-greedy optimizer would have shipped while computing the WRONG function -- detected here only because we ran an exact equivalence check after the fact. The gated pass keeps exactly the relative-phase decompositions that are provably valid (the compute/uncompute ones) and falls back to exact Toffolis everywhere else, so it is sound by construction.

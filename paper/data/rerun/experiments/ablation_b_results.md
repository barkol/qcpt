# Safety-ablation row (b): our context analysis WITHOUT the verification gate

Apply the selector's relative-phase **admissibility flagging** (structural compute/uncompute pairing `find_relative_phase_safe_sites` + the phase-observability test `is_phase_unobservable` on the reachable subspace), commit the relative-phase gadget at every flagged site, but **skip** the per-instance `ExactEquivalenceVerifier` gate. Each result is then audited with the exact verifier (`allow_permutation=True`); a failure is a silent error the analysis alone admitted.


## Per-circuit results

| circuit | qubits | relphase-safe (truth) | sites flagged | audited correct? |
|---|---:|:--:|---:|:--:|
| `ripple_carry_adder_2b` | 6 | safe | 4 | yes |
| `controlled_adder_2b` | 7 | UNSAFE | 8 | **NO** |
| `grover_oracle_mcx3` | 6 | safe | 4 | yes |
| `grover_oracle_mcx4` | 8 | safe | 6 | yes |
| `grover_native_mcx4` | 5 | safe | 0 | yes |
| `compute_uncompute_pair` | 4 | safe | 2 | yes |
| `nested_compute_uncompute` | 6 | safe | 4 | yes |
| `live_and_chain` | 5 | UNSAFE | 2 | **NO** |
| `phase_sensitive_oracle` | 3 | UNSAFE | 1 | **NO** |
| `single_live_toffoli` | 3 | UNSAFE | 1 | **NO** |
| `half_uncomputed` | 5 | UNSAFE | 3 | **NO** |
| `adder_live_carry` | 5 | UNSAFE | 3 | **NO** |

## Aggregate (row (b))

**Our analysis-only (un-gated) pass introduces 6 of 12 silent errors, flagging 38 relative-phase sites across the suite.**


Contrast with (a) count-greedy = 6/12 and (c) gated = 0/12. The gap between (a) and (b) is the contribution of the context analysis; the gap between (b) and (c) is the contribution of the verification gate.


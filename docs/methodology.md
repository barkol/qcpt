# Methodology

Condensed from the paper's §3 (Method) and §4 (Verification gate). Read this
document to understand what the code in `toffoli_optimizer/` does without
opening the manuscript. Section numbers refer to the arXiv preprint
[2606.31791](https://arxiv.org/abs/2606.31791).

## 1. Objective — error budget, not gate count

For a target device with two-qubit-gate infidelity `p_{2q}` and one-qubit
infidelity `p_{1q}`, the emitted circuit's fidelity is dominated by the
sum of two-qubit-gate errors along the compiled path. We therefore score
every candidate circuit `C` by the first-order additive budget

    B(C) = n_{2q}(C) · p_{2q} + n_{1q}(C) · p_{1q}

implemented in `toffoli_optimizer.core.error_model.HardwareErrorModel`.
This is a surrogate — see `experiments/noisy_sim.py` for a density-matrix
validation that the surrogate's ranking translates into a real
state-fidelity gain.

## 2. Toffoli decomposition taxonomy

Each Toffoli site is compiled by selecting one of

- **EXACT** — the standard 6-CX Toffoli. Unitary-equivalent everywhere;
  always sound.
- **RCCX (relative-phase)** — Margolus / RCCX with 3 CX. Introduces a
  relative phase on `|110⟩ ↔ |111⟩`; sound only in contexts where the
  phase is either unobservable or cancelled by a partner substitution.
- **Approximate on reachable subspace** — a bounded-error decomposition
  that is exact on the input subspace actually reachable at this site.
- **Ancilla-augmented** — cheaper CX-count using a borrowed or dirty
  ancilla; requires the ancilla to be genuinely available.

Implemented in `toffoli_optimizer.core.decomposition_selector` as
`append_exact_ccx`, `append_relative_phase_ccx`, and the
`ErrorBudgetSelector.select` dispatch.

## 3. Admissibility analyses

Before an alternative can enter the selection pool for a given site, one
of two admissibility analyses must certify it:

### 3.1 Reachable-subspace analysis
`toffoli_optimizer.core.reachable_subspace` propagates a symbolic
description of which basis states can actually appear at each cut of the
circuit. A relative-phase Toffoli whose phase acts only on states outside
this subspace is safe. This admits a large fraction of Grover-oracle and
compute/uncompute Toffolis without any exact check.

### 3.2 Phase-observability analysis
`toffoli_optimizer.core.phase_observability` traces whether the phase
introduced by an RCCX is observable at circuit output — either measured
directly or interfering with another kickback. Unobservable phases are
free to introduce.

### 3.3 Context analysis
`toffoli_optimizer.core.context_analysis.find_relative_phase_safe_sites`
locates compute/uncompute pairs — the archetypal safe context in which
two RCCX substitutions cancel a shared phase exactly.

## 4. Verification gate

When an admissibility analysis does not conclusively certify a
substitution, the **verification gate** falls back to a proof. Two
backends dispatch by width:

### 4.1 Exact truth-table / unitary verifier
`toffoli_optimizer.core.equivalence_verifier.ExactEquivalenceVerifier`
compares two circuits either by their classical truth-tables (when both
are classical-reversible) or by dense unitary equivalence up to global
phase and output permutation. Handles ancilla and reachable-subspace
comparison. Sound but limited to ~10-12 qubits by dense-unitary cost.

### 4.2 Scalable decision-diagram verifier
`toffoli_optimizer.core.scalable_verification.verify_scalable` uses
MQT QCEC (decision-diagram equivalence checking) as the fallback backend
above the exhaustive limit. The paper documents the crossover at
`n ≈ 11-12` with a >3000× speedup at `n=12` and sub-25 ms cost through
`n=22`.

The verifier is used **conservatively**: a positive answer certifies
equivalence; an inconclusive answer keeps the exact decomposition. This
is why the safety ablation reports 0 silent errors on the gated pass.

## 5. What the count-greedy baseline gets wrong

The naive pattern-matching substitution deployed in current optimizers
replaces every Toffoli with the cheapest same-permutation RCCX
unconditionally. On the paper's 12-circuit primary suite this **silently
corrupts 6 of 12 circuits** (see `experiments/safety_experiment.py`,
Fig. `safety`). Our audit against a deployed open optimizer additionally
flagged **66 library rewrites** as non-equivalent under exact truth-table
comparison — see the `test_verified_patterns.py` suite.

## 6. What the verification gate buys back

On the same 12-circuit suite the gated pass emits **170 two-qubit gates
vs 281 for exact-only** at infidelity 1.736 vs 2.744 — a 39.5% count
reduction and 36.7% infidelity reduction with **every substitution
verified**. See `experiments/baselines.py`, Fig. `budget`.

On the larger 20-circuit / 12-24 qubit scale suite the aggregate
reduction is 15.6% two-qubit / 11.7% infidelity, workload-dependent:
Grover oracles ≈49%, carry-lookahead adders ≈28%, ripple adders ≈6%,
modular-increment / array-multiplier 0% (correctly left exact). See
`experiments/scale_eval.py`, Fig. `scale`.

## 7. External application

The state-resetting (quantum synchronizing-word) circuits of Stempin
*et al.* (arXiv:2504.01106) contain a structural compute/uncompute
`T_j^† S T_j` at every "letter" application — exactly the context the
selector exploits. On the smallest instance (8-state, 3-letter, 7 qubits)
the full phase-aware pass removes **48.8%** of the native two-qubit
gates with every substitution verified; the four decision-diagram-certified
instances (7-12 qubits) average ≈27% reduction. See
`experiments/sync_benchmark.py`, Table `reset`.

#!/usr/bin/env python3
"""Tests for the bounded-approximate-on-the-reachable-subspace admission path.

Run with:  python test_reachable_approx.py

This is the quantitative refinement of the exact-on-reachable check: a cheaper
context-specialised candidate (here a CONTROL-DROP of a Toffoli) is admitted iff its
worst-case phase-insensitive deviation, measured ONLY on a sound over-approximation
of the reachable subspace, is within a tolerance epsilon. epsilon == 0 recovers the
exact-on-reachable case.

Covered scenarios (matching the task spec):

  (a) PROVABLY-|0> CONTROL.  A Toffoli whose control is provably |0> on every
      reachable input (restricted input space) is DROPPED EXACTLY (epsilon == 0):
      the selector emits identity for it, and the result is observationally
      equivalent to a real CCX on every reachable input.

  (b) TUNABLE-DEVIATION CANDIDATE.  A candidate that deviates from CCX by a small,
      controllable amount on the reachable subspace is ADMITTED at a chosen epsilon
      and REJECTED at a tighter epsilon. (Driven at the verifier level so the
      deviation is a continuous knob; basis-state reachability alone only resolves
      0-vs-sqrt(2) deviations -- see the soundness note in the results file.)

  (c) GENUINELY-|1> CONTROL.  A Toffoli whose control IS set on reachable mass has a
      control-drop deviation of sqrt(2); it is NOT dropped at a small epsilon and the
      selector keeps the exact (or relative-phase) decomposition.
"""

import math
import sys

import numpy as np
from qiskit import QuantumCircuit

from toffoli_optimizer.core.optimizer import (
    ErrorBudgetSelector,
    ExactEquivalenceVerifier,
)
from toffoli_optimizer.core.error_model import HardwareErrorModel


# ---------------------------------------------------------------------------------
# (a) Provably-|0> control -> dropped exactly at epsilon == 0.
# ---------------------------------------------------------------------------------
def test_provably_zero_control_dropped_exactly():
    """A CCX(0,1,2) whose control 0 is pinned to |0> by the input space collapses to
    IDENTITY on the reachable subspace. With epsilon == 0 the selector drops it
    exactly (0 two-qubit gates for that Toffoli) and certifies it."""
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)

    # The circuit is only ever run on inputs where qubit 0 (a control) is |0>: the
    # |11> branch can never fire, so the Toffoli is the identity on these inputs.
    input_space = [x for x in range(1 << 3) if not (x & 1)]  # bit 0 == 0

    sel = ErrorBudgetSelector(epsilon=0.0)
    out = sel.select(qc, input_space=input_space)
    rep = out["report"]
    print("[a] " + rep.summary())

    assert len(rep["approx_admitted"]) == 1, (
        f"expected exactly one approx admission, got {rep['approx_admitted']}")
    adm = rep["approx_admitted"][0]
    assert adm["keep"] == [], f"control-0-|0> must drop to IDENTITY, got keep={adm['keep']}"
    assert adm["max_deviation"] == 0.0, f"exact drop must have 0 deviation, got {adm['max_deviation']}"
    assert rep["epsilon_spent_total"] == 0.0
    assert rep["two_qubit_after"] == 0, f"dropped Toffoli should cost 0 two-qubit gates, got {rep['two_qubit_after']}"
    assert rep["verified"], "exact-on-reachable drop must certify"

    # OBSERVATIONAL EQUIVALENCE on the reachable inputs: selected output == CCX output
    # (here == input, since the |11> branch never fires) for every reachable basis input.
    ref = QuantumCircuit(3)
    ref.ccx(0, 1, 2)
    v = ExactEquivalenceVerifier()
    ok, _md, info = v.verify_on_reachable_basis_approx(
        ref, out["circuit"], epsilon=0.0, reachable_states=input_space)
    assert ok, f"dropped circuit not observationally equivalent on reachable inputs: {info}"
    print("[a] PASS: provably-|0> control dropped to identity exactly (epsilon=0), verified on reachable inputs")


# ---------------------------------------------------------------------------------
# (b) Tunable-deviation candidate: admitted at a loose epsilon, rejected at a tight one.
# ---------------------------------------------------------------------------------
def test_tunable_deviation_admitted_then_rejected():
    """A candidate whose reachable-subspace deviation is a controllable theta is
    admitted when epsilon >= theta and rejected when epsilon < theta.

    We drive the verifier directly so the deviation is a smooth knob. ``orig`` is the
    identity (so CCX-output == input on the single reachable basis input |0..0>);
    ``cand`` applies a small Ry(2*theta) on the target, giving an output that deviates
    from the identity by exactly ||Ry(2 theta)|0> - |0>|| = 2 sin(theta/...).  We read
    the exact deviation from the verifier and bracket epsilon around it."""
    n = 3
    theta = 0.05  # small rotation angle

    orig = QuantumCircuit(n)          # identity: output == input on every input
    cand = QuantumCircuit(n)
    cand.ry(2 * theta, n - 1)         # small coherent error on the target wire

    reachable = [0]  # only |000> reaches the gate
    v = ExactEquivalenceVerifier()

    # Exact phase-insensitive deviation of Ry(2 theta)|0> from |0>:
    #   Ry(2 theta)|0> = cos(theta)|0> + sin(theta)|1>
    #   min-phase distance to |0> = sqrt(2 - 2|cos theta|) = 2 sin(theta/2).
    expected = 2.0 * math.sin(theta / 2.0)
    _adm, dev, info = v.verify_on_reachable_basis_approx(orig, cand, expected, reachable)
    print(f"[b] measured deviation = {dev:.6g}, expected {expected:.6g}")
    assert abs(dev - expected) < 1e-9, f"deviation {dev} != expected {expected}"

    # Admitted at a loose epsilon (just above the deviation) ...
    adm_loose, _d, _i = v.verify_on_reachable_basis_approx(orig, cand, dev + 1e-3, reachable)
    assert adm_loose, "candidate must be admitted when epsilon > deviation"

    # ... and rejected at a tighter epsilon (just below it).
    adm_tight, _d, _i = v.verify_on_reachable_basis_approx(orig, cand, dev - 1e-3, reachable)
    assert not adm_tight, "candidate must be rejected when epsilon < deviation"

    # epsilon == 0 rejects any nonzero deviation -> recovers the exact-on-reachable case.
    adm_zero, _d, _i = v.verify_on_reachable_basis_approx(orig, cand, 0.0, reachable)
    assert not adm_zero, "epsilon == 0 must reject a nonzero-deviation candidate"
    print("[b] PASS: tunable deviation admitted at epsilon>=theta, rejected at epsilon<theta, rejected at epsilon=0")


# ---------------------------------------------------------------------------------
# (c) Genuinely-|1> control on reachable mass: NOT dropped at small epsilon.
# ---------------------------------------------------------------------------------
def test_set_control_not_dropped():
    """A CCX(0,1,2) run on the full basis (both controls range over {0,1}, so the
    |11> branch IS reachable) cannot be control-dropped within a small epsilon: any
    drop deviates by sqrt(2) on the firing input. The selector keeps the exact
    decomposition (no approx admission)."""
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)

    # Even a generous epsilon below sqrt(2) must NOT admit a control drop.
    sel = ErrorBudgetSelector(epsilon=1.0, phase_aware=False)
    out = sel.select(qc)  # default all_basis -> |11> control branch reachable
    rep = out["report"]
    print("[c] " + rep.summary())

    assert len(rep["approx_admitted"]) == 0, (
        f"a genuinely-set control must NOT be dropped, got {rep['approx_admitted']}")
    assert rep["epsilon_spent_total"] == 0.0
    assert rep["two_qubit_after"] == rep["two_qubit_before"] == 6, (
        f"no approx admission -> exact 6-CX Toffoli kept, got {rep['two_qubit_after']}")
    assert rep["verified"]

    # The kept circuit is exactly a CCX.
    ref = QuantumCircuit(3)
    ref.ccx(0, 1, 2)
    v = ExactEquivalenceVerifier()
    ok, _p, info = v.verify(ref, out["circuit"], allow_permutation=False)
    assert ok, f"kept circuit not equivalent to CCX: {info}"
    print("[c] PASS: genuinely-set control NOT dropped; exact CCX kept and verified")


# ---------------------------------------------------------------------------------
# (d) sqrt(2) sanity: drop-to-identity deviation on a firing basis input is sqrt(2).
# ---------------------------------------------------------------------------------
def test_drop_to_identity_deviation_is_sqrt2_on_firing_input():
    """Direct verifier check: on the |11>-control input, dropping CCX to identity
    deviates by exactly sqrt(2) (phase-insensitive), confirming the metric's scale."""
    orig = QuantumCircuit(3)
    orig.ccx(0, 1, 2)
    cand = QuantumCircuit(3)  # identity (dropped)

    firing = 0b011  # qubit0=1, qubit1=1, qubit2=0 -> CCX flips qubit2
    v = ExactEquivalenceVerifier()
    _adm, dev, _info = v.verify_on_reachable_basis_approx(orig, cand, 0.0, [firing])
    assert abs(dev - math.sqrt(2.0)) < 1e-9, f"firing-input drop deviation {dev} != sqrt(2)"
    # On a non-firing input the deviation is 0.
    _adm2, dev0, _i2 = v.verify_on_reachable_basis_approx(orig, cand, 0.0, [0b001])
    assert dev0 == 0.0, f"non-firing input deviation should be 0, got {dev0}"
    print("[d] PASS: drop-to-identity deviation = sqrt(2) on firing input, 0 on non-firing input")


def main():
    test_provably_zero_control_dropped_exactly()
    test_tunable_deviation_admitted_then_rejected()
    test_set_control_not_dropped()
    test_drop_to_identity_deviation_is_sqrt2_on_firing_input()
    print("\nAll bounded-approximate-on-reachable tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

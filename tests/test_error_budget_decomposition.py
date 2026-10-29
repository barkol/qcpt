#!/usr/bin/env python3
"""Tests for error-budget-aware, context-verified Toffoli decomposition selection.

Run with:  python test_error_budget_decomposition.py

Covers the three required scenarios:

  1. COMPUTE/UNCOMPUTE PAIR.  CCX(0,1,2); <op uses 2 as control>; CCX(0,1,2):
     the selector detects the relative-phase-safe site, verifies the substitution
     of the cheap (Margolus) gadget into BOTH gates, applies it, and reports a
     strictly lower two-qubit count and infidelity than the exact-only baseline.

  2. STANDALONE CCX.  A single CCX with no uncompute: no admissible site is found
     (or verification fails), so NO relative-phase gadget is applied; the output
     falls back to the exact decomposition and is still verified-correct.

  3. INVARIANTS.  infidelity(selected) <= infidelity(exact_only) always, and the
     selected output always verifies against the exact-only circuit.
"""

import sys

from qiskit import QuantumCircuit

from toffoli_optimizer.core.optimizer import (
    ErrorBudgetSelector,
    HardwareErrorModel,
    ExactEquivalenceVerifier,
)
from toffoli_optimizer.core.context_analysis import find_relative_phase_safe_sites
from toffoli_optimizer.core.reachable_subspace import reachable_overapprox


def _compute_uncompute_circuit():
    """CCX(0,1,2); CX(2,3); CCX(0,1,2) -- t=2 used only as a control in between."""
    qc = QuantumCircuit(4)
    qc.ccx(0, 1, 2)   # compute t = a&b into qubit 2
    qc.cx(2, 3)       # use 2 ONLY as a control
    qc.ccx(0, 1, 2)   # uncompute
    return qc


def test_compute_uncompute_pair_applies_relative_phase():
    qc = _compute_uncompute_circuit()

    # The site detector must find exactly one admissible compute/uncompute site.
    sites = find_relative_phase_safe_sites(qc)
    assert len(sites) == 1, f"expected 1 safe site, found {len(sites)}: {sites}"
    site = sites[0]
    assert site.controls == (0, 1) and site.target == 2, f"bad site qubits: {site}"

    sel = ErrorBudgetSelector()
    out = sel.select(qc)
    rep = out["report"]

    print("[1] " + rep.summary())

    # The relative-phase gadget was applied (to the pair) and verified.
    assert rep["sites_found"] == 1
    assert rep["sites_applied"] == 1, "compute/uncompute pair should be applied"
    assert rep["sites_rejected"] == 0
    assert rep["verified"], "selected circuit must verify against exact-only"

    # Strictly cheaper than exact-only on BOTH the 2q count and the infidelity.
    assert rep["two_qubit_after"] < rep["two_qubit_before"], (
        f"2q count not reduced: {rep['two_qubit_before']} -> {rep['two_qubit_after']}"
    )
    assert rep["infidelity_after"] < rep["infidelity_before"], (
        f"infidelity not reduced: {rep['infidelity_before']} -> {rep['infidelity_after']}"
    )

    # Concrete numbers: exact-only = 2 Toffolis * 6 CX + 1 CX = 13 two-qubit gates;
    # relative-phase = 2 * 3 CX + 1 CX = 7 two-qubit gates.
    exact = sel.decompose_exact_only(qc)
    assert rep["two_qubit_before"] == 13, rep["two_qubit_before"]
    assert rep["two_qubit_after"] == 7, rep["two_qubit_after"]

    # Independent re-verification of the produced circuit vs the exact baseline.
    v = ExactEquivalenceVerifier()
    ok, _p, info = v.verify(exact, out["circuit"], allow_permutation=False)
    assert ok, f"independent verification failed: {info}"
    print("[1] PASS: relative-phase applied to pair, verified, 13->7 two-qubit gates")


def test_standalone_ccx_falls_back_to_exact():
    """With phase_aware=False (the old pair-only contract) a standalone CCX gets NO
    relative-phase gadget and falls back to the exact decomposition.

    (With phase_aware=True the standalone CCX feeding an implicit basis read has a
    provably-unobservable phase and IS substituted soundly -- that is covered by the
    new test_phase_unobservable_admits_standalone.)"""
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)  # standalone -- no uncompute

    sites = find_relative_phase_safe_sites(qc)
    assert len(sites) == 0, f"standalone CCX must have NO safe site, got {sites}"

    sel = ErrorBudgetSelector(phase_aware=False)
    out = sel.select(qc)
    rep = out["report"]
    print("[2] " + rep.summary())

    assert rep["sites_applied"] == 0, "standalone CCX must not get a relative-phase gadget"
    assert rep["verified"], "fallback exact output must verify"
    # No saving possible (pair-only path) -> equal cost to exact-only.
    assert rep["two_qubit_after"] == rep["two_qubit_before"] == 6

    # Output must be exactly equivalent to a real CCX.
    ref = QuantumCircuit(3)
    ref.ccx(0, 1, 2)
    v = ExactEquivalenceVerifier()
    ok, _p, info = v.verify(ref, out["circuit"], allow_permutation=False)
    assert ok, f"standalone fallback not equivalent to CCX: {info}"
    print("[2] PASS: standalone CCX (phase_aware=False) -> exact decomposition, verified-correct")


def test_unsafe_window_rejected():
    """A 'pair' whose target is WRITTEN in between (e.g. an X on t) is not a valid
    compute/uncompute site; the selector must not apply a relative-phase gadget."""
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    qc.x(2)          # WRITES the target -> breaks phase cancellation
    qc.ccx(0, 1, 2)

    sites = find_relative_phase_safe_sites(qc)
    assert len(sites) == 0, f"unsafe window must yield no site, got {sites}"

    sel = ErrorBudgetSelector()
    out = sel.select(qc)
    rep = out["report"]
    print("[3] " + rep.summary())
    assert rep["sites_applied"] == 0
    assert rep["verified"], "output must still be verified-correct"
    print("[3] PASS: unsafe (target-written) window not substituted, output verified")


def test_invariants_over_several_circuits():
    """infidelity(selected) <= infidelity(exact_only) always; output always
    verifies against the exact-only circuit (allow_permutation as needed)."""
    circuits = {
        "pair": _compute_uncompute_circuit(),
        "standalone": _single_ccx(),
        "two_pairs": _two_nested_pairs(),
        "pair_then_standalone": _pair_then_standalone(),
    }
    # allow_permutation only affects the exact-unitary path; the phase-aware path
    # certifies phase-insensitively on the reachable basis inside select().
    sel = ErrorBudgetSelector(allow_permutation=True)
    em = HardwareErrorModel()
    v = ExactEquivalenceVerifier()

    for name, qc in circuits.items():
        exact = sel.decompose_exact_only(qc)
        out = sel.select(qc)
        sel_circ = out["circuit"]
        rep = out["report"]

        inf_exact = em.circuit_infidelity(exact)
        inf_sel = em.circuit_infidelity(sel_circ)
        assert inf_sel <= inf_exact + 1e-15, (
            f"[{name}] infidelity increased: {inf_sel} > {inf_exact}"
        )

        # SOUNDNESS: if a phase-aware (unobservable-phase) substitution was made, the
        # output intentionally differs from exact-only by a relative phase, so the
        # independent check must be phase-insensitive on the reachable basis; else the
        # strict exact-unitary check applies.
        if rep.get("phase_aware_admitted"):
            reachable = reachable_overapprox(qc, len(qc.data), input_space="all_basis")
            ok, _perm, info = v.verify_on_reachable_basis(exact, sel_circ, reachable)
        else:
            ok, _perm, info = v.verify(exact, sel_circ, allow_permutation=True)
        assert ok, f"[{name}] selected not sound vs exact-only: {info}"
        assert rep["verified"], f"[{name}] report says not verified"
        print(f"[inv:{name}] inf {inf_exact:.4g} -> {inf_sel:.4g}, verified, "
              f"2q {rep['two_qubit_before']}->{rep['two_qubit_after']}")
    print("[inv] PASS: infidelity never increased and every output verified")


# ---- helper circuits -------------------------------------------------------------
def _single_ccx():
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    return qc


def _two_nested_pairs():
    qc = QuantumCircuit(5)
    qc.ccx(0, 1, 2)   # outer compute
    qc.ccx(2, 3, 4)   # inner compute (uses 2 as control -> safe for outer)
    qc.cx(4, 3)
    qc.ccx(2, 3, 4)   # inner uncompute
    qc.ccx(0, 1, 2)   # outer uncompute
    return qc


def _pair_then_standalone():
    qc = QuantumCircuit(4)
    qc.ccx(0, 1, 2)
    qc.cx(2, 3)
    qc.ccx(0, 1, 2)   # closes the pair
    qc.ccx(0, 1, 3)   # standalone afterward
    return qc


def main():
    test_compute_uncompute_pair_applies_relative_phase()
    test_standalone_ccx_falls_back_to_exact()
    test_unsafe_window_rejected()
    test_invariants_over_several_circuits()
    print("\nAll error-budget decomposition tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

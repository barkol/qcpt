#!/usr/bin/env python3
"""Regression tests: admissibility must hold on the reachable SUBSPACE, not per basis state.

Run with:  python test_superposition_soundness.py

Background (counterexample of 2026-10-01): with both controls at 1,
the target at 0 and a Hadamard on the target before and after the Toffoli, Qiskit's
RCCX agrees with CCX up to a phase on EACH reachable basis state |110>, |111>, but the
two phases differ, and the final Hadamard turns the difference into a different target
outcome. A basis-state-wise "agrees up to phase" check accepts RCCX; the subspace
check (one phase for the whole reachable subspace) rejects it.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

from toffoli_optimizer.core.subspace_check import (
    check_on_subspace, certify_on_input_subspace, project_support)
from toffoli_optimizer.core.decomposition_selector import (
    ErrorBudgetSelector, append_relative_phase_ccx)
from toffoli_optimizer.core.reachable_subspace import reachable_overapprox


def _u(build, n=3):
    qc = QuantumCircuit(n)
    build(qc)
    return Operator(qc).data


U_CCX = _u(lambda q: q.ccx(0, 1, 2))
U_RCCX = _u(lambda q: q.rccx(0, 1, 2))


def test_editor_counterexample_basiswise_accepts_subspace_rejects():
    reach = [0b011, 0b111]  # q0=1, q1=1, t in {0,1} (little-endian: bit j = qubit j)
    # Basis-state-wise: each column equal up to its own phase -> would be accepted.
    for x in reach:
        a, b = U_RCCX[:, x], U_CCX[:, x]
        assert abs(abs(np.vdot(b, a)) - 1) < 1e-9
    # Subspace (one phase): rejected.
    ok, dev, _ = check_on_subspace(U_RCCX, U_CCX, reach, epsilon=0.0)
    assert not ok and dev > 0.5, dev
    # And the substitution really changes the outcome.
    def run(gate):
        c = QuantumCircuit(3); c.x(0); c.x(1); c.h(2); gate(c); c.h(2)
        return Statevector(c).probabilities([2])
    p_ccx, p_rccx = run(lambda c: c.ccx(0, 1, 2)), run(lambda c: c.rccx(0, 1, 2))
    assert not np.allclose(p_ccx, p_rccx)
    print("[1] PASS: editor's RCCX counterexample: basis-wise check accepts, subspace check rejects "
          f"(deviation {dev:.3f}); outcomes differ {np.round(p_ccx,3)} vs {np.round(p_rccx,3)}")


def test_editor_circuit_through_pipeline_is_exact_and_certified():
    c = QuantumCircuit(3); c.x(0); c.x(1); c.h(2); c.ccx(0, 1, 2); c.h(2)
    res = ErrorBudgetSelector().select(c)
    rep = res["report"]
    assert not rep["phase_aware_admitted"], rep["phase_aware_admitted"]
    assert rep["verified"] is True
    print("[2] PASS: pipeline keeps the Toffoli exact on the editor's circuit "
          f"(certification={rep['verify_info'].get('certification')})")


def test_fig1b_control_interference_rejected_and_forced_gadget_detected():
    # Fig. 1(b) of the revised paper: H on control q0 after g.
    c = QuantumCircuit(3); c.ccx(0, 1, 2); c.h(0)
    res = ErrorBudgetSelector(semantics="program").select(c)
    rep = res["report"]
    assert not rep["phase_aware_admitted"]
    # Forcing the gadget is observably wrong on a superposed data input ...
    forced = QuantumCircuit(3); append_relative_phase_ccx(forced, 0, 1, 2); forced.h(0)
    exact = ErrorBudgetSelector().decompose_exact_only(c)
    inputs = [x for x in range(8) if not (x >> 2) & 1]  # data q0,q1 free, ancilla a=0
    ok, dev, _ = certify_on_input_subspace(
        Operator(exact).data, Operator(forced).data, inputs, mode="observational")
    assert not ok and dev > 0.5, dev
    # ... while every BASIS input alone would not reveal it.
    for x in inputs:
        sv_e = Statevector.from_int(x, 8).evolve(exact).probabilities()
        sv_f = Statevector.from_int(x, 8).evolve(forced).probabilities()
        assert np.allclose(sv_e, sv_f)
    print(f"[3] PASS: Fig. 1(b) rejected; forced gadget caught by the subspace certificate "
          f"(deviation {dev:.3f}) though invisible on basis inputs")


def test_control_drop_rejected_when_control_in_superposition():
    c = QuantumCircuit(3); c.h(0); c.x(1); c.ccx(0, 1, 2)
    res = ErrorBudgetSelector(epsilon=0.0).select(c, input_space=[0])
    rep = res["report"]
    # q1 is provably 1, so dropping q1 (keep q0 -> CX(q0, t)) is exact on the
    # reachable subspace even though q0 is in superposition; dropping q0 is not.
    kept = [d["keep"] for d in rep["approx_admitted"]]
    assert kept == [[0]], kept
    assert rep["verified"] is True
    print("[4] PASS: control-drop admitted only for the provably-set control; "
          "superposed control kept; output certified on the input subspace")


def test_default_subroutine_semantics_never_uses_U():
    # A lone, measured Toffoli: (U) would admit the gadget in "program" semantics,
    # but the default "subroutine" semantics must keep it exact (unitary changes).
    c = QuantumCircuit(3); c.ccx(0, 1, 2)
    rep = ErrorBudgetSelector().select(c)["report"]
    assert not rep["phase_aware_admitted"] and rep["semantics"] == "subroutine"
    assert rep["verified"] is True
    rep_p = ErrorBudgetSelector(semantics="program").select(c)["report"]
    assert rep_p["phase_aware_admitted"] and rep_p["semantics"] == "observational"
    assert rep_p["verified"] is True
    print("[6] PASS: default subroutine semantics keeps a lone Toffoli exact; "
          "program semantics admits it under (U), certified observationally")


def test_gate_level_states_not_used_as_circuit_inputs():
    # Audit finding B4: X q0; X q1; CCX with input |000>. The control-drop to identity
    # must NOT be admitted (both controls are 1 at the gate).
    c = QuantumCircuit(3); c.x(0); c.x(1); c.ccx(0, 1, 2)
    res = ErrorBudgetSelector(epsilon=0.0).select(c, input_space=[0])
    rep = res["report"]
    assert all(d["keep"] != [] for d in rep["approx_admitted"]), rep["approx_admitted"]
    sv = Statevector.from_int(0, 8).evolve(res["circuit"]).probabilities()
    assert abs(sv[0b111] - 1) < 1e-9
    assert rep["verified"] is True and not rep["fell_back_to_exact"]
    print("[7] PASS: X;X;CCX on |000> keeps the Toffoli's effect (target flipped)")


def test_window_pairs_mirror_adder():
    # Condition (W): mirrored MAJ/UMA-style pair whose window writes a control
    # (structural rule rejects it) but whose segment equals the exact one.
    c = QuantumCircuit(5)
    c.ccx(0, 1, 2); c.cx(3, 2); c.cx(4, 0); c.cx(4, 0); c.cx(3, 2); c.ccx(0, 1, 2)
    rep = ErrorBudgetSelector().select(c)["report"]
    rep_c = ErrorBudgetSelector(window_pairs=False).select(c)["report"]
    assert rep_c["sites_applied"] == 0, "structural rule must reject (target and control written)"
    assert len(rep["window_pairs_admitted"]) == 1
    assert rep["verified"] is True and rep["semantics"] == "subroutine"
    # A window that breaks the cancellation must be rejected.
    d = QuantumCircuit(3); d.ccx(0, 1, 2); d.h(0); d.ccx(0, 1, 2)
    rep_d = ErrorBudgetSelector().select(d)["report"]
    assert not rep_d["window_pairs_admitted"], rep_d["window_pairs_admitted"]
    assert rep_d["verified"] is True
    print(f"[8] PASS: window pairs certified on the segment (admitted {len(rep['window_pairs_admitted'])}, "
          f"structural {rep_c['sites_applied']}); H inside the window rejected")


def test_mirror_gadget_on_clean_ancilla():
    # Condition (R): a live AND written into a clean ancilla (pinned to |0>) gets
    # the mirrored gadget, exact on the reachable subspace; unpinned it does not.
    c = QuantumCircuit(3); c.h(0); c.ccx(0, 1, 2); c.cx(2, 1)
    rep_p = ErrorBudgetSelector().select(c, pinned_zero=[2])["report"]
    rep_f = ErrorBudgetSelector().select(c)["report"]
    assert [d["gadget"] for d in rep_p["rphase_admitted"]] == ["relphase_m"], rep_p["rphase_admitted"]
    assert rep_p["verified"] is True and rep_p["semantics"] == "subroutine"
    assert not rep_f["rphase_admitted"]
    print("[9] PASS: mirrored gadget admitted under (R) for an AND into a pinned clean ancilla, "
          "certified on the input subspace; rejected when the ancilla is free")


def test_window_pairs_never_share_an_endpoint():
    # Audit finding (round 2): three identical Toffolis g0, g3, g5 must not form two
    # windows (0,3) and (3,5) sharing g3.
    from toffoli_optimizer.core.window_pairs import mirror_pair_candidates
    c = QuantumCircuit(4)
    c.ccx(0, 1, 2); c.cx(3, 0); c.cx(3, 0); c.ccx(0, 1, 2); c.cx(2, 3); c.ccx(0, 1, 2)
    cands = mirror_pair_candidates(c)
    ends = [k for p in cands for k in p]
    assert len(ends) == len(set(ends)), cands
    rep = ErrorBudgetSelector().select(c)["report"]
    assert rep["verified"] is True
    print(f"[10] PASS: mirror candidates are disjoint {cands}; output certified")


def test_pairs_remain_unitary_exact():
    c = QuantumCircuit(4); c.ccx(0, 1, 2); c.cx(2, 3); c.ccx(0, 1, 2)
    res = ErrorBudgetSelector().select(c)
    rep = res["report"]
    assert rep["sites_applied"] == 1 and rep["verified"] is True
    assert rep["semantics"] == "subroutine"
    print("[5] PASS: compute/uncompute pair admitted under (C), unitary-exact, subroutine semantics")


if __name__ == "__main__":
    test_editor_counterexample_basiswise_accepts_subspace_rejects()
    test_editor_circuit_through_pipeline_is_exact_and_certified()
    test_fig1b_control_interference_rejected_and_forced_gadget_detected()
    test_control_drop_rejected_when_control_in_superposition()
    test_default_subroutine_semantics_never_uses_U()
    test_gate_level_states_not_used_as_circuit_inputs()
    test_window_pairs_mirror_adder()
    test_mirror_gadget_on_clean_ancilla()
    test_window_pairs_never_share_an_endpoint()
    test_pairs_remain_unitary_exact()
    print("\nAll superposition-soundness tests passed.")

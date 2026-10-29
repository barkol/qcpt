"""Hardened versions of weak tests from tests/*.py (audit 2026-10-02, E_testy).

Design rules applied throughout:
  * never trust ``report["verified"]``: after a failed certificate the selector falls
    back to the all-exact circuit and still reports ``verified=True`` (fail closed),
    so that flag cannot fail. Positive tests assert ``fell_back_to_exact is False``
    and re-certify the output with an INDEPENDENT numpy check written here;
  * every rejection test also shows that the rejected substitution really is wrong
    (the guard is load-bearing);
  * randomised tests are seeded.
"""

import itertools
import math

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

from toffoli_optimizer.core import decomposition_selector as DS
from toffoli_optimizer.core.decomposition_selector import (
    ErrorBudgetSelector, append_relative_phase_ccx, append_relative_phase_ccx_mirror,
    append_control_drop)
from toffoli_optimizer.core.equivalence_verifier import ExactEquivalenceVerifier
from toffoli_optimizer.core.orientation import PairCost, orient
from toffoli_optimizer.core.phase_observability import (
    is_phase_unobservable, default_affected_qubits)
from toffoli_optimizer.core.reach_local import local_reachable
from toffoli_optimizer.core.reachable_subspace import reachable_overapprox
from toffoli_optimizer.core.subspace_check import (
    check_on_subspace, certify_on_input_subspace)
from toffoli_optimizer.core import window_pairs as WP


# ------------------------------------------------------------------ independent oracles
def _cols(U_ex, U_sel, inputs):
    I = sorted(set(inputs))
    return U_sel[:, I], U_ex[:, I]


def indep_subroutine_equal(U_ex, U_sel, inputs, tol=1e-7):
    """U_sel[:, I] == e^{i phi} U_ex[:, I] with ONE phase (exact equality test)."""
    A, B = _cols(U_ex, U_sel, inputs)
    ov = np.vdot(B.ravel(), A.ravel())
    if abs(ov) < 1e-12:
        return False
    return float(np.max(np.abs(A - (ov / abs(ov)) * B))) < tol


def indep_observational_equal(U_ex, U_sel, inputs, tol=1e-7):
    """Exists a diagonal unitary Phi with U_sel[:, I] == Phi U_ex[:, I] (one phase per
    output row). Equivalent to equal basis-measurement statistics for EVERY input state
    supported on span{|x>: x in I}, superpositions included."""
    A, B = _cols(U_ex, U_sel, inputs)
    for y in range(A.shape[0]):
        k = int(np.argmax(np.abs(B[y])))
        if abs(B[y, k]) < 1e-12:
            if np.max(np.abs(A[y])) > tol:
                return False
            continue
        ph = A[y, k] / B[y, k]
        if abs(abs(ph) - 1) > 1e-6 or np.max(np.abs(A[y] - ph * B[y])) > tol:
            return False
    return True


def _U(qc):
    return Operator(qc).data


def _inputs(n, pins=()):
    pm = sum(1 << q for q in pins)
    return [x for x in range(1 << n) if not x & pm]


def _assert_certified(res, inputs, mode="subroutine"):
    rep = res["report"]
    assert rep["fell_back_to_exact"] is False, "selector fell back: its own certificate failed"
    U_ex, U_sel = _U(res["exact"]), _U(res["circuit"])
    f = indep_subroutine_equal if mode == "subroutine" else indep_observational_equal
    assert f(U_ex, U_sel, inputs), f"output not {mode}-equal to the exact lowering on the inputs"


U_CCX = _U((lambda q: (q.ccx(0, 1, 2), q)[1])(QuantumCircuit(3)))
U_RCCX = _U((lambda q: (q.rccx(0, 1, 2), q)[1])(QuantumCircuit(3)))


# ------------------------------------------------------------------ oracles self-test
def test_independent_oracles_discriminate():
    """The oracles themselves must not be vacuous."""
    reach = [0b011, 0b111]
    assert indep_subroutine_equal(U_CCX, np.exp(0.7j) * U_CCX, range(8))
    assert not indep_subroutine_equal(U_CCX, U_RCCX, reach)
    D = np.diag(np.exp(1j * np.arange(8)))
    assert indep_observational_equal(U_CCX, D @ U_CCX, range(8))
    assert not indep_subroutine_equal(U_CCX, D @ U_CCX, range(8))
    # for a permutation matrix column phases ARE row phases; use a generic unitary
    rng = np.random.default_rng(5)
    q, r = np.linalg.qr(rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8)))
    U = q * (np.diag(r) / np.abs(np.diag(r)))
    assert indep_observational_equal(U, D @ U, range(8))
    assert not indep_observational_equal(U, U @ D, range(8))


# ------------------------------------------------------------------ fail-closed contract
def test_failed_certificate_falls_back_and_verified_flag_stays_true(monkeypatch):
    """Documents why ``assert rep['verified']`` is vacuous: a FAILED certificate still
    yields verified=True (with fell_back_to_exact=True and the exact circuit)."""
    monkeypatch.setenv("QCPT_CERT_CPP", "0")                  # the injected failure is in the numpy certificate
    monkeypatch.setattr(DS, "certify_on_input_subspace",
                        lambda *a, **k: (False, 1.0, {"method": "forced_fail"}))
    c = QuantumCircuit(3); c.h(0); c.ccx(0, 1, 2); c.cx(2, 1)
    res = ErrorBudgetSelector().select(c, pinned_zero=[2])
    rep = res["report"]
    assert rep["verified"] is True and rep["fell_back_to_exact"] is True
    assert rep["rphase_admitted"] == [] and res["actions"] == {}
    assert _U(res["circuit"]).shape == _U(res["exact"]).shape
    assert indep_subroutine_equal(_U(res["exact"]), _U(res["circuit"]), range(8))


# ------------------------------------------------------------------ subspace checks
def test_check_on_subspace_one_phase_not_basiswise():
    reach = [0b011, 0b111]
    ok, dev, _ = check_on_subspace(U_RCCX, U_CCX, reach, 0.0)
    assert not ok and dev > 0.5
    # one basis state: the phase is global there, so it must be accepted
    ok1, dev1, _ = check_on_subspace(U_RCCX, U_CCX, [0b011], 0.0)
    assert ok1 and dev1 == 0.0
    ok2, _, _ = check_on_subspace(np.exp(1.1j) * U_CCX, U_CCX, range(8), 0.0)
    assert ok2


def test_certify_subroutine_rejects_relative_phase_accepts_global():
    reach = [0b011, 0b111]
    assert not certify_on_input_subspace(U_CCX, U_RCCX, reach, "subroutine")[0]
    assert certify_on_input_subspace(U_CCX, np.exp(2.3j) * U_CCX, range(8), "subroutine")[0]
    assert certify_on_input_subspace(U_CCX, U_RCCX, [0b011], "subroutine")[0]
    # a wrong function is rejected in both modes
    U_I = np.eye(8)
    assert not certify_on_input_subspace(U_CCX, U_I, range(8), "subroutine")[0]
    assert not certify_on_input_subspace(U_CCX, U_I, range(8), "observational")[0]


def test_certify_observational_row_phase_vs_column_phase():
    rng = np.random.default_rng(11)
    q, r = np.linalg.qr(rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8)))
    U = q * (np.diag(r) / np.abs(np.diag(r)))
    D = np.diag(np.exp(1j * rng.uniform(0, 2 * np.pi, 8)))
    assert certify_on_input_subspace(U, D @ U, range(8), "observational")[0]      # row phases: invisible
    assert not certify_on_input_subspace(U, U @ D, range(8), "observational")[0]  # column phases: visible
    assert not certify_on_input_subspace(U, D @ U, range(8), "subroutine")[0]


def test_certify_fast_path_borderline_uses_full_minimisation():
    """Column phases (0,...,0,delta): trace-phase deviation ~delta in (atol, 2 atol],
    minimal deviation ~delta/2 <= atol. Only the full minimisation certifies it."""
    k, delta, atol = 64, 1.6e-8, 1e-8
    B = np.eye(k, dtype=complex)
    A = B @ np.diag([1.0] * (k - 1) + [np.exp(1j * delta)])
    ok, dev, info = certify_on_input_subspace(B, A, range(k), "subroutine", atol=atol)
    assert ok and dev <= atol, (dev, info)
    # and a clear violation (delta = 10 atol) is rejected
    A2 = B @ np.diag([1.0] * (k - 1) + [np.exp(1j * 10 * atol)])
    assert not certify_on_input_subspace(B, A2, range(k), "subroutine", atol=atol)[0]


def test_verifier_distinguishes_relative_phase():
    """The ExactEquivalenceVerifier is used as the independent oracle by many tests;
    it must not be phase-blind."""
    v = ExactEquivalenceVerifier()
    a = QuantumCircuit(3); a.ccx(0, 1, 2)
    b = QuantumCircuit(3); b.rccx(0, 1, 2)
    c = QuantumCircuit(3); c.ccx(0, 1, 2); c.global_phase = 0.9
    assert not v.verify(a, b, allow_permutation=False)[0]
    assert v.verify(a, c, allow_permutation=False)[0]
    m = QuantumCircuit(3); append_relative_phase_ccx(m, 0, 1, 2)
    assert not v.verify(a, m, allow_permutation=False)[0]


# ------------------------------------------------------------------ pipeline (independent certificate)
def test_editor_circuit_no_gadget_and_exact():
    c = QuantumCircuit(3); c.x(0); c.x(1); c.h(2); c.ccx(0, 1, 2); c.h(2)
    res = ErrorBudgetSelector().select(c)
    rep = res["report"]
    assert not (rep["phase_aware_admitted"] or rep["rphase_admitted"]
                or rep["window_pairs_admitted"] or rep["sites_applied"])
    _assert_certified(res, range(8))


def test_control_drop_only_provably_set_control_independent():
    c = QuantumCircuit(3); c.h(0); c.x(1); c.ccx(0, 1, 2)
    res = ErrorBudgetSelector(epsilon=0.0).select(c, input_space=[0])
    assert [d["keep"] for d in res["report"]["approx_admitted"]] == [[0]]
    _assert_certified(res, [0])
    # load-bearing: dropping q0 instead (keep q1) is observably wrong on |000>
    sel = ErrorBudgetSelector()
    bad = sel._build(c, {2: ("control_drop", (1,))})
    assert not indep_subroutine_equal(_U(res["exact"]), _U(bad), [0])


def test_gate_level_state_not_used_as_input_independent():
    c = QuantumCircuit(3); c.x(0); c.x(1); c.ccx(0, 1, 2)
    res = ErrorBudgetSelector(epsilon=0.0).select(c, input_space=[0])
    _assert_certified(res, [0])
    sv = Statevector.from_int(0, 8).evolve(res["circuit"]).probabilities()
    assert abs(sv[0b111] - 1) < 1e-9


def test_mirror_gadget_clean_ancilla_independent_and_load_bearing():
    c = QuantumCircuit(3); c.h(0); c.ccx(0, 1, 2); c.cx(2, 1)
    res = ErrorBudgetSelector().select(c, pinned_zero=[2])
    assert [d["gadget"] for d in res["report"]["rphase_admitted"]] == ["relphase_m"]
    _assert_certified(res, _inputs(3, [2]))
    # same gadget with the ancilla NOT pinned is wrong -> the pin is what admits it
    forced = ErrorBudgetSelector()._build(c, {1: ("relphase_m",)})
    assert not indep_subroutine_equal(_U(res["exact"]), _U(forced), range(8))
    res_f = ErrorBudgetSelector().select(c)
    assert not res_f["report"]["rphase_admitted"]
    _assert_certified(res_f, range(8))


def test_window_pair_mirror_adder_independent():
    c = QuantumCircuit(5)
    c.ccx(0, 1, 2); c.cx(3, 2); c.cx(4, 0); c.cx(4, 0); c.cx(3, 2); c.ccx(0, 1, 2)
    res = ErrorBudgetSelector().select(c)
    assert [w["pair"] for w in res["report"]["window_pairs_admitted"]] == [(0, 5)]
    _assert_certified(res, range(32))
    # an H inside the window breaks the cancellation: not admitted, no fallback needed
    d = QuantumCircuit(3); d.ccx(0, 1, 2); d.h(0); d.ccx(0, 1, 2)
    res_d = ErrorBudgetSelector().select(d)
    assert not res_d["report"]["window_pairs_admitted"]
    _assert_certified(res_d, range(8))
    forced = ErrorBudgetSelector()._build(d, {0: ("relphase",), 2: ("relphase",)})
    assert not indep_subroutine_equal(_U(res_d["exact"]), _U(forced), range(8))


def test_window_pairs_disjoint_endpoints_independent():
    c = QuantumCircuit(4)
    c.ccx(0, 1, 2); c.cx(3, 0); c.cx(3, 0); c.ccx(0, 1, 2); c.cx(2, 3); c.ccx(0, 1, 2)
    cands = WP.mirror_pair_candidates(c)
    ends = [k for pr in cands for k in pr]
    assert len(ends) == len(set(ends)), cands
    res = ErrorBudgetSelector().select(c)
    ends_adm = [k for w in res["report"]["window_pairs_admitted"] for k in w["pair"]]
    assert len(ends_adm) == len(set(ends_adm))
    _assert_certified(res, range(16))


def test_pair_unitary_exact_independent():
    c = QuantumCircuit(4); c.ccx(0, 1, 2); c.cx(2, 3); c.ccx(0, 1, 2)
    res = ErrorBudgetSelector().select(c)
    rep = res["report"]
    assert rep["sites_applied"] == 1 and rep["two_qubit_after"] == 7
    _assert_certified(res, range(16))


def test_program_semantics_lone_toffoli_observational_not_subroutine():
    c = QuantumCircuit(3); c.ccx(0, 1, 2)
    res = ErrorBudgetSelector(semantics="program").select(c)
    assert len(res["report"]["phase_aware_admitted"]) == 1
    _assert_certified(res, range(8), mode="observational")
    # (U) really changes the unitary: NOT subroutine-equal
    assert not indep_subroutine_equal(_U(res["exact"]), _U(res["circuit"]), range(8))
    res_s = ErrorBudgetSelector().select(c)
    assert not res_s["report"]["phase_aware_admitted"]
    _assert_certified(res_s, range(8))


def test_fig1b_rejected_and_forced_gadget_observably_wrong():
    c = QuantumCircuit(3); c.ccx(0, 1, 2); c.h(0)
    res = ErrorBudgetSelector(semantics="program").select(c)
    assert not res["report"]["phase_aware_admitted"]
    _assert_certified(res, range(8), mode="observational")
    forced = ErrorBudgetSelector()._build(c, {0: ("relphase",)})
    assert not indep_observational_equal(_U(res["exact"]), _U(forced), _inputs(3, [2]))


# ------------------------------------------------------------------ window laminarity
def _crossing_circuit():
    # Found by a seeded search: without the laminarity rule windows (4,7) and (1,5)
    # are both admitted, cross, and the whole circuit is no longer certified.
    qc = QuantumCircuit(4)
    qc.x(1); qc.ccx(0, 1, 3); qc.cx(2, 3); qc.x(3); qc.ccx(0, 1, 2)
    qc.ccx(0, 1, 3); qc.cx(1, 0); qc.ccx(0, 1, 2)
    return qc


def test_crossing_windows_never_admitted():
    qc = _crossing_circuit()
    cands = WP.mirror_pair_candidates(qc)
    assert (1, 5) in cands and (4, 7) in cands       # the crossing candidates exist
    res = ErrorBudgetSelector().select(qc)
    wins = [w["pair"] for w in res["report"]["window_pairs_admitted"]]
    for a, b in itertools.combinations(wins, 2):
        assert not WP._crosses(a, b), wins
    _assert_certified(res, range(16))
    # load-bearing: admitting both crossing windows is wrong
    both = ErrorBudgetSelector()._build(qc, {1: ("relphase",), 5: ("relphase",),
                                             4: ("relphase",), 7: ("relphase",)})
    assert not indep_subroutine_equal(_U(res["exact"]), _U(both), range(16))


# ------------------------------------------------------------------ orientation
def _orient(qc, pins, cost):
    sel = ErrorBudgetSelector()
    res = sel.select(qc, pinned_zero=pins)
    out, info = orient(sel, qc, res, cost, pinned_zero=pins)
    return res, out, info


def test_orient_single_swap_rejected_inside_orient():
    """Standalone Margolus at CCX(0,1,2) admitted under (R) because abt=100 is
    unreachable (t = a AND NOT b is computed first); its swap moves the phase to
    abt=010, which IS reachable, so orient() itself must reject that swap."""
    qc = QuantumCircuit(4)
    qc.cx(1, 3); qc.x(3); qc.ccx(0, 3, 2); qc.ccx(0, 1, 2); qc.h(2)
    cost = PairCost({(1, 2): 0.05, (0, 2): 0.001}, default=0.01)
    res, out, info = _orient(qc, [2, 3], cost)
    assert res["actions"].get(3) == ("relphase",)
    assert info["rejected_R"] == 1 and info["oriented"] == 1 and info["kept_default"] is False
    assert info["actions"][3] == ("relphase",)          # not swapped
    assert indep_subroutine_equal(_U(res["exact"]), _U(out), _inputs(4, [2, 3]))
    # load-bearing: the swapped gadget at index 3 would be wrong
    bad = ErrorBudgetSelector()._build(qc, {**info["actions"], 3: ("relphase", "swap")})
    assert not indep_subroutine_equal(_U(res["exact"]), _U(bad), _inputs(4, [2, 3]))


def test_orient_window_group_swap_rejected_by_final_certificate():
    """A (W) window certified in the default orientation whose swapped orientation
    no longer cancels: only the whole-circuit re-check in orient() catches it."""
    qc = QuantumCircuit(4)
    qc.ccx(0, 1, 2); qc.x(3); qc.ccx(2, 0, 3); qc.cx(1, 2); qc.ccx(0, 1, 2)
    cost = PairCost({(1, 2): 0.05, (0, 2): 0.001}, default=0.01)
    res, out, info = _orient(qc, [], cost)
    assert [w["pair"] for w in res["report"]["window_pairs_admitted"]] == [(0, 4)]
    assert info["certificate"] is False and info["kept_default"] is True
    assert indep_subroutine_equal(_U(res["exact"]), _U(out), range(16))
    swapped = ErrorBudgetSelector()._build(qc, {0: ("relphase", "swap"), 4: ("relphase", "swap")})
    assert not indep_subroutine_equal(_U(res["exact"]), _U(swapped), range(16))


def test_orient_pair_output_independently_certified():
    qc = QuantumCircuit(4); qc.ccx(0, 1, 2); qc.cx(2, 3); qc.ccx(0, 1, 2)
    cost = PairCost({(1, 2): 0.05, (0, 2): 0.001}, default=0.01)
    res, out, info = _orient(qc, [2], cost)
    assert info["oriented"] == 2 and info["certificate"] is True
    assert indep_subroutine_equal(_U(res["exact"]), _U(out), _inputs(4, [2]))


# ------------------------------------------------------------------ reachability (property tests)
def _simulate(qc, s, upto):
    for inst in qc.data[:upto]:
        qb = [qc.find_bit(q).index for q in inst.qubits]
        n = inst.operation.name
        if n == "x":
            s ^= 1 << qb[0]
        elif n == "swap":
            if ((s >> qb[0]) ^ (s >> qb[1])) & 1:
                s ^= (1 << qb[0]) | (1 << qb[1])
        elif all((s >> c) & 1 for c in qb[:-1]):
            s ^= 1 << qb[-1]
    return s


def _random_classical(rng, n, L):
    qc = QuantumCircuit(n)
    for _ in range(L):
        k = int(rng.integers(4))
        q = [int(x) for x in rng.choice(n, 3, replace=False)]
        if k == 0: qc.x(q[0])
        elif k == 1: qc.cx(q[0], q[1])
        elif k == 2: qc.ccx(*q)
        else: qc.swap(q[0], q[1])
    return qc


def test_local_reachable_equals_bruteforce():
    rng = np.random.default_rng(2026)
    for _ in range(60):
        n = int(rng.integers(3, 7))
        qc = _random_classical(rng, n, int(rng.integers(1, 10)))
        g = int(rng.integers(0, len(qc.data) + 1))
        pins = [int(x) for x in rng.choice(n, int(rng.integers(0, n)), replace=False)]
        qbs = tuple(int(x) for x in rng.choice(n, 3, replace=False))
        got = local_reachable(qc, g, qbs, pins)
        exp = set()
        for x in _inputs(n, pins):
            s = _simulate(qc, x, g)
            exp.add(sum(((s >> q) & 1) << j for j, q in enumerate(qbs)))
        assert got == sorted(exp), (qc, g, pins, qbs)


def test_overapprox_contains_true_support_random():
    rng = np.random.default_rng(99)
    gates_1q = ["h", "x", "t", "ry"]
    for _ in range(80):
        n = int(rng.integers(2, 5))
        qc = QuantumCircuit(n)
        for _ in range(int(rng.integers(1, 7))):
            k = int(rng.integers(5))
            q = [int(x) for x in rng.choice(n, 2, replace=False)]
            if k < 3:
                g = gates_1q[int(rng.integers(4))]
                (qc.ry(0.7, q[0]) if g == "ry" else getattr(qc, g)(q[0]))
            elif k == 3: qc.cx(q[0], q[1])
            else: qc.swap(q[0], q[1])
        space = sorted({int(x) for x in rng.integers(0, 1 << n, int(rng.integers(1, 4)))})
        over = reachable_overapprox(qc, len(qc.data), input_space=space)
        for x in space:
            sv = Statevector.from_int(x, 1 << n).evolve(qc).data
            supp = {i for i, a in enumerate(sv) if abs(a) > 1e-9}
            assert supp <= over, (qc, space, supp, sorted(over))


def test_overapprox_clean_bits_from_nonzero_input():
    qc = QuantumCircuit(3); qc.h(0); qc.cx(2, 1)   # qubit 2 clean = 1, so q1 flips
    over = reachable_overapprox(qc, 2, input_space=[0b100])
    assert over == frozenset({0b110, 0b111}), sorted(over)


# ------------------------------------------------------------------ phase observability
def test_phase_taint_propagates_through_classical_gates():
    c = QuantumCircuit(4); c.ccx(0, 1, 2); c.cx(2, 3); c.h(3)
    assert not is_phase_unobservable(c, 0, default_affected_qubits(c, 0))
    d = QuantumCircuit(4); d.ccx(0, 1, 2); d.swap(2, 3); d.h(3)
    assert not is_phase_unobservable(d, 0, default_affected_qubits(d, 0))


# ------------------------------------------------------------------ verify gate (depth optimizer)







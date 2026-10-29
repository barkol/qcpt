"""Adversarial soundness tests for the certification core (toffoli_optimizer/core).

Ground truth everywhere is computed independently of the package's own certificate
code: dense ``qiskit.quantum_info.Operator`` of the ORIGINAL circuit (not the
package's exact decomposition) and a direct column/row comparison.

Semantics checked
-----------------
* subroutine:    U' P_in = e^{i theta} U P_in  (one phase), P_in = span of valid
                 inputs (pinned qubits |0>, all other qubits free);
* observational: U' P_in = Phi U P_in with Phi diagonal (one phase per output row),
                 i.e. equal computational-basis output distributions for every
                 input state on span(P_in).

Bugs found during the audit are pinned as ``xfail(strict=True)`` tests at the end of
this file (section "confirmed bugs").
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from toffoli_optimizer.core import decomposition_selector as ds
from toffoli_optimizer.core import subspace_check as sc
from toffoli_optimizer.core.context_analysis import find_relative_phase_safe_sites
from toffoli_optimizer.core.decomposition_selector import (
    ErrorBudgetSelector, append_exact_ccx, append_relative_phase_ccx,
    append_relative_phase_ccx_mirror, _U_CCX, _U_MARG, _U_MARG_M)
from toffoli_optimizer.core.orientation import (
    PairCost, orient, pair_cost_from_calibration, _gadget_cost, _swapped_local_unitary)
from toffoli_optimizer.core.phase_observability import (
    _phase_unobservable_in_forward_cone, default_affected_qubits, is_phase_unobservable)
from toffoli_optimizer.core.reach_local import local_reachable, propagate_prefix
from toffoli_optimizer.core.reachable_subspace import (
    reachable_basis_states, reachable_overapprox)
from toffoli_optimizer.core.subspace_check import (
    certify_on_input_subspace, check_on_subspace, gadget_unitary, project_support,
    qcec_certify_on_subspace)
from toffoli_optimizer.core.window_pairs import (
    _crosses, admit_window_pairs, mirror_pair_candidates, window_certified)


# =============================================================================
# independent helpers
# =============================================================================
def rand_circ(rng, n, m, classical_bias=0.7, allow_mcx=False, extra_diag=False):
    """Random circuit over X/CX/CCX (+ H/T/S/RZ/CZ/SWAP, optionally MCX, CY/CCZ)."""
    qc = QuantumCircuit(n)
    for _ in range(m):
        r = rng.random()
        if r < 0.35:
            a, b, t = (int(x) for x in rng.choice(n, 3, replace=False))
            qc.ccx(a, b, t)
        elif r < 0.55:
            a, b = (int(x) for x in rng.choice(n, 2, replace=False))
            qc.cx(a, b)
        elif r < 0.65:
            qc.x(int(rng.integers(n)))
        elif r < 0.65 + 0.35 * (1 - classical_bias):
            g = int(rng.integers(8 if extra_diag else 6))
            q = int(rng.integers(n))
            if g == 0:
                qc.h(q)
            elif g == 1:
                qc.t(q)
            elif g == 2:
                qc.s(q)
            elif g == 3:
                qc.rz(float(rng.uniform(0, 2 * np.pi)), q)
            elif g == 4:
                a, b = (int(x) for x in rng.choice(n, 2, replace=False))
                qc.cz(a, b)
            elif g == 5:
                a, b = (int(x) for x in rng.choice(n, 2, replace=False))
                qc.swap(a, b)
            elif g == 6:
                a, b = (int(x) for x in rng.choice(n, 2, replace=False))
                qc.cy(a, b)
            else:
                a, b, c = (int(x) for x in rng.choice(n, 3, replace=False))
                qc.ccz(a, b, c)
        else:
            if allow_mcx and n >= 4 and rng.random() < 0.2:
                qs = [int(x) for x in rng.choice(n, 4, replace=False)]
                qc.mcx(qs[:3], qs[3])
            else:
                a, b, t = (int(x) for x in rng.choice(n, 3, replace=False))
                qc.ccx(a, b, t)
    return qc


def valid_inputs(n, pins):
    pm = sum(1 << q for q in pins)
    return [x for x in range(1 << n) if not x & pm]


def subroutine_dev(U, V, cols):
    """max-entry deviation of V[:,cols] from e^{i th} U[:,cols], best trace phase."""
    A, B = V[:, cols], U[:, cols]
    tr = np.trace(B.conj().T @ A)
    if abs(tr) < 1e-12:
        return np.inf
    return float(np.abs(A - (tr / abs(tr)) * B).max())


def observational_dev(U, V, cols):
    """0 iff every output row of V[:,cols] is a phase times the row of U[:,cols]."""
    A, B = V[:, cols], U[:, cols]
    na, nb = np.linalg.norm(A, axis=1), np.linalg.norm(B, axis=1)
    inner = np.abs(np.einsum("ij,ij->i", B.conj(), A))
    return float(max(np.abs(na - nb).max(), np.abs(inner - na * nb).max()))


def brute_min_phase_opnorm(A, B, grid=721):
    """Independent min_theta ||A - e^{i theta} B||_2: dense grid, then Brent refine
    (scipy) around the best grid points."""
    from scipy.optimize import minimize_scalar

    def f(t):
        return float(np.linalg.norm(A - np.exp(1j * t) * B, 2))
    ths = np.linspace(-np.pi, np.pi, grid)
    vals = np.array([f(t) for t in ths])
    best = float(vals.min())
    h = ths[1] - ths[0]
    for i in np.argsort(vals)[:3]:
        r = minimize_scalar(f, bounds=(ths[i] - h, ths[i] + h), method="bounded",
                            options={"xatol": 1e-13})
        best = min(best, float(r.fun))
    return best


def prefix(qc, g):
    out = QuantumCircuit(qc.num_qubits)
    for inst in qc.data[:g]:
        out.append(inst)
    return out


def support_union(qc, g, inputs):
    U = Operator(prefix(qc, g)).data
    return set(np.nonzero(np.abs(U[:, inputs]) > 1e-10)[0].tolist())


# =============================================================================
# 1. gadget unitaries
# =============================================================================
def _diag_rel(U):
    D = _U_CCX.conj().T @ U
    assert np.abs(D - np.diag(np.diag(D))).max() < 1e-12, "gadget is not CCX * diagonal"
    # gadget = CCX D = D CCX (D trivial on the a=b=1 branch)
    assert np.allclose(U @ _U_CCX.conj().T, D)
    return np.diag(D)


def _local(a, b, t):
    return a | (b << 1) | (t << 2)


def test_exact_gadget_is_ccx():
    assert np.allclose(gadget_unitary(append_exact_ccx), _U_CCX, atol=1e-12)


def test_margolus_phase_only_on_abt_100():
    d = _diag_rel(_U_MARG)
    exp = np.ones(8)
    exp[_local(1, 0, 0)] = -1
    assert np.allclose(d, exp, atol=1e-12)


def test_mirror_phase_only_on_abt_101():
    d = _diag_rel(_U_MARG_M)
    exp = np.ones(8)
    exp[_local(1, 0, 1)] = -1
    assert np.allclose(d, exp, atol=1e-12)


@pytest.mark.parametrize("kind,branch", [("relphase", (0, 1, 0)), ("relphase_m", (0, 1, 1))])
def test_swapped_gadget_moves_phase(kind, branch):
    d = _diag_rel(_swapped_local_unitary(kind))
    exp = np.ones(8)
    exp[_local(*branch)] = -1
    assert np.allclose(d, exp, atol=1e-12)


def test_gadget_cx_counts():
    for fn, ncx in ((append_exact_ccx, 6), (append_relative_phase_ccx, 3),
                    (append_relative_phase_ccx_mirror, 3)):
        qc = QuantumCircuit(3)
        fn(qc, 0, 1, 2)
        assert qc.count_ops().get("cx", 0) == ncx


# =============================================================================
# 2. whole selector: end-to-end soundness against the ORIGINAL circuit
# =============================================================================
def _sweep(seed, count, semantics, allow_mcx=False, window_pairs=True):
    rng = np.random.default_rng(seed)
    nontrivial = 0
    for it in range(count):
        n = int(rng.integers(3, 7))
        qc = rand_circ(rng, n, int(rng.integers(2, 12)),
                       classical_bias=float(rng.choice([0.6, 0.9, 1.0])), allow_mcx=allow_mcx)
        pins = tuple(sorted(int(x) for x in rng.choice(n, int(rng.integers(0, n)), replace=False)))
        res = ErrorBudgetSelector(semantics=semantics, window_pairs=window_pairs).select(
            qc, pinned_zero=pins)
        rep = res["report"]
        U, V = Operator(qc).data, Operator(res["circuit"]).data
        cols = valid_inputs(n, pins)
        if semantics == "subroutine":
            assert rep["semantics"] == "subroutine"
        if rep["semantics"] == "subroutine":
            d = subroutine_dev(U, V, cols)
        else:
            d = observational_dev(U, V, cols)
        assert d < 1e-7, (it, pins, rep.summary(), str(qc))
        # the reported 2q count is the real one
        assert rep["two_qubit_after"] == sum(
            1 for i in res["circuit"].data if i.operation.num_qubits == 2)
        nontrivial += rep["two_qubit_after"] < rep["two_qubit_before"]
    return nontrivial


def test_selector_subroutine_sound_random():
    nontrivial = _sweep(seed=101, count=300, semantics="subroutine")
    assert nontrivial > 150  # the test is not vacuous: most circuits were optimised


def test_selector_program_observationally_sound_random():
    nontrivial = _sweep(seed=202, count=150, semantics="program")
    assert nontrivial > 75


def test_selector_mcx_sound_without_windows():
    # 4-qubit MCX gates (crash in the window path is pinned separately below)
    _sweep(seed=303, count=120, semantics="subroutine", allow_mcx=True, window_pairs=False)


def test_selector_explicit_input_space_sound():
    """Restricted input domains (arbitrary sets of basis inputs): soundness on span(I)."""
    rng = np.random.default_rng(505050)
    nontrivial = 0
    for it in range(120):
        n = int(rng.integers(3, 7))
        qc = rand_circ(rng, n, int(rng.integers(2, 12)),
                       classical_bias=float(rng.choice([0.6, 1.0])))
        I = sorted({int(x) for x in rng.choice(1 << n, int(rng.integers(1, 1 << n)))})
        sem = "program" if rng.random() < 0.25 else "subroutine"
        res = ErrorBudgetSelector(semantics=sem).select(qc, input_space=I)
        rep = res["report"]
        U, V = Operator(qc).data, Operator(res["circuit"]).data
        d = subroutine_dev(U, V, I) if rep["semantics"] == "subroutine" \
            else observational_dev(U, V, I)
        assert d < 1e-7, (it, I, rep.summary(), str(qc))
        nontrivial += rep["two_qubit_after"] < rep["two_qubit_before"]
    assert nontrivial > 50


def test_selector_mirror_structured_windows_laminar():
    """S . M . S^-1 circuits stress conditions (C)/(W): soundness + laminar windows,
    no shared endpoints, each Toffoli in at most one window."""
    rng = np.random.default_rng(404)
    n_w = 0
    for it in range(200):
        n = int(rng.integers(3, 7))
        S = rand_circ(rng, n, int(rng.integers(1, 6)), classical_bias=1.0)
        M = rand_circ(rng, n, int(rng.integers(0, 4)), classical_bias=float(rng.choice([0.5, 1.0])))
        qc = QuantumCircuit(n)
        qc.compose(S, inplace=True)
        qc.compose(M, inplace=True)
        qc.compose(S.inverse(), inplace=True)
        pins = tuple(int(x) for x in rng.choice(n, int(rng.integers(0, n)), replace=False))
        res = ErrorBudgetSelector().select(qc, pinned_zero=pins)
        rep = res["report"]
        cols = valid_inputs(n, pins)
        assert subroutine_dev(Operator(qc).data, Operator(res["circuit"]).data, cols) < 1e-7
        wins = [tuple(s[:2]) for s in rep["applied_sites"]] + \
               [tuple(w["pair"]) for w in rep["window_pairs_admitted"]]
        n_w += len(rep["window_pairs_admitted"])
        ends = [e for w in wins for e in w]
        assert len(ends) == len(set(ends))
        for a in wins:
            for b in wins:
                if a != b:
                    assert not _crosses(a, b)
        # every window endpoint got a gadget
        for i in ends:
            assert res["actions"][i][0] in ("relphase", "relphase_m")
    assert n_w > 50


def test_selector_fail_closed_dense(monkeypatch):
    """A failing dense certificate must return the all-exact circuit."""
    monkeypatch.setenv("QCPT_CERT_CPP", "0")                  # the injected failure is in the numpy certificate
    monkeypatch.setattr(ds, "certify_on_input_subspace",
                        lambda *a, **k: (False, 1.0, {"forced": True}))
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    res = ErrorBudgetSelector().select(qc, pinned_zero=(2,))
    rep = res["report"]
    assert rep["fell_back_to_exact"] and res["actions"] == {}
    assert rep["two_qubit_after"] == rep["two_qubit_before"] == 6
    assert np.allclose(Operator(res["circuit"]).data, Operator(qc).data, atol=1e-9)


def test_selector_fail_closed_qcec(monkeypatch):
    """Above 12 qubits the QCEC branch is used; a failing QCEC must fall back."""
    monkeypatch.setenv("QCPT_BLOCK_CERT", "0")                # the injected failure is in the QCEC certificate
    monkeypatch.setattr(sc, "qcec_certify_on_subspace", lambda *a, **k: (False, {"forced": 1}))
    n = 13
    qc = QuantumCircuit(n)
    qc.ccx(0, 1, 2)
    qc.cx(2, 3)
    qc.ccx(0, 1, 2)
    res = ErrorBudgetSelector().select(qc, pinned_zero=(n - 1,))
    assert res["report"]["fell_back_to_exact"]
    assert res["report"]["two_qubit_after"] == res["report"]["two_qubit_before"]


def test_selector_fail_closed_exact_verifier():
    """No pins / no (R): final certificate is the exact verifier; if it refuses, exact."""
    class Refuse:
        calls = 0

        def verify(self, *a, **k):
            Refuse.calls += 1
            return False, None, {"reason": "refused"}
    qc = QuantumCircuit(4)
    qc.ccx(0, 1, 2)
    qc.cx(2, 3)
    qc.ccx(0, 1, 2)
    res = ErrorBudgetSelector(verifier=Refuse()).select(qc)
    assert res["report"]["fell_back_to_exact"] or res["report"]["two_qubit_after"] == \
        res["report"]["two_qubit_before"]
    assert np.allclose(Operator(res["circuit"]).data, Operator(qc).data, atol=1e-9)


def test_selector_and_into_clean_ancilla_uses_mirror_R():
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    res = ErrorBudgetSelector().select(qc, pinned_zero=(2,))
    assert res["actions"] == {0: ("relphase_m",)}
    U, V = Operator(qc).data, Operator(res["circuit"]).data
    assert subroutine_dev(U, V, valid_inputs(3, (2,))) < 1e-9
    # without the pin the gadget would be wrong on abt=101 -> must stay exact
    res2 = ErrorBudgetSelector().select(qc)
    assert res2["actions"] == {}


def test_selector_relative_phase_counterexample_rejected():
    """Docstring counterexample of subspace_check: controls=1, H on target around CCX.
    Basis-state-wise checks are blind; the subspace check must reject."""
    qc = QuantumCircuit(3)
    qc.h(2)
    qc.ccx(0, 1, 2)
    qc.h(2)
    for pins in ((), (0,), (1,)):
        res = ErrorBudgetSelector().select(qc, pinned_zero=pins)
        assert subroutine_dev(Operator(qc).data, Operator(res["circuit"]).data,
                              valid_inputs(3, pins)) < 1e-9


# =============================================================================
# 3. reachability
# =============================================================================
def test_reach_local_equals_brute_force_classical():
    rng = np.random.default_rng(505)
    for it in range(400):
        n = int(rng.integers(3, 7))
        qc = rand_circ(rng, n, int(rng.integers(1, 10)), classical_bias=1.0,
                       allow_mcx=True)
        pins = [int(x) for x in rng.choice(n, int(rng.integers(0, n)), replace=False)]
        g = int(rng.integers(0, len(qc.data) + 1))
        qs = tuple(int(x) for x in rng.choice(n, 3, replace=False))
        truth = project_support(support_union(qc, g, valid_inputs(n, pins)), qs)
        assert local_reachable(qc, g, qs, pins) == truth, (it, g, qs, pins, str(qc))


def test_reach_local_swap_and_mcx_exact():
    qc = QuantumCircuit(5)
    qc.x(0)
    qc.swap(0, 3)
    qc.mcx([1, 2, 3], 4)
    qc.cx(4, 0)
    for g in range(len(qc.data) + 1):
        truth = project_support(support_union(qc, g, valid_inputs(5, (1,))), (0, 3, 4))
        assert local_reachable(qc, g, (0, 3, 4), (1,)) == truth


def test_reach_local_refuses_nonclassical_and_wide():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.ccx(0, 1, 2)
    assert local_reachable(qc, 1, (0, 1, 2)) is None
    assert local_reachable(qc, 0, (0, 1, 2)) == list(range(8))
    qc2 = QuantumCircuit(4)
    qc2.ccx(0, 1, 2)
    assert propagate_prefix(qc2, 1, (), max_free_bits=3) is None
    # open-controlled gates have a different name and must not be treated as classical
    qc3 = QuantumCircuit(3)
    qc3.cx(0, 1, ctrl_state=0)
    qc3.ccx(0, 1, 2)
    assert local_reachable(qc3, 1, (0, 1, 2)) is None


def test_reachable_overapprox_is_superset():
    rng = np.random.default_rng(606)
    for it in range(500):
        n = int(rng.integers(3, 7))
        qc = rand_circ(rng, n, int(rng.integers(1, 10)),
                       classical_bias=float(rng.choice([0.5, 0.9, 1.0])), allow_mcx=True)
        if rng.random() < 0.4:
            inputs = sorted({int(x) for x in rng.choice(1 << n, int(rng.integers(1, 1 << n)))})
        else:
            pins = [int(x) for x in rng.choice(n, int(rng.integers(0, n)), replace=False)]
            inputs = valid_inputs(n, pins)
        g = int(rng.integers(0, len(qc.data) + 1))
        over = reachable_overapprox(qc, g, input_space=inputs)
        assert support_union(qc, g, inputs) <= set(over), (it, g, inputs, str(qc))


def test_reachable_overapprox_cap_widens_to_full_space():
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.h(1)
    qc.h(2)
    qc.ccx(0, 1, 3)
    over = reachable_overapprox(qc, 3, input_space=[0], max_basis=4)
    assert set(over) == set(range(16))
    # classical prefix but too many inputs -> full space, never an exception
    qc2 = QuantumCircuit(3)
    qc2.ccx(0, 1, 2)
    assert set(reachable_overapprox(qc2, 1, max_basis=2)) == set(range(8))


def test_reachable_basis_states_exact_and_refuses():
    qc = QuantumCircuit(3)
    qc.x(0)
    qc.cx(0, 1)
    qc.ccx(0, 1, 2)
    assert reachable_basis_states(qc, 3, input_space=[0, 4]) == frozenset({0b111, 0b011})
    qc.h(0)
    with pytest.raises(ValueError):
        reachable_basis_states(qc, 4)
    with pytest.raises(ValueError):
        reachable_basis_states(qc, 99)


# =============================================================================
# 4. subspace checks
# =============================================================================
def _rand_unitary(rng, d):
    z = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def test_check_on_subspace_matches_brute_min():
    rng = np.random.default_rng(707)
    for it in range(60):
        d = 8
        U = _rand_unitary(rng, d)
        # partially aligned: random diagonal phases on a subset
        ph = np.exp(1j * rng.uniform(0, 2 * np.pi)) * np.ones(d)
        k = int(rng.integers(0, d))
        ph[:k] *= np.exp(1j * rng.normal(scale=0.3, size=k))
        V = np.diag(ph) @ U if rng.random() < 0.5 else U @ np.diag(ph)
        cols = sorted({int(x) for x in rng.choice(d, int(rng.integers(1, d + 1)))})
        ok, dev, info = check_on_subspace(V, U, cols, 0.0)
        true = brute_min_phase_opnorm(V[:, cols], U[:, cols])
        # the package's value is a genuine function value (>= min) by construction;
        # the brute minimiser is itself only accurate to ~1e-7 on this nonsmooth f.
        A, B = V[:, cols], U[:, cols]
        assert dev == 0.0 or dev >= true - 1e-6
        if dev > 0.0:
            th = info["theta"]
            assert np.linalg.norm(A - np.exp(1j * th) * B, 2) <= dev + 1e-6
        assert dev <= true + 1e-6
        if true > 1e-6:
            assert not ok
        if true < 1e-12:
            assert ok


def test_check_on_subspace_edge_cases():
    U = _U_CCX
    assert check_on_subspace(U, U, [])[0]
    assert check_on_subspace(np.exp(0.7j) * U, U, range(8)) [:2] == (True, 0.0)
    # Margolus: one phase on the whole subspace only if abt=100 excluded
    no100 = [x for x in range(8) if x != _local(1, 0, 0)]
    assert check_on_subspace(_U_MARG, U, no100)[0]
    ok, dev, _ = check_on_subspace(_U_MARG, U, range(8))
    assert not ok and abs(dev - math.sqrt(2)) < 1e-6   # min_th |1 - e^{i th}| with both +-1
    # only abt=100 reachable -> phase -1 is a GLOBAL phase there -> admissible
    assert check_on_subspace(_U_MARG, U, [_local(1, 0, 0)])[0]
    # relative phase between two reachable states -> rejected
    assert not check_on_subspace(_U_MARG, U, [_local(1, 0, 0), _local(0, 0, 0)])[0]
    # epsilon variant
    assert check_on_subspace(_U_MARG, U, range(8), epsilon=1.5)[0]


def test_certify_on_input_subspace_phases():
    rng = np.random.default_rng(808)
    U = _rand_unitary(rng, 16)
    cols = [0, 3, 5, 9, 12]
    assert certify_on_input_subspace(U, np.exp(1.1j) * U, cols)[0]
    D = np.diag(np.exp(1j * rng.uniform(0, 2 * np.pi, 16)))
    # row phases: observational yes, subroutine no
    assert not certify_on_input_subspace(U, D @ U, cols, "subroutine")[0]
    assert certify_on_input_subspace(U, D @ U, cols, "observational")[0]
    # column (input) relative phase: neither (it is visible to interference)
    C = np.diag(np.exp(1j * rng.uniform(0, 2 * np.pi, 16)))
    assert not certify_on_input_subspace(U, U @ C, cols, "observational")[0]
    # difference only OUTSIDE the inputs -> certified
    V = U.copy()
    V[:, 1] = U[:, 2]
    V[:, 2] = U[:, 1]
    assert certify_on_input_subspace(U, V, cols, "subroutine")[0]
    with pytest.raises(ValueError):
        certify_on_input_subspace(U, U, cols, "nope")


def test_certify_on_input_subspace_never_accepts_beyond_atol():
    """Soundness of the factor-2 fast path: whenever it accepts (tolerance 0), the
    independently computed minimum deviation is within atol."""
    rng = np.random.default_rng(909)
    atol = 1e-8
    for it in range(200):
        d = 8
        U = _rand_unitary(rng, d)
        E = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
        E *= float(rng.choice([0.3, 0.9, 1.5, 2.5, 10])) * atol / np.linalg.norm(E, 2)
        V = np.exp(1j * rng.uniform(0, 6.3)) * U + E
        cols = sorted({int(x) for x in rng.choice(d, int(rng.integers(1, d + 1)))})
        ok, dev, _ = certify_on_input_subspace(U, V, cols, "subroutine", atol=atol)
        A, B = V[:, cols], U[:, cols]
        tr = np.trace(B.conj().T @ A)
        # independent UPPER bound on the minimum (any theta gives one)
        ub = min(brute_min_phase_opnorm(A, B),
                 float(np.linalg.norm(A - tr / abs(tr) * B, 2)))
        if ok:
            assert ub <= atol * (1 + 1e-6), (it, ub)
        else:
            assert ub > atol


def test_project_support():
    assert project_support([0b1010, 0b0001], [1, 3]) == [0b00, 0b11]
    assert project_support([0b1010, 0b0001], [0, 2]) == [0b00, 0b01]
    assert project_support([], [0]) == []


def test_qcec_certificate_semantics():
    e = QuantumCircuit(2)
    e.x(0)
    s = QuantumCircuit(2)
    s.cz(0, 1)
    s.x(0)                                       # differs only when pinned q1 = 1
    assert qcec_certify_on_subspace(e, s, (1,))[0]
    assert not qcec_certify_on_subspace(e, s, ())[0]
    s2 = QuantumCircuit(2)
    s2.z(0)
    s2.x(0)                                      # relative phase on valid inputs
    assert not qcec_certify_on_subspace(e, s2, (1,))[0]
    s3 = QuantumCircuit(2)
    s3.x(0)
    s3.cx(0, 1)                                  # leaves the pinned qubit dirty
    assert not qcec_certify_on_subspace(e, s3, (1,))[0]
    s4 = QuantumCircuit(2, global_phase=0.7)
    s4.x(0)
    assert qcec_certify_on_subspace(e, s4, (1,))[0]
    assert not qcec_certify_on_subspace(e, e, (), observational=True)[0]


def test_qcec_pinned_qubit_in_the_middle():
    # pinned qubit is not the last one: with_ancillas must reorder correctly
    e = QuantumCircuit(3)
    e.cx(0, 2)
    s = QuantumCircuit(3)
    s.cx(1, 2)                                  # equal iff... never with q1 free
    s.cx(0, 2)
    assert qcec_certify_on_subspace(e, s, (1,))[0]
    assert not qcec_certify_on_subspace(e, s, (0,))[0]


# =============================================================================
# 5. window pairs
# =============================================================================
def test_mirror_pair_candidates_properties():
    rng = np.random.default_rng(1001)
    for it in range(300):
        n = int(rng.integers(3, 6))
        qc = rand_circ(rng, n, int(rng.integers(2, 14)), classical_bias=0.9)
        excl = {int(x) for x in rng.choice(len(qc.data), int(rng.integers(0, 3)))} \
            if len(qc.data) else set()
        pairs = mirror_pair_candidates(qc, exclude=excl)
        idx = [k for p in pairs for k in p]
        assert len(idx) == len(set(idx))
        assert not (set(idx) & excl)
        assert [j - i for i, j in pairs] == sorted(j - i for i, j in pairs)
        key = {}
        for k, inst in enumerate(qc.data):
            if inst.operation.name == "ccx":
                qb = [qc.find_bit(q).index for q in inst.qubits]
                key[k] = (frozenset(qb[:2]), qb[2])
        for i, j in pairs:
            assert i < j and key[i] == key[j]
            # j is the NEAREST later Toffoli with the same key
            assert not any(i < k < j and key[k] == key[i] for k in key)


def test_crosses():
    assert _crosses((0, 5), (3, 8)) and _crosses((3, 8), (0, 5))
    assert _crosses((0, 5), (5, 8)) and _crosses((0, 5), (0, 3))
    assert not _crosses((0, 9), (3, 5)) and not _crosses((0, 2), (3, 5))


def test_window_certified_against_dense_segment():
    rng = np.random.default_rng(1101)
    seen_true = seen_false = 0
    for it in range(300):
        n = int(rng.integers(3, 6))
        qc = rand_circ(rng, n, int(rng.integers(3, 10)), classical_bias=0.8)
        pairs = mirror_pair_candidates(qc)
        for i, j in pairs:
            ok, _ = window_certified(qc, i, j, {i, j}, append_relative_phase_ccx)
            ref = QuantumCircuit(n)
            cand = QuantumCircuit(n)
            for k in range(i, j + 1):
                inst = qc.data[k]
                ref.append(inst)
                if k in (i, j):
                    qb = [qc.find_bit(q).index for q in inst.qubits]
                    append_relative_phase_ccx(cand, *qb)
                else:
                    cand.append(inst)
            A, B = Operator(cand).data, Operator(ref).data
            truth = brute_min_phase_opnorm(A, B, grid=361) < 1e-7
            assert ok == truth, (it, i, j, str(qc))
            seen_true += ok
            seen_false += not ok
    assert seen_true > 20 and seen_false > 20


def test_window_certified_refusals():
    qc = QuantumCircuit(3, 1)
    qc.ccx(0, 1, 2)
    qc.measure(0, 0)
    qc.ccx(0, 1, 2)
    assert not window_certified(qc, 0, 2, {0, 2}, append_relative_phase_ccx)[0]
    qc2 = QuantumCircuit(5)
    qc2.ccx(0, 1, 2)
    qc2.cx(3, 4)
    qc2.ccx(0, 1, 2)
    assert not window_certified(qc2, 0, 2, {0, 2}, append_relative_phase_ccx,
                                max_window_qubits=4)[0]
    assert window_certified(qc2, 0, 2, {0, 2}, append_relative_phase_ccx)[0]


def test_admit_window_pairs_laminar_with_accepted():
    rng = np.random.default_rng(1201)
    for it in range(200):
        n = int(rng.integers(3, 6))
        S = rand_circ(rng, n, int(rng.integers(1, 6)), classical_bias=1.0)
        qc = QuantumCircuit(n)
        qc.compose(S, inplace=True)
        qc.compose(S.inverse(), inplace=True)
        cands = mirror_pair_candidates(qc)
        acc = [cands[0]] if cands and rng.random() < 0.5 else []
        cur = set(acc[0]) if acc else set()
        gad, log = admit_window_pairs(qc, cur, append_relative_phase_ccx, accepted_windows=acc)
        wins = acc + [tuple(w["pair"]) for w in log]
        assert gad >= cur and gad == cur | {k for w in log for k in w["pair"]}
        ends = [e for w in wins for e in w]
        assert len(ends) == len(set(ends))
        for a in wins:
            for b in wins:
                if a != b:
                    assert not _crosses(a, b)
        # whole circuit with all gadgets equals the original up to a global phase
        sel = ErrorBudgetSelector()
        built = sel._build(qc, {k: ("relphase",) for k in gad})
        if all(window_certified(qc, *w, gad, append_relative_phase_ccx)[0] for w in acc):
            assert subroutine_dev(Operator(qc).data, Operator(built).data,
                                  list(range(1 << n))) < 1e-7


# =============================================================================
# 6. context analysis and phase observability
# =============================================================================
def test_structural_sites_cancel_when_controls_in_same_order():
    """Condition (C): for every structural site whose two Toffolis list the controls
    in the same order, replacing both by Margolus leaves the FULL unitary unchanged
    up to a global phase (independent of the selector's verifier)."""
    rng = np.random.default_rng(1301)
    checked = 0
    for it in range(400):
        n = int(rng.integers(3, 6))
        qc = rand_circ(rng, n, int(rng.integers(2, 12)), classical_bias=0.6, extra_diag=True)
        for s in find_relative_phase_safe_sites(qc):
            qi = [qc.find_bit(q).index for q in qc.data[s.compute_idx].qubits]
            qj = [qc.find_bit(q).index for q in qc.data[s.uncompute_idx].qubits]
            assert qi[2] == qj[2] == s.target and set(qi[:2]) == set(qj[:2])
            if qi != qj:
                continue
            built = ErrorBudgetSelector()._build(
                qc, {s.compute_idx: ("relphase",), s.uncompute_idx: ("relphase",)})
            exact = ErrorBudgetSelector().decompose_exact_only(qc)
            assert subroutine_dev(Operator(exact).data, Operator(built).data,
                                  list(range(1 << n))) < 1e-7, (it, s, str(qc))
            checked += 1
    assert checked > 30


def test_forward_cone_unobservable_is_sound():
    """If the forward-cone test says unobservable, replacing THAT single Toffoli by the
    Margolus gadget keeps every output row equal up to a phase on the full space."""
    rng = np.random.default_rng(1401)
    hits = 0
    for it in range(400):
        n = int(rng.integers(3, 6))
        qc = rand_circ(rng, n, int(rng.integers(2, 10)), classical_bias=0.7, extra_diag=True)
        for g, inst in enumerate(qc.data):
            if inst.operation.name != "ccx":
                continue
            if not _phase_unobservable_in_forward_cone(qc, g, default_affected_qubits(qc, g)):
                continue
            built = ErrorBudgetSelector()._build(qc, {g: ("relphase",)})
            assert observational_dev(Operator(qc).data, Operator(built).data,
                                     list(range(1 << n))) < 1e-7, (it, g, str(qc))
            hits += 1
    assert hits > 50


def test_pair_reason_requires_both_halves():
    """Reason (a) of is_phase_unobservable is a statement about the PAIR: it returns
    True for one half, yet replacing only that half is observable. The selector must
    therefore never use reason (a) for a single gate (it skips paired indices)."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.ccx(0, 1, 2)
    qc.ccx(0, 1, 2)
    qc.h(0)
    assert is_phase_unobservable(qc, 1, default_affected_qubits(qc, 1))
    single = ErrorBudgetSelector()._build(qc, {1: ("relphase",)})
    assert observational_dev(Operator(qc).data, Operator(single).data, list(range(8))) > 0.1
    res = ErrorBudgetSelector(semantics="program").select(qc)
    assert observational_dev(Operator(qc).data, Operator(res["circuit"]).data,
                             list(range(8))) < 1e-9
    assert not res["report"]["phase_aware_admitted"]


def test_forward_cone_rejects_interference():
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    qc.cx(2, 0)
    qc.h(0)
    assert not _phase_unobservable_in_forward_cone(qc, 0, {0, 1, 2})
    qc2 = QuantumCircuit(4)
    qc2.ccx(0, 1, 2)
    qc2.h(3)
    qc2.cx(2, 3)
    qc2.measure_all()
    assert _phase_unobservable_in_forward_cone(qc2, 0, {0, 1, 2})
    assert not _phase_unobservable_in_forward_cone(qc2, 0, set())


# =============================================================================
# 7. orientation
# =============================================================================
def _cx_cost(qc, cost):
    return sum(cost(*[qc.find_bit(q).index for q in i.qubits])
               for i in qc.data if i.operation.name == "cx")


def test_orientation_random_sound_and_never_more_expensive():
    rng = np.random.default_rng(1501)
    oriented = 0
    for it in range(150):
        n = int(rng.integers(3, 7))
        qc = rand_circ(rng, n, int(rng.integers(2, 10)),
                       classical_bias=float(rng.choice([0.8, 1.0])))
        pins = tuple(int(x) for x in rng.choice(n, int(rng.integers(0, n)), replace=False))
        cost = PairCost({(i, j): float(rng.uniform(0.001, 0.05))
                         for i in range(n) for j in range(i + 1, n)}, default=0.01)
        sem = "program" if rng.random() < 0.2 else "subroutine"
        sel = ErrorBudgetSelector(semantics=sem)
        res = sel.select(qc, pinned_zero=pins)
        out, info = orient(sel, qc, res, cost, pinned_zero=pins)
        cols = valid_inputs(n, pins)
        U, V = Operator(qc).data, Operator(out).data
        if res["report"]["semantics"] == "subroutine":
            assert subroutine_dev(U, V, cols) < 1e-7
        else:
            assert observational_dev(U, V, cols) < 1e-7
        assert _cx_cost(out, cost) <= _cx_cost(res["circuit"], cost) + 1e-12
        if out is not res["circuit"]:
            oriented += 1
            assert info["certificate"] is True
            assert _cx_cost(out, cost) < _cx_cost(res["circuit"], cost)
    assert oriented > 10


def test_orientation_equal_costs_never_swaps():
    rng = np.random.default_rng(1601)
    for it in range(60):
        n = int(rng.integers(3, 6))
        qc = rand_circ(rng, n, int(rng.integers(2, 8)), classical_bias=1.0)
        pins = tuple(int(x) for x in rng.choice(n, int(rng.integers(0, n)), replace=False))
        sel = ErrorBudgetSelector()
        res = sel.select(qc, pinned_zero=pins)
        out, info = orient(sel, qc, res, PairCost({}, default=0.02), pinned_zero=pins)
        assert info["oriented"] == 0 and out is res["circuit"]


def test_orientation_failed_certificate_keeps_default(monkeypatch):
    from toffoli_optimizer.core import orientation as om
    monkeypatch.setattr(om, "_certify", lambda *a, **k: False)
    qc = QuantumCircuit(4)
    qc.ccx(0, 1, 2)
    qc.cx(2, 3)
    qc.ccx(0, 1, 2)
    sel = ErrorBudgetSelector()
    res = sel.select(qc, pinned_zero=(2,))
    out, info = orient(sel, qc, res, PairCost({(1, 2): 0.05, (0, 2): 0.001}, 0.01),
                       pinned_zero=(2,))
    assert out is res["circuit"] and info["kept_default"] and info["certificate"] is False


def test_gadget_cost_and_path_cost():
    cost = PairCost({(1, 2): 0.05, (0, 2): 0.001}, default=0.01)
    assert _gadget_cost(0, 1, 2, False, cost) == pytest.approx(2 * 0.05 + 0.001)
    assert _gadget_cost(0, 1, 2, True, cost) == pytest.approx(2 * 0.001 + 0.05)
    assert cost(2, 1) == cost(1, 2)
    # line 0-1-2: adjacent = w, two hops = 3*w01 + w12
    pc = pair_cost_from_calibration({(0, 1): 0.01, (1, 2): 0.02}, [0, 1, 2])
    w = lambda e: -math.log(1 - e)
    assert pc(0, 1) == pytest.approx(w(0.01))
    assert pc(0, 2) == pytest.approx(3 * w(0.01) + w(0.02))
    # disconnected physical qubit -> infinite cost (never chosen as the doubled pair)
    pc2 = pair_cost_from_calibration({(0, 1): 0.01}, [0, 1, 5])
    assert math.isinf(pc2(0, 2))
    assert _gadget_cost(0, 1, 2, False, pc2) == math.inf


# =============================================================================
# 8. confirmed bugs (xfail strict: must start passing once fixed)
# =============================================================================
def test_select_rejects_measurement_with_a_clear_error():
    qc = QuantumCircuit(3, 3)
    qc.ccx(0, 1, 2)
    qc.measure(range(3), range(3))
    with pytest.raises(ValueError):
        ErrorBudgetSelector(semantics="program").select(qc)


def test_certify_empty_inputs():
    ok, dev, _ = certify_on_input_subspace(np.eye(4), np.eye(4), [])
    assert ok and dev == 0.0


def test_select_empty_input_space():
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    ErrorBudgetSelector().select(qc, input_space=[])


def test_mcx_site_inside_window_is_not_a_relphase_site():
    qc = QuantumCircuit(5)
    qc.ccx(0, 1, 4)
    qc.x(4)
    qc.mcx([0, 1, 2], 3)
    qc.mcx([0, 1, 2], 3)
    qc.x(4)
    qc.ccx(0, 1, 4)
    res = ErrorBudgetSelector().select(qc)
    assert Operator(res["circuit"]).equiv(Operator(qc))


def test_mcx_pair_is_not_a_relphase_site():
    """The relative-phase gadget has two controls: a pair of 4-qubit MCX gates is not reported as a site."""
    qc = QuantumCircuit(4)
    qc.mcx([0, 1, 2], 3)
    qc.mcx([0, 1, 2], 3)
    res = ErrorBudgetSelector(window_pairs=False).select(qc)
    assert not res["report"]["applied_sites"]
    assert Operator(res["circuit"]).equiv(Operator(qc))


def test_wide_circuit_without_pins_falls_back_to_exact():
    """Limitation: above 12 qubits, without pinned qubits and without (R) gadgets the final certificate is the dense
    exact verifier, which refuses; the pass then fails closed and returns the exact circuit. With one pin, QCEC certifies."""
    n = 13
    qc = QuantumCircuit(n)
    qc.ccx(0, 1, 2)
    qc.cx(2, 3)
    qc.ccx(0, 1, 2)
    for q in range(4, n):
        qc.cx(q - 1, q)
    with_pin = ErrorBudgetSelector().select(qc, pinned_zero=(n - 1,))["report"]
    assert not with_pin["fell_back_to_exact"]
    no_pin = ErrorBudgetSelector().select(qc)
    assert no_pin["report"]["fell_back_to_exact"]
    assert no_pin["circuit"].count_ops() == no_pin["exact"].count_ops()


def test_bug_qcec_accepts_small_rotation():
    e = QuantumCircuit(2)
    e.x(0)
    s = QuantumCircuit(2)
    s.x(0)
    s.rx(1e-4, 0)
    # QCEC is nondeterministic here (~75 % 'equivalent_up_to_global_phase', else
    # 'no_information'), so repeat: a sound certificate must never accept.
    assert not any(qcec_certify_on_subspace(e, s, ())[0] for _ in range(20))


def test_bug_inconclusive_final_verdict_is_not_fail_closed():
    class Inconclusive:
        def verify(self, *a, **k):
            return None, None, {"reason": "inconclusive"}
    qc = QuantumCircuit(4)
    qc.ccx(0, 1, 2)
    qc.cx(2, 3)
    qc.ccx(0, 1, 2)
    rep = ErrorBudgetSelector(verifier=Inconclusive()).select(qc)["report"]
    assert rep["verified"] is True
    assert rep["fell_back_to_exact"] or rep["two_qubit_after"] == rep["two_qubit_before"]


def _wire_permuted(U, perm, n):
    """Rows of U (indexed by wire bits) reordered so that bit i is the logical qubit i: wire perm[i] -> bit i."""
    idx = [sum(((x >> perm[i]) & 1) << i for i in range(n)) for x in range(1 << n)]
    inv = np.empty(1 << n, dtype=int)
    for x, y in enumerate(idx):
        inv[y] = x
    return U[inv, :]


def test_elide_swaps_removes_swaps_and_reports_the_output_permutation():
    qc = QuantumCircuit(5)
    qc.ccx(0, 1, 2); qc.swap(2, 3); qc.cx(3, 4); qc.swap(0, 4); qc.ccx(1, 3, 2); qc.swap(1, 2)
    res = ErrorBudgetSelector(elide_swaps=True).select(qc, pinned_zero=(4,))
    assert not any(g.operation.name == "swap" for g in res["circuit"].data)
    assert res["report"]["swaps_elided"] == 3
    perm = res["output_permutation"]
    assert sorted(perm) == list(range(5))
    cols = [x for x in range(32) if not (x >> 4) & 1]
    A = Operator(qc).data[:, cols]
    B = _wire_permuted(Operator(res["circuit"]).data, perm, 5)[:, cols]
    ph = np.vdot(A.ravel(), B.ravel()); ph /= abs(ph)
    assert np.abs(B - ph * A).max() < 1e-9


def test_elide_swaps_is_off_by_default():
    qc = QuantumCircuit(3)
    qc.swap(0, 1); qc.ccx(0, 1, 2)
    res = ErrorBudgetSelector().select(qc)
    assert "output_permutation" not in res


def test_zx_stage_keeps_the_certificate_and_never_gets_worse():
    pytest.importorskip("pyzx")
    qc = QuantumCircuit(4)
    qc.ccx(0, 1, 2); qc.cx(2, 3); qc.cx(2, 3); qc.h(3); qc.h(3); qc.ccx(0, 1, 2); qc.cx(0, 3)
    base = ErrorBudgetSelector().select(qc)
    res = ErrorBudgetSelector(simplify="zx").select(qc)
    em = ErrorBudgetSelector().error_model
    assert em.two_qubit_count(res["circuit"]) <= em.two_qubit_count(base["circuit"])
    assert "simplify" in res["report"]
    assert Operator(res["circuit"]).equiv(Operator(qc))


def test_zx_stage_fails_closed_on_a_wrong_candidate(monkeypatch):
    pytest.importorskip("pyzx")
    import toffoli_optimizer.core.zx_simplify as Z
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2); qc.cx(0, 2); qc.ccx(0, 1, 2)
    bad = QuantumCircuit(3); bad.x(0)                              # cheaper but not equivalent
    monkeypatch.setattr(Z, "zx_simplify", lambda c: bad)
    res = ErrorBudgetSelector(simplify="zx").select(qc)
    assert res["report"]["simplify"]["chosen"] is None
    assert Operator(res["circuit"]).equiv(Operator(qc))


def test_zx_option_is_validated():
    with pytest.raises(ValueError):
        ErrorBudgetSelector(simplify="bogus")


def test_peephole_stage_keeps_the_certificate_and_never_gets_worse():
    qc = QuantumCircuit(4)
    qc.ccx(0, 1, 2); qc.cx(2, 3); qc.cx(1, 3); qc.cx(2, 3); qc.ccx(0, 1, 2); qc.cx(0, 3); qc.cx(0, 3)
    base = ErrorBudgetSelector().select(qc)
    res = ErrorBudgetSelector(simplify="peephole").select(qc)
    em = ErrorBudgetSelector().error_model
    assert em.circuit_infidelity(res["circuit"]) <= em.circuit_infidelity(base["circuit"])
    assert res["report"]["simplify"]["mode"] == "peephole"
    assert Operator(res["circuit"]).equiv(Operator(qc))


def test_peephole_stage_fails_closed_on_a_wrong_candidate(monkeypatch):
    import toffoli_optimizer.core.peephole as P
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2); qc.cx(0, 2); qc.ccx(0, 1, 2)
    bad = QuantumCircuit(3); bad.x(0)
    monkeypatch.setattr(P, "peephole", lambda c, max_rounds=6: bad)
    res = ErrorBudgetSelector(simplify="peephole").select(qc)
    assert res["report"]["simplify"]["chosen"] is None
    assert Operator(res["circuit"]).equiv(Operator(qc))


def test_simplify_option_accepts_the_documented_modes():
    for m in (None, "zx", "peephole", "zx+peephole"):
        ErrorBudgetSelector(simplify=m)


def _cpp_binary():
    from toffoli_optimizer.core import cppcert
    return cppcert.available()


def test_cpp_certificate_agrees_with_the_numpy_certificate():
    if not _cpp_binary():
        pytest.skip("cpp/qcpt not built (make -C cpp)")
    from toffoli_optimizer.core.cppcert import certify_equal_up_to_phase
    qc = QuantumCircuit(5)
    qc.ccx(0, 1, 2); qc.cx(2, 3); qc.h(4); qc.ccx(0, 1, 2); qc.t(3); qc.ccx(3, 4, 1)
    ex = ErrorBudgetSelector().decompose_exact_only(qc)
    sel = ErrorBudgetSelector().select(qc)["circuit"]
    cols = list(range(32))
    assert certify_equal_up_to_phase(ex, sel, cols) is not None
    bad = ex.copy(); bad.rx(1e-3, 2)
    assert certify_equal_up_to_phase(ex, bad, cols) is None
    wrong = ex.copy(); wrong.z(0)
    assert certify_equal_up_to_phase(ex, wrong, cols) is None
    # a pure global phase difference is accepted, a relative phase between inputs is not
    ph = ex.copy(); ph.global_phase += 0.7
    assert certify_equal_up_to_phase(ex, ph, cols) is not None


def test_identical_circuits_skip_the_certificate():
    qc = QuantumCircuit(4)
    qc.ccx(0, 1, 2); qc.cx(2, 3)
    res = ErrorBudgetSelector().select(qc)
    assert res["report"]["verify_info"].get("certification") in ("exact_unitary", "input_subspace_subroutine")


def test_block_certificate_accepts_local_differences_and_rejects_wrong_blocks():
    from toffoli_optimizer.core.blockcert import block_certificate
    a = QuantumCircuit(6)
    for _ in range(3):
        a.h(0); a.cx(0, 1); a.t(1); a.cx(2, 3); a.ccx(3, 4, 5)
    b = a.copy()
    ok, info = block_certificate(a, b)
    assert ok and info["blocks"] == 0
    # an equivalent local rewrite (cx cx = identity inserted) is certified
    b = QuantumCircuit(6)
    for k in range(3):
        b.h(0); b.cx(0, 1); b.cx(4, 5); b.cx(4, 5); b.t(1); b.cx(2, 3); b.ccx(3, 4, 5)
    ok, info = block_certificate(a, b, budget_frac=1.0)
    assert ok
    # a wrong rewrite must NOT be certified
    c = a.copy()
    c.x(2)
    assert block_certificate(a, c, budget_frac=1.0)[0] is None
    c2 = QuantumCircuit(6)
    for _ in range(3):
        c2.h(0); c2.cx(0, 1); c2.tdg(1); c2.cx(2, 3); c2.ccx(3, 4, 5)
    assert block_certificate(a, c2, budget_frac=1.0)[0] is None


def test_block_certificate_slides_a_commuting_middle():
    """A compute/uncompute pair with a long middle on other qubits (controls only used as controls) is checked on its own qubits."""
    from toffoli_optimizer.core.blockcert import block_certificate
    def circ(pair_gadget):
        qc = QuantumCircuit(8)
        pair_gadget(qc, True)
        for _ in range(40):                                   # middle: only uses qubits 0 and 1 as controls, never the target 2
            qc.cx(0, 3); qc.ccx(1, 3, 4); qc.t(0); qc.cx(1, 5); qc.h(6); qc.cx(6, 7)
        pair_gadget(qc, False)
        return qc
    def exact(qc, first): qc.ccx(0, 1, 2)
    def relphase(qc, first):
        qc.ry(0.25 * 3.141592653589793, 2); qc.cx(1, 2); qc.ry(0.25 * 3.141592653589793, 2); qc.cx(0, 2)
        qc.ry(-0.25 * 3.141592653589793, 2); qc.cx(1, 2); qc.ry(-0.25 * 3.141592653589793, 2)
    ok, info = block_certificate(circ(exact), circ(relphase), budget_frac=1.0)
    assert ok and info["slid_gates"] > 0


def _mc_circuit():
    qc = QuantumCircuit(7)
    qc.ccx(0, 1, 2); qc.cx(3, 4); qc.h(5); qc.ccx(0, 1, 2)          # compute / uncompute pair, the middle does not touch the target
    qc.ccx(3, 4, 5); qc.cx(0, 6); qc.t(1); qc.ccx(3, 4, 5)
    qc.ccx(0, 2, 4); qc.cx(5, 6); qc.ccx(0, 2, 4)
    return qc


def test_monte_carlo_certificate_is_opt_in_and_reports_its_failure_probability():
    qc = _mc_circuit()
    res = ErrorBudgetSelector().select(qc)
    assert res["report"]["verify_info"].get("method") != "monte_carlo"
    res = ErrorBudgetSelector(certificate="monte_carlo").select(qc)
    vi = res["report"]["verify_info"]
    assert vi["method"] in ("monte_carlo", "monte_carlo_blocks") and vi["failure_probability"] < 1e-9 and not res["report"]["fell_back_to_exact"]
    assert Operator(res["circuit"]).equiv(Operator(qc))


def test_monte_carlo_certificate_rejects_wrong_circuits_and_fails_closed(monkeypatch):
    from toffoli_optimizer.core.cppcert import mc_compare
    qc = _mc_circuit()
    ex = ErrorBudgetSelector().decompose_exact_only(qc)
    full = range(1 << 7)
    assert mc_compare(ex, ex.copy(), full, seed=1)[0] == "equal"
    for k, bad in enumerate((lambda c: c.z(0), lambda c: c.rx(1e-4, 3), lambda c: c.cx(0, 6))):
        c = ex.copy(); bad(c)
        assert mc_compare(ex, c, full, seed=10 + k)[0] == "different"
    # a global phase is allowed, a relative phase between inputs is not
    c = ex.copy(); c.global_phase += 0.7
    assert mc_compare(ex, c, full, seed=3)[0] == "equal"
    # the selector falls back when the random test finds a counterexample
    import toffoli_optimizer.core.cppcert as cc
    monkeypatch.setattr(cc, "mc_compare", lambda *a, **k: ("different", {"method": "monte_carlo"}))
    res = ErrorBudgetSelector(certificate="monte_carlo").select(qc)
    assert res["report"]["fell_back_to_exact"]


def test_monte_carlo_option_is_validated_and_numpy_fallback_agrees(monkeypatch):
    with pytest.raises(ValueError):
        ErrorBudgetSelector(certificate="bogus")
    with pytest.raises(ValueError):
        ErrorBudgetSelector(certificate="monte_carlo", mc_vectors=1)
    import toffoli_optimizer.core.cppcert as cc
    qc = _mc_circuit()
    ex = ErrorBudgetSelector().decompose_exact_only(qc)
    monkeypatch.setattr(cc, "available", lambda: False)         # force the numpy path
    assert cc.mc_compare(ex, ex.copy(), range(1 << 7), seed=5)[0] == "equal"
    bad = ex.copy(); bad.z(2)
    assert cc.mc_compare(ex, bad, range(1 << 7), seed=5)[0] == "different"


def test_monte_carlo_blocks_certify_wide_circuits_and_reject_wrong_ones():
    """Blocks of difference are tested with random vectors on the qubits they touch, so the width of the circuit does not matter."""
    from toffoli_optimizer.core.blockcert import block_certificate
    from toffoli_optimizer.core.cppcert import mc_compare
    n = 20
    qc = QuantumCircuit(n)
    for b in range(0, 15, 3):
        qc.ccx(b, b + 1, b + 2); qc.cx(b + 3, b + 4); qc.h(b + 5); qc.ccx(b, b + 1, b + 2)
    sel = ErrorBudgetSelector(certificate="monte_carlo").select(qc)
    vi = sel["report"]["verify_info"]
    assert vi["method"] in ("monte_carlo_blocks", "monte_carlo") and vi["failure_probability"] < 1e-9
    assert not sel["report"]["fell_back_to_exact"] and sel["report"]["two_qubit_after"] < sel["report"]["two_qubit_before"]
    ex = sel["exact"]
    bad = sel["circuit"].copy(); bad.rx(1e-3, 7)
    cmp = lambda la, lb, cols, tol: (lambda r: (r[0], tol if r[0] == "equal" else None))(mc_compare(la, lb, range(1 << la.num_qubits), eps=tol))
    assert block_certificate(ex, bad, max_qubits=22, compare=cmp, work_cost=lambda g, t: g * 24 * 2.0 ** t)[0] is None


def test_counterexample_of_the_stepwise_argument_is_handled_by_pairs_and_the_certificate():
    """C0 = H_a CCX(a,b,t) H_a h2 h3 H_a with h2 = h3 = CCX(a,c,d); a, b, t, d start in |0>, c = |1>. The support of h2 is not {a=0}
    (a is released by the Hadamard), h2 and h3 form a pair under (C); the first Toffoli has a control that is |0>."""
    qc = QuantumCircuit(5)
    qc.x(3); qc.h(0); qc.ccx(0, 1, 2); qc.h(0); qc.ccx(0, 3, 4); qc.ccx(0, 3, 4); qc.h(0)
    res = ErrorBudgetSelector().select(qc, pinned_zero=(0, 1, 2, 4))
    rep = res["report"]
    assert not rep["fell_back_to_exact"] and (rep["two_qubit_before"], rep["two_qubit_after"]) == (18, 6)
    assert rep["sites_applied"] == 1

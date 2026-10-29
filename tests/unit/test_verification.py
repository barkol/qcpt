"""Adversarial unit tests for the verification layer.

Scope: equivalence_verifier, scalable_verification, subspace_check.qcec_certify_on_subspace /
with_ancillas, toffoli_count_reducer, pattern_library_module.

Ground truth everywhere is a dense ``qiskit.quantum_info.Operator`` on small circuits.
Tests marked ``xfail(strict=True)`` are minimal reproducers of CONFIRMED bugs (audit
2026-10-02, report B_weryfikacja.md); they flip to XPASS (= failure) once fixed.
"""

from __future__ import annotations

import itertools
import random

import numpy as np
import pytest
from qiskit import AncillaRegister, ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import CCXGate, CXGate, MCXGate, XGate
from qiskit.quantum_info import Operator

from toffoli_optimizer.core import scalable_verification as sv
from toffoli_optimizer.core.equivalence_verifier import ExactEquivalenceVerifier
from toffoli_optimizer.core.scalable_verification import ScalableVerifyResult, verify_scalable
from toffoli_optimizer.core.subspace_check import qcec_certify_on_subspace, with_ancillas
from toffoli_optimizer.core.toffoli_count_reducer import ToffoliCountReducer, count_toffoli

try:  # pragma: no cover - environment probe
    import mqt.qcec  # noqa: F401
    HAS_QCEC = True
except Exception:  # pragma: no cover
    HAS_QCEC = False

needs_qcec = pytest.mark.skipif(not HAS_QCEC, reason="mqt.qcec not installed")


# =============================================================== ground truth
def _eq_up_to_phase(A: np.ndarray, B: np.ndarray, atol: float = 1e-8) -> bool:
    """A == e^{i phi} B for one phi (A, B same shape)."""
    if A.shape != B.shape:
        return False
    k = np.unravel_index(np.argmax(np.abs(B)), B.shape)
    if abs(B[k]) < 1e-12:
        return np.max(np.abs(A)) < atol
    if abs(A[k]) < 1e-12:
        return False
    ph = B[k] / A[k]
    return abs(abs(ph) - 1.0) < 1e-6 and np.allclose(A * ph, B, atol=atol)


def _perm_matrix(sigma, n):
    """P with output bit j of the rewrite moved to position sigma[j]."""
    d = 1 << n
    P = np.zeros((d, d))
    for y in range(d):
        yp = 0
        for j in range(n):
            if (y >> j) & 1:
                yp |= 1 << sigma[j]
        P[yp, y] = 1.0
    return P


def dense_truth(original, rewritten, allow_permutation=False) -> bool:
    """Ground truth of the verifier's contract: extra high-index qubits of
    ``rewritten`` are clean ancillas (|0> in, |0> out), equality up to ONE global
    phase, optionally up to an output-wire permutation of the data wires."""
    n, nr = original.num_qubits, rewritten.num_qubits
    if nr < n:
        return False
    Uo = Operator(original).data
    Ur = Operator(rewritten).data
    d = 1 << n
    if nr > n and np.max(np.abs(Ur[d:, :d])) > 1e-9:
        return False
    B = Ur[:d, :d]
    if _eq_up_to_phase(B, Uo):
        return True
    if allow_permutation:
        return any(_eq_up_to_phase(_perm_matrix(s, n) @ B, Uo)
                   for s in itertools.permutations(range(n)))
    return False


def subspace_truth(exact, selected, pinned) -> bool:
    """U_sel P = e^{i theta} U_ex P, P = projector onto pinned qubits = |0>."""
    n = exact.num_qubits
    mask = sum(1 << p for p in pinned)
    cols = [x for x in range(1 << n) if not x & mask]
    return _eq_up_to_phase(Operator(selected).data[:, cols], Operator(exact).data[:, cols])


# ============================================================ random circuits
_CLASSICAL = ("x", "cx", "ccx", "swap")
_QUANTUM = ("x", "cx", "ccx", "h", "t", "tdg", "s", "rz")


def rand_circ(rng, n, m, gates=_QUANTUM, qubits=None):
    qubits = list(range(n)) if qubits is None else list(qubits)
    qc = QuantumCircuit(n)
    for _ in range(m):
        g = rng.choice(gates)
        if g == "ccx" and len(qubits) < 3:
            g = "cx"
        if g in ("cx", "swap") and len(qubits) < 2:
            g = "x"
        q = rng.sample(qubits, min(3, len(qubits)))
        if g == "cx":
            qc.cx(q[0], q[1])
        elif g == "ccx":
            qc.ccx(q[0], q[1], q[2])
        elif g == "swap":
            qc.swap(q[0], q[1])
        elif g == "rz":
            qc.rz(rng.choice([1, 2, 3, 5]) * np.pi / 4, q[0])
        else:
            getattr(qc, g)(q[0])
    return qc


def _mutate(rng, qc, gates, n_r):
    """Return a circuit of width n_r that is qc (on the low wires) plus a random
    perturbation -- sometimes trivial, so both verdicts occur."""
    out = QuantumCircuit(n_r)
    kind = rng.choice(["same", "prepend", "append", "phase", "cancel_pair", "anc_gadget"])
    if kind == "prepend":
        out.compose(rand_circ(rng, n_r, 1, gates), inplace=True)
    out.compose(qc, qubits=range(qc.num_qubits), inplace=True)
    if kind == "append":
        out.compose(rand_circ(rng, n_r, 1, gates), inplace=True)
    elif kind == "phase":
        out.global_phase += 0.37
    elif kind == "cancel_pair":
        g = rand_circ(rng, n_r, 1, ("x", "cx", "ccx", "swap"))
        out.compose(g, inplace=True)
        out.compose(g, inplace=True)
    elif kind == "anc_gadget" and n_r > qc.num_qubits:
        c = rng.sample(range(qc.num_qubits), 2)
        a = qc.num_qubits
        out.ccx(c[0], c[1], a)
        if rng.random() < 0.5:
            out.cx(a, rng.randrange(qc.num_qubits))
        out.ccx(c[0], c[1], a)  # restore ancilla
        if rng.random() < 0.3:
            out.x(a)            # ...or leave it dirty
    return out


# ========================================================== ExactEquivalenceVerifier
V = ExactEquivalenceVerifier()


@pytest.mark.parametrize("seed", range(4))
def test_truth_table_path_matches_dense_truth(seed):
    rng = random.Random(1000 + seed)
    for _ in range(60):
        n = rng.choice([2, 3])
        a = rand_circ(rng, n, rng.randint(1, 6), _CLASSICAL)
        b = _mutate(rng, a, _CLASSICAL, n + rng.choice([0, 0, 1]))
        ok, perm, info = V.verify(a, b)
        assert info.get("method") == "truth_table"
        assert bool(ok) == dense_truth(a, b), (a, b, info)
        if ok:
            assert perm == {i: i for i in range(n)}


@pytest.mark.parametrize("seed", range(4))
def test_unitary_path_matches_dense_truth(seed):
    rng = random.Random(2000 + seed)
    for _ in range(50):
        n = rng.choice([2, 3])
        a = rand_circ(rng, n, rng.randint(1, 6))
        b = _mutate(rng, a, _QUANTUM, n + rng.choice([0, 0, 1]))
        ok, _perm, info = V.verify(a, b)
        assert bool(ok) == dense_truth(a, b), (a, b, info)


@pytest.mark.parametrize("seed", range(3))
def test_permutation_option_matches_dense_truth(seed):
    rng = random.Random(3000 + seed)
    for _ in range(40):
        n = 3
        gates = _CLASSICAL if rng.random() < 0.5 else _QUANTUM
        a = rand_circ(rng, n, rng.randint(1, 5), gates)
        b = a.copy()
        if rng.random() < 0.6:  # output relabeling
            p, q = rng.sample(range(n), 2)
            b.swap(p, q)
        if rng.random() < 0.3:
            b.compose(rand_circ(rng, n, 1, gates), inplace=True)
        ok, perm, info = V.verify(a, b, allow_permutation=True)
        assert bool(ok) == dense_truth(a, b, allow_permutation=True), (a, b, info)
        if ok:
            # the returned map must actually reproduce the original
            sigma = [None] * n
            for orig_w, rew_w in perm.items():
                sigma[rew_w] = orig_w
            Ur = Operator(b).data
            assert _eq_up_to_phase(_perm_matrix(sigma, n) @ Ur, Operator(a).data)


def test_permutation_map_semantics_is_orig_to_rewritten():
    a = QuantumCircuit(3); a.h(2); a.swap(0, 2)   # output wire 0 carries H(input 2)
    b = QuantumCircuit(3); b.h(2)                 # ... which the rewrite leaves on wire 2
    ok, perm, _ = V.verify(a, b, allow_permutation=True)
    assert ok and perm[0] == 2
    assert not V.verify(a, b)[0]


def test_global_phase_only_is_equivalent_relative_phase_is_not():
    a = QuantumCircuit(3); a.ccx(0, 1, 2)
    b = QuantumCircuit(3); b.ccx(0, 1, 2); b.global_phase = 1.234
    assert V.verify(a, b)[0]
    # RCCX = CCX up to a RELATIVE phase -> must be rejected
    c = QuantumCircuit(3); c.rccx(0, 1, 2)
    assert not V.verify(a, c)[0]
    # S vs Z on a superposition: differ by a relative phase
    d = QuantumCircuit(1); d.h(0); d.s(0)
    e = QuantumCircuit(1); e.h(0); e.z(0)
    assert not V.verify(d, e)[0]
    # T vs global-phase-shifted identity on |1>
    f = QuantumCircuit(1); f.t(0)
    g = QuantumCircuit(1); g.global_phase = np.pi / 8
    assert not V.verify(f, g)[0]


@pytest.mark.parametrize("gate,width,ref", [
    (CXGate(ctrl_state=0), 2, "cx"),
    (CCXGate(ctrl_state=1), 3, "ccx"),
    (MCXGate(3, ctrl_state=2), 4, "mcx"),
    (XGate().control(3, ctrl_state="010"), 4, "mcx"),
])
def test_open_controls_are_not_confused_with_closed_controls(gate, width, ref):
    a = QuantumCircuit(width); a.append(gate, range(width))
    b = QuantumCircuit(width)
    if ref == "cx":
        b.cx(0, 1)
    elif ref == "ccx":
        b.ccx(0, 1, 2)
    else:
        b.mcx([0, 1, 2], 3)
    assert not V.verify(a, b)[0]
    assert V.verify(a, a.copy())[0]


@pytest.mark.parametrize("classical", [True, False])
def test_dirty_ancilla_is_rejected_clean_ancilla_accepted(classical):
    a = QuantumCircuit(2); a.cx(0, 1)
    if not classical:
        a.h(0); a.h(0)
    clean = QuantumCircuit(3); clean.cx(0, 2); clean.cx(2, 1); clean.cx(0, 2)
    if not classical:
        clean.h(0); clean.h(0)
    dirty = QuantumCircuit(3); dirty.cx(0, 2); dirty.cx(2, 1)   # ancilla keeps a copy
    if not classical:
        dirty.h(0); dirty.h(0)
    assert V.verify(a, clean)[0] is True
    ok, _, info = V.verify(a, dirty)
    assert ok is False and "ancilla" in info["reason"]


def test_ancilla_dependence_on_nonzero_ancilla_inputs_is_irrelevant():
    # rewrite behaves arbitrarily when ancilla=1 at input; must still verify
    a = QuantumCircuit(2); a.cx(0, 1)
    b = QuantumCircuit(3); b.cx(0, 1); b.ccx(2, 0, 1)
    assert V.verify(a, b)[0]
    assert dense_truth(a, b)


def test_fewer_qubits_and_too_wide_are_rejected():
    a = QuantumCircuit(3); a.h(0)
    b = QuantumCircuit(2); b.h(0)
    assert V.verify(a, b)[0] is False
    small = ExactEquivalenceVerifier(max_qubits_unitary=3)
    c = QuantumCircuit(4); c.h(0)
    ok, _, info = small.verify(c, c.copy())
    assert ok is False and "too large" in info["reason"]


@pytest.mark.parametrize("op", ["measure", "reset"])
def test_non_unitary_ops_never_certified(op):
    a = QuantumCircuit(2, 1); a.h(0)
    if op == "measure":
        a.measure(0, 0)
    else:
        a.reset(1)
    b = a.copy()
    try:
        ok = V.verify(a, b)[0]
    except Exception:
        ok = False   # raising is fail-closed
    assert ok is False


def test_reachable_basis_is_phase_insensitive_but_output_sensitive():
    a = QuantumCircuit(3); a.ccx(0, 1, 2)
    r = QuantumCircuit(3); r.rccx(0, 1, 2)
    ok, _, info = V.verify_on_reachable_basis(a, r, range(8))
    assert ok, info  # RCCX = CCX up to per-basis-state phase
    w = QuantumCircuit(3); w.cx(0, 2)
    ok, _, info = V.verify_on_reachable_basis(a, w, range(8))
    assert not ok and info["witness_input"] in (1, 5)
    # ... but agrees on inputs where q1 = 1
    assert V.verify_on_reachable_basis(a, w, [2, 3, 6, 7])[0]
    # out-of-range input
    assert not V.verify_on_reachable_basis(a, a, [8])[0]
    # dirty ancilla on a reachable input
    d = QuantumCircuit(4); d.ccx(0, 1, 2); d.cx(0, 3)
    ok, _, info = V.verify_on_reachable_basis(a, d, [0, 2])
    assert ok  # q0 = 0 -> ancilla clean
    ok, _, info = V.verify_on_reachable_basis(a, d, [1])
    assert not ok and "ancilla" in info["reason"]


def test_reachable_basis_superposed_outputs_need_global_phase_match():
    a = QuantumCircuit(2); a.h(0)
    b = QuantumCircuit(2); b.h(0); b.z(0)   # |0> -> |->: differs from |+>
    assert not V.verify_on_reachable_basis(a, b, [0])[0]
    c = QuantumCircuit(2); c.h(0); c.global_phase = 0.9
    assert V.verify_on_reachable_basis(a, c, [0, 1, 2, 3])[0]


@pytest.mark.parametrize("seed", range(3))
def test_reachable_basis_approx_matches_bruteforce_phase_min(seed):
    rng = random.Random(4000 + seed)
    for _ in range(15):
        n = 3
        a = rand_circ(rng, n, 4)
        b = rand_circ(rng, n + rng.choice([0, 1]), 4)
        S = rng.sample(range(1 << n), 3)
        ok, dev, info = V.verify_on_reachable_basis_approx(a, b, 0.3, S)
        Uo, Ur = Operator(a).data, Operator(b).data
        want = 0.0
        for x in S:
            v = Uo[:, x]
            u = Ur[: 1 << n, x]
            phis = np.linspace(0, 2 * np.pi, 4001)
            want = max(want, min(np.linalg.norm(u - np.exp(1j * p) * v) for p in phis))
        assert dev == pytest.approx(want, abs=2e-3)
        assert ok == (dev <= 0.3 + V.tol)
    with pytest.raises(ValueError):
        V.verify_on_reachable_basis_approx(a, a, -1.0, [0])


# ====================================================================== verify_scalable
def test_verify_scalable_exhaustive_path_matches_dense():
    rng = random.Random(5000)
    for _ in range(40):
        a = rand_circ(rng, 3, 5)
        b = _mutate(rng, a, _QUANTUM, 3 + rng.choice([0, 1]))
        r = verify_scalable(a, b)
        assert r.method == "exhaustive"
        assert r.equivalent == dense_truth(a, b)
        assert bool(r) == r.equivalent


@needs_qcec
def test_verify_scalable_qcec_path_matches_dense_random():
    """No SWAP gates here (see the xfail below for the SWAP false positive)."""
    rng = random.Random(5100)
    gates = ("x", "cx", "ccx", "h", "t", "tdg", "s", "rz")
    for _ in range(200):
        a = rand_circ(rng, 3, rng.randint(2, 7), gates)
        b = _mutate(rng, a, gates, 3 + rng.choice([0, 1]))
        r = verify_scalable(a, b, prefer="qcec")
        assert r.method == "qcec"
        truth = dense_truth(a, b)
        if r.equivalent is True:
            assert truth, (a, b, r)              # soundness
        if truth:
            assert r.equivalent is True, (a, b, r)  # completeness on these sizes


@needs_qcec
def test_verify_scalable_qcec_rejects_relative_phase_rewrite():
    a = QuantumCircuit(3); a.ccx(0, 1, 2)
    b = QuantumCircuit(3); b.rccx(0, 1, 2)
    r = verify_scalable(a, b, prefer="qcec")
    assert r.equivalent is False and not r


@needs_qcec
def test_verify_scalable_auto_dispatch_uses_qcec_above_crossover():
    a = QuantumCircuit(13); a.h(0); a.ccx(0, 1, 2)
    b = a.copy(); b.t(3); b.tdg(3)
    r = verify_scalable(a, b)
    assert r.method == "qcec" and r.equivalent is True
    assert ExactEquivalenceVerifier(max_qubits_unitary=4).verify_scalable(
        QuantumCircuit(5), QuantumCircuit(5)).method == "qcec"


@needs_qcec
def test_verify_scalable_timeout_is_undecided_never_true():
    rng = random.Random(1)
    n = 24
    a = QuantumCircuit(n)
    for _ in range(600):
        q = rng.sample(range(n), 3)
        r = rng.random()
        if r < .3:
            a.h(q[0])
        elif r < .5:
            a.t(q[0])
        elif r < .8:
            a.cx(q[0], q[1])
        else:
            a.ccx(*q)
    b = a.copy(); b.t(0); b.tdg(0)
    res = verify_scalable(a, b, timeout_s=1e-3)
    assert res.equivalent is None and not res
    assert res.detail["equivalence_criterion"] == "no_information"
    ok, info = qcec_certify_on_subspace(a, b, timeout=1e-3)
    assert ok is False and info["qcec"] == "no_information"


class _FakeRes:
    def __init__(self, name):
        self.equivalence = f"EquivalenceCriterion.{name}"


@pytest.mark.parametrize("name,expected", [
    ("equivalent", True),
    ("equivalent_up_to_global_phase", True),
    ("not_equivalent", False),
    ("probably_not_equivalent", False),
    ("probably_equivalent", None),
    ("no_information", None),
])
def test_qcec_verdict_mapping(monkeypatch, name, expected):
    import mqt.qcec as qcec_mod
    monkeypatch.setattr(qcec_mod, "verify", lambda *a, **k: _FakeRes(name))
    monkeypatch.setattr(sv, "_qcec_available", lambda: True)
    r = verify_scalable(QuantumCircuit(1), QuantumCircuit(1), prefer="qcec")
    assert r.equivalent is expected
    assert bool(r) is (expected is True)


def test_qcec_equivalent_up_to_phase_is_not_a_proof(monkeypatch):
    import mqt.qcec as qcec_mod
    monkeypatch.setattr(qcec_mod, "verify", lambda *a, **k: _FakeRes("equivalent_up_to_phase"))
    monkeypatch.setattr(sv, "_qcec_available", lambda: True)
    r = verify_scalable(QuantumCircuit(1), QuantumCircuit(1), prefer="qcec")
    assert r.equivalent is not True


def test_verify_scalable_no_backend_is_undecided(monkeypatch):
    monkeypatch.setattr(sv, "_qcec_available", lambda: False)
    monkeypatch.setattr(sv, "_pyzx_available", lambda: False)
    a = QuantumCircuit(3)
    r = verify_scalable(a, a.copy(), exhaustive_max_qubits=2)
    assert r.equivalent is None and r.method == "none" and not r
    r = verify_scalable(a, a.copy(), prefer="qcec")
    assert r.equivalent is None and r.method == "none"
    r = verify_scalable(a, a.copy(), prefer="pyzx")
    assert r.equivalent is None and r.method == "none"


@needs_qcec
def test_qcec_swap_is_not_identity_auto_dispatch():
    a = QuantumCircuit(13); a.swap(0, 1)
    b = QuantumCircuit(13)
    assert verify_scalable(a, b).equivalent is not True   # auto -> qcec at n=13


@needs_qcec
def test_qcec_three_cx_swap_is_not_identity():
    a = QuantumCircuit(3); a.cx(0, 1); a.cx(1, 0); a.cx(0, 1)
    assert verify_scalable(a, QuantumCircuit(3), prefer="qcec").equivalent is not True


# ============================================================= qcec_certify_on_subspace
def test_with_ancillas_moves_pinned_last_and_preserves_unitary():
    rng = random.Random(6000)
    for _ in range(20):
        c = rand_circ(rng, 4, 6)
        pinned = rng.sample(range(4), rng.choice([0, 1, 2]))
        w = with_ancillas(c, pinned)
        assert w.num_qubits == 4
        assert [r.name for r in w.qregs][-1:] == (["anc0"] if pinned else ["d"])
        if pinned:
            assert isinstance(w.qregs[-1], AncillaRegister)
        free = [q for q in range(4) if q not in pinned]
        order = free + list(pinned)   # new position k holds old qubit order[k]
        ref = QuantumCircuit(4)
        ref.compose(c, qubits=[order.index(q) for q in range(4)], inplace=True)
        assert Operator(w).equiv(Operator(ref))


def test_qcec_certify_observational_fails_closed():
    a = QuantumCircuit(2); a.h(0)
    ok, info = qcec_certify_on_subspace(a, a.copy(), (), observational=True)
    assert ok is False and "reason" in info


def test_qcec_certify_unavailable_fails_closed(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake(name, *args, **kw):
        if name.startswith("mqt"):
            raise ImportError("no qcec")
        return real_import(name, *args, **kw)
    monkeypatch.setattr(builtins, "__import__", fake)
    a = QuantumCircuit(2); a.h(0)
    ok, info = qcec_certify_on_subspace(a, a.copy(), (1,))
    assert ok is False and "unavailable" in info["reason"]


@needs_qcec
@pytest.mark.parametrize("variant", ["cx", "cz", "ch", "ccx", "crz"])
def test_qcec_certify_accepts_difference_only_on_pinned_one_inputs(variant):
    ex = QuantumCircuit(4); ex.h(0); ex.ccx(0, 1, 2); ex.t(2); ex.cx(2, 3)
    sel = QuantumCircuit(4)
    p = 3   # pinned qubit, touched by ex later
    {"cx": lambda: sel.cx(p, 0), "cz": lambda: sel.cz(p, 1), "ch": lambda: sel.ch(p, 2),
     "ccx": lambda: sel.ccx(p, 0, 1), "crz": lambda: sel.crz(0.7, p, 0)}[variant]()
    sel.compose(ex, inplace=True)
    assert subspace_truth(ex, sel, (p,))
    assert not Operator(ex).equiv(Operator(sel))          # NOT fully equivalent
    ok, info = qcec_certify_on_subspace(ex, sel, (p,))
    assert ok, info


@needs_qcec
def test_qcec_certify_rejects_difference_on_valid_inputs():
    ex = QuantumCircuit(4); ex.ccx(0, 1, 2); ex.cx(2, 3)
    for extra in (lambda c: c.t(0), lambda c: c.cx(0, 1), lambda c: c.cz(0, 1),
                  lambda c: c.rccx(0, 1, 2)):
        sel = ex.copy(); extra(sel)
        assert not subspace_truth(ex, sel, (3,))
        ok, info = qcec_certify_on_subspace(ex, sel, (3,))
        assert ok is False, info
    gp = ex.copy(); gp.global_phase = 2.0
    assert qcec_certify_on_subspace(ex, gp, (3,))[0]


@needs_qcec
def test_qcec_certify_rejects_flipped_pinned_qubit():
    ex = QuantumCircuit(4); ex.ccx(0, 1, 2)
    sel = ex.copy(); sel.x(3)          # dirty the pinned qubit
    assert not subspace_truth(ex, sel, (3,))
    assert qcec_certify_on_subspace(ex, sel, (3,))[0] is False


@needs_qcec
def test_qcec_certify_rejects_flip_after_restore():
    ex = QuantumCircuit(2); ex.cx(1, 0); ex.cx(1, 0); ex.t(1)
    sel = ex.copy(); sel.x(0)
    assert not subspace_truth(ex, sel, (0,))
    assert qcec_certify_on_subspace(ex, sel, (0,))[0] is False


@needs_qcec
def test_qcec_certify_rejects_swap_vs_identity():
    a = QuantumCircuit(3); a.swap(0, 1)
    assert qcec_certify_on_subspace(a, QuantumCircuit(3), (2,))[0] is False


def _random_subspace_cases(seed, count):
    rng = random.Random(seed)
    gates = ("h", "t", "tdg", "s", "x", "cx", "ccx", "rz")
    for _ in range(count):
        n = 4
        ex = rand_circ(rng, n, 8, gates)
        pinned = rng.sample(range(n), rng.choice([1, 2]))
        kind = rng.choice(["pinctrl", "append", "prepend", "same"])
        sel = QuantumCircuit(n)
        if kind == "pinctrl":
            p = rng.choice(pinned)
            d = [q for q in range(n) if q != p]
            rng.choice([lambda: sel.cx(p, d[0]), lambda: sel.cz(p, d[0]),
                        lambda: sel.ccx(p, d[0], d[1])])()
            sel.compose(ex, inplace=True)
        elif kind == "append":
            sel.compose(ex, inplace=True)
            sel.compose(rand_circ(rng, n, 1, gates), inplace=True)
        elif kind == "prepend":
            sel.compose(rand_circ(rng, n, 1, gates), inplace=True)
            sel.compose(ex, inplace=True)
        else:
            sel.compose(ex, inplace=True)
            sel.global_phase = 0.4
        yield ex, sel, pinned


@needs_qcec
def test_qcec_certify_random_soundness():
    fp = []
    for ex, sel, pinned in _random_subspace_cases(7, 300):
        ok, info = qcec_certify_on_subspace(ex, sel, pinned)
        if ok and not subspace_truth(ex, sel, pinned):
            fp.append(info)
    assert not fp, f"{len(fp)} false certificates"


@needs_qcec
def test_qcec_certify_random_soundness_with_proposed_fix(monkeypatch):
    """The fix (ZX off, no permutation elision) is sound AND complete on the same
    300 random cases."""
    import mqt.qcec as qcec_mod
    real = qcec_mod.verify

    def patched(a, b, **kw):
        kw.setdefault("run_zx_checker", False)
        kw.setdefault("elide_permutations", False)
        return real(a, b, **kw)
    monkeypatch.setattr(qcec_mod, "verify", patched)
    for ex, sel, pinned in _random_subspace_cases(7, 300):
        ok, info = qcec_certify_on_subspace(ex, sel, pinned)
        assert ok == subspace_truth(ex, sel, pinned), info


def test_qcec_certify_with_measurement_fails_closed():
    a = QuantumCircuit(2, 1); a.h(0); a.measure(0, 0)
    ok, _ = qcec_certify_on_subspace(a, a.copy(), (1,))
    assert ok is False                # no unitary certificate for a circuit with a measurement


# ===================================================================== count reducer
def _reducer_equivalent(orig, out, perm):
    n = orig.num_qubits
    sigma = [None] * n
    for o, f in perm.items():
        sigma[f] = o
    Ur = Operator(out).data
    d = 1 << n
    if out.num_qubits > n and np.max(np.abs(Ur[d:, :d])) > 1e-9:
        return False
    return _eq_up_to_phase(_perm_matrix(sigma, n) @ Ur[:d, :d], Operator(orig).data)


def _reducer_circ(rng, n, m, classical):
    qc = QuantumCircuit(n)
    for _ in range(m):
        q = rng.sample(range(n), 3)
        r = rng.random()
        if r < 0.45:
            c = q if rng.random() < .5 else [0, 1, rng.choice(range(2, n))]
            qc.ccx(*c)
        elif r < 0.65:
            qc.cx(q[0], q[1])
        elif r < 0.75:
            qc.swap(q[0], q[1])
        elif r < 0.85 or classical:
            qc.x(q[0])
        else:
            rng.choice([qc.h, qc.t, qc.s])(q[0])
    return qc


@pytest.mark.parametrize("seed", range(3))
def test_reducer_output_is_equivalent_random(seed):
    rng = random.Random(7000 + seed)
    reduced = 0
    for _ in range(80):
        n = rng.choice([4, 5])
        qc = _reducer_circ(rng, n, rng.randint(3, 9), rng.random() < 0.6)
        ap = rng.random() < 0.5
        R = ToffoliCountReducer(allow_permutation=ap)
        out = R.reduce_toffoli_count(qc)
        rep = R.report
        assert rep["ccx_after"] <= rep["ccx_before"] == count_toffoli(qc)
        assert rep["ccx_after"] == count_toffoli(out)
        if not ap:
            assert rep["output_permutation"] == {i: i for i in range(n)}
        assert _reducer_equivalent(qc, out, rep["output_permutation"]), rep
        reduced += rep["reduced"] > 0
    assert reduced > 5   # the passes actually fired


def test_reducer_named_rewrites():
    R = ToffoliCountReducer()
    qc = QuantumCircuit(4); qc.ccx(0, 1, 2); qc.x(3); qc.ccx(0, 1, 2)
    out = R.reduce_toffoli_count(qc)
    assert count_toffoli(out) == 0 and Operator(out).equiv(Operator(qc))
    qc = QuantumCircuit(4); qc.ccx(0, 1, 2); qc.cx(2, 3); qc.ccx(0, 1, 2)
    out = R.reduce_toffoli_count(qc)
    assert count_toffoli(out) == 1 and Operator(out).equiv(Operator(qc))
    qc = QuantumCircuit(5)
    for t in (2, 3, 4):
        qc.ccx(0, 1, t)
    out = R.reduce_toffoli_count(qc)
    assert out.num_qubits == 6 and count_toffoli(out) == 2
    assert _reducer_equivalent(qc, out, {i: i for i in range(5)})
    # a non-cancelling pair (control touched in between) must survive
    qc = QuantumCircuit(3); qc.ccx(0, 1, 2); qc.h(0); qc.ccx(0, 1, 2)
    out = R.reduce_toffoli_count(qc)
    assert count_toffoli(out) == 2 and Operator(out).equiv(Operator(qc))


def test_reducer_permutation_tracking():
    qc = QuantumCircuit(5)
    qc.ccx(0, 1, 2); qc.swap(2, 3); qc.ccx(0, 1, 2)
    qc.ccx(0, 1, 4); qc.swap(4, 2); qc.ccx(0, 1, 4)
    R = ToffoliCountReducer(allow_permutation=True)
    out = R.reduce_toffoli_count(qc)
    perm = R.report["output_permutation"]
    assert count_toffoli(out) < count_toffoli(qc)
    assert perm != {i: i for i in range(5)}
    assert _reducer_equivalent(qc, out, perm)
    assert not Operator(out).equiv(Operator(qc))


def test_reducer_wide_nonclassical_reverts_fail_closed():
    qc = QuantumCircuit(14); qc.h(13); qc.ccx(0, 1, 2); qc.ccx(0, 1, 2)
    R = ToffoliCountReducer()
    out = R.reduce_toffoli_count(qc)
    assert count_toffoli(out) == 2
    assert R.report["all_rewrites_verified"] is False
    assert "REVERTED:final_verify_failed" in R.report["rewrites_applied"]


def test_reducer_named_register():
    qc = QuantumCircuit(QuantumRegister(3, "a")); qc.ccx(0, 1, 2); qc.ccx(0, 1, 2)
    out = ToffoliCountReducer().reduce_toffoli_count(qc)
    assert count_toffoli(out) == 0


def test_reducer_circuit_with_measurement():
    qc = QuantumCircuit(3, 1); qc.ccx(0, 1, 2); qc.ccx(0, 1, 2); qc.measure(2, 0)
    out = ToffoliCountReducer().reduce_toffoli_count(qc)
    # no crash; the exact verifier has no verdict for circuits with classical bits, so the input is returned
    assert out.num_clbits == 1 and sum(i.operation.name == "measure" for i in out.data) == 1


# =================================================================== pattern library















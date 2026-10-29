"""Targeted tests for the rarely exercised branches of the maintained core.

Every test checks behaviour against an independent ground truth (dense
``qiskit.quantum_info.Operator`` / ``Statevector`` on small circuits, or a hand
computation), or against the documented fail-closed contract (undecided / False
verdicts, ValueError) when a backend is unavailable or misbehaves.
"""

import math
import sys
import types

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.quantum_info import Operator, Statevector

from toffoli_optimizer.core import context_analysis as ca
from toffoli_optimizer.core import decomposition_selector as ds
from toffoli_optimizer.core import orientation as ori
from toffoli_optimizer.core import reachable_subspace as rs
from toffoli_optimizer.core import scalable_verification as sv
from toffoli_optimizer.core import subspace_check as sc
from toffoli_optimizer.core.equivalence_verifier import ExactEquivalenceVerifier
from toffoli_optimizer.core.error_model import HardwareErrorModel
from toffoli_optimizer.core.reach_local import local_reachable
from toffoli_optimizer.core.toffoli_count_reducer import ToffoliCountReducer, count_toffoli
from toffoli_optimizer.core.window_pairs import window_certified


# --------------------------------------------------------------------------- helpers
def _equiv(a, b):
    return Operator(a).equiv(Operator(b))


def _support(qc, x):
    """Basis support of qc|x> (ground truth for reachability)."""
    sv_ = Statevector.from_int(x, 2 ** qc.num_qubits).evolve(qc)
    return {i for i, amp in enumerate(sv_.data) if abs(amp) > 1e-9}


def _block_qcec(monkeypatch):
    """Make ``import mqt.qcec`` / ``from mqt import qcec`` fail."""
    import mqt
    monkeypatch.setitem(sys.modules, "mqt.qcec", None)
    monkeypatch.delattr(mqt, "qcec", raising=False)


def _fake_pyzx(raise_on_parse=False):
    """A stand-in ``pyzx`` whose equality test is the dense Operator ground truth."""
    from qiskit import qasm2

    mod = types.ModuleType("pyzx")

    class Circuit:
        def __init__(self, qc):
            self.qc = qc

        @classmethod
        def from_qasm(cls, text):
            if raise_on_parse:
                raise RuntimeError("parse failure")
            return cls(qasm2.loads(text))

        def verify_equality(self, other):
            return Operator(self.qc).equiv(Operator(other.qc))

    mod.Circuit = Circuit
    return mod


# =========================================================== context_analysis
class TestContextAnalysis:
    def test_two_qubit_gate_named_mcx_is_not_a_site(self):
        g = Gate("mcx", 2, [])
        qc = QuantumCircuit(2)
        qc.append(g, [0, 1])
        qc.append(g, [0, 1])
        assert ca._gate_qubits(qc, qc.data[0]) is None
        assert ca.find_relative_phase_safe_sites(qc) == []

    def test_barrier_and_id_do_not_break_pair_and_pair_is_exact(self):
        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)
        qc.barrier()
        qc.id(2)
        qc.ccx(0, 1, 2)
        sites = ca.find_relative_phase_safe_sites(qc)
        assert sites == [ca.RelativePhaseSite(0, 3, (0, 1), 2)]
        # Ground truth: both Toffolis -> Margolus gadget gives the same unitary.
        sub = QuantumCircuit(3)
        ds.append_relative_phase_ccx(sub, 0, 1, 2)
        sub.id(2)
        ds.append_relative_phase_ccx(sub, 0, 1, 2)
        assert _equiv(sub, QuantumCircuit(3))   # CCX.CCX = I

    @pytest.mark.parametrize("op,qubit,expect_site", [
        ("measure", 3, True), ("measure", 2, False), ("measure", 0, False),
        ("reset", 3, True), ("reset", 1, False)])
    def test_measure_reset_inside_window(self, op, qubit, expect_site):
        qc = QuantumCircuit(4, 1)
        qc.ccx(0, 1, 2)
        if op == "measure":
            qc.measure(qubit, 0)
        else:
            qc.reset(qubit)
        qc.ccx(0, 1, 2)
        sites = ca.find_relative_phase_safe_sites(qc)
        assert bool(sites) is expect_site

    @pytest.mark.parametrize("pair,expect_site", [((3, 4), True), ((2, 3), False), ((0, 4), False)])
    def test_swap_inside_window(self, pair, expect_site):
        qc = QuantumCircuit(5)
        qc.ccx(0, 1, 2)
        qc.swap(*pair)
        qc.ccx(0, 1, 2)
        sites = ca.find_relative_phase_safe_sites(qc)
        assert bool(sites) is expect_site
        if expect_site:
            # the admitted pair really cancels its relative phases (exact unitary)
            sub = QuantumCircuit(5)
            ds.append_relative_phase_ccx(sub, 0, 1, 2)
            sub.swap(*pair)
            ds.append_relative_phase_ccx(sub, 0, 1, 2)
            assert _equiv(sub, qc)


# ====================================================== decomposition_selector
class _LinearErrorModel(HardwareErrorModel):
    """Infidelity = number of two-qubit gates (not capped at 1): lets the
    bounded-approximate budget test pass with a nonzero deviation."""

    def circuit_infidelity(self, circuit):
        return float(self.two_qubit_count(circuit))


class TestDecompositionSelector:
    def test_control_drop_rejects_two_kept_controls(self):
        with pytest.raises(ValueError, match="0 or 1 controls"):
            ds.append_control_drop(QuantumCircuit(3), 0, 1, 2, (0, 1))

    def test_constructor_validates_semantics_and_epsilon(self):
        with pytest.raises(ValueError, match="semantics"):
            ds.ErrorBudgetSelector(semantics="bogus")
        with pytest.raises(ValueError, match="epsilon"):
            ds.ErrorBudgetSelector(epsilon=-1e-3)

    def test_build_rejects_unknown_action(self):
        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)
        with pytest.raises(ValueError, match="unknown action"):
            ds.ErrorBudgetSelector()._build(qc, {0: ("bogus",)})

    def test_reach_circuit_inserts_h_after_perturbed_gate(self):
        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)
        qc.cx(2, 0)
        out, idx_map = ds.ErrorBudgetSelector._reach_circuit(qc, {(0, 2)})
        names = [(i.operation.name, [out.find_bit(q).index for q in i.qubits]) for i in out.data]
        assert names == [("ccx", [0, 1, 2]), ("h", [2]), ("cx", [2, 0])]
        assert idx_map == [0, 2]

    def test_bounded_approximate_admission_and_certificate(self):
        """epsilon > 0 with a budget that pays for a nonzero deviation: both
        Toffolis collapse to identity; the second one's reachability must be
        computed on the perturbed (H-inserted) circuit; the certificate must
        bound the true deviation, checked here against a dense-Operator scan."""
        qc = QuantumCircuit(4)
        qc.ccx(0, 1, 2)
        qc.ccx(1, 2, 3)
        sel = ds.ErrorBudgetSelector(error_model=_LinearErrorModel(), epsilon=3.0,
                                     window_pairs=False)
        res = sel.select(qc)
        rep = res["report"]
        assert res["actions"] == {0: ("control_drop", ()), 1: ("control_drop", ())}
        assert not rep["fell_back_to_exact"] and rep["verified"] is True
        assert rep["two_qubit_after"] == 0
        # Deviation of identity vs CCX on the full local space: sqrt(2).
        for d in rep["approx_admitted"]:
            assert d["condition"] == "R-eps"
            assert d["max_deviation"] == pytest.approx(math.sqrt(2), abs=1e-9)
        # The second gate saw the perturbed circuit: its target 2 is free after
        # gate 0, so the over-approximation is the whole 16-state space.
        assert rep["approx_admitted"][1]["reachable_inputs"] == 16
        assert rep["epsilon_spent_total"] == pytest.approx(2 * math.sqrt(2))
        # Independent ground truth: min over a global phase of the operator norm.
        A = Operator(res["circuit"]).data
        B = Operator(res["exact"]).data
        devs = [np.linalg.norm(A - np.exp(1j * th) * B, 2)
                for th in np.linspace(-np.pi, np.pi, 2001)]
        true_dev = min(devs)
        assert true_dev <= rep["epsilon_spent_total"] + 1e-9
        assert rep["verify_info"]["certified_deviation"] == pytest.approx(true_dev, abs=5e-3)

    def test_default_model_never_pays_for_a_nonzero_deviation(self):
        """With the shipped (capped, 1 - prod) infidelity the saving is <= 1, while a
        control drop that differs from CCX on a reachable basis column deviates by
        >= sqrt(2) (two orthogonal basis vectors). So only exact-on-reachable drops
        (deviation 0) can ever be admitted, however large epsilon is."""
        rng = np.random.default_rng(20261002)
        for _ in range(4):
            qc = QuantumCircuit(4)
            for _ in range(3):
                a, b, t = rng.choice(4, size=3, replace=False)
                qc.ccx(int(a), int(b), int(t))
            res = ds.ErrorBudgetSelector(epsilon=2.0, window_pairs=False).select(
                qc, input_space=[0, 1, 2, 3])
            rep = res["report"]
            assert all(d["max_deviation"] == 0.0 for d in rep["approx_admitted"])
            assert rep["epsilon_spent_total"] == 0.0
            # exact on the declared input domain (columns 0..3)
            A = Operator(res["circuit"]).data[:, :4]
            B = Operator(qc).data[:, :4]
            k = np.unravel_index(np.argmax(np.abs(B)), B.shape)
            assert np.allclose(A, (A[k] / B[k]) * B, atol=1e-8)

    def test_no_approximation_without_epsilon(self):
        qc = QuantumCircuit(4)
        qc.ccx(0, 1, 2)
        qc.ccx(1, 2, 3)
        res = ds.ErrorBudgetSelector(error_model=_LinearErrorModel(), epsilon=0.0,
                                     window_pairs=False).select(qc)
        assert res["report"]["approx_admitted"] == []
        assert _equiv(res["circuit"], qc)


# ========================================================= equivalence_verifier
class TestEquivalenceVerifier:
    def test_truth_table_skips_barrier_and_id(self):
        a = QuantumCircuit(3)
        a.ccx(0, 1, 2)
        a.barrier()
        a.id(1)
        a.x(0)
        b = QuantumCircuit(3)
        b.ccx(0, 1, 2)
        b.x(0)
        ok, perm, info = ExactEquivalenceVerifier().verify(a, b)
        assert ok and info["method"] == "truth_table" and perm == {0: 0, 1: 1, 2: 2}
        b2 = QuantumCircuit(3)
        b2.x(0)
        ok2, _, info2 = ExactEquivalenceVerifier().verify(a, b2)
        assert not ok2 and info2["reason"] == "classical output mismatch"

    def test_equal_up_to_global_phase_degenerate_inputs(self):
        v = ExactEquivalenceVerifier()
        assert v._equal_up_to_global_phase(np.eye(2), np.eye(4)) is False
        Z = np.zeros((2, 2))
        assert v._equal_up_to_global_phase(Z, Z)
        assert not v._equal_up_to_global_phase(np.eye(2), Z)

    def test_reachable_basis_rejects_narrower_rewrite(self):
        ok, perm, info = ExactEquivalenceVerifier().verify_on_reachable_basis(
            QuantumCircuit(3), QuantumCircuit(2), [0])
        assert ok is False and perm is None and "fewer qubits" in info["reason"]

    def test_reachable_basis_approx_rejects_narrower_rewrite(self):
        ok, dev, info = ExactEquivalenceVerifier().verify_on_reachable_basis_approx(
            QuantumCircuit(3), QuantumCircuit(2), 0.5, [0])
        assert ok is False and dev == float("inf") and "fewer qubits" in info["reason"]

    @pytest.mark.parametrize("bad", [-1, 4, 7])
    def test_reachable_basis_approx_rejects_out_of_range_input(self, bad):
        ok, dev, info = ExactEquivalenceVerifier().verify_on_reachable_basis_approx(
            QuantumCircuit(2), QuantumCircuit(2), 0.5, [0, bad])
        assert ok is False and dev == float("inf") and "out of data range" in info["reason"]


# ================================================================== error_model
class TestErrorModelBarrier:
    def _circ(self, barrier):
        qc = QuantumCircuit(2)
        qc.x(0)
        if barrier:
            qc.barrier(0, 1)
        qc.x(1)
        return qc

    def test_barrier_synchronises_alap_schedule(self):
        tau = 36e-9
        em = HardwareErrorModel(t_1q=tau)
        # Hand-computed ALAP schedule (reversed list): x(1) ends at tau, the
        # barrier lifts qubit 0 to tau, x(0) then occupies [tau, 2 tau].
        assert em.idle_times(self._circ(True)) == pytest.approx({0: tau, 1: 0.0})
        aw = em.active_windows(self._circ(True))
        assert aw[0] == pytest.approx((0.0, 2 * tau))
        assert aw[1] == pytest.approx((tau, 2 * tau))
        # Without the barrier the two X gates run in parallel: nobody idles.
        assert em.idle_times(self._circ(False)) == pytest.approx({0: 0.0, 1: 0.0})
        assert em.active_windows(self._circ(False))[0] == pytest.approx((0.0, tau))


# ================================================================== orientation
class TestOrientation:
    def test_dijkstra_skips_stale_queue_entry(self):
        def w(e):
            return -math.log(1 - e)
        edges = {(0, 1): 0.5, (0, 2): 0.01, (2, 1): 0.01, (1, 3): 0.9}
        cost = ori.pair_cost_from_calibration(edges, layout=[0, 3])
        # Cheapest route 0-2-1-3; last hop 1-3: 3*sum(w) - 2*w_last.
        d = w(0.01) + w(0.01) + w(0.9)
        assert cost(0, 3) == pytest.approx(3 * d - 2 * w(0.9))
        assert cost(3, 0) == cost(0, 3)

    def test_edge_errors_from_properties(self):
        def gate(name, qubits, err=None):
            params = [{"name": "gate_length", "value": 1e-7}]
            if err is not None:
                params.append({"name": "gate_error", "value": err})
            return {"gate": name, "qubits": qubits, "parameters": params}
        props = {"gates": [
            gate("cz", [0, 1], 0.02),
            gate("cx", [1, 0], 0.005),     # same pair, lower error wins
            gate("ecr", [1, 2], 1.0),      # broken edge (error 1) ignored
            gate("ecr", [2, 3], 0.03),
            gate("sx", [0], 0.001),        # single-qubit ignored
            gate("cz", [3, 4]),            # no gate_error parameter: ignored
            gate("rzz", [4, 5], 0.01),     # not a listed 2q gate name
        ]}
        assert ori.edge_errors_from_properties(props) == {(0, 1): 0.005, (2, 3): 0.03}

    def test_group_with_non_gadget_member_is_not_oriented(self):
        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)
        qc.ccx(0, 1, 2)
        result = {
            "circuit": qc,
            "actions": {0: ("control_drop", ()), 1: ("relphase",)},
            "report": {"applied_sites": [(0, 1)], "window_pairs_admitted": []},
        }
        cost = ori.PairCost({(1, 2): 10.0}, default=0.1)   # would favour swapping
        out, info = ori.orient(ds.ErrorBudgetSelector(), qc, result, cost)
        assert out is qc
        assert info["groups"] == 0 and info["oriented"] == 0

    def test_certify_without_dense_inputs_uses_qcec(self):
        pytest.importorskip("mqt.qcec")
        exact = QuantumCircuit(3)
        ds.append_exact_ccx(exact, 0, 1, 2)
        same = exact.copy()
        assert ori._certify(exact, same, None, ()) is True
        wrong = exact.copy()
        wrong.x(2)
        assert ori._certify(exact, wrong, None, ()) is False
        # relative-phase gadget alone is NOT the exact CCX: must fail closed
        rp = QuantumCircuit(3)
        ds.append_relative_phase_ccx(rp, 0, 1, 2)
        assert ori._certify(exact, rp, None, ()) is False


# ================================================================== reach_local
def test_local_reachable_none_for_nonclassical_prefix_and_exact_otherwise():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.ccx(0, 1, 2)
    assert local_reachable(qc, 1, (0, 1, 2)) is None
    qc2 = QuantumCircuit(3)
    qc2.x(0)
    qc2.barrier()
    qc2.id(1)
    qc2.ccx(0, 1, 2)
    prefix = QuantumCircuit(3)
    prefix.x(0)
    for pinned in [(), (0,), (0, 1, 2)]:
        pm = sum(1 << q for q in pinned)
        truth = sorted(set().union(*(_support(prefix, x) for x in range(8) if not x & pm)))
        assert local_reachable(qc2, 3, (0, 1, 2), pinned_zero=pinned) == truth
    assert local_reachable(qc2, 3, (0, 1, 2), pinned_zero=(0,)) == [1, 3, 5, 7]


# =========================================================== reachable_subspace
class TestReachableSubspace:
    def test_exact_states_skip_barrier_and_id(self):
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.barrier()
        qc.id(1)
        qc.cx(0, 1)
        got = rs.reachable_basis_states(qc, len(qc.data), input_space=[0, 1])
        truth = _support(qc, 0) | _support(qc, 1)
        assert got == frozenset(truth) == frozenset({0, 3})

    def test_apply_classical_rejects_nonclassical_gate(self):
        with pytest.raises(ValueError, match="non-classical"):
            rs._apply_classical("h", [0], 0)

    def test_unknown_input_space_string_rejected(self):
        qc = QuantumCircuit(2)
        qc.x(0)
        with pytest.raises(ValueError, match="unknown input_space"):
            rs.reachable_basis_states(qc, 1, input_space="ancilla_zero")

    @pytest.mark.parametrize("idx", [-1, 3])
    def test_overapprox_gate_index_out_of_range(self, idx):
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.h(1)
        with pytest.raises(ValueError, match="out of range"):
            rs.reachable_overapprox(qc, idx)

    def test_overapprox_nonclassical_prefix_with_barrier_is_sound_and_tight(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.barrier()
        qc.cx(1, 2)
        x0 = 0b010
        got = rs.reachable_overapprox(qc, len(qc.data), input_space=[x0])
        truth = _support(qc, x0)
        assert truth <= set(got)
        assert got == frozenset({0b110, 0b111})

    def test_overapprox_empty_input_space(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(1, 2)
        got = rs.reachable_overapprox(qc, len(qc.data), input_space=[])
        # only the H-tainted qubit 0 varies; clean qubits take the representative 0
        assert got == frozenset({0b000, 0b001})


# ======================================================== scalable_verification
def _pair(equal=True, n=3):
    a = QuantumCircuit(n)
    a.h(0)
    a.ccx(0, 1, 2)
    b = QuantumCircuit(n)
    b.h(0)
    ds.append_exact_ccx(b, 0, 1, 2)
    if not equal:
        b.z(1)
    return a, b


class TestScalableVerification:
    def test_qcec_unavailable_large_circuit_is_undecided(self, monkeypatch):
        _block_qcec(monkeypatch)
        monkeypatch.setitem(sys.modules, "pyzx", None)
        assert sv._qcec_available() is False
        assert sv._pyzx_available() is False
        a, b = _pair(equal=False)
        r = sv.verify_scalable(a, b, exhaustive_max_qubits=1)
        assert r.equivalent is None and r.method == "none" and not r
        r2 = sv.verify_scalable(a, b, prefer="qcec")
        assert r2.equivalent is None and "mqt.qcec" in r2.detail["reason"]
        r3 = sv.verify_scalable(a, b, prefer="pyzx")
        assert r3.equivalent is None and "pyzx" in r3.detail["reason"]

    def test_qcec_option_typeerror_fails_closed(self, monkeypatch):
        qcec = pytest.importorskip("mqt.qcec")

        def boom(*args, **kwargs):
            raise TypeError("unexpected keyword argument 'elide_permutations'")
        a, b = _pair(equal=True)
        assert sv.verify_scalable(a, b, prefer="qcec").equivalent is True  # control
        monkeypatch.setattr(qcec, "verify", boom)
        r = sv.verify_scalable(a, b, prefer="qcec")
        assert r.equivalent is None and r.method == "qcec" and not r
        assert "unsupported" in r.detail["reason"]

    def test_pyzx_backend_maps_verdicts(self, monkeypatch):
        _block_qcec(monkeypatch)
        monkeypatch.setitem(sys.modules, "pyzx", _fake_pyzx())
        assert sv._pyzx_available() is True
        a, b = _pair(equal=True)
        r = sv.verify_scalable(a, b, exhaustive_max_qubits=1)   # auto -> pyzx
        assert r.method == "pyzx" and r.equivalent is True and r.up_to_global_phase
        a, c = _pair(equal=False)
        r = sv.verify_scalable(a, c, prefer="pyzx")
        assert r.method == "pyzx" and r.equivalent is False

    def test_pyzx_width_mismatch_is_undecided(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pyzx", _fake_pyzx())
        a, _ = _pair()
        b = QuantumCircuit(4)
        b.h(0)
        b.ccx(0, 1, 2)
        r = sv.verify_scalable(a, b, prefer="pyzx")
        assert r.method == "pyzx" and r.equivalent is None and r.n_qubits == 4

    def test_pyzx_exception_is_undecided(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pyzx", _fake_pyzx(raise_on_parse=True))
        a, b = _pair(equal=True)
        r = sv.verify_scalable(a, b, prefer="pyzx")
        assert r.equivalent is None and "raised" in r.detail["reason"]


# ================================================================ subspace_check
def test_qcec_subspace_certificate_typeerror_fails_closed(monkeypatch):
    qcec = pytest.importorskip("mqt.qcec")
    exact = QuantumCircuit(3)
    exact.ccx(0, 1, 2)
    sel = QuantumCircuit(3)
    ds.append_exact_ccx(sel, 0, 1, 2)
    ok, _ = sc.qcec_certify_on_subspace(exact, sel)
    assert ok is True  # control: a real certificate exists

    def boom(*args, **kwargs):
        raise TypeError("bad option")
    monkeypatch.setattr(qcec, "verify", boom)
    ok, info = sc.qcec_certify_on_subspace(exact, sel)
    assert ok is False and "unsupported" in info["reason"]


# ================================================================== window_pairs
def test_window_with_wrong_gadget_has_zero_overlap_and_is_rejected():
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    qc.x(2)
    qc.ccx(0, 1, 2)
    # "gadget" H on the target: H X H = Z, orthogonal to the true segment X.
    ok, info = window_certified(qc, 0, 2, {0, 2}, lambda c, a, b, t: c.h(t))
    assert ok is False and "zero overlap" in info["reason"]
    # X on the target does not commute with the Margolus phase: rejected too.
    ok2, _ = window_certified(qc, 0, 2, {0, 2}, ds.append_relative_phase_ccx)
    assert ok2 is False
    # A window that only reads the target is certified for the Margolus gadget,
    # and the certificate agrees with the dense whole-circuit check.
    qc3 = QuantumCircuit(4)
    qc3.ccx(0, 1, 2)
    qc3.cx(2, 3)
    qc3.ccx(0, 1, 2)
    ok3, _ = window_certified(qc3, 0, 2, {0, 2}, ds.append_relative_phase_ccx)
    sub = QuantumCircuit(4)
    ds.append_relative_phase_ccx(sub, 0, 1, 2)
    sub.cx(2, 3)
    ds.append_relative_phase_ccx(sub, 0, 1, 2)
    assert ok3 is True and _equiv(sub, qc3)


# ======================================================== toffoli_count_reducer
class _RaisingVerifier(ExactEquivalenceVerifier):
    def verify(self, *a, **k):
        raise RuntimeError("backend crashed")


class _NoPermVerifier(ExactEquivalenceVerifier):
    """A verifier that reports success without a permutation (allowed by the
    reducer's contract: None means identity wiring)."""

    def verify(self, original, rewritten, allow_permutation=False):
        ok, _perm, info = super().verify(original, rewritten, allow_permutation)
        return ok, None, info


class _LyingFinalVerifier(ExactEquivalenceVerifier):
    """Truthful on windows; on the final whole-circuit check it reports a
    permutation that differs from the one the reducer tracked."""

    def __init__(self, target):
        super().__init__()
        self.target = target

    def verify(self, original, rewritten, allow_permutation=False):
        ok, perm, info = super().verify(original, rewritten, allow_permutation)
        if original is self.target and ok:
            return ok, {i: i for i in range(original.num_qubits)}, info
        return ok, perm, info


def _perm_window_circuit():
    qc = QuantumCircuit(4)
    qc.ccx(0, 1, 2)
    qc.cx(2, 3)
    qc.swap(2, 3)
    qc.ccx(0, 1, 2)
    return qc


class TestToffoliCountReducer:
    def test_ccz_pair_with_permuted_qubits_cancels(self):
        qc = QuantumCircuit(3)
        qc.ccz(0, 1, 2)
        qc.ccz(2, 0, 1)
        r = ToffoliCountReducer()
        out = r.reduce_toffoli_count(qc)
        assert count_toffoli(out) == 0
        assert _equiv(out, qc)
        assert "inverse_pair_annihilation" in r.report["rewrites_applied"]

    def test_verbose_prints_report(self, capsys):
        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)
        qc.ccx(0, 1, 2)
        r = ToffoliCountReducer(verbose=True)
        r.reduce_toffoli_count(qc)
        printed = capsys.readouterr().out
        assert "'ccx_before': 2" in printed and "'ccx_after': 0" in printed

    @pytest.mark.parametrize("builder", ["conjugation", "fanout"])
    def test_crashing_verifier_fails_closed(self, builder):
        qc = QuantumCircuit(5)
        if builder == "conjugation":
            qc.ccx(0, 1, 2)
            qc.cx(2, 3)
            qc.ccx(0, 1, 2)
        else:
            qc.ccx(0, 1, 2)
            qc.ccx(0, 1, 3)
            qc.ccx(0, 1, 4)
        r = ToffoliCountReducer(verifier=_RaisingVerifier())
        out = r.reduce_toffoli_count(qc)
        assert count_toffoli(out) == count_toffoli(qc)
        assert _equiv(out, qc)
        assert r.report["all_rewrites_verified"] is False

    def test_verifier_without_permutation_is_read_as_identity(self):
        qc = QuantumCircuit(4)
        qc.ccx(0, 1, 2)
        qc.cx(2, 3)
        qc.ccx(0, 1, 2)
        r = ToffoliCountReducer(verifier=_NoPermVerifier())
        out = r.reduce_toffoli_count(qc)
        assert count_toffoli(out) == 1
        assert _equiv(out, qc)
        assert r.report["output_permutation"] == {i: i for i in range(4)}

    def test_final_permutation_mismatch_reverts_to_input(self):
        qc = _perm_window_circuit()
        r = ToffoliCountReducer(verifier=_LyingFinalVerifier(qc))
        out = r.reduce_toffoli_count(qc, allow_permutation=True)
        assert any(s.startswith("expand_to_cancel(perm=") for s in r.report["rewrites_applied"])
        assert r.report["rewrites_applied"][-1] == "REVERTED:final_verify_failed"
        assert r.report["all_rewrites_verified"] is False
        assert r.report["output_permutation"] == {i: i for i in range(4)}
        assert _equiv(out, qc) and count_toffoli(out) == 2

    def test_expand_to_cancel_keeps_prefix_and_suffix(self):
        qc = QuantumCircuit(6)
        for q in (4, 5, 4, 5):          # four non-atomic prefix gates
            qc.h(q)
        qc.ccx(0, 1, 2)
        qc.cx(2, 3)
        qc.ccx(0, 1, 2)
        qc.t(3)                          # suffix gate
        qc.cx(3, 5)
        r = ToffoliCountReducer()
        out = r.reduce_toffoli_count(qc)
        assert count_toffoli(out) == 1
        assert "expand_to_cancel" in r.report["rewrites_applied"]
        assert _equiv(out, qc)
        names = [i.operation.name for i in out.data]
        assert names[:4] == ["h"] * 4 and names[-2:] == ["t", "cx"]

    def test_trivial_swap_conjugation_offers_only_verbatim(self):
        qc = QuantumCircuit(4)
        qc.ccx(0, 1, 2)
        qc.swap(2, 3)
        qc.swap(2, 3)
        qc.ccx(0, 1, 2)
        r = ToffoliCountReducer()
        cands = r._enumerate_expansions(qc, list(qc.data), 4, allow_permutation=True)
        assert len(cands) == 1
        assert [op.name for op, _q, _c in cands[0]] == ["ccx", "swap", "swap", "ccx"]
        # and the reducer as a whole still removes the pair, exactly
        out = r.reduce_toffoli_count(qc)
        assert count_toffoli(out) == 0 and _equiv(out, qc)

    def test_fanout_preserves_between_and_trailing_gates(self):
        qc = QuantumCircuit(6)
        qc.ccx(0, 1, 2)
        qc.x(5)                          # between the run, disjoint
        qc.ccx(0, 1, 3)
        qc.ccx(0, 1, 4)
        qc.cx(4, 5)                      # after the run
        r = ToffoliCountReducer(enable_expand_to_cancel=False)
        out = r.reduce_toffoli_count(qc)
        assert out.num_qubits == 7 and count_toffoli(out) == 2
        assert any(s.startswith("fanout_cse") for s in r.report["rewrites_applied"])
        # Ground truth: on the ancilla=|0> subspace out == qc, with no leakage.
        U = Operator(out).data
        R = Operator(qc).data
        dim = 1 << 6
        assert np.allclose(U[dim:, :dim], 0)
        k = np.unravel_index(np.argmax(np.abs(R)), R.shape)
        assert np.allclose(U[:dim, :dim], (U[:dim, :dim][k] / R[k]) * R)

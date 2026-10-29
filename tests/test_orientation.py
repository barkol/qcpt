"""Calibration-aware orientation: picks the cheaper control for the doubled CX,
keeps every output certified, and never orients a gadget into an inadmissible phase."""
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from toffoli_optimizer.core.decomposition_selector import ErrorBudgetSelector
from toffoli_optimizer.core.orientation import PairCost, orient
from toffoli_optimizer.core.subspace_check import certify_on_input_subspace


def _cx_pairs(qc):
    out = {}
    for ins in qc.data:
        if ins.operation.num_qubits == 2:
            k = tuple(sorted(qc.find_bit(q).index for q in ins.qubits)); out[k] = out.get(k, 0) + 1
    return out


def _check(qc, pins, cost):
    sel = ErrorBudgetSelector()
    res = sel.select(qc, pinned_zero=pins)
    out, info = orient(sel, qc, res, cost, pinned_zero=pins)
    pm = sum(1 << q for q in pins)
    inputs = [x for x in range(2 ** qc.num_qubits) if not x & pm]
    ok = certify_on_input_subspace(Operator(res["exact"]).data, Operator(out).data, inputs, "subroutine")[0]
    return res, out, info, ok


def test_pair_oriented_to_cheap_edge():
    # compute/uncompute pair on controls 0,1 and ancilla 2. The default gadget doubles
    # the CX on (b, t) = (1, 2); make that edge expensive so the swap must win.
    qc = QuantumCircuit(4); qc.ccx(0, 1, 2); qc.cx(2, 3); qc.ccx(0, 1, 2)
    cost = PairCost({(1, 2): 0.05, (0, 2): 0.001}, default=0.01)
    res, out, info, ok = _check(qc, [2], cost)
    assert ok and info["oriented"] == 2 and info["certificate"] is True
    assert _cx_pairs(res["circuit"])[(1, 2)] > _cx_pairs(res["circuit"])[(0, 2)]
    assert _cx_pairs(out)[(1, 2)] < _cx_pairs(out)[(0, 2)]


def test_mirror_single_oriented_and_certified():
    # AND into a clean ancilla, then the ancilla is read: (R) with the mirrored gadget.
    qc = QuantumCircuit(3); qc.ccx(0, 1, 2); qc.h(0)
    cost = PairCost({(1, 2): 0.05, (0, 2): 0.001}, default=0.01)
    res, out, info, ok = _check(qc, [2], cost)
    assert ok and info["oriented"] == 1 and info["rejected_R"] == 0
    assert _cx_pairs(out)[(1, 2)] < _cx_pairs(out)[(0, 2)]


def test_margolus_single_swap_rejected_when_phase_becomes_reachable():
    # Margolus admitted under (R) only because the branch abt=100 is unreachable
    # (a=1 forces b=1 here); swapping moves the phase to abt=010, which IS reachable.
    from toffoli_optimizer.core.orientation import _swapped_local_unitary
    from toffoli_optimizer.core.decomposition_selector import _U_CCX, _U_MARG
    from toffoli_optimizer.core.subspace_check import check_on_subspace
    local = [0b000, 0b010, 0b011, 0b110, 0b111]   # bit j = qubit j of (a, b, t); no a=1,b=0
    assert check_on_subspace(_U_MARG, _U_CCX, local, 0.0)[0]
    assert not check_on_subspace(_swapped_local_unitary("relphase"), _U_CCX, local, 0.0)[0]


def test_no_change_when_costs_equal():
    qc = QuantumCircuit(4); qc.ccx(0, 1, 2); qc.cx(2, 3); qc.ccx(0, 1, 2)
    res, out, info, ok = _check(qc, [2], PairCost({}, default=0.01))
    assert ok and info["oriented"] == 0


if __name__ == "__main__":
    test_pair_oriented_to_cheap_edge(); test_mirror_single_oriented_and_certified(); test_no_change_when_costs_equal(); print("ok")

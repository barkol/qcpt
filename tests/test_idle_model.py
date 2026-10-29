"""Optional idle term of HardwareErrorModel: off by default (numbers unchanged),
charges only waiting time after a qubit's first gate, and agrees with the
Pauli-twirled damping formula."""
import math
from qiskit import QuantumCircuit
from toffoli_optimizer.core.error_model import HardwareErrorModel


def test_default_unchanged():
    qc = QuantumCircuit(3); qc.h(0); qc.cx(0, 1); qc.cx(1, 2)
    a = HardwareErrorModel().circuit_infidelity(qc)
    assert abs(a - (1 - (1 - 1e-3) * (1 - 1e-2) ** 2)) < 1e-15


def test_idle_only_after_first_gate():
    # q0 does its H, then waits while q1-q2 run 10 CX; q3 never acts (no charge).
    qc = QuantumCircuit(4); qc.h(0)
    for _ in range(10): qc.cx(1, 2)
    m = HardwareErrorModel(t1=100e-6, t2=50e-6, t_2q=100e-9, t_1q=50e-9)
    tau = m.idle_times(qc)
    assert 3 not in tau
    # ALAP: H on q0 is scheduled last, so q0 does not wait at all.
    assert abs(tau[0]) < 1e-15 and abs(tau[1]) < 1e-15
    qc2 = QuantumCircuit(3); qc2.cx(0, 1)
    for _ in range(10): qc2.cx(1, 2)
    qc2.cx(0, 1)                      # q0 is entangled first, then waits for the chain
    tau2 = m.idle_times(qc2)
    assert abs(tau2[0] - 10 * 100e-9) < 1e-12 and abs(tau2[1]) < 1e-12


def test_idle_formula():
    m = HardwareErrorModel(t1=100e-6, t2=50e-6, t_2q=1e-6, t_1q=0.0)
    qc = QuantumCircuit(3); qc.cx(0, 1)
    for _ in range(5): qc.cx(1, 2)
    qc.cx(0, 1)
    # q0 waits 5 us; q2 waits 1 us after its last gate (the final cx(0,1)).
    tau = 5e-6; t2_ = 1e-6
    a = 1 - math.exp(-tau / 100e-6); b = 1 - math.exp(-tau / 50e-6)
    expect = 1 - a / 2 - (b / 2 - a / 4)
    a2 = 1 - math.exp(-t2_ / 100e-6); b2 = 1 - math.exp(-t2_ / 50e-6)
    expect *= 1 - a2 / 2 - (b2 / 2 - a2 / 4)
    assert abs(m.idle_fidelity(qc) - expect) < 1e-12



def test_zz_and_excess_terms():
    import math
    qc = QuantumCircuit(2); qc.cx(0, 1)
    for _ in range(9): qc.x(0)
    qc.cx(0, 1)
    base = HardwareErrorModel()
    m = HardwareErrorModel(zz_hz=5e3, t_2q=100e-9, t_1q=100e-9, p_excess_2q=2e-3)
    T = 11 * 100e-9  # both qubits active from the first to the last gate
    zz = 1 - 0.75 * math.sin(math.pi * 5e3 * T) ** 2
    expect = (1 - base.circuit_infidelity(qc)) * zz * (1 - 2e-3) ** 2
    assert abs((1 - m.circuit_infidelity(qc)) - expect) < 1e-12
    assert HardwareErrorModel(zz_hz=5e3, zz_pairs=[(0, 2)]).zz_fidelity(qc) == 1.0


if __name__ == "__main__":
    test_default_unchanged(); test_idle_only_after_first_gate(); test_idle_formula(); test_zz_and_excess_terms(); print("ok")

"""Unit tests for toffoli_optimizer.core.error_model.HardwareErrorModel.

Every expected value is recomputed independently here (hand formulas / an
independent ALAP scheduler), never by calling the model's own helpers.
``xfail(strict=True)`` tests are reproducers of confirmed bugs (audit 2026-10-02).
"""

from __future__ import annotations

import math
import random

import pytest
from hypothesis import given, settings, strategies as st
from qiskit import QuantumCircuit, transpile

from toffoli_optimizer.core import error_model as em
from toffoli_optimizer.core.error_model import HardwareErrorModel


# ----------------------------------------------------------------- independent oracle
def _err(name, nq, p1, p2, pr):
    if name in ("barrier", "id"):
        return 0.0
    if name == "measure":
        return pr
    if name in ("ccx", "ccz"):
        return 1 - (1 - p2) ** 6
    return p2 if nq >= 2 else p1


def _plain_infidelity(qc, p1, p2, pr=0.0):
    f = 1.0
    for inst in qc.data:
        f *= 1 - _err(inst.operation.name, len(inst.qubits), p1, p2, pr)
    return 1 - f


def _alap_idle(qc, t1q, t2q):
    """Independent forward ALAP: schedule ASAP on the reversed circuit, then each
    qubit idles (first-gate-start .. end) minus busy time."""
    n = qc.num_qubits
    dur = {}
    ops = []
    for inst in qc.data:
        nm = inst.operation.name
        qs = [qc.find_bit(q).index for q in inst.qubits]
        if nm in ("barrier",):
            d = None
        elif nm in ("id", "rz", "measure"):
            d = 0.0
        elif nm in ("ccx", "ccz"):
            d = 6 * t2q + 8 * t1q
        else:
            d = t2q if len(qs) >= 2 else t1q
        ops.append((nm, qs, d))
    end = [0.0] * n                     # reversed-time frontier
    start_rev = [None] * n
    busy = [0.0] * n
    for nm, qs, d in reversed(ops):
        t = max(end[q] for q in qs)
        if d is None:
            for q in qs:
                end[q] = t
            continue
        for q in qs:
            end[q] = t + d
            if d > 0:
                busy[q] += d
                start_rev[q] = t + d
    return {q: start_rev[q] - busy[q] for q in range(n) if start_rev[q] is not None}


def _idle_channel_fidelity(tau, T1, T2):
    a = 1 - math.exp(-tau / T1)
    b = 1 - math.exp(-tau / T2)
    px = py = a / 4
    pz = max(0.0, b / 2 - a / 4)
    return 1 - px - py - pz


# ---------------------------------------------------------------------- counting
def test_counts_basic():
    qc = QuantumCircuit(4, 1)
    qc.h(0); qc.x(1); qc.rz(0.3, 2)          # 3 x 1q
    qc.cx(0, 1); qc.cz(1, 2); qc.swap(2, 3)  # 3 x 2q
    qc.ccx(0, 1, 2)                          # 6 x 2q
    qc.ccz(1, 2, 3)                          # 6 x 2q
    qc.barrier(); qc.id(0)                   # free
    qc.measure(0, 0)                         # not a gate
    m = HardwareErrorModel()
    assert m.two_qubit_count(qc) == 3 + 6 + 6
    assert m.one_qubit_count(qc) == 3
    assert em.two_qubit_count(qc) == 15
    rep = m.report(qc)
    assert rep["two_qubit_count"] == 15 and rep["one_qubit_count"] == 3
    assert rep["fidelity"] == pytest.approx(1 - rep["infidelity"])


def test_gate_error_values():
    m = HardwareErrorModel(p2q=0.02, p1q=0.003, p_readout=0.05)
    assert m.gate_error("cx", 2) == 0.02
    assert m.gate_error("h", 1) == 0.003
    assert m.gate_error("barrier", 4) == 0.0 and m.gate_error("id", 1) == 0.0
    assert m.gate_error("measure", 1) == 0.05
    assert m.gate_error("ccx", 3) == pytest.approx(1 - 0.98 ** 6)
    assert m.gate_error("CCX", 3) == pytest.approx(1 - 0.98 ** 6)  # case-insensitive


_GATES = ["h", "x", "t", "rz", "cx", "cz", "swap", "ccx", "ccz", "barrier", "id", "measure"]


def _build(ops, n=4):
    qc = QuantumCircuit(n, n)
    for g, qs in ops:
        if g == "rz":
            qc.rz(0.1, qs[0])
        elif g in ("cx", "cz", "swap"):
            getattr(qc, g)(qs[0], qs[1])
        elif g in ("ccx", "ccz"):
            getattr(qc, g)(qs[0], qs[1], qs[2])
        elif g == "barrier":
            qc.barrier(*qs[:2])
        elif g == "measure":
            qc.measure(qs[0], qs[0])
        else:
            getattr(qc, g)(qs[0])
    return qc


_op = st.tuples(st.sampled_from(_GATES), st.permutations(range(4)))


@settings(max_examples=60, deadline=None, derandomize=True)
@given(st.lists(_op, max_size=25),
       st.floats(0, 0.2), st.floats(0, 0.05), st.floats(0, 0.1))
def test_infidelity_is_one_minus_product(ops, p2, p1, pr):
    qc = _build(ops)
    m = HardwareErrorModel(p2q=p2, p1q=p1, p_readout=pr)
    assert m.circuit_infidelity(qc) == pytest.approx(_plain_infidelity(qc, p1, p2, pr), abs=1e-12)
    assert em.circuit_infidelity(qc) == pytest.approx(_plain_infidelity(qc, 1e-3, 1e-2), abs=1e-12)
    # defaults-off extensions reproduce the plain model exactly
    m0 = HardwareErrorModel(p2q=p2, p1q=p1, p_readout=pr, t1=None, t2=None,
                            zz_hz=0.0, p_excess_2q=0.0)
    assert m0.circuit_infidelity(qc) == m.circuit_infidelity(qc)
    assert m.idle_fidelity(qc) == 1.0 and m.zz_fidelity(qc) == 1.0


def test_infidelity_monotone_in_gates():
    m = HardwareErrorModel()
    qc = QuantumCircuit(2)
    prev = m.circuit_infidelity(qc)
    for k in range(10):
        qc.cx(0, 1) if k % 2 else qc.h(0)
        cur = m.circuit_infidelity(qc)
        assert cur > prev
        prev = cur


# --------------------------------------------------------------------------- idle
def test_idle_times_hand_example():
    m = HardwareErrorModel(t1=100e-6, t2=80e-6, t_2q=70e-9, t_1q=30e-9)
    qc = QuantumCircuit(3)
    qc.cx(0, 1); qc.x(1); qc.x(1)
    qc.x(2)
    idle = m.idle_times(qc)
    # ALAP: q1 runs x,x at the end; the cx starts 2 t1q before them; q0 waits 2 t1q
    assert idle[0] == pytest.approx(2 * 30e-9)
    assert idle[1] == pytest.approx(0.0)
    assert idle[2] == pytest.approx(0.0)          # x(2) is pushed to the end
    # an initial x on q0 is scheduled late (ALAP) and does not add idling
    qc2 = QuantumCircuit(2); qc2.x(0); qc2.x(1); qc2.x(1); qc2.x(1); qc2.cx(0, 1)
    assert m.idle_times(qc2) == pytest.approx({0: 0.0, 1: 0.0})


@pytest.mark.parametrize("seed", range(5))
def test_idle_matches_independent_scheduler(seed):
    rng = random.Random(seed)
    ops = [(rng.choice(_GATES[:-1]), rng.sample(range(4), 4)) for _ in range(20)]
    qc = _build(ops)
    m = HardwareErrorModel(t1=50e-6, t2=40e-6)
    got = m.idle_times(qc)
    want = _alap_idle(qc, m.t_1q, m.t_2q)
    assert set(got) == set(want)
    for q in got:
        assert got[q] == pytest.approx(want[q], abs=1e-15)
    f = 1.0
    for tau in want.values():
        f *= _idle_channel_fidelity(tau, 50e-6, 40e-6)
    assert m.idle_fidelity(qc) == pytest.approx(f, rel=1e-12)
    base = _plain_infidelity(qc, m.p1q, m.p2q)
    assert m.circuit_infidelity(qc) == pytest.approx(1 - (1 - base) * f, rel=1e-12)


def test_idle_channel_limits():
    m = HardwareErrorModel(t1=1e-6, t2=1e-6)
    qc = QuantumCircuit(2); qc.cx(0, 1)
    for _ in range(2000):
        qc.x(1)
    f = m.idle_fidelity(qc)
    tau = m.idle_times(qc)[0]
    # long wait -> fully depolarised in the twirled AD+PD channel: 1 - 1/4 - 1/2
    assert tau > 50e-6
    assert f == pytest.approx(0.25, abs=1e-6)


# ----------------------------------------------------------------------------- ZZ
def test_zz_fidelity_hand_example():
    zeta = 2e5
    m = HardwareErrorModel(zz_hz=zeta, t_2q=100e-9, t_1q=50e-9)
    qc = QuantumCircuit(3)
    qc.cx(0, 1); qc.x(0); qc.x(0); qc.x(0)
    win = m.active_windows(qc)
    T_total = 100e-9 + 3 * 50e-9
    assert win[0] == pytest.approx((0.0, T_total))
    assert win[1] == pytest.approx((0.0, T_total))
    want = 1 - 0.75 * math.sin(math.pi * zeta * T_total) ** 2
    assert m.zz_fidelity(qc) == pytest.approx(want)
    base = _plain_infidelity(qc, m.p1q, m.p2q)
    assert m.circuit_infidelity(qc) == pytest.approx(1 - (1 - base) * want)
    # explicit layout: a pair that is never jointly active costs nothing
    m2 = HardwareErrorModel(zz_hz=zeta, zz_pairs=[(1, 2)])
    assert m2.zz_fidelity(qc) == 1.0
    # sin^2 periodicity: zeta*T = 1 -> no error
    m3 = HardwareErrorModel(zz_hz=1 / T_total, t_2q=100e-9, t_1q=50e-9)
    assert m3.zz_fidelity(qc) == pytest.approx(1.0)


def test_p_excess_2q_factor():
    qc = QuantumCircuit(3); qc.cx(0, 1); qc.ccx(0, 1, 2); qc.h(2)
    m = HardwareErrorModel(p_excess_2q=0.01)
    base = _plain_infidelity(qc, 1e-3, 1e-2)
    assert m.circuit_infidelity(qc) == pytest.approx(1 - (1 - base) * 0.99 ** 7)


# --------------------------------------------------------------------- validation
@pytest.mark.parametrize("kw", [
    {"p2q": 1.0}, {"p2q": -0.1}, {"p1q": 1.0}, {"p1q": -1e-9}, {"p_readout": 1.0},
    {"p2q": float("nan")}, {"t1": 1e-6}, {"t2": 1e-6}, {"t1": 0.0, "t2": 1e-6},
    {"t1": 1e-6, "t2": -1.0}, {"zz_hz": -1.0}, {"p_excess_2q": 1.0},
    {"p_excess_2q": -0.5},
])
def test_parameter_validation(kw):
    with pytest.raises(ValueError):
        HardwareErrorModel(**kw)


def test_nan_coherence_times_rejected():
    with pytest.raises(ValueError):
        HardwareErrorModel(t1=float("nan"), t2=1e-6)


# ------------------------------------------------------------------ confirmed bugs
def _decomposed_cost(qc, m):
    d = transpile(qc, basis_gates=["cx", "u"], optimization_level=0)
    return m.two_qubit_count(d), m.circuit_infidelity(d)


@pytest.mark.xfail(strict=True, reason=(
    "BUG 🟠: two_qubit_count docstring says ccx/ccz/MCX are charged their standard "
    "decomposition cost and the module 'never under-counts', but 'mcx' (3+ controls) "
    "is charged ONE 2q gate (qiskit: 14 CX). error_model.py:43-46,149-153"))
def test_mcx_not_undercounted():
    m = HardwareErrorModel()
    qc = QuantumCircuit(4); qc.mcx([0, 1, 2], 3)
    assert m.two_qubit_count(qc) >= _decomposed_cost(qc, m)[0]


@pytest.mark.xfail(strict=True, reason=(
    "BUG 🟠: cswap / rccx / c3sx (atomic 3-4 qubit gates) are charged ONE 2q gate "
    "(8 / 3 / 20 CX after decomposition). error_model.py:43-46"))
@pytest.mark.parametrize("gate", ["cswap", "rccx", "c3sx"])
def test_other_multiqubit_gates_not_undercounted(gate):
    m = HardwareErrorModel()
    qc = QuantumCircuit(4)
    if gate == "c3sx":
        from qiskit.circuit.library import C3SXGate
        qc.append(C3SXGate(), [0, 1, 2, 3])
    else:
        getattr(qc, gate)(0, 1, 2)
    assert m.two_qubit_count(qc) >= _decomposed_cost(qc, m)[0]


@pytest.mark.xfail(strict=True, reason=(
    "BUG 🔵: ccx comment says '+ 1q gates, costed separately below', but the 9 "
    "single-qubit gates of the textbook Toffoli are never charged, so an atomic CCX "
    "costs less than its own decomposition. error_model.py:44,126-129"))
def test_ccx_infidelity_not_below_its_decomposition():
    m = HardwareErrorModel()
    qc = QuantumCircuit(3); qc.ccx(0, 1, 2)
    assert m.circuit_infidelity(qc) >= _decomposed_cost(qc, m)[1] - 1e-15


@pytest.mark.xfail(strict=True, reason=(
    "BUG 🔵: 'delay' is charged as a single-qubit gate (p1q) and counted by "
    "one_qubit_count; idle time is the idle model's job. error_model.py:131-133,156-166"))
def test_delay_is_not_a_gate():
    m = HardwareErrorModel()
    qc = QuantumCircuit(1); qc.delay(100, 0)
    assert m.one_qubit_count(qc) == 0 and m.circuit_infidelity(qc) == 0.0

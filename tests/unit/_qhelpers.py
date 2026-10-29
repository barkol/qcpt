"""Shared helpers for the compiler / depth-optimizer unit tests.

Ground truth is ``qiskit.quantum_info.Operator`` on small circuits (<= 7 qubits).
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

BASIS = ['id', 'rz', 'sx', 'x', 'cx']


def op(circ):
    """Operator of ``circ`` in the *virtual* qubit order (undoes a transpile layout)."""
    if getattr(circ, 'layout', None) is not None:
        return Operator.from_circuit(circ)
    return Operator(circ)


def equiv(a, b):
    """Unitary equivalence up to global phase (layout-aware)."""
    oa, ob = op(a), op(b)
    if oa.dim != ob.dim:
        return False
    return oa.equiv(ob)


def clean_ancilla_equiv(circ, ref, ancillas, rel_phase=False, atol=1e-8):
    """Check ``circ`` implements ``ref`` when ``ancillas`` start in |0> and are restored.

    ``ref`` acts on the same number of qubits as ``circ`` (it simply never touches
    the ancillas). With ``rel_phase=True`` only the moduli are compared (i.e. the
    weaker "relative-phase" contract: right classical action, arbitrary diagonal
    phases).
    """
    n = circ.num_qubits
    mask = sum(1 << a for a in ancillas)
    cols = [i for i in range(2 ** n) if i & mask == 0]
    U = op(circ).data[:, cols]
    R = op(ref).data[:, cols]
    if rel_phase:
        return np.allclose(np.abs(U), np.abs(R), atol=atol)
    # global phase from the largest reference entry
    k = np.unravel_index(np.argmax(np.abs(R)), R.shape)
    if abs(U[k]) < 1e-9:
        return False
    phase = U[k] / R[k]
    return np.allclose(U, phase * R, atol=atol)


def random_circuit(n, m, rng, gates=('x', 'cx', 'ccx', 'mcx', 'h', 't', 's', 'rz', 'swap'),
                   inverse_pairs=True, h_layer_if_mcx=True):
    """Random circuit over the requested gate alphabet.

    Random qubit orders, repeats and (optionally) adjacent inverse pairs.

    qiskit's transpiler synthesises MCX using *idle qubits assumed to be in |0>*
    (``qubits_initially_zero``), which is correct for a program started in
    |0...0> but not as a unitary.  To keep the unitary ground truth meaningful we
    open MCX-containing circuits with an H layer, so no qubit is "known zero".
    """
    qc = QuantumCircuit(n)
    gates = [g for g in gates if {'cx': 2, 'swap': 2, 'ccx': 3, 'mcx': 4}.get(g, 1) <= n]
    picks = [gates[int(rng.integers(len(gates)))] for _ in range(m)]
    if h_layer_if_mcx and 'mcx' in picks:
        for q in range(n):
            qc.h(q)
    for g in picks:
        q = [int(x) for x in rng.permutation(n)]
        reps = 2 if (inverse_pairs and rng.random() < 0.15) else 1
        for _ in range(reps):
            if g == 'x':
                qc.x(q[0])
            elif g == 'h':
                qc.h(q[0])
            elif g == 't':
                qc.t(q[0])
            elif g == 's':
                qc.s(q[0])
            elif g == 'rz':
                qc.rz(float(rng.normal()), q[0])
            elif g == 'cx':
                qc.cx(q[0], q[1])
            elif g == 'swap':
                qc.swap(q[0], q[1])
            elif g == 'ccx':
                qc.ccx(q[0], q[1], q[2])
            elif g == 'mcx':
                qc.mcx(q[:3], q[3])
    return qc


def independent_counts(circ):
    """Gate statistics computed directly from circuit.data (no count_ops)."""
    names = [inst.operation.name for inst in circ.data]
    return {
        'size': len(names),
        'cx': names.count('cx'),
        't': names.count('t') + names.count('tdg'),
    }


def independent_depth(circ):
    """ASAP depth over qubits and clbits, ignoring barriers (qiskit's convention)."""
    level = {}
    depth = 0
    for inst in circ.data:
        if inst.operation.name == 'barrier':
            continue
        wires = list(inst.qubits) + list(inst.clbits)
        if not wires:
            continue
        lv = max(level.get(w, 0) for w in wires) + 1
        for w in wires:
            level[w] = lv
        depth = max(depth, lv)
    return depth

"""General simplification by the ZX-calculus (optional, needs PyZX).

``zx_simplify(circuit)`` lowers the circuit to {cx, h, x, z, s, sdg, t, tdg, rz}, reduces its ZX-diagram
(``full_reduce``), extracts a circuit and returns it as a Qiskit circuit, or None if PyZX is not installed or the
conversion fails. The result is equal to the input only up to a global phase and the rotation angles are rounded
to rational multiples of pi by PyZX, so it is never used without the whole-circuit certificate
(see ``ErrorBudgetSelector(simplify="zx")``)."""
from __future__ import annotations

from typing import Optional

from qiskit import QuantumCircuit, qasm2, transpile

_BASIS = ["cx", "h", "x", "z", "s", "sdg", "t", "tdg", "rz"]


def zx_available() -> bool:
    try:
        import pyzx  # noqa: F401
        return True
    except Exception:
        return False


def zx_simplify(circuit: QuantumCircuit) -> Optional[QuantumCircuit]:
    try:
        import pyzx as zx
    except Exception:
        return None
    try:
        lowered = transpile(circuit, basis_gates=_BASIS, optimization_level=0)
        zc = zx.Circuit.from_qasm(qasm2.dumps(lowered))
        g = zc.to_graph()
        zx.simplify.full_reduce(g, quiet=True)
        out = zx.extract_circuit(g.copy()).to_basic_gates()
        out = zx.basic_optimization(out)
        res = qasm2.loads(out.to_qasm(), custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    except Exception:
        return None
    if res.num_qubits != circuit.num_qubits:
        return None
    return res

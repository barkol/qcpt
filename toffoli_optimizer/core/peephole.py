"""General peephole simplification of a circuit (commutation-aware CNOT cancellation, resynthesis of two-qubit blocks).

``peephole(circuit)`` lowers the circuit to {cx, u} and repeats, until the number of CX stops decreasing,
CommutativeCancellation (cancellations through commuting gates, e.g. across the boundaries of neighbouring
decompositions), Collect2qBlocks + ConsolidateBlocks + UnitarySynthesis (every two-qubit block is resynthesised with the
minimal number of CX) and the merging of single-qubit gates. These are the passes of the Qiskit transpiler at level 2/3 without the
elision of permutations, so the qubits keep their wires. The result is equal to the input up to a global phase; it is
never used without the whole-circuit certificate (see ``ErrorBudgetSelector(simplify=...)``)."""
from __future__ import annotations

from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (CommutativeCancellation, Collect2qBlocks, ConsolidateBlocks,
                                      Optimize1qGatesDecomposition, UnitarySynthesis)

_BASIS = ["cx", "u"]


def peephole(circuit: QuantumCircuit, max_rounds: int = 6) -> QuantumCircuit:
    best = transpile(circuit, basis_gates=_BASIS, optimization_level=0)
    pm = PassManager([CommutativeCancellation(), Collect2qBlocks(), ConsolidateBlocks(basis_gates=_BASIS),
                      UnitarySynthesis(basis_gates=_BASIS), Optimize1qGatesDecomposition(basis=["u"])])
    key = lambda c: (c.count_ops().get("cx", 0), c.size())          # two-qubit gates first, then the total number of gates
    for _ in range(max_rounds):
        nxt = pm.run(best)
        if not key(nxt) < key(best):
            break
        best = nxt
    return best

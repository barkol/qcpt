"""Real-application benchmark: a quantum synchronizing-word (state-resetting)
circuit, following the construction of Stempin et al., "Quantum Resetting
Protocols Based on Synchronizing Words" (arXiv:2504.01106) and the
qutrit/qudit synchronizing-words framework (arXiv:2502.09522).

Construction (faithful to the paper's per-step operator):
    each automaton "letter" j applies  U_j = T_j^dagger . S . T_j
where
  * T_j is a *controlled permutation* of the binary-encoded state register,
    conditioned on the auxiliary "letter" qubit c_j (T = |0><0|⊗I + |1><1|⊗Tg);
    realised as a multi-controlled gate, which decomposes into a CCX pair that
    computes a predicate ancilla and a CCX that uncomputes it
    (standard MCX -> CCX-with-borrowed-ancilla);
  * S is the shift permutation (a binary increment of the state register);
  * T_j^dagger inverts T_j (the controlled permutation is uncomputed).

So every letter contributes a compute/uncompute conjugation around the shift:
exactly the context the context-verified pass certifies and exploits. We build
the circuit with CCX gates only (the pass targets CCX) and run it through the
ErrorBudgetSelector, reporting the *measured* two-qubit-gate / infidelity
reduction and whether every emitted substitution is verified.

Run:
    pip install -e ".[experiments]"
    python experiments/sync_benchmark.py
"""
from __future__ import annotations

import sys

from _paths import EXPERIMENTS_DIR  # noqa: F401  (must be first)

from qiskit import QuantumCircuit, QuantumRegister  # noqa: E402

from toffoli_optimizer.core.optimizer import (  # noqa: E402
    ErrorBudgetSelector,
    HardwareErrorModel,
)


def _mcx3_via_ccx(qc, c0, c1, c2, tgt, anc):
    """3-control X (c0,c1,c2 -> tgt) using one borrowed ancilla, as a
    compute/uncompute CCX triple: anc=c0&c1, tgt ^= anc&c2, uncompute anc."""
    qc.ccx(c0, c1, anc)
    qc.ccx(anc, c2, tgt)
    qc.ccx(c0, c1, anc)


def _shift(qc, s):
    """Binary increment (mod 2^m) of the state register s -> the shift S.
    Carry chain uses CCX for m>=3 (a 'live' permutation, not uncomputed)."""
    m = len(s)
    # increment: flip MSB-carry first (high-order), then ripple down
    for i in range(m - 1, 1, -1):
        # carry into bit i if all lower bits are 1: approximate ripple with CCX
        qc.ccx(s[0], s[i - 1], s[i])
    if m >= 2:
        qc.cx(s[0], s[1])
    qc.x(s[0])


def resetting_circuit(m: int = 3, n_letters: int = 7) -> QuantumCircuit:
    """Synchronizing-word resetting circuit on a 2^m-state register.

    m       : state-register qubits (encodes n = 2^m automaton nodes)
    n_letters: length of the applied (synchronizing) word
    """
    s = QuantumRegister(m, "s")        # binary-encoded automaton state
    c = QuantumRegister(n_letters, "c")  # one 'letter' qubit per step
    a = QuantumRegister(1, "a")         # borrowed predicate ancilla
    qc = QuantumCircuit(s, c, a, name=f"resetting_n{2**m}_L{n_letters}")

    for j in range(n_letters):
        cj = c[j]
        # T_j : controlled permutation Tg of the state, conditioned on letter cj.
        #       Tg here is the transposition flipping s[m-1] when (s0,s1)=11,
        #       so controlled-Tg is the 3-control X (cj,s0,s1 -> s_{m-1}).
        _mcx3_via_ccx(qc, cj, s[0], s[1], s[m - 1], a[0])
        # S : shift the state register
        _shift(qc, s)
        # T_j^dagger : uncompute the controlled permutation (CCX/MCX self-inverse)
        _mcx3_via_ccx(qc, cj, s[0], s[1], s[m - 1], a[0])
    return qc


def run(qc):
    em = HardwareErrorModel(p2q=4.0e-3, p1q=2.0e-4)   # IonQ-Forte-like
    sel = ErrorBudgetSelector(error_model=em, phase_aware=True)
    from _clean import clean_ancillas
    res = sel.select(qc, pinned_zero=clean_ancillas(qc))
    rep = res["report"]
    nccx = sum(1 for inst in qc.data if inst.operation.name == "ccx")
    b, af = rep["two_qubit_before"], rep["two_qubit_after"]
    ib, ia = rep["infidelity_before"], rep["infidelity_after"]
    print(f"{qc.name:22s} q={qc.num_qubits:2d} ccx={nccx:2d} "
          f"2q {b:3d}->{af:3d} ({100*(b-af)/b:4.1f}%)  "
          f"infid {ib:.4f}->{ia:.4f} ({100*(ib-ia)/ib:4.1f}%)  "
          f"verified={rep['verified']}")
    return rep


if __name__ == "__main__":
    print("Quantum synchronizing-word / resetting circuits "
          "(Stempin et al., arXiv:2504.01106 / 2502.09522)\n")
    for m, L in [(3, 3), (3, 5), (3, 7)]:
        run(resetting_circuit(m, L))
        sys.stdout.flush()

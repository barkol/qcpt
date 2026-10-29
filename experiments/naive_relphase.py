"""Relative-phase (Margolus / RCCX) Toffoli substitution passes.

`naive_relphase` is the COUNT-GREEDY, context-free baseline: it is the kind of
pass a pattern-matched / Maslov-template optimizer performs when it trusts its
rewrite library. It replaces EVERY CCX with the cheap relative-phase Toffoli
(qiskit `RCCXGate`, the Margolus gate), unconditionally. The Margolus gate
agrees with CCX on the computational action of the AND but applies an extra
relative phase of -1 to the |11>|1> -> |11>|0> branch. This is cheaper (fewer
T gates / lower depth) but it is only function-preserving when that phase is
later cancelled (compute/uncompute). Applied blindly it SILENTLY changes the
function of the circuit.

`gated_relphase` performs the SAME per-gate substitution, but wraps each
substitution in a per-window verification gate using QCPT's committed
ExactEquivalenceVerifier: it keeps a substitution only if replacing that single
CCX (in the context of the whole circuit) is proven equivalent to the original,
otherwise it falls back to the exact CCX. This makes aggressive selection SOUND.
"""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit.library import RCCXGate


def _ccx_positions(circuit: QuantumCircuit):
    """Indices into circuit.data of every CCX instruction."""
    return [i for i, inst in enumerate(circuit.data) if inst.operation.name == "ccx"]


def _rebuild_with_substitutions(circuit: QuantumCircuit, substitute: set) -> QuantumCircuit:
    """Return a copy of `circuit` where the CCX instructions whose data-index is
    in `substitute` are replaced by an RCCX (relative-phase) gate on the same
    qubits; all other instructions are copied verbatim."""
    new = QuantumCircuit(*circuit.qregs, *circuit.cregs, name=circuit.name)
    for i, inst in enumerate(circuit.data):
        if i in substitute and inst.operation.name == "ccx":
            new.append(RCCXGate(), inst.qubits)
        else:
            new.append(inst.operation, inst.qubits, inst.clbits)
    return new


def naive_relphase(circuit: QuantumCircuit) -> QuantumCircuit:
    """Count-greedy: replace EVERY CCX with a relative-phase (RCCX) Toffoli.

    No verification. Returns the (possibly silently incorrect) rewritten circuit.
    The number of substitutions applied is stored on `.relphase_applied`.
    """
    pos = set(_ccx_positions(circuit))
    out = _rebuild_with_substitutions(circuit, pos)
    out.relphase_applied = len(pos)
    return out


def _verifies(circuit, verifier, subset) -> bool:
    """True iff substituting RCCX at the data-indices in `subset` keeps the
    circuit function-equivalent to the original. A per-gate RCCX substitution
    never permutes data wires, so we use allow_permutation=False -- which also
    keeps the gate cheap (the exact-unitary permutation search is factorial in
    the qubit count)."""
    candidate = _rebuild_with_substitutions(circuit, subset)
    ok, _perm, _info = verifier.verify(circuit, candidate, allow_permutation=False)
    return ok


def gated_relphase(circuit: QuantumCircuit, verifier) -> QuantumCircuit:
    """Per-instance VERIFIED relative-phase substitution -- the SOUND pass.

    Greedily grows a committed set of CCX -> RCCX substitutions, but commits a
    candidate only after the EXACT verifier proves the substituted circuit still
    computes the original function. Anything that cannot be proven equivalent is
    left as an exact Toffoli. The result is therefore sound by construction,
    while still harvesting every relative-phase decomposition that IS valid.

    Two commit shapes are tried per candidate, against the current committed set:

      * singleton: replace this one CCX (valid when its phase is otherwise
        irrelevant / already cancelled);
      * pair-completion: replace this CCX together with one not-yet-committed
        CCX (this is what captures compute/uncompute pairs -- the Margolus gate
        is self-inverse, so the spurious phase of the compute is cancelled by
        the uncompute only when BOTH are substituted).

    The scan is repeated to a fixpoint so that a substitution rejected early can
    succeed once its partner is available. Every committed superset is verified,
    so soundness never depends on the search order.

    Returns the rewritten circuit; `.relphase_applied` counts kept substitutions.
    """
    pos = _ccx_positions(circuit)
    committed: set = set()

    changed = True
    while changed:
        changed = False
        for p in pos:
            if p in committed:
                continue
            # try this CCX alone
            if _verifies(circuit, verifier, committed | {p}):
                committed = committed | {p}
                changed = True
                continue
            # try completing a compute/uncompute pair with another free CCX
            for q in pos:
                if q == p or q in committed:
                    continue
                if _verifies(circuit, verifier, committed | {p, q}):
                    committed = committed | {p, q}
                    changed = True
                    break

    out = _rebuild_with_substitutions(circuit, committed)
    out.relphase_applied = len(committed)
    return out

"""Phase-observability: can a relative phase introduced at a gate ever be measured?

A relative-phase (Margolus) Toffoli agrees with exact CCX as a Boolean function but
stamps a relative phase on the |11> control branch. Replacing an exact CCX by the
gadget is SOUND exactly when that extra phase can never influence any measurement
outcome. There are two independent sound reasons that can hold:

  (a) PHASE CANCELS.  The gate is one half of a verified compute/uncompute pair, so
      the compute's relative phase is exactly undone by the uncompute (this is the
      pre-existing ``context_analysis`` criterion). The pair is substituted together.

  (b) PHASE NEVER BECOMES OBSERVABLE.  Downstream of the gate, the qubits carrying
      the potential phase only ever feed CLASSICAL-reversible operations
      (x/cx/ccx/mcx/swap) and end at a computational-basis read. A relative phase on
      a computational-basis amplitude is invisible to a computational-basis
      measurement, and a classical-reversible op is a permutation matrix -- it maps a
      basis state to a basis state, never interfering two branches -- so it cannot
      turn a relative phase into a population difference. The phase therefore remains
      a global-on-its-branch, unobservable phase forever. The MOMENT a non-classical
      gate (h, ry, t, controlled-phase, any basis change) touches a tainted qubit,
      two branches can interfere and the phase becomes observable -- so we must
      reject.

This module implements a CONSERVATIVE, sound test for (b), and combines it with (a).
"is unsure -> return False" is the rule everywhere: a false "observable" only costs
us a missed optimization; a false "unobservable" would be a soundness bug.

SOUNDNESS OF THE FORWARD-CONE TEST (b).  Let ``T`` be the set of qubits that carry
the relative phase right after the gate (the gate's controls and target). Walk the
instructions after the gate. Maintain ``tainted = T``. For each instruction:

  * Classical-reversible gate (x/cx/ccx/mcx/swap): it is a permutation of basis
    states. It can MOVE the phase to other qubits (a CX from a tainted control, a
    swap of a tainted qubit, a tainted control of a CCX), so we grow ``tainted`` to
    include every qubit of any gate that touches a tainted qubit. But it never makes
    the phase observable.
  * measure / reset on tainted qubits: a computational-basis measurement of a
    relative-phased basis amplitude is unaffected by the phase -> still safe. (reset
    likewise.)
  * ANY non-classical gate (h/ry/rz/t/u/cp/cz/... -- anything not in the classical
    set, and not measure/reset/barrier) acting on a tainted qubit: it can interfere
    branches -> the phase may become observable -> UNSAFE, return False.
  * A non-classical gate that does NOT touch any tainted qubit is irrelevant.

If we reach the end of the circuit without any non-classical gate touching a tainted
qubit, the phase only ever fed classical-reversible ops and basis reads, so it is
unobservable -- return True. This is sound: growing ``tainted`` is an
over-approximation of where the phase can be, and we reject on the first chance of
interference.
"""

from __future__ import annotations

from typing import Iterable, Set

from .context_analysis import find_relative_phase_safe_sites

# Classical-reversible permutation gates: cannot turn a relative phase into an
# observable population difference.
_CLASSICAL_GATES = {"x", "cx", "ccx", "mcx", "mcx_gray", "swap", "id", "barrier"}

# Measurement-like ops that read in the computational basis (phase-insensitive).
_BASIS_READ = {"measure", "reset"}


def _qubit_indices(circuit, inst):
    return [circuit.find_bit(q).index for q in inst.qubits]


def _gate_in_verified_pair(circuit, gate_index) -> bool:
    """True iff ``gate_index`` is the compute or uncompute half of a detected pair."""
    for site in find_relative_phase_safe_sites(circuit):
        if gate_index in (site.compute_idx, site.uncompute_idx):
            return True
    return False


def is_phase_unobservable(
    circuit,
    gate_index: int,
    affected_qubits: Iterable[int],
) -> bool:
    """Sound test: can a relative phase introduced at ``gate_index`` be measured?

    Returns True only if it is PROVABLY unobservable, via either reason (a) the gate
    is in a verified compute/uncompute pair (phase cancels), or (b) the forward cone
    from ``affected_qubits`` only ever feeds classical-reversible ops and a
    computational-basis read (no downstream interference). Conservative: any
    uncertainty -> False.

    Parameters
    ----------
    circuit : QuantumCircuit
    gate_index : int
        Index into ``circuit.data`` of the (relative-phase candidate) CCX.
    affected_qubits : iterable of int
        Qubits that carry the potential relative phase immediately after the gate --
        for a CCX(a,b,t) this is {a, b, t} (the |11> branch lives on the controls and
        the phase rides the whole control/target register).

    Returns
    -------
    bool
    """
    # Reason (a): a verified compute/uncompute pair cancels the phase exactly.
    if _gate_in_verified_pair(circuit, gate_index):
        return True

    # Reason (b): forward-cone classicality to a basis read.
    return _phase_unobservable_in_forward_cone(circuit, gate_index, affected_qubits)


def _phase_unobservable_in_forward_cone(
    circuit, gate_index, affected_qubits
) -> bool:
    """Sound forward-cone test (reason (b)). See module docstring for the argument."""
    data = circuit.data
    n = len(data)
    tainted: Set[int] = set(affected_qubits)
    if not tainted:
        return False  # nothing identified -> be conservative

    for k in range(gate_index + 1, n):
        inst = data[k]
        name = inst.operation.name.lower()
        if name in ("barrier", "id"):
            continue
        qb = _qubit_indices(circuit, inst)
        touches = any(q in tainted for q in qb)

        if name in _BASIS_READ:
            # A computational-basis read is insensitive to a relative phase on a
            # basis amplitude. It does not propagate the phase. Safe; do not taint.
            continue

        if name in _CLASSICAL_GATES:
            # Permutation gate. If it touches the tainted set it can MOVE the phase
            # onto its other qubits -> grow the taint. It never makes it observable.
            if touches:
                tainted.update(qb)
            continue

        # Any other (non-classical) gate: h, ry, rz, t, u, p, cp, cz, cy, ...
        # If it touches a tainted qubit it can interfere branches -> observable.
        if touches:
            return False
        # Does not touch the phase -> irrelevant, ignore.

    # Reached the end with no interfering gate on a tainted qubit: unobservable.
    return True


def default_affected_qubits(circuit, gate_index) -> Set[int]:
    """The qubits carrying the relative phase of the CCX at ``gate_index``.

    For a CCX/MCX the relative phase rides the controls and target; we return all of
    the gate's qubits.
    """
    inst = circuit.data[gate_index]
    return set(_qubit_indices(circuit, inst))

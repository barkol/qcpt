"""Context analysis: where is a relative-phase Toffoli decomposition *admissible*?

The cheap relative-phase (Margolus) Toffoli implements CCX(a,b,t) correctly only
UP TO A RELATIVE PHASE on the target subspace: it flips ``t`` exactly when
``a & b``, like a real Toffoli, but it also stamps a state-dependent phase. That
phase is harmless precisely when it is later *uncomputed* -- i.e. when the gate is
one half of a compute/uncompute pair whose two relative phases cancel.

The canonical safe pattern is

    CCX(a, b, t)              # compute:  t ^= a&b   (+ relative phase phi)
    ... gates that use t only as a CONTROL, never re-touching a or b's logical
        roles in a way that breaks the pairing ...
    CCX(a, b, t)              # uncompute: t ^= a&b   (- relative phase phi)

If between the two matched gates the target ``t`` is only ever read (used as a
control) and never written (never a target, never permuted by a swap), and the
controls ``a, b`` are not disturbed, then replacing BOTH Toffolis by the
relative-phase gadget leaves the overall unitary unchanged on the data subspace:
the relative phase of the compute is exactly undone by the uncompute.

This module finds candidate sites of that form. It is deliberately CONSERVATIVE:
it only reports a (compute_idx, uncompute_idx, qubits) site that *looks* safe by
this structural rule. The decomposition selector then VERIFIES each reported site
exactly (via ExactEquivalenceVerifier) before committing the substitution, so a
false positive here can never produce an incorrect circuit -- it is merely
rejected downstream.
"""

from __future__ import annotations

from typing import List, NamedTuple


class RelativePhaseSite(NamedTuple):
    """A compute/uncompute CCX pair where a relative-phase decomposition is
    structurally admissible (pending exact verification)."""
    compute_idx: int       # index into circuit.data of the compute CCX
    uncompute_idx: int     # index into circuit.data of the matching uncompute CCX
    controls: tuple        # (a, b) control qubit indices (sorted)
    target: int            # t target qubit index


_CCX_NAMES = {"ccx", "mcx", "mcx_gray"}


def _qubit_indices(circuit, inst):
    return [circuit.find_bit(q).index for q in inst.qubits]


def _gate_qubits(circuit, inst):
    """Return (controls_set, target) for a CCX/MCX, else None."""
    name = inst.operation.name.lower()
    if name not in _CCX_NAMES:
        return None
    qb = _qubit_indices(circuit, inst)
    if len(qb) != 3:
        return None  # the relative-phase gadget has exactly two controls (CCX or a 3-qubit MCX)
    return frozenset(qb[:-1]), qb[-1]


def find_relative_phase_safe_sites(circuit) -> List[RelativePhaseSite]:
    """Detect compute/uncompute CCX/MCX pairs admitting a relative-phase decomp.

    A pair (i, j), i < j, qualifies when:

      * instruction i and j are both CCX/MCX with the SAME control set and SAME
        target t;
      * between i and j the target t is used ONLY as a control (never a target,
        never moved by a swap, never measured/reset);
      * between i and j none of the controls is the target of any gate (their
        logical value -- and hence the a&b the gadget recomputes -- is preserved);
      * i and j are matched greedily (innermost pairing), so nested compute/
        uncompute structures pair correctly.

    Returns a list of ``RelativePhaseSite``. Conservative: a site is reported only
    if it passes every structural check; the selector still verifies it exactly.
    """
    data = circuit.data
    n = len(data)

    # Collect indices of CCX/MCX gates with their (controls, target).
    ccx = {}
    for idx in range(n):
        info = _gate_qubits(circuit, data[idx])
        if info is not None:
            ccx[idx] = info  # idx -> (controls_frozenset, target)

    sites: List[RelativePhaseSite] = []
    used = set()  # indices already consumed as a compute or uncompute

    # Greedy innermost matching: for each potential compute i (ascending), find the
    # NEAREST later unused j with identical (controls, target); if the window (i, j) is not
    # safe, an even-further j is also unsafe (its window is a superset), so i stays unpaired.
    # The candidates of one (controls, target) key are kept in a sorted list, so the search is linear.
    from bisect import bisect_right
    by_key: dict = {}
    for idx in sorted(ccx):
        by_key.setdefault((ccx[idx][0], ccx[idx][1]), []).append(idx)
    for i in sorted(ccx):
        if i in used:
            continue
        controls_i, target_i = ccx[i]
        lst = by_key[(controls_i, target_i)]
        k = bisect_right(lst, i)
        while k < len(lst) and lst[k] in used:
            k += 1
        if k < len(lst) and _window_is_safe(circuit, i, lst[k], controls_i, target_i):
            j = lst[k]
            sites.append(
                RelativePhaseSite(
                    compute_idx=i,
                    uncompute_idx=j,
                    controls=tuple(sorted(controls_i)),
                    target=target_i,
                )
            )
            used.add(i)
            used.add(j)
    return sites


def _window_is_safe(circuit, i, j, controls, target) -> bool:
    """Is the open window (i, j) safe for relative-phase substitution of i and j?

    Safe means: between the compute (i) and uncompute (j), the target ``target``
    is used only as a control (read, never written), and none of ``controls`` is
    written. Any gate that writes the target or a control -- as its own target, as
    a swap operand, or via measure/reset -- breaks the phase-cancellation and makes
    the window unsafe.
    """
    data = circuit.data
    protected = set(controls) | {target}
    for k in range(i + 1, j):
        inst = data[k]
        name = inst.operation.name.lower()
        if name in ("barrier", "id"):
            continue
        qb = _qubit_indices(circuit, inst)

        if name in ("measure", "reset"):
            # Any measure/reset on a protected qubit breaks the pairing.
            if any(q in protected for q in qb):
                return False
            continue

        if name == "swap":
            # A swap WRITES both its operands; if either touches a protected qubit
            # the protected wire's value moves -> unsafe.
            if any(q in protected for q in qb):
                return False
            continue

        # x / cx / ccx / mcx and any other unitary: the LAST qubit(s) it writes.
        # For controlled-X family the written qubit is the final target qubit;
        # treat conservatively -- the target qubit of the gate is qb[-1].
        if name in ("x",):
            written = {qb[0]}
        elif name in ("cx", "ccx", "mcx", "mcx_gray", "cz", "ccz", "cy"):
            # controlled-X family: target is last qubit. (cz/ccz are symmetric,
            # but treating the last qubit as "written" is the conservative choice.)
            written = {qb[-1]}
        else:
            # Unknown / single- or multi-qubit gate: conservatively treat EVERY
            # qubit it acts on as written.
            written = set(qb)

        if written & protected:
            return False
    return True

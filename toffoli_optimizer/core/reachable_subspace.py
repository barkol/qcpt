"""Reachable computational-basis subspace at a gate in a Toffoli network.

A relative-phase (Margolus) Toffoli computes the SAME Boolean function as an exact
CCX but stamps a relative phase on the |11> control branch. Whether substituting it
is sound depends on (a) which inputs actually *reach* the gate, and (b) whether the
resulting phase is ever *observable*. This module answers (a): it computes, or
soundly over-approximates, the set of computational-basis states that can appear at
a given gate, given an input space.

WHY A BASIS-STATE SET SUFFICES (and is exact for classical-reversible prefixes).
If every gate BEFORE ``gate_index`` is classical-reversible (x / cx / ccx / mcx /
swap), then the prefix is a permutation ``pi`` of the 2^n computational basis. Any
computational-basis input ``x`` maps to the single basis state ``pi(x)`` at the
gate -- no superposition is created. So the *reachable set* at the gate is exactly
``{ pi(x) : x in input_space }``, a finite set of basis indices. Enumerating it is
the exact reachable set, and the reachable SUBSPACE is its span (superpositions
of reachable basis states included).

If the prefix contains a superposition-creating gate (h / ry / t / ...), a single
basis input can spread over many basis states. We then fall back to a SOUND
OVER-APPROXIMATION: the set of basis states whose amplitude *could* be nonzero is a
superset of the truly-reachable basis states. The admissibility conditions are
OPERATOR identities on the span of the set with one phase (``subspace_check``); an
identity  U_d P_S = e^{i theta} U P_S  on span(S) restricts to the same identity on
span(R) for any R contained in S. So checking on a superset is sound for accepting a
rewrite: it can only make us reject -- never wrongly accept -- a substitution.

Two entry points:

* ``reachable_basis_states(circuit, gate_index, input_space='all_basis')`` -- EXACT
  enumeration for a classical-reversible prefix. Returns a ``frozenset`` of basis
  indices (little-endian: qubit j == bit j). Raises if the prefix is not classical-
  reversible or if the enumeration would be too large (caller should fall back to
  the over-approximation).

* ``reachable_overapprox(circuit, gate_index, ...)`` -- a cheap SOUND superset that
  always succeeds. See its docstring for the soundness argument.
"""

from __future__ import annotations

from typing import FrozenSet, Iterable, Optional, Set, Union

_CLASSICAL_GATES = {"x", "cx", "ccx", "mcx", "mcx_gray", "swap", "id", "barrier"}

# Guard exact enumeration from blowing up. 2^20 basis states is already 1M ints.
_DEFAULT_MAX_BASIS = 1 << 18


def _qubit_indices(circuit, inst):
    return [circuit.find_bit(q).index for q in inst.qubits]


def _is_classical_name(name: str) -> bool:
    return name.lower() in _CLASSICAL_GATES


def _apply_classical(name: str, qb, state: int) -> int:
    """Apply one classical-reversible gate to a packed basis state ``state``.

    ``qb`` are the gate's qubit indices; ``state`` is an int with bit j == qubit j.
    Returns the new packed basis state. Caller guarantees ``name`` is classical.
    """
    name = name.lower()
    if name in ("barrier", "id"):
        return state
    if name == "x":
        return state ^ (1 << qb[0])
    if name == "cx":
        if (state >> qb[0]) & 1:
            return state ^ (1 << qb[1])
        return state
    if name == "ccx":
        if ((state >> qb[0]) & 1) and ((state >> qb[1]) & 1):
            return state ^ (1 << qb[2])
        return state
    if name in ("mcx", "mcx_gray"):
        if all((state >> c) & 1 for c in qb[:-1]):
            return state ^ (1 << qb[-1])
        return state
    if name == "swap":
        b0 = (state >> qb[0]) & 1
        b1 = (state >> qb[1]) & 1
        if b0 != b1:
            return state ^ ((1 << qb[0]) | (1 << qb[1]))
        return state
    raise ValueError(f"_apply_classical called with non-classical gate {name!r}")


def _normalize_input_space(input_space, n: int) -> Iterable[int]:
    """Turn ``input_space`` into an iterable of basis indices.

    * ``'all_basis'``         -> every basis state 0 .. 2^n - 1.
    * ``'ancilla_zero'``      -> basis states whose highest bits are 0 is NOT known
                                 here (we do not know the ancilla split), so this is
                                 treated the same as a caller-provided explicit set.
    * an iterable of ints     -> used verbatim (e.g. a known input domain).
    """
    if input_space == "all_basis":
        return range(1 << n)
    if isinstance(input_space, str):
        raise ValueError(f"unknown input_space string {input_space!r}")
    return input_space


def reachable_basis_states(
    circuit,
    gate_index: int,
    input_space: Union[str, Iterable[int]] = "all_basis",
    max_basis: int = _DEFAULT_MAX_BASIS,
) -> FrozenSet[int]:
    """Exact set of computational-basis states reachable at ``gate_index``.

    Forward-propagates ``input_space`` through instructions ``0 .. gate_index - 1``.
    Requires that PREFIX to be classical-reversible (only x/cx/ccx/mcx/swap); then
    the prefix is a basis permutation and the reachable set is exact and finite.

    Parameters
    ----------
    circuit : QuantumCircuit
    gate_index : int
        Index into ``circuit.data`` of the gate whose reachable inputs we want. The
        returned set is the set of basis states present *just before* this gate.
    input_space : 'all_basis' | iterable of ints
        The set of computational-basis inputs the whole circuit may be run on. Use a
        restricted iterable when the gate is only ever fed a known sub-domain (e.g.
        ancilla known to start in |0>); a smaller input space yields a smaller, still
        exact, reachable set.
    max_basis : int
        Refuse to enumerate if the working set would exceed this many states (the
        caller should fall back to ``reachable_overapprox``).

    Returns
    -------
    frozenset[int]
        Basis indices reachable at the gate (little-endian bit j == qubit j).

    Raises
    ------
    ValueError
        If any prefix gate is not classical-reversible (superposition possible -> a
        basis-set is not exact; use ``reachable_overapprox`` instead), or if the
        working set exceeds ``max_basis``.
    """
    n = circuit.num_qubits
    data = circuit.data
    if not (0 <= gate_index <= len(data)):
        raise ValueError(f"gate_index {gate_index} out of range")

    states: Set[int] = set(_normalize_input_space(input_space, n))
    if len(states) > max_basis:
        raise ValueError(
            f"input space too large for exact enumeration "
            f"({len(states)} > {max_basis}); use reachable_overapprox"
        )

    for k in range(gate_index):
        inst = data[k]
        name = inst.operation.name.lower()
        if not _is_classical_name(name):
            raise ValueError(
                f"prefix gate #{k} ({name!r}) is not classical-reversible; "
                f"exact basis enumeration is unsound -- use reachable_overapprox"
            )
        qb = _qubit_indices(circuit, inst)
        states = {_apply_classical(name, qb, s) for s in states}
        # A classical-reversible map is a bijection, so |states| never grows; this
        # guard is belt-and-braces against a future non-bijective addition.
        if len(states) > max_basis:
            raise ValueError(
                f"reachable set exceeded {max_basis} states at prefix gate #{k}"
            )

    return frozenset(states)


def prefix_is_classical(circuit, gate_index: int) -> bool:
    """True iff every instruction before ``gate_index`` is classical-reversible."""
    data = circuit.data
    for k in range(min(gate_index, len(data))):
        if not _is_classical_name(data[k].operation.name.lower()):
            return False
    return True


def reachable_overapprox(
    circuit,
    gate_index: int,
    input_space: Union[str, Iterable[int]] = "all_basis",
    max_basis: int = _DEFAULT_MAX_BASIS,
) -> FrozenSet[int]:
    """A cheap SOUND OVER-APPROXIMATION of the reachable basis set at ``gate_index``.

    Returns a SUPERSET of the truly reachable computational-basis states. Useful when
    the exact enumeration is impossible (a superposition-creating prefix gate) or too
    large.

    SOUNDNESS OF OVER-APPROXIMATION.  Admissibility of a substitution is an OPERATOR
    identity on the span of the reachable basis states with ONE phase for the whole
    subspace (condition (R), ``subspace_check.check_on_subspace``), not a predicate
    checked basis state by basis state. If ``R`` is the true reachable set and
    ``S ⊇ R`` is our over-approximation, then span(R) ⊆ span(S) and

        U_d P_S = e^{i theta} U P_S   =>   U_d P_R = e^{i theta} U P_R

    (restrict both sides to the smaller subspace; the phase stays the same). So
    checking on the superset can only cause us to *reject* a valid substitution --
    never to *accept* an invalid one. Hence any superset is sound for admissibility. We return the SMALLEST cheap
    superset we can justify:

    * If the prefix up to ``gate_index`` is classical-reversible, delegate to the
      exact ``reachable_basis_states`` (the exact set is its own tightest superset).
    * Otherwise, identify the qubits that the prefix could have driven into a
      non-classical (superposition) state -- the ``tainted`` qubits: any qubit
      touched by a non-classical gate, plus any qubit later entangled with one via a
      classical gate (a CX from a tainted control taints its target; a tainted qubit
      participating in any multi-qubit gate taints all of that gate's qubits). On the
      classical ("clean") qubits we still know the exact bit value reachable from the
      input space; on the tainted qubits we conservatively allow BOTH 0 and 1. The
      returned superset is { clean-bits fixed } x { all 2^t assignments of tainted
      bits }. This is a sound superset because the true state's clean bits match
      (those gates were classical and deterministic) and its tainted bits are among
      all assignments by definition.

    Returns
    -------
    frozenset[int]
        A superset of the reachable basis states. Always succeeds (may be large; it
        is capped at ``max_basis`` by widening to the full 2^n space, which is the
        trivially-sound superset of last resort).
    """
    n = circuit.num_qubits
    data = circuit.data
    if not (0 <= gate_index <= len(data)):
        raise ValueError(f"gate_index {gate_index} out of range")

    if prefix_is_classical(circuit, gate_index):
        try:
            return reachable_basis_states(circuit, gate_index, input_space, max_basis)
        except ValueError:
            # too large: fall through to the trivially-sound full space
            return frozenset(range(1 << n))

    # Non-classical prefix: keep exact bits on clean qubits, taint the rest. A qubit
    # is tainted if EITHER it is touched by / entangled with a non-classical gate, OR
    # the input space does not pin its initial bit to a constant. Clean qubits are
    # then deterministic and propagated exactly by a single representative input.
    tainted = _compute_tainted_qubits(circuit, gate_index, input_space, n)
    if len(tainted) >= n:
        return frozenset(range(1 << n))

    clean = [q for q in range(n) if q not in tainted]

    # Determine the fixed bit value of each clean qubit. Clean qubits are, by
    # construction, pinned to a constant by the input space and only touched by
    # classical gates whose other operands are also clean+constant -- so a single
    # representative input propagates them exactly. Build that representative.
    rep = _representative_clean_input(circuit, gate_index, input_space, clean, n)

    # Enumerate all assignments of the tainted bits, with clean bits = rep's bits.
    clean_mask = 0
    for q in clean:
        if (rep >> q) & 1:
            clean_mask |= 1 << q
    tainted_list = sorted(tainted)
    t = len(tainted_list)
    # The over-approx has exactly 2^t elements (one per tainted-bit assignment). If
    # that exceeds the cap, widen to the trivially-sound full 2^n space.
    if (1 << t) > max_basis:
        return frozenset(range(1 << n))

    out: Set[int] = set()
    for assign in range(1 << t):
        s = clean_mask
        for bit, q in enumerate(tainted_list):
            if (assign >> bit) & 1:
                s |= 1 << q
        out.add(s)
    return frozenset(out)


def local_overapprox(circuit, gate_index: int, qubits, input_space="all_basis",
                     max_basis: int = _DEFAULT_MAX_BASIS) -> list:
    """Local support on ``qubits`` of ``reachable_overapprox`` without materialising the global set.

    For ``input_space == 'all_basis'`` the over-approximation is the whole space: a classical-reversible prefix is a bijection of
    the 2^n basis states (or the set is widened to the full space when 2^n exceeds ``max_basis``), and with a non-classical prefix
    every qubit is tainted. The local support is then every pattern on ``qubits``; the answer is the same as projecting
    ``reachable_overapprox`` but costs nothing instead of building sets of 2^n integers."""
    if input_space == "all_basis":
        if not (0 <= gate_index <= len(circuit.data)):
            raise ValueError(f"gate_index {gate_index} out of range")
        return list(range(1 << len(qubits)))
    from .subspace_check import project_support
    return project_support(reachable_overapprox(circuit, gate_index, input_space=input_space, max_basis=max_basis), qubits)


def _compute_tainted_qubits(circuit, gate_index, input_space, n) -> Set[int]:
    """Qubits whose bit value before ``gate_index`` is NOT a known classical constant.

    Tainted if: touched by a non-classical gate; entangled (via any multi-qubit gate)
    with a tainted qubit; or not pinned to a constant by ``input_space``.
    """
    data = circuit.data
    tainted: Set[int] = set()

    # Qubits not fixed to a constant by the input space are tainted from the start.
    tainted |= _non_constant_input_qubits(input_space, n)

    for k in range(min(gate_index, len(data))):
        inst = data[k]
        name = inst.operation.name.lower()
        qb = _qubit_indices(circuit, inst)
        if name in ("barrier", "id"):
            continue
        if not _is_classical_name(name):
            # A non-classical gate can create superposition on every qubit it touches.
            tainted.update(qb)
            continue
        # Classical gate: if any operand is tainted, the controlled/swap interaction
        # can spread non-classical correlation -- conservatively taint all operands.
        if any(q in tainted for q in qb):
            tainted.update(qb)
    return tainted


def _non_constant_input_qubits(input_space, n) -> Set[int]:
    """Qubits whose initial bit varies across ``input_space`` (hence not a constant)."""
    if input_space == "all_basis":
        return set(range(n))
    states = list(input_space)
    if not states:
        return set()
    nonconst: Set[int] = set()
    or_bits = 0
    and_bits = (1 << n) - 1
    for s in states:
        or_bits |= s
        and_bits &= s
    # bit q is constant iff or_bits and and_bits agree on it; else it varies.
    varying = or_bits ^ and_bits
    for q in range(n):
        if (varying >> q) & 1:
            nonconst.add(q)
    return nonconst


def _representative_clean_input(circuit, gate_index, input_space, clean, n) -> int:
    """Propagate one representative input through the prefix, reading only clean bits.

    Clean qubits are pinned by the input space and only ever interact (classically)
    with other clean, constant qubits, so any representative input yields the same
    clean-bit values at the gate. We pick the input space's first element (or 0).
    """
    data = circuit.data
    if input_space == "all_basis":
        rep = 0
    else:
        states = list(input_space)
        rep = states[0] if states else 0

    for k in range(min(gate_index, len(data))):
        inst = data[k]
        name = inst.operation.name.lower()
        if not _is_classical_name(name):
            # Non-classical gate: clean bits (by construction) are not among its
            # operands, so skip it -- it cannot change a clean qubit's value.
            continue
        qb = _qubit_indices(circuit, inst)
        rep = _apply_classical(name, qb, rep)
    # Mask to clean bits only; tainted bits in rep are meaningless and overwritten.
    clean_mask = 0
    for q in clean:
        clean_mask |= 1 << q
    return rep & clean_mask

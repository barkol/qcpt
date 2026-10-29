"""Exact local reachability for classical-reversible prefixes (vectorised).

For a Toffoli g on qubits (a, b, t), condition (R) only needs the set L of local
basis strings on (a, b, t) that some valid input can produce at g's input. When the
prefix before g consists of X/CX/CCX/MCX/SWAP gates, every basis input maps to one
basis state, so L is computed EXACTLY by propagating all valid inputs through the
prefix. We do this with numpy on an array of up to 2^22 integers, which covers the
12-24-qubit circuits of the paper when their ancillas are pinned to |0>.

If the prefix contains any other gate the function returns None and the caller
falls back to the sound over-approximation of ``reachable_subspace``.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

_CLASSICAL = {"x", "cx", "ccx", "mcx", "mcx_gray", "swap", "id", "barrier"}


def _inputs(n: int, pinned_zero: Iterable[int], max_free_bits: int):
    pinned = set(pinned_zero)
    free = [q for q in range(n) if q not in pinned]
    if len(free) > max_free_bits:
        return None
    k = len(free)
    idx = np.arange(1 << k, dtype=np.uint64)
    s = np.zeros(1 << k, dtype=np.uint64)
    for bit, q in enumerate(free):
        s |= ((idx >> np.uint64(bit)) & np.uint64(1)) << np.uint64(q)
    return s


def propagate_prefix(circuit, gate_index: int, pinned_zero=(), max_free_bits: int = 22):
    """Basis states (as uint64 array) at the input of ``gate_index``, or None."""
    n = circuit.num_qubits
    s = _inputs(n, pinned_zero, max_free_bits)
    if s is None:
        return None
    one = np.uint64(1)
    for inst in circuit.data[:gate_index]:
        name = inst.operation.name.lower()
        if name not in _CLASSICAL:
            return None
        qb = [circuit.find_bit(q).index for q in inst.qubits]
        if name in ("id", "barrier"):
            continue
        if name == "x":
            s ^= one << np.uint64(qb[0])
        elif name == "swap":
            a, b = qb
            ba = (s >> np.uint64(a)) & one
            bb = (s >> np.uint64(b)) & one
            diff = ba ^ bb
            s ^= (diff << np.uint64(a)) | (diff << np.uint64(b))
        else:  # cx / ccx / mcx: controls qb[:-1], target qb[-1]
            m = np.ones_like(s)
            for c in qb[:-1]:
                m &= (s >> np.uint64(c)) & one
            s ^= m << np.uint64(qb[-1])
    return s


def local_reachable(circuit, gate_index: int, qubits: Sequence[int], pinned_zero=(),
                    max_free_bits: int = 22) -> Optional[list]:
    """Exact set of local strings on ``qubits`` (bit j = qubits[j]) reaching
    ``gate_index``, or None if the prefix is not classical or too many free bits."""
    s = propagate_prefix(circuit, gate_index, pinned_zero, max_free_bits)
    if s is None:
        return None
    loc = np.zeros_like(s)
    for j, q in enumerate(qubits):
        loc |= ((s >> np.uint64(q)) & np.uint64(1)) << np.uint64(j)
    return sorted(int(v) for v in np.unique(loc))


class PrefixReach:
    """Incremental ``local_reachable``: the basis states at the input of gate k are advanced from the previous query, so a
    sweep over increasing gate indices costs one pass over the circuit instead of one pass per query. After the first
    non-classical gate every answer is None (the caller then uses the sound over-approximation)."""

    def __init__(self, circuit, pinned_zero=(), max_free_bits: int = 22):
        self.c = circuit
        self.pinned = tuple(pinned_zero)
        self.mfb = max_free_bits
        self.gates = None
        self.reset()

    def reset(self):
        self.s = _inputs(self.c.num_qubits, self.pinned, self.mfb)
        self.pos = 0
        self.dead = self.s is None

    def _gates(self):
        if self.gates is None:
            c = self.c
            self.gates = [(inst.operation.name.lower(), [c.find_bit(q).index for q in inst.qubits]) for inst in c.data]
        return self.gates

    def advance(self, gate_index: int):
        if gate_index < self.pos:
            self.reset()
        if self.dead:
            return None
        one = np.uint64(1)
        s = self.s
        gates = self._gates()
        for k in range(self.pos, gate_index):
            name, qb = gates[k]
            if name not in _CLASSICAL:
                self.dead = True
                return None
            if name in ("id", "barrier"):
                continue
            if name == "x":
                s ^= one << np.uint64(qb[0])
            elif name == "swap":
                a, b = qb
                diff = ((s >> np.uint64(a)) ^ (s >> np.uint64(b))) & one
                s ^= (diff << np.uint64(a)) | (diff << np.uint64(b))
            else:
                m = (s >> np.uint64(qb[0])) & one
                for c_ in qb[1:-1]:
                    m &= (s >> np.uint64(c_)) & one
                s ^= m << np.uint64(qb[-1])
        self.pos = gate_index
        return s

    def local(self, gate_index: int, qubits: Sequence[int]) -> Optional[list]:
        s = self.advance(gate_index)
        if s is None:
            return None
        loc = np.zeros_like(s)
        for j, q in enumerate(qubits):
            loc |= ((s >> np.uint64(q)) & np.uint64(1)) << np.uint64(j)
        if len(qubits) <= 16:
            present = np.zeros(1 << len(qubits), dtype=bool)
            present[loc.astype(np.int64)] = True
            return [int(v) for v in np.nonzero(present)[0]]
        return sorted(int(v) for v in np.unique(loc))

"""Window-certified relative-phase pairs (condition (W) of the paper).

The structural compute/uncompute detector (``context_analysis``) admits a pair
(g, g') of Toffolis on the same controls and target only if no gate in between
writes a control or the target. Then the diagonal phase D of the gadget commutes
with the segment between them and cancels (condition (C)).

Many reversible-arithmetic circuits violate that syntactic rule but still cancel
the phase: in a Cuccaro adder the MAJ and UMA blocks are mirror images, the CX gates
between a Toffoli and its mirror image are undone symmetrically, and the phases of
nested gadget pairs cancel jointly.

Condition (W): the pair is admissible if the SEGMENT from g to g' (inclusive), with
both Toffolis replaced by gadgets (and any already-admitted gadgets inside), is equal
to the exact segment up to one global phase, as an operator on the qubits the segment
touches. Equality of the segment operator implies equality of the whole circuit up to
that phase, i.e. subroutine equivalence; (C) is the special case in which D commutes
with the window.

The check is exact (dense unitary of the segment) and its cost depends only on the
number of qubits the segment touches, not on the width of the circuit. Candidates are
processed from the shortest window to the longest, so inner pairs are admitted
before the outer pairs whose cancellation relies on them.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Tuple

import numpy as np

_CCX = {"ccx", "mcx", "mcx_gray"}
_NONUNITARY = {"measure", "reset", "barrier"}


def _ccx_info(circuit, inst):
    name = inst.operation.name.lower()
    qb = [circuit.find_bit(q).index for q in inst.qubits]
    if name in _CCX and len(qb) == 3:
        return frozenset(qb[:2]), qb[2]
    return None


def mirror_pair_candidates(circuit, exclude=()) -> List[Tuple[int, int]]:
    """Pairs (i, j), i < j, of 3-qubit Toffolis with the same controls and target,
    j the nearest later such Toffoli. Sorted by window length (shortest first).
    Linear in the number of gates: the Toffolis of one (controls, target) key are kept in an ascending list and the nearest later
    one is found by bisection."""
    from bisect import bisect_right
    excl = set(exclude)
    info = {}
    by_key: dict = {}
    for i, inst in enumerate(circuit.data):
        ci = _ccx_info(circuit, inst)
        if ci is not None:
            info[i] = ci
            by_key.setdefault(ci, []).append(i)
    pairs, used = [], set()
    for i in sorted(info):
        if i in excl or i in used:
            continue
        lst = by_key[info[i]]
        k = bisect_right(lst, i)
        if k < len(lst):
            j = lst[k]                    # nearest later Toffoli with the same key
            if j not in excl and j not in used:
                pairs.append((i, j))
                used.update((i, j))   # every Toffoli belongs to at most one pair
    pairs.sort(key=lambda p: p[1] - p[0])
    return pairs


class _RangeTree:
    """Iterative segment tree over positions 0..n-1 with point 'chmax/chmin' updates and range queries (O(log n))."""

    def __init__(self, n: int, take_max: bool):
        self.size = 1
        while self.size < n + 2:
            self.size *= 2
        self.max = take_max
        self.neutral = -1 if take_max else 1 << 60
        self.t = [self.neutral] * (2 * self.size)

    def update(self, pos: int, val: int):
        i = pos + self.size
        f = max if self.max else min
        while i >= 1:
            self.t[i] = f(self.t[i], val)
            i >>= 1

    def query(self, lo: int, hi: int) -> int:
        """over positions lo <= p < hi"""
        res = self.neutral
        f = max if self.max else min
        lo += self.size
        hi += self.size
        while lo < hi:
            if lo & 1:
                res = f(res, self.t[lo]); lo += 1
            if hi & 1:
                hi -= 1; res = f(res, self.t[hi])
            lo >>= 1
            hi >>= 1
        return res


class _WindowFamily:
    """Laminar family of windows with an O(log n) test whether a new window crosses (or shares an endpoint with) one of them."""

    def __init__(self, n: int, windows=()):
        self.end = set()
        self.right_by_left = _RangeTree(n, True)      # at a left endpoint k: its right endpoint l (query: max over lefts inside)
        self.left_by_right = _RangeTree(n, False)     # at a right endpoint l: its left endpoint k (query: min over rights inside)
        for w in windows:
            self.add(w)

    def add(self, w):
        i, j = w
        self.end.update((i, j))
        self.right_by_left.update(i, j)
        self.left_by_right.update(j, i)

    def crosses(self, w) -> bool:
        i, j = w
        if i in self.end or j in self.end:
            return True
        if self.right_by_left.query(i + 1, j) > j:      # i < k < j < l
            return True
        return self.left_by_right.query(i + 1, j) < i   # k < i < l < j


# ---------------------------------------------------------------- fast exact check for monomial segments
_MONO_CACHE: dict = {}


def _mono_from_matrix(U):
    A = np.abs(U)
    if U.shape[0] > 8 or not (np.allclose(A.sum(axis=0), 1, atol=1e-9) and np.allclose(A.sum(axis=1), 1, atol=1e-9) and np.allclose(A, np.round(A), atol=1e-9)):
        return None
    perm = np.argmax(A, axis=0)
    return perm.astype(np.int64), np.array([U[perm[c], c] for c in range(U.shape[0])], dtype=complex)


def _mono_table(op):
    """(perm, phase) of a gate on <= 3 qubits if it is monomial (one nonzero of modulus 1 per column), else None. Local bit r = qubit r of the gate."""
    try:
        params = tuple(float(x) for x in op.params)
    except Exception:
        return None
    key = (op.name, tuple(np.round(params, 12)), op.num_qubits)
    if key in _MONO_CACHE:
        return _MONO_CACHE[key]
    from qiskit.quantum_info import Operator
    try:
        res = _mono_from_matrix(Operator(op).data)
    except Exception:
        res = None
    _MONO_CACHE[key] = res
    return res


def _gadget_table(append_gadget):
    key = ("gadget", id(append_gadget))
    if key not in _MONO_CACHE:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator
        qc = QuantumCircuit(3)
        append_gadget(qc, 0, 1, 2)
        _MONO_CACHE[key] = _mono_from_matrix(Operator(qc).data)
    return _MONO_CACHE[key]


def _mono_apply(table, qb, out, ph):
    perm, phase = table
    loc = np.zeros_like(out)
    for r, q in enumerate(qb):
        loc |= ((out >> q) & 1) << r
    new = perm[loc]
    for r, q in enumerate(qb):
        out = (out & ~(1 << q)) | (((new >> r) & 1) << q)
    return out, ph * phase[loc]


def _fast_monomial_equal(ops_ref, ops_cand, nq, atol):
    """Exact test 'cand == phase * ref' for monomial segments by acting on all 2^nq basis states."""
    N = 1 << nq
    o1 = np.arange(N, dtype=np.int64); p1 = np.ones(N, dtype=complex)
    o2 = o1.copy(); p2 = p1.copy()
    for t, qb in ops_ref:
        o1, p1 = _mono_apply(t, qb, o1, p1)
    for t, qb in ops_cand:
        o2, p2 = _mono_apply(t, qb, o2, p2)
    if not np.array_equal(o1, o2):
        return False, None
    r = p2 / p1
    ok = bool(np.allclose(r, r[0], atol=atol) and abs(abs(r[0]) - 1.0) < atol)      # one global phase of modulus 1 on every basis state
    return ok, (float(np.angle(r[0])) if ok else None)


def window_certified(circuit, i: int, j: int, gadget_idx, append_gadget: Callable,
                     max_window_qubits: int = 12, atol: float = 1e-9) -> Tuple[bool, dict]:
    """Condition (W) for the segment [i, j] with gadgets at every index in
    ``gadget_idx`` (which must contain i and j)."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator

    ops = circuit.data[i:j + 1]
    qubits = sorted({circuit.find_bit(q).index for inst in ops for q in inst.qubits})
    if len(qubits) > max_window_qubits:
        return False, {"reason": f"window touches {len(qubits)} qubits > {max_window_qubits}"}
    if any(inst.operation.name.lower() in _NONUNITARY for inst in ops):
        return False, {"reason": "non-unitary operation inside the window"}
    pos = {q: k for k, q in enumerate(qubits)}
    if os.environ.get("QCPT_FAST_WINDOW", "1") == "1":                  # exact O(2^k)-per-gate test when every gate is monomial; dense fallback otherwise
        gt = _gadget_table(append_gadget)
        if gt is not None:
            ops_ref, ops_cand = [], []
            for k, inst in enumerate(ops, start=i):
                t = _mono_table(inst.operation)
                if t is None:
                    ops_ref = None
                    break
                qb = [pos[circuit.find_bit(q).index] for q in inst.qubits]
                ops_ref.append((t, qb)); ops_cand.append((gt if k in gadget_idx else t, qb))
            if ops_ref is not None:
                ok, phi = _fast_monomial_equal(ops_ref, ops_cand, len(qubits), atol)
                return ok, {"window_qubits": len(qubits), "window_gates": len(ops), "global_phase": phi}
    ref = QuantumCircuit(len(qubits))
    cand = QuantumCircuit(len(qubits))
    for k, inst in enumerate(ops, start=i):
        qb = [pos[circuit.find_bit(q).index] for q in inst.qubits]
        ref.append(inst.operation, [ref.qubits[x] for x in qb])
        if k in gadget_idx:
            append_gadget(cand, *qb)
        else:
            cand.append(inst.operation, [cand.qubits[x] for x in qb])
    A = Operator(cand).data
    B = Operator(ref).data
    tr = np.trace(B.conj().T @ A)
    if abs(tr) < 1e-12:
        return False, {"reason": "segment not equal (zero overlap)", "window_qubits": len(qubits)}
    phase = tr / abs(tr)
    ok = bool(np.allclose(A, phase * B, atol=atol))
    return ok, {"window_qubits": len(qubits), "window_gates": len(ops), "global_phase": (float(np.angle(phase)) if ok else None)}


def _crosses(a, b):
    """Windows that cross or share an endpoint (not nested, not disjoint)."""
    (i, j), (k, l) = a, b
    if {i, j} & {k, l}:
        return True
    return (i < k < j < l) or (k < i < l < j)


def admit_window_pairs(circuit, current_gadgets, append_gadget: Callable,
                       max_window_qubits: int = 12, exclude=(), accepted_windows=()) -> Tuple[set, List[dict]]:
    """Greedily admit mirror pairs that pass condition (W), innermost first.

    Admitted windows (including ``accepted_windows`` from the structural pass) must
    form a laminar family: a new window may contain or be disjoint from an accepted
    one, never cross it. Then the certificate of every window stays valid when later,
    larger windows are added, and the whole circuit equals the exact one up to a
    global phase. Returns the enlarged gadget index set and a log of admitted pairs."""
    gadgets = set(current_gadgets)
    windows = list(accepted_windows)
    family = _WindowFamily(len(circuit.data), windows)
    log = []
    for i, j in mirror_pair_candidates(circuit, exclude=set(exclude) | gadgets):
        if family.crosses((i, j)):
            continue
        trial = gadgets | {i, j}
        ok, info = window_certified(circuit, i, j, trial, append_gadget, max_window_qubits)
        if ok:
            gadgets = trial
            windows.append((i, j))
            family.add((i, j))
            log.append({"pair": (i, j), "condition": "W", **info})
    return gadgets, log

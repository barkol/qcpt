"""Whole-circuit certificate by decomposition into blocks of difference (no dense propagation of the whole circuit).

``exact`` and ``selected`` are aligned gate by gate. Outside the blocks where they differ the two circuits contain the same
gates on the same qubits, so with A = X0 a1 X1 a2 X2 ... and B = X0 b1 X1 b2 X2 ... (X_k common stretches, a_k and b_k the
blocks of difference) it is enough that every block is equal up to a phase as an operator on the qubits it touches (identity
on the rest): then B = (prod_k e^{i th_k}) A, i.e. the whole circuits are equal up to ONE global phase on the FULL input space.
The statement is rigorous whatever the alignment is: a wrong alignment can only make a block fail, never pass.

A failing block is merged with the following blocks of difference (1, 2, 4, ... of them, so a block is re-checked O(log m)
times and the attempts grow geometrically). A common stretch that lies between two blocks is slid past the next block when
every gate of the stretch commutes with every gate of that block (two gates commute if each qubit they share is a diagonal one
for both gates; a1 S a2 = a1 a2 S in both circuits), so a compute ... uncompute pair is checked on the few qubits of its gadgets
and not on those of the arbitrarily long middle part. If a block touches more than ``max_qubits`` qubits, or the work (gates x
4^qubits over the blocks actually compared) exceeds ``max_work`` or a fraction budget_frac (0.1) of the cost of the dense certificate, the certificate
is undecided (None) and the caller uses the dense / QCEC certificate, so the worst case costs at most ~1.5x the dense one.

Numerical tolerance: the deviation of the product is at most the sum of the block deviations (unitary invariance), each block
is compared with atol / (number of blocks), so the reported total stays <= atol."""
from __future__ import annotations

from typing import Optional, Tuple

_R = 12            # number of consecutive equal gates that re-synchronise the two circuits
_W = 400           # look-ahead for the re-synchronisation (gates)

_DIAG_ALL = {"z", "s", "sdg", "t", "tdg", "rz", "p", "u1", "cz", "cp", "cu1", "crz", "ccz", "rzz", "id"}
_CTRL_LAST = {"cx", "cy", "ch", "crx", "cry", "csx", "cu", "cu3", "ccx", "mcx", "c3x", "c4x", "mcx_gray", "cnot"}


def _keys(circ):
    qidx = {q: i for i, q in enumerate(circ.qubits)}
    out = []
    for inst in circ.data:
        op = inst.operation
        if op.name == "barrier":
            continue
        try:
            par = tuple(round(float(p), 12) for p in op.params)
        except Exception:
            par = tuple(repr(p) for p in op.params)
        out.append((op.name, tuple(qidx[q] for q in inst.qubits), par, inst))
    return out


def _sig(k):
    return k[:3]


def _roles(g):
    """(mask of touched qubits, mask of qubits on which the gate is NOT diagonal in the computational basis). Unknown gates
    count as non-diagonal on every qubit they touch (conservative)."""
    name, qs = g[0], g[1]
    m = 0
    for q in qs:
        m |= 1 << q
    if name in _DIAG_ALL:
        return m, 0
    if name in _CTRL_LAST and len(qs) >= 2:
        return m, 1 << qs[-1]
    if name == "cswap" and len(qs) == 3:
        return m, (1 << qs[1]) | (1 << qs[2])
    return m, m


def _local_circuit(circ_cls, gates, touched):
    pos = {q: j for j, q in enumerate(touched)}
    sub = circ_cls(len(touched))
    for g in gates:
        sub.append(g[3].operation, [pos[q] for q in g[1]])
    return sub


def block_certificate(a, b, max_qubits: int = 12, atol: float = 1e-8, compare=None,
                      max_work: float = 5e10, budget_frac: float = 0.1, work_cost=None, cache=None, tol=None) -> Tuple[Optional[bool], dict]:
    """(True, info) if certified equal up to one global phase on the full input space, else (None, info) = undecided."""
    from qiskit import QuantumCircuit
    if compare is None:
        from .cppcert import compare_up_to_phase as compare
    if a.num_qubits != b.num_qubits:
        return None, {"reason": "qubit count differs"}
    ka, kb = _keys(a), _keys(b)
    sa, sb = [_sig(k) for k in ka], [_sig(k) for k in kb]
    na, nb = len(sa), len(sb)

    # ---- alignment: blocks of difference [(i0, i1, j0, j1)] ----
    blocks = []
    i = j = 0
    while i < na and j < nb:
        if sa[i] == sb[j]:
            i += 1
            j += 1
            continue
        i0, j0 = i, j
        resync = None
        if na - i0 > _R and nb - j0 > _R:
            posb = {}
            for dj in range(min(_W, nb - j0 - _R + 1)):
                posb.setdefault(tuple(sb[j0 + dj:j0 + dj + _R]), dj)
            best = None
            for di in range(min(_W, na - i0 - _R + 1)):
                dj = posb.get(tuple(sa[i0 + di:i0 + di + _R]))
                if dj is not None and (best is None or di + dj < best[0]):
                    best = (di + dj, di, dj)
                if best is not None and di > best[0]:
                    break
            if best is not None:
                resync = (best[1], best[2])
        if resync is None:
            blocks.append((i0, na, j0, nb))
            i, j = na, nb
            break
        blocks.append((i0, i0 + resync[0], j0, j0 + resync[1]))
        i, j = i0 + resync[0], j0 + resync[1]
    if i < na or j < nb:
        blocks.append((i, na, j, nb))
    if not blocks:
        return True, {"method": "block_diff", "blocks": 0, "certified_deviation": 0.0}

    # ---- verification with merge-on-failure and sliding of commuting common stretches ----
    cache = {} if cache is None else cache          # results by canonical block pattern (shareable between calls)
    rcache = {}

    def roles(g):
        key = (g[0], g[1])
        r = rcache.get(key)
        if r is None:
            r = rcache[key] = _roles(g)
        return r

    work_total = 0.0
    if work_cost is None:                                     # dense blocks: g 4^t, budget relative to the dense certificate of the whole circuit
        work_cost = lambda g, t: g * 4.0 ** t
        budget = min(max_work, budget_frac * (na + nb) * 4.0 ** min(a.num_qubits, 12)) if a.num_qubits <= 12 else max_work
    else:                                                     # a custom cost (e.g. Monte Carlo blocks: g K 2^t): only the absolute budget
        budget = max_work
    n_total = len(blocks)
    tol = atol / max(1, n_total) if tol is None else tol      # a fixed per-block tolerance makes the cache reusable; the total is n_blocks x tol
    total_dev = 0.0
    verified = 0
    slid = 0
    k = 0
    while k < len(blocks):
        i0, i1, j0, j1 = blocks[k]
        cur_a, cur_b = list(ka[i0:i1]), list(kb[j0:j1])
        pend = []
        step = 1
        while True:
            touched = sorted({q for g in cur_a for q in g[1]} | {q for g in cur_b for q in g[1]})
            if not touched:
                break
            if len(touched) > max_qubits:
                return None, {"reason": "block touches %d qubits" % len(touched), "blocks": n_total}
            pos = {q: t for t, q in enumerate(touched)}
            ck = (tol, len(touched), tuple((g[0], tuple(pos[q] for q in g[1]), g[2]) for g in cur_a),
                  tuple((g[0], tuple(pos[q] for q in g[1]), g[2]) for g in cur_b))
            hit = cache.get(ck)
            if hit is None:
                work_total += work_cost(len(cur_a) + len(cur_b) + 1, len(touched))
                if work_total > budget:
                    return None, {"reason": "work budget exceeded", "blocks": n_total, "work": work_total}
                la = _local_circuit(QuantumCircuit, cur_a, touched)
                lb = _local_circuit(QuantumCircuit, cur_b, touched)
                hit = cache[ck] = compare(la, lb, list(range(1 << len(touched))), tol)
            st, fro = hit
            if st == "equal":
                total_dev += fro or 0.0
                verified += 1
                break
            take = min(step, len(blocks) - k - 1)
            if take <= 0:
                return None, {"reason": "block not equal", "blocks": n_total}
            for _ in range(take):
                pi1 = i1
                ni0, ni1, nj0, nj1 = blocks.pop(k + 1)
                pend += ka[pi1:ni0]                                   # common stretch between the two blocks (same in B)
                new_a, new_b = ka[ni0:ni1], kb[nj0:nj1]
                tb = nd = 0
                for g in new_a + new_b:
                    t_, n_ = roles(g)
                    tb |= t_
                    nd |= n_
                if all((roles(g)[1] & tb) == 0 and (roles(g)[0] & nd) == 0 for g in pend):
                    slid += len(pend)                                  # the stretch commutes with the new block: leave it behind
                else:
                    cur_a += pend
                    cur_b += pend
                    pend = []
                cur_a += new_a
                cur_b += new_b
                i1, j1 = ni1, nj1
            step *= 2
        k += 1
    return True, {"method": "block_diff", "blocks": n_total, "checked": verified, "distinct": len(cache),
                  "slid_gates": slid, "certified_deviation": total_dev}

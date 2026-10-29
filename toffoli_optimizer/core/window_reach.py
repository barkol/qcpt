"""Windowed resynthesis with reachable-state context.

A window (a contiguous run of gates on at most k_max qubits) only has to act correctly on the local states that the valid inputs of the whole circuit can bring to its
input: the projection onto the window qubits of the states reached by the prefix (ancillas start in |0>, controls of live registers are classical). The replacement then
matches the window (permutation and Z8 phases, global phase free) on that set of local states, which is exact for the whole circuit up to a global phase, because the
qubits outside the window are not touched and the set is the union over all reachable full states. This is what lets a window lose a Toffoli that the context never fully
uses (relative-phase Toffolis, uncomputation pairs); the windows of resynth.window_resynthesize are exact on all 2^k local states and cannot do that.
The circuit has to be classical-reversible (permutation part defined by the library gates); the final circuit is certified by the caller against the exact circuit."""
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from . import resynth as RS
from . import cppsearch

_GP = {}


def _gate_perm(name, k_args):
    """Permutation of the 2^len(args) local states of one library gate (cached by name and arity)."""
    key = (name, len(k_args))
    if key not in _GP:
        perm, _ = RS._table_norm(RS.emit(len(k_args), [(name, tuple(range(len(k_args))))])); _GP[key] = perm
    return _GP[key]


def _apply(states, name, args):
    perm = _gate_perm(name, args); loc = np.zeros(len(states), dtype=np.int64)
    for x, q in enumerate(args): loc |= ((states >> q) & 1) << x
    new = perm[loc]; out = states.copy()
    for x, q in enumerate(args): out = (out & ~(1 << q)) | (((new >> x) & 1) << q)
    return out


def reach_sets(n, gates, valid_inputs):
    """List of arrays: the full-register states that can arrive at the input of gate i (i = 0..len(gates)), over all valid inputs."""
    s = np.unique(np.asarray(valid_inputs, dtype=np.int64)); out = [s]
    for g in gates: s = np.unique(_apply(s, g[0], tuple(g[1]))); out.append(s)
    return out


def _local(states, Q):
    loc = np.zeros(len(states), dtype=np.int64)
    for x, q in enumerate(Q): loc |= ((states >> q) & 1) << x
    return np.unique(loc)


def _task(a):
    k, inputs, target, phase, c, beam = a
    return cppsearch.beam_search(k, inputs, target, phase, c - 1, beam=beam)


def window_resynthesize_reach(n, gates, valid_inputs, k_max=5, min_cost=3, beam=3000, total_s=600.0, nproc=24, failed=None, verbose=False, max_len=None):
    """Repeated windowed resynthesis with reachable-state context; returns the new gate list [(name, args)...]. Each accepted window is exact on its reachable local states."""
    t0 = time.time(); gates = [(g[0], tuple(g[1])) for g in gates]; failed = set() if failed is None else failed; tabs = {}
    while time.time() - t0 < total_s:
        S = reach_sets(n, gates, valid_inputs); cands = {}
        for i in range(len(gates)):
            qs = set(); j = i
            while j < len(gates) and (max_len is None or j - i < max_len):
                nq = qs | set(gates[j][1])
                if len(nq) > k_max: break
                qs = nq; j += 1
                if len(qs) < 2: continue
                c = sum(RS.gate_cost(g[0]) for g in gates[i:j])
                if c < min_cost: continue
                Q = sorted(qs); loc = {q: x for x, q in enumerate(Q)}; sub = tuple((g[0], tuple(loc[q] for q in g[1])) for g in gates[i:j]); k = len(Q)
                if (k, sub) not in tabs: tabs[(k, sub)] = RS._table_norm(RS.emit(k, list(sub)))
                perm, ph = tabs[(k, sub)]; R = _local(S[i], Q); key = (k, perm[R].tobytes() + R.tobytes(), ph[R].tobytes(), c, beam)
                if key in failed: continue
                if key not in cands: cands[key] = [(k, [int(v) for v in R], [int(v) for v in perm[R]], [int(v) for v in ph[R]], c, beam), []]
                cands[key][1].append((i, j, Q))
        if not cands: break
        keys = list(cands)
        with ThreadPoolExecutor(nproc) as ex: res = list(ex.map(_task, [cands[kk][0] for kk in keys]))
        hits = []
        for kk, r in zip(keys, res):
            if r is None: failed.add(kk); continue
            for (i, j, Q) in cands[kk][1]: hits.append((cands[kk][0][4] - r[0], i, j, Q, r[1]))
        if not hits: break
        hits.sort(key=lambda h: (-h[0], h[1])); used = []; chosen = []
        for h in hits:
            if all(h[2] <= u0 or h[1] >= u1 for u0, u1 in used): used.append((h[1], h[2])); chosen.append(h)
        for sv, i, j, Q, new in sorted(chosen, key=lambda h: -h[1]):
            gates[i:j] = [(g[0], tuple(Q[x] for x in g[1])) for g in new]
            if verbose: print("window", Q, "saves", sv, "cost now", sum(RS.gate_cost(g[0]) for g in gates), flush=True)
    return gates

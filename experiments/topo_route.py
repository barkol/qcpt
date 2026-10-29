"""Gate-level router for the topology-aware variant of QCPT (extracted unchanged from topo/topo_resynth.py): SABRE-style routing of a library-gate circuit on the
subgraph of couplers of the chosen layout (SWAP = 3 CX), the check that the routed circuit equals the original up to a relabelling of qubits and a global phase,
and the windowed resynthesis restricted to the same couplers (toffoli_optimizer.core.resynth.window_resynthesize with edges=...)."""
import os, sys
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path[:0] = [R]
import numpy as np
from collections import deque
from qiskit.quantum_info import Operator
from toffoli_optimizer.core import resynth as RS

L = RS._local_gadgets(); P = RS._parity_gadgets()
def req_pairs(name, args):
    if name in ("cx", "cz"): return [tuple(args)]
    if name in L: return [(args[i], args[j]) for i, j in RS._local_pairs(name, L[name][1], 3)]
    if name in P: return [(args[i], args[j]) for i, j in RS._local_pairs(name, P[name][2], P[name][1])]
    return []
def _dist(n, E):
    adj = {i: set() for i in range(n)}
    for a, b in E: adj[a].add(b); adj[b].add(a)
    D = {}
    for s0 in range(n):
        d = {s0: 0}; dq = deque([s0])
        while dq:
            u = dq.popleft()
            for w in adj[u]:
                if w not in d: d[w] = d[u] + 1; dq.append(w)
        D[s0] = d
    return adj, D
def _exact_swaps(pos, prs, n, E):
    """Minimal swap sequence (BFS over placements) that makes all pairs adjacent; fallback of the heuristic router."""
    Es = {frozenset(e) for e in E}; edges = sorted(E); st0 = tuple(pos)
    sat = lambda st: all(frozenset((st[u], st[v])) in Es for u, v in prs)
    prev = {st0: None}; dq = deque([st0]); goal = st0 if sat(st0) else None
    while dq and goal is None:
        st = dq.popleft()
        for a, b in edges:
            nx = tuple(b if x == a else a if x == b else x for x in st)
            if nx in prev: continue
            prev[nx] = (st, (a, b)); dq.append(nx)
            if sat(nx): goal = nx; break
    seq = []; st = goal
    while prev[st] is not None: st, sw = prev[st]; seq.append(sw)
    return seq[::-1]
def sabre(gates, n, E, pos0, rng, W=0.5, EXT=20, DECAY=0.001):
    """SABRE-style router at the level of library gates: front layer of the dependency graph, SWAP chosen by distance of front-layer pairs
    plus a look-ahead set, decay against swap loops. Returns (gates in node language, final placement). SWAP = 3 CX."""
    adj, D = _dist(n, E); Es = {frozenset(e) for e in E}; m = len(gates); prs = [req_pairs(g[0], g[1]) for g in gates]
    last = {}; succ = [[] for _ in range(m)]; indeg = [0] * m
    for i, g in enumerate(gates):
        for q in set(g[1]):
            if q in last: succ[last[q]].append(i); indeg[i] += 1
            last[q] = i
    pos = list(pos0); out = []; front = [i for i in range(m) if indeg[i] == 0]; done = 0; decay = [1.0] * n; stall = 0
    ok = lambda i: all(frozenset((pos[u], pos[v])) in Es for u, v in prs[i])
    def sw(a, b):
        nonlocal out
        out += [("cx", (a, b)), ("cx", (b, a)), ("cx", (a, b))]
        for l in range(n):
            if pos[l] == a: pos[l] = b
            elif pos[l] == b: pos[l] = a
    while done < m:
        moved = True
        while moved:
            moved = False
            for i in list(front):
                if ok(i):
                    out.append((gates[i][0], tuple(pos[q] for q in gates[i][1]))); front.remove(i); done += 1; moved = True; stall = 0; decay = [1.0] * n
                    for k in succ[i]:
                        indeg[k] -= 1
                        if indeg[k] == 0: front.append(k)
        if done >= m: break
        if stall > 4 * n:                                         # release valve: exact placement for the first blocked gate
            for a, b in _exact_swaps(pos, prs[front[0]], n, E): sw(a, b)
            stall = 0; continue
        ext = []; seen = set(front); dq = deque(front)
        while dq and len(ext) < EXT:
            i = dq.popleft()
            for k in succ[i]:
                if k not in seen: seen.add(k); ext.append(k); dq.append(k)
        nodes = {pos[q] for i in front for u, v in prs[i] for q in (u, v)}
        cands = sorted({(min(a, b), max(a, b)) for a in nodes for b in adj[a]})
        def score(a, b):
            p2 = [b if x == a else a if x == b else x for x in pos]
            f = sum(sum(D[p2[u]][p2[v]] - 1 for u, v in prs[i]) for i in front) / len(front)
            e = (sum(sum(D[p2[u]][p2[v]] - 1 for u, v in prs[i]) for i in ext) / len(ext)) if ext else 0.0
            return max(decay[a], decay[b]) * (f + W * e)
        sc = [(score(a, b), rng.random(), (a, b)) for a, b in cands]; a, b = min(sc)[2]
        sw(a, b); decay[a] += DECAY; decay[b] += DECAY; stall += 1
    return out, tuple(pos)
def route(gates, n, E, trials=40, seed=0):
    import random; rng = random.Random(seed); best = None
    for t in range(trials):
        pos0 = list(range(n))
        if t: rng.shuffle(pos0)
        pos0 = tuple(pos0)
        for _ in range(2):                                        # forward-backward passes refine the initial placement (as in SABRE)
            _, f = sabre(gates, n, E, pos0, rng); _, pos0 = sabre(gates[::-1], n, E, f, rng)
        out, fin = sabre(gates, n, E, pos0, rng); c = sum(RS.gate_cost(g[0]) for g in out)
        if best is None or c < best[0]: best = (c, out, pos0, fin)
    return best[1], best[2], best[3]
def check_routed(gl, routed, pos0, fin, n):
    """U_routed[perm_out(y), perm_in(x)] = phase * U_orig[y, x]: routed circuit equals the original up to relabelling of qubits and a global phase."""
    U0 = Operator(RS.emit(n, gl)).data; U1 = Operator(RS.emit(n, routed)).data; N = 1 << n
    pin = lambda x, p: sum(((x >> l) & 1) << p[l] for l in range(n)); I = [pin(x, pos0) for x in range(N)]; O = [pin(y, fin) for y in range(N)]
    V = U1[np.ix_(O, I)]; a = np.vdot(U0.ravel(), V.ravel()); return float(np.abs(V - (a / abs(a)) * U0).max())

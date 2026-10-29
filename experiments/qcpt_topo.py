"""Topology-aware variant of QCPT for the hardware runs ("qct_topo"): the (frozen) QCPT circuit is routed by QCPT itself on the couplers of the chosen layout (gate-level SABRE-style router,
SWAP = 3 CX) and resynthesized in windows restricted to those couplers, then sent already routed; Qiskit does not move it. The routed circuit R takes logical qubit l from node
pos0[l] to node fin[l] (checked against the original up to a relabelling and a global phase). The mirror test needs no extra SWAPs for that: H layer and the reference circuit E are
placed in the output frame (logical l on node fin[l]), R^dagger carries the qubits back to pos0, and the last H layer acts there; the ideal return probability is 1 as before.
Node i of the circuit sits on physical qubit lay[i]. Routed circuits are cached per circuit and layout (resumable)."""
import json, os, time
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, Statevector
import topo_route as TR
from toffoli_optimizer.core import resynth as RS

_PATH_CACHE = {}

def trees(k):
    """All spanning trees of k labelled nodes (Pruefer decoding): 3 for k = 3, 16 for k = 4."""
    import itertools
    if k == 3: yield from ({(0, 1), (1, 2)}, {(0, 1), (0, 2)}, {(0, 2), (1, 2)}); return
    for seq in itertools.product(range(k), repeat=k - 2):
        deg = [1] * k
        for x in seq: deg[x] += 1
        E = set()
        for x in seq:
            leaf = min(i for i in range(k) if deg[i] == 1); E.add(tuple(sorted((leaf, x)))); deg[leaf] -= 1; deg[x] -= 1
        u, v = [i for i in range(k) if deg[i] == 1][:2]; E.add((u, v)); yield E

def has_cycle(pairs):
    """Do the required couplers contain a cycle (a triangle among three qubits)? Heavy-hex has none, so such a gate cannot be placed."""
    parent = {}
    def find(x):
        while parent.setdefault(x, x) != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in {tuple(sorted(p)) for p in pairs}:
        ra, rb = find(a), find(b)
        if ra == rb: return True
        parent[ra] = rb
    return False

def path_equivalent(name, k, maxdeg=3):
    """Cheapest exact equivalent (same permutation and Z8 phases up to a global phase) of a k-qubit library gate whose couplers contain a cycle, built from library gates restricted to the couplers of
    a spanning tree (searched over all trees of k nodes). At equal cost a path (max degree 2) is preferred to a star: it fits any connected set of qubits. Returns (cost, [(gate, local qubits)])."""
    if (name, k, maxdeg) in _PATH_CACHE: return _PATH_CACHE[(name, k, maxdeg)]
    os.environ['QCPT_GPU'] = '0'; perm, ph = RS._table_norm(RS.emit(k, [(name, tuple(range(k)))])); base = RS.gate_cost(name); best = None
    for T in trees(k):
        deg = max(sum(1 for e in T if v in e) for v in range(k))
        if deg > maxdeg: continue
        lib = [g for g in RS.library(k, coupling=frozenset(T)) if not g[0].startswith('mined')]
        for mc in range(base, base + 10):
            r = None
            for beam, ds in (((3000, 8), (30000, 10)) if k == 3 else ((3000, 8),)):
                r = RS.bidir_search(k, list(range(1 << k)), perm, mc, beam=beam, lam=0.6, depth_side=ds, lib=lib, target_phase=ph)
                if r: break
            if r:
                if best is None or (r[0], deg) < best[0]: best = ((r[0], deg), [(g[0], tuple(g[1])) for g in r[1]])
                break
    if best is None: raise RuntimeError(f"no tree-coupled equivalent of {name} found")
    _PATH_CACHE[(name, k, maxdeg)] = (best[0][0], best[1]); return _PATH_CACHE[(name, k, maxdeg)]

def eliminate_triangles(gl, maxdeg=3):
    """Replace every gate whose required couplers contain a cycle by its tree-coupled equivalent; returns the new gate list and the list of (gate, original cost, new cost)."""
    out, log = [], []
    for g in gl:
        name, args = g[0], tuple(g[1])
        if len(args) in (3, 4) and has_cycle(TR.req_pairs(name, list(args))):
            cost, rep = path_equivalent(name, len(args), maxdeg); log.append((name, RS.gate_cost(name), cost)); out += [(h[0], tuple(args[i] for i in h[1])) for h in rep]
        else: out.append(g)
    return out, log

def alive_edges(bk): return {(a, b) for (a, b), v in bk.target['cz'].items() if v is not None and v.error is not None and v.error < 0.5}
def n2q(t): return sum(1 for i in t.data if i.operation.num_qubits == 2 and i.operation.name != 'barrier')

def routed_circuit(nm, n, gl, lay, bk, cache_dir, win_s):
    os.makedirs(cache_dir, exist_ok=True); path = os.path.join(cache_dir, f"{nm}__{'-'.join(map(str, lay))}.json")
    if os.path.exists(path):
        d = json.load(open(path)); return [(g[0], tuple(g[1])) for g in d['gates']], d['pos0'], d['fin'], d
    gl0 = gl
    for maxdeg in (3, 2):                                        # first trees (a star needs a node of degree 3 in the layout), then paths only
        gl, tri_log = eliminate_triangles(gl0, maxdeg)
        edg = alive_edges(bk); E = {(i, j) for i in range(n) for j in range(i + 1, n) if (lay[i], lay[j]) in edg or (lay[j], lay[i]) in edg}; t0 = time.time()
        try: routed, pos0, fin = TR.route(gl, n, E, trials=40); break
        except (KeyError, RuntimeError):
            if maxdeg == 2: raise
    U0_ = Operator(RS.emit(n, gl0)).data; U1_ = Operator(RS.emit(n, gl)).data; a_ = np.vdot(U0_.ravel(), U1_.ravel()); dev0 = float(np.abs(U1_ - (a_ / abs(a_)) * U0_).max())
    if dev0 > 1e-7: raise RuntimeError(f"{nm}: triangle elimination changed the circuit (dev {dev0:.1e})")
    dev = TR.check_routed(gl, routed, pos0, fin, n)
    if dev > 1e-7: raise RuntimeError(f"{nm}: routed circuit differs from the original (dev {dev:.1e})")
    new = RS.window_resynthesize(n, routed, k_max=4, total_s=win_s, edges=E)
    U0 = Operator(RS.emit(n, routed)).data; U1 = Operator(RS.emit(n, new)).data; a = np.vdot(U0.ravel(), U1.ravel()); dev2 = float(np.abs(U1 - (a / abs(a)) * U0).max())
    if dev2 > 1e-7: raise RuntimeError(f"{nm}: windows changed the routed circuit (dev {dev2:.1e})")
    Es = {frozenset(e) for e in E}; left = sum(1 for g in new for p in TR.req_pairs(g[0], list(g[1])) if frozenset(p) not in Es)
    d = dict(triangles=[list(x) for x in tri_log], dev_triangles=dev0, gates=[[g[0], list(g[1])] for g in new], pos0=[int(x) for x in pos0], fin=[int(x) for x in fin], dev=dev, dev_windows=dev2, non_adjacent_left=left, seconds=round(time.time() - t0, 1),
             cost_routed=sum(RS.gate_cost(g[0]) for g in routed), cost_windows=sum(RS.gate_cost(g[0]) for g in new))
    json.dump(d, open(path, 'w')); return [(g[0], tuple(g[1])) for g in new], d['pos0'], d['fin'], d

def sabre_routed(nm, n, gl, lay, bk, seeds=200, level=3, L=None):
    """Second candidate route: the logical circuit routed by Qiskit (SABRE, optimization level ``level``) on the couplers of the layout, best of ``seeds`` seeds
    by the number of two-qubit gates. Same convention as the QCPT router: R takes logical qubit l from node pos0[l] to node fin[l]. Returns (Rc, pos0, fin, n2q)
    or None; the result is checked against the logical circuit (up to the permutation and a global phase)."""
    from qiskit.transpiler import CouplingMap
    edg = alive_edges(bk); E = [(i, j) for i in range(n) for j in range(i + 1, n) if (lay[i], lay[j]) in edg or (lay[j], lay[i]) in edg]
    cm = CouplingMap(E + [(b, a) for a, b in E]); L = RS.emit(n, gl) if L is None else L; best = None
    for sd in range(seeds):
        t = transpile(L, coupling_map=cm, basis_gates=["cz", "rz", "sx", "x"], optimization_level=level, seed_transpiler=sd)
        key = (n2q(t), len(t.data))
        if best is None or key < best[0]: best = (key, t)
    if best is None: return None
    t = best[1]; lo = t.layout
    pos0 = [int(lo.initial_layout[q]) for q in L.qubits] if False else [int(x) for x in lo.initial_index_layout(filter_ancillas=True)]
    fin = [int(x) for x in lo.final_index_layout(filter_ancillas=True)]
    Rc = QuantumCircuit(n)
    for i in t.data: Rc.append(i.operation, [Rc.qubits[t.find_bit(q).index] for q in i.qubits])
    U = Operator.from_circuit(t).data if n <= 12 else None
    if U is not None:
        a = np.vdot(Operator(L).data.ravel(), U.ravel()); dev = float(np.abs(U - (a / abs(a)) * Operator(L).data).max())
        if dev > 1e-7: return None
    return Rc, pos0, fin, best[0][0]

def route_best(nm, n, gl, lay, bk, cache_dir, win_s, seeds=200, extra=()):
    """Portfolio of the QCPT router (with windows) and the SABRE route: the candidate with fewer two-qubit gates after translation to CZ."""
    gates, pos0, fin, info = routed_circuit(nm, n, gl, lay, bk, cache_dir, win_s); Rq = RS.emit(n, gates)
    edg = alive_edges(bk)
    from qiskit.transpiler import CouplingMap
    E = [(i, j) for i in range(n) for j in range(i + 1, n) if (lay[i], lay[j]) in edg or (lay[j], lay[i]) in edg]
    c_q = n2q(transpile(Rq, coupling_map=CouplingMap(E + [(b, a) for a, b in E]), basis_gates=["cz", "rz", "sx", "x"], optimization_level=0, seed_transpiler=11))
    sab = sabre_routed(nm, n, gl, lay, bk, seeds) if os.environ.get("QCPT_TOPO_SABRE", "1") == "1" else None
    info = dict(info, n2q_qcpt_router=c_q, n2q_sabre=(sab[3] if sab else None))
    cands = [("qcpt", c_q, (Rq, pos0, fin))]
    if sab is not None: cands.append(("sabre", sab[3], sab[:3]))
    if os.environ.get("QCPT_TOPO_SABRE", "1") == "1":
        for label, circ in extra:                                  # further logical circuits that are equivalent on the input subspace (e.g. the output of stage 1)
            r = sabre_routed(nm, n, gl, lay, bk, seeds, L=circ)
            info["n2q_sabre_" + label] = (r[3] if r else None)
            if r is not None: cands.append(("sabre_" + label, r[3], r[:3]))
    label, c, (Rc, p0, fi) = min(cands, key=lambda x: x[1])
    info["chosen"] = label; info["n2q_chosen"] = c
    return Rc, p0, fi, info

def topo_variant(nm, n, gl, lay, bk, E_circ, data, cache_dir, win_s=20.0, extra=()):
    """Returns the mirror circuit on nodes (to be transposed with initial_layout=lay, no routing needed for the V part), its ideal return probability and counts of two-qubit gates of V alone."""
    Rc, pos0, fin, info = route_best(nm, n, gl, lay, bk, cache_dir, win_s, extra=extra)
    Em = QuantumCircuit(n); Em.compose(E_circ, qubits=[fin[l] for l in range(n)], inplace=True)
    M = QuantumCircuit(n)
    for q in data: M.h(fin[q])
    M.compose(Em, inplace=True); M.barrier(); M.compose(Rc.inverse(), inplace=True)
    for q in data: M.h(pos0[q])
    p0i = float(abs(Statevector(M).data[0]) ** 2); M.measure_all()
    own = n2q(transpile(Rc, bk, initial_layout=lay, optimization_level=0, seed_transpiler=11))              # V alone as routed by QCPT
    qis = n2q(transpile(RS.emit(n, gl), bk, initial_layout=lay, optimization_level=2, seed_transpiler=11))   # V alone as routed by Qiskit on the same layout
    return M, p0i, own, qis, info

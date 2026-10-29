"""QCP, etap 2: certyfikowana resynteza obwodu klasycznego (NCT) jako funkcji netto na podprzestrzeni wejsc.

Obwod wejsciowy (ccx/cx/x) jest permutacja bazy obliczeniowej. Szukamy tanszego obwodu z bramek MONOMIALNYCH z fazami z grupy Z8
(potegi omega = exp(i pi/4)): X, CX, CZ, CCX, CCZ, diagonalne T/Tdg/S/Sdg/Z (koszt 0 CX), gadzety Margolusa (3 CX, 4 warianty),
RCCX (Maslov, 3 CX) i jego odwrotnosc. Stan przeszukiwania to odwzorowanie wejscie -> (wyjscie, faza); cel to ta sama permutacja
z dokladnoscia do jednej globalnej fazy. Przeszukiwanie dwukierunkowe (wiazki od identycznosci i od celu, bramki odwrotne po stronie celu,
spotkanie przez haszowanie). Wynik jest ZAWSZE certyfikowany osobno (certify_on_input_subspace), wiec poprawnosc nie zalezy od heurystyki.
Koszt = liczba CX (Margolus/RCCX = 3, CCX/CCZ = 6, CZ = 1)."""
from __future__ import annotations
import itertools
import os
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator


def classical_target(circuit: QuantumCircuit, inputs):
    """Wyjscia (int, bit q = kubit q) obwodu z bramek ccx/cx/x dla zadanych wejsc; ValueError dla innych bramek."""
    ops = []
    for inst in circuit.data:
        nm = inst.operation.name
        if nm == "barrier": continue
        if nm not in ("ccx", "cx", "x"): raise ValueError(f"bramka {nm} nie jest klasyczna")
        ops.append((nm, [circuit.find_bit(q).index for q in inst.qubits]))
    out = []
    for x in inputs:
        for nm, q in ops:
            if nm == "x": x ^= 1 << q[0]
            elif nm == "cx": x ^= ((x >> q[0]) & 1) << q[1]
            else: x ^= (((x >> q[0]) & (x >> q[1])) & 1) << q[2]
        out.append(x)
    return np.array(out)


def _rccx_dg():
    c = QuantumCircuit(3); c.rccx(0, 1, 2); return c.inverse()


def _local_gadgets():
    """Lokalne (3-kubitowe) bramki monomialne: nazwa -> (koszt CX, funkcja emisji(qc, a, b, t))."""
    from .decomposition_selector import append_relative_phase_ccx as marg, append_relative_phase_ccx_mirror as mirr, append_exact_ccx
    G = {}
    G["mg1"] = (3, lambda qc, a, b, t: marg(qc, a, b, t)); G["mg2"] = (3, lambda qc, a, b, t: marg(qc, b, a, t))
    G["mg3"] = (3, lambda qc, a, b, t: mirr(qc, a, b, t)); G["mg4"] = (3, lambda qc, a, b, t: mirr(qc, b, a, t))
    G["rccx1"] = (3, lambda qc, a, b, t: qc.rccx(a, b, t)); G["rccx2"] = (3, lambda qc, a, b, t: qc.rccx(b, a, t))
    G["rccx1d"] = (3, lambda qc, a, b, t: qc.compose(_rccx_dg(), [a, b, t], inplace=True)); G["rccx2d"] = (3, lambda qc, a, b, t: qc.compose(_rccx_dg(), [b, a, t], inplace=True))
    G["ccx"] = (6, lambda qc, a, b, t: append_exact_ccx(qc, a, b, t)); G["ccz"] = (6, lambda qc, a, b, t: qc.ccz(a, b, t))
    return G


def _tables3(fn):
    """Tablice (perm, faza Z8) 3-kubitowej bramki monomialnej z jej macierzy: indeks lokalny l = bit0 + 2 bit1 + 4 bit2 (kubity 0,1,2)."""
    c = QuantumCircuit(3); fn(c, 0, 1, 2); U = Operator(c).data; perm = np.zeros(8, int); ph = np.zeros(8, int)
    for l in range(8):
        col = U[:, l]; o = int(np.argmax(np.abs(col))); assert abs(abs(col[o]) - 1) < 1e-9 and np.sum(np.abs(col) > 1e-9) == 1, "bramka nie jest monomialna"
        k = np.angle(col[o]) / (np.pi / 4); assert abs(k - round(k)) < 1e-6, f"faza spoza Z8: {k}"
        perm[l] = o; ph[l] = int(round(k)) % 8
    return perm, ph


def _parity_gadgets(extra=True):
    """Margolus gadgets whose middle control is the parity of A (|A|=2) or whose outer controls are parities of B (|B|=2):
    Ry(-q) CX(B->t) Ry(-q) CX(A->t) Ry(q) CX(B->t) Ry(q), each parity realised by one CX per member (cost |A|+2|B|).
    Local qubit order: A..., B..., t. 'mgq' is the mirror image (angle signs flipped)."""
    import math
    q = math.pi / 4; G = {}
    def make(na, nb, sg):
        def emit_(qc, *qs):
            A, B, t = qs[:na], qs[na:na + nb], qs[-1]
            qc.ry(-sg * q, t)
            for b in B: qc.cx(b, t)
            qc.ry(-sg * q, t)
            for a in A: qc.cx(a, t)
            qc.ry(sg * q, t)
            for b in B: qc.cx(b, t)
            qc.ry(sg * q, t)
        return emit_
    for na, nb in ((2, 1), (1, 2)):
        for nm, sg in (("mgp", 1), ("mgq", -1)): G[f"{nm}{na}{nb}"] = (na + 2 * nb, na + nb + 1, make(na, nb, sg))
    def make_signs(na, nb, sgs):
        def emit_(qc, *qs):
            A, B, t = qs[:na], qs[na:na + nb], qs[-1]
            qc.ry(sgs[0] * q, t)
            for b in B: qc.cx(b, t)
            qc.ry(sgs[1] * q, t)
            for a in A: qc.cx(a, t)
            qc.ry(sgs[2] * q, t)
            for b in B: qc.cx(b, t)
            qc.ry(sgs[3] * q, t)
        return emit_
    G["mgr21"] = (4, 4, make_signs(2, 1, (-1, 1, 1, -1))); G["mgs21"] = (4, 4, make_signs(2, 1, (1, -1, -1, 1)))   # sign patterns not reachable from the other gadgets at this cost
    if extra:                                               # remaining monomial sign patterns (equal to the others up to free gates; used as single moves by the template search)
        for nm, sg in (("mgt21", (-1, -1, -1, -1)), ("mgu21", (1, 1, 1, 1)), ("mgv21", (-1, 1, -1, 1)), ("mgw21", (1, -1, 1, -1))): G[nm] = (4, 4, make_signs(2, 1, sg))
    return G


def _tablesk(fn, k):
    """(perm, Z8 phase) tables of a k-qubit monomial gate from its matrix; local index bit i = local qubit i."""
    c = QuantumCircuit(k); fn(c, *range(k)); U = Operator(c).data; N = 1 << k; perm = np.zeros(N, int); ph = np.zeros(N, int)
    for l in range(N):
        col = U[:, l]; o = int(np.argmax(np.abs(col))); assert abs(abs(col[o]) - 1) < 1e-9 and np.sum(np.abs(col) > 1e-9) == 1, "bramka nie jest monomialna"
        k8 = np.angle(col[o]) / (np.pi / 4); assert abs(k8 - round(k8)) < 1e-6, f"faza spoza Z8: {k8}"
        perm[l] = o; ph[l] = int(round(k8)) % 8
    return perm, ph


MINED = {}          # name -> (nlocal, cost, subcircuit); gadgets learned from accepted circuits of the language-model search


def register_mined(items):
    """items: dicts with keys k, cost, qasm (a monomial gadget on k local qubits). Names are mined<i>."""
    from qiskit import qasm2
    MINED.clear()
    for i, it in enumerate(items):
        MINED[f"mined{i}"] = (it["k"], it["cost"], qasm2.loads(it["qasm"], custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS))


def _table_norm(sub):
    """(perm, Z8 phase) of a monomial k-qubit circuit with the global phase removed (phase of column 0 set to 0). Only the phases RELATIVE to column 0 must be multiples of pi/4;
    the global phase of the circuit is arbitrary (a Qiskit transpilation leaves it at odd multiples of pi/8)."""
    U = Operator(sub).data; N = U.shape[0]; perm = np.zeros(N, int); ang = np.zeros(N)
    for l in range(N):
        col = U[:, l]; o = int(np.argmax(np.abs(col))); assert abs(abs(col[o]) - 1) < 1e-7 and np.sum(np.abs(col) > 1e-7) == 1
        perm[l] = o; ang[l] = np.angle(col[o])
    k8 = (ang - ang[0]) / (np.pi / 4); assert np.all(np.abs(k8 - np.round(k8)) < 1e-6)
    return perm, np.round(k8).astype(int) % 8


_PAIRS = {}


def _local_pairs(name, fn, k):
    """Local qubit pairs that carry a two-qubit gate in the emission of a k-qubit gadget (undirected, cached per gadget name)."""
    if name not in _PAIRS:
        from qiskit import transpile
        c = QuantumCircuit(k); fn(c, *range(k)); t = transpile(c, basis_gates=["cx", "u"], optimization_level=0)
        _PAIRS[name] = {tuple(sorted(t.find_bit(q).index for q in i.qubits)) for i in t.data if i.operation.num_qubits == 2}
    return _PAIRS[name]


def _topo_filter(G, coupling):
    """Keep gates whose every two-qubit interaction lies on an edge of `coupling` (set of frozenset logical pairs)."""
    E = {frozenset(e) for e in coupling}; L = _local_gadgets(); P = _parity_gadgets(); out = []
    for g in G:
        nm, qs = g[0], g[1]
        if nm in ("cx", "cz"): ok = frozenset(qs) in E
        elif nm in L: ok = all(frozenset((qs[i], qs[j])) in E for i, j in _local_pairs(nm, L[nm][1], 3))
        elif nm in P: ok = all(frozenset((qs[i], qs[j])) in E for i, j in _local_pairs(nm, P[nm][2], P[nm][1]))
        elif nm in MINED: ok = all(frozenset((qs[i], qs[j])) in E for i, j in _local_pairs(nm, lambda c, *a, _s=MINED[nm][2]: c.compose(_s, list(a), inplace=True), MINED[nm][0]))
        else: ok = True                                   # one-qubit gates
        if ok: out.append(g)
    return out


def library(n: int, parity: bool = True, extra_signs: bool = False, coupling=None):
    """Lista bramek (nazwa, argumenty, koszt, perm, faza, perm_odwr, faza_odwr); tablice indeksowane stanem PRZED bramka."""
    idx = np.arange(1 << n); bit = lambda q: (idx >> q) & 1; G = []
    def add(name, args, cost, perm, ph):
        inv = np.zeros_like(perm); inv[perm] = idx; phi = np.zeros_like(ph); phi[perm] = (-ph) % 8      # odwrotnosc: wejscie z = perm[y], faza -ph[y]
        G.append((name, args, cost, perm.astype(np.int64), (ph % 8).astype(np.int8), inv.astype(np.int64), phi.astype(np.int8)))
    for t in range(n): add("x", (t,), 0, idx ^ (1 << t), np.zeros(1 << n, np.int8))
    for q in range(n):
        for nm, e in (("t", 1), ("tdg", 7), ("s", 2), ("sdg", 6), ("z", 4)): add(nm, (q,), 0, idx.copy(), (e * bit(q)).astype(np.int8))
    for c in range(n):
        for t in range(n):
            if c != t: add("cx", (c, t), 1, idx ^ (bit(c) << t), np.zeros(1 << n, np.int8))
    for a, b in itertools.combinations(range(n), 2): add("cz", (a, b), 1, idx.copy(), (4 * bit(a) * bit(b)).astype(np.int8))
    L = {nm: (cost, fn, *_tables3(fn)) for nm, (cost, fn) in _local_gadgets().items()}
    for t in range(n):
        for a, b in itertools.combinations([q for q in range(n) if q != t], 2):
            loc = bit(a) + 2 * bit(b) + 4 * bit(t)
            for nm, (cost, fn, p3, f3) in L.items():
                o = p3[loc]; perm = (idx & ~((1 << a) | (1 << b) | (1 << t))) | ((o & 1) << a) | (((o >> 1) & 1) << b) | (((o >> 2) & 1) << t)
                add(nm, (a, b, t), cost, perm, f3[loc].astype(np.int8))
    if parity:
        for nm, (cost, k, fn) in _parity_gadgets().items():
            if not extra_signs and nm in ("mgt21", "mgu21", "mgv21", "mgw21"): continue
            p_k, f_k = _tablesk(fn, k); na = int(nm[-2]); nb = int(nm[-1])
            for t in range(n):
                rest = [q for q in range(n) if q != t]
                for A in itertools.combinations(rest, na):
                    for B in itertools.combinations([q for q in rest if q not in A], nb):
                        qs = list(A) + list(B) + [t]; loc = np.zeros(1 << n, np.int64)
                        for i, q in enumerate(qs): loc |= bit(q) << i
                        o = p_k[loc]; perm = idx & ~sum(1 << q for q in qs)
                        for i, q in enumerate(qs): perm = perm | (((o >> i) & 1) << q)
                        add(nm, tuple(qs), cost, perm, f_k[loc].astype(np.int8))
    if MINED:
        seen = {}
        for g in G: seen[(bytes(g[3].tobytes()), bytes(((g[4].astype(np.int16) - g[4][0]) % 8).astype(np.int8).tobytes()))] = g[2]
        for nm, (k, cost, sub) in MINED.items():
            p_k, f_k = _table_norm(sub)
            for qs in itertools.permutations(range(n), k):
                loc = np.zeros(1 << n, np.int64)
                for i, q in enumerate(qs): loc |= bit(q) << i
                o = p_k[loc]; perm = idx & ~sum(1 << q for q in qs)
                for i, q in enumerate(qs): perm = perm | (((o >> i) & 1) << q)
                ph = f_k[loc].astype(np.int8); key = (bytes(perm.astype(np.int64).tobytes()), bytes(((ph.astype(np.int16) - ph[0]) % 8).astype(np.int8).tobytes()))
                if key in seen and seen[key] <= cost: continue
                seen[key] = cost; add(nm, tuple(qs), cost, perm, ph)
    if coupling is not None: G = _topo_filter(G, coupling)
    return G


def emit(n: int, gates, circuit_name="resynth") -> QuantumCircuit:
    L = _local_gadgets(); L.update({k: (v[0], v[2]) for k, v in _parity_gadgets().items()}); L.update({k: (v[1], (lambda sub: (lambda qc_, *qs_: qc_.compose(sub, list(qs_), inplace=True)))(v[2])) for k, v in MINED.items()}); qc = QuantumCircuit(n, name=circuit_name)
    simple = {"x": lambda a: qc.x(*a), "cx": lambda a: qc.cx(*a), "cz": lambda a: qc.cz(*a), "t": lambda a: qc.t(*a), "tdg": lambda a: qc.tdg(*a),
              "s": lambda a: qc.s(*a), "sdg": lambda a: qc.sdg(*a), "z": lambda a: qc.z(*a)}
    for g in gates:
        name, args = g[0], g[1]
        if name in simple: simple[name](args)
        else: L[name][1](qc, *args)
    return qc


import multiprocessing as _mp
_CTX = {}


def _nproc():
    import os
    return int(os.environ.get("QCPT_NPROC", min(12, os.cpu_count() or 1)))


def _chunk_work(arg):
    """Candidates of one chunk of library gates: top 2*beam children by f = g + lam*h (module level so a forked pool can run it)."""
    c0, chunk = arg; C = _CTX; Y, S, g, P, F, goal, cost, pc, cd = (C[k] for k in ("Y", "S", "g", "P", "F", "goal", "cost", "pc", "cd")); Bn, lim, beam, lam = C["Bn"], C["lim"], C["beam"], C["lam"]
    Yn = P[c0:c0 + chunk][:, Y]; Sn = (S[None] + F[c0:c0 + chunk][:, Y]) & 7
    rel = (Sn - Sn[..., :1]) & 7; hh = pc[Yn ^ goal].sum(-1) + cd[rel].sum(-1); gc = g[None, :] + cost[c0:c0 + chunk, None]
    f = np.where(gc > lim, np.inf, gc + lam * hh).ravel()
    if C.get('noise'): f = f + C['noise'] * np.random.default_rng((C['seed'] + c0) & 0x7fffffff).standard_normal(f.size)
    k = min(beam * 2, f.size - 1)
    sel = np.argpartition(f, k)[:k]; sel = sel[np.isfinite(f[sel])]; gi, pi = np.divmod(sel, Bn)
    return f[sel], gc[gi, pi], gi + c0, pi


_GPUC = {}


def _gpu_on(Bn, Gn, m):
    """GPU path for the beam expansion: opt out with QCPT_GPU=0; small expansions stay on the CPU (launch overhead)."""
    if os.environ.get("QCPT_GPU", "1") == "0" or Bn * Gn * m < int(os.environ.get("QCPT_GPU_MIN", 3_000_000)): return False
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _gpu_tab(P, F):
    import torch
    key = (id(P), P.ctypes.data, P.shape)
    if key not in _GPUC:
        if len(_GPUC) > 8: _GPUC.clear()
        _GPUC[key] = (P, torch.as_tensor(P.astype(np.int32)).cuda(), torch.as_tensor(F.astype(np.int32)).cuda())
    return _GPUC[key][1:]


def _gpu_cands(C):
    """Same candidates as the CPU chunks (top 2*beam children by f = g + lam*h), computed on the GPU in integer arithmetic (f scaled by 1000)."""
    import torch
    dev = torch.device("cuda"); Y, S, g, P, F, goal, cost, pc, cd = (C[k] for k in ("Y", "S", "g", "P", "F", "goal", "cost", "pc", "cd")); Bn, lim, beam, lam = C["Bn"], C["lim"], C["beam"], C["lam"]
    Pt, Ft = _gpu_tab(P, F); Gn, N = Pt.shape; m = Y.shape[1]
    Yf = torch.as_tensor(Y.astype(np.int64)).to(dev).reshape(-1); St = torch.as_tensor(S.astype(np.int32)).to(dev); gt = torch.as_tensor(g.astype(np.int32)).to(dev)
    goalt = torch.as_tensor(np.asarray(goal).astype(np.int32)).to(dev); pct = torch.as_tensor(pc.astype(np.int32)).to(dev); cdt = torch.as_tensor(cd.astype(np.int32)).to(dev)
    costt = torch.as_tensor(cost.astype(np.int32)).to(dev); L = int(round(lam * 1000)); BIG = 2 ** 30
    grp = int(C.get("chunk", 32)); ch = max(grp, int(float(os.environ.get("QCPT_GPU_ELEMS", 6e7)) // (Bn * m)) // grp * grp); k = min(beam * 2, grp * Bn - 1); fs, gs, gis, pis = [], [], [], []
    for c0 in range(0, Gn, ch):                    # top-2*beam per group of `grp` library gates, exactly the candidate pool of the CPU chunks, but several groups per GPU batch
        c1 = min(Gn, c0 + ch); n_ = c1 - c0
        Yn = Pt[c0:c1].index_select(1, Yf).view(n_, Bn, m); Sn = (St[None] + Ft[c0:c1].index_select(1, Yf).view(n_, Bn, m)) & 7
        rel = (Sn - Sn[..., :1]) & 7; hh = pct[Yn ^ goalt].sum(-1) + cdt[rel].sum(-1); del Yn, Sn, rel
        gc = gt[None, :] + costt[c0:c1, None]; f = torch.where(gc > lim, torch.full_like(gc, BIG), gc * 1000 + L * hh); npad = (-n_) % grp
        if npad: f = torch.cat([f, torch.full((npad, Bn), BIG, dtype=f.dtype, device=dev)]); gc = torch.cat([gc, torch.zeros((npad, Bn), dtype=gc.dtype, device=dev)])
        ng = f.shape[0] // grp; v, ix = torch.topk(f.view(ng, grp * Bn), min(k, grp * Bn), dim=1, largest=False, sorted=False)
        grow = torch.arange(ng, device=dev)[:, None] * grp + torch.div(ix, Bn, rounding_mode="floor"); pi = ix - torch.div(ix, Bn, rounding_mode="floor") * Bn
        keep = v < BIG; v, grow, pi = v[keep], grow[keep], pi[keep]
        fs.append(v); gs.append(gc[grow, pi]); gis.append(grow + c0); pis.append(pi)
    f = torch.cat(fs); o = torch.sort(f, stable=True).indices
    cat = lambda xs: torch.cat(xs)[o].cpu().numpy()
    return f[o].cpu().numpy().astype(np.float64) / 1000.0, cat(gs), cat(gis), cat(pis)


def _gpu_cands_safe(C):
    """_gpu_cands with a fallback for a full GPU: retry with batches of half the size (QCPT_GPU_ELEMS), give up after five halvings."""
    import torch
    el0 = os.environ.get("QCPT_GPU_ELEMS"); el = float(el0 or 6e7)
    try:
        for _ in range(6):
            try:
                os.environ["QCPT_GPU_ELEMS"] = str(el); return _gpu_cands(C)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache(); el /= 2
        return None
    finally:
        if el0 is None: os.environ.pop("QCPT_GPU_ELEMS", None)
        else: os.environ["QCPT_GPU_ELEMS"] = el0


def bidir_search(n: int, inputs, target, max_cost: int, beam: int = 6000, lam: float = 0.6, depth_side: int = 8, seed: int = 0, chunk: int = 32, lib=None, verbose=False, target_phase=None, diversity=None, noise=None):
    """Przeszukiwanie dwukierunkowe w Z8-monomialnej przestrzeni stanow. Zwraca (koszt, lista bramek) lub None."""
    import os as _os
    diversity = float(_os.environ.get('QCPT_DIV', 0.0)) if diversity is None else diversity; noise = float(_os.environ.get('QCPT_NOISE', 0.0)) if noise is None else noise
    lib = lib or library(n); Gn = len(lib); dt = np.uint8 if n <= 8 else np.uint16
    perms = np.stack([g[3] for g in lib]).astype(dt); phs = np.stack([g[4] for g in lib]).astype(np.uint8); cost = np.array([g[2] for g in lib], dtype=np.int32)
    iperms = np.stack([g[5] for g in lib]).astype(dt); iphs = np.stack([g[6] for g in lib]).astype(np.uint8)
    X = np.asarray(inputs, dtype=dt); T = np.asarray(target, dtype=dt); m = len(X)
    pc = np.array([bin(i).count("1") for i in range(1 << n)], dtype=np.int16); cd = np.array([0, 1, 2, 3, 4, 3, 2, 1], dtype=np.int16)
    rng = np.random.default_rng(seed); R1 = rng.integers(1, 2**62, size=m, dtype=np.uint64); R2 = rng.integers(1, 2**62, size=m, dtype=np.uint64)

    def canon(Y, S):                                   # skrot: wyjscia i fazy wzgledem pierwszego wejscia (globalna faza)
        rel = (S - S[..., :1]) & 7
        return (Y * R1 + rel * R2).sum(-1), rel

    class Side:
        def __init__(self, Y0, goal, P, F):
            self.Y, self.S = Y0, np.zeros((1, m), np.uint8); self.g = np.zeros(1, np.int32); self.goal = goal; self.P, self.F = P, F
            self.layers = [(np.array([-1]), np.array([-1]), self.g.copy())]; self.seen = {}
    Fw = Side(X[None].copy(), T, perms, phs); Bw = Side(T[None].copy(), X, iperms, iphs)     # strona celu uzywa bramek odwrotnych
    if target_phase is not None: Bw.S = np.asarray(target_phase, np.uint8)[None].copy()
    best = [max_cost + 1, None]; tab = {"F": {}, "B": {}}
    for sd, so in (("F", Fw), ("B", Bw)):
        h0 = int(canon(so.Y, so.S)[0][0]); tab[sd][h0] = (0, 0, 0); so.seen[h0] = 0
    if int(canon(Fw.Y, Fw.S)[0][0]) == int(canon(Bw.Y, Bw.S)[0][0]): return (0, [])

    def path(side, depth, idx):
        P = []
        while depth > 0:
            par, gate, _ = side.layers[depth]; P.append(int(gate[idx])); idx = int(par[idx]); depth -= 1
        return list(reversed(P))

    def meet(sd, depth, hashes, gs):
        tb = tab["B" if sd == "F" else "F"]
        for i, (h, g_) in enumerate(zip(hashes.tolist(), gs.tolist())):
            if h in tb:
                od, oi, og = tb[h]
                if g_ + og < best[0]: best[0] = g_ + og; best[1] = (sd, depth, i, "B" if sd == "F" else "F", od, oi)

    def expand(side, sd, depth):
        Y, S, g = side.Y, side.S, side.g; Bn = len(Y); lim = best[0] - 1; cf_, cg_, cgt_, cpar_ = [], [], [], []
        _CTX.update(Y=Y, S=S, g=g, P=side.P, F=side.F, goal=side.goal, cost=cost, pc=pc, cd=cd, lim=lim, beam=beam, Bn=Bn, lam=lam, noise=noise, seed=seed * 1000003 + depth * 131 + (7 if sd == 'B' else 0))
        starts = list(range(0, Gn, chunk)); npr = _nproc() if (Gn >= 4 * chunk and Bn * Gn * m > 2_000_000) else 1
        res = None
        if _gpu_on(Bn, Gn, m) and not noise:
            res = _gpu_cands_safe(_CTX)                    # None when the GPU is out of memory even with small batches -> CPU path
        if res is not None: res = [res]
        elif npr > 1:
            with _mp.get_context("fork").Pool(npr) as pool: res = pool.map(_chunk_work, [(c0, chunk) for c0 in starts])
        else: res = [_chunk_work((c0, chunk)) for c0 in starts]
        for r in res:
            if r is not None: cf_.append(r[0]); cg_.append(r[1]); cgt_.append(r[2]); cpar_.append(r[3])
        cf = np.concatenate(cf_)
        if cf.size == 0: return False
        cg = np.concatenate(cg_); cgate = np.concatenate(cgt_); cpar = np.concatenate(cpar_); order = np.argsort(cf, kind="stable"); cf, cg, cgate, cpar = cf[order], cg[order], cgate[order], cpar[order]
        if diversity > 0: ncap = min(len(cf), int(beam * (4 + 4 * diversity))); cf, cg, cgate, cpar = cf[:ncap], cg[:ncap], cgate[:ncap], cpar[:ncap]
        Ycs, Scs, hss, uniq, seen, pos, total = [], [], [], [], set(), 0, len(cf)
        while pos < total:                                 # candidates in order of f, deduplicated block by block until `beam` unique ones are found (no cap, as in the first version)
            e = min(total, pos + max(4 * beam, 4096)); Yb = side.P[cgate[pos:e, None], Y[cpar[pos:e]]]; Sb = (S[cpar[pos:e]] + side.F[cgate[pos:e, None], Y[cpar[pos:e]]]) & 7; hb, _ = canon(Yb, Sb)
            for i, h in enumerate(hb.tolist(), pos):
                if h in seen: continue
                old = side.seen.get(h)
                if old is not None and old <= cg[i]: continue
                seen.add(h); uniq.append(i)
            Ycs.append(Yb); Scs.append(Sb); hss.append(hb); pos = e
            if diversity == 0 and len(uniq) >= beam: break
        Yc, Sc, hs = np.concatenate(Ycs), np.concatenate(Scs), np.concatenate(hss); cf, cg, cgate, cpar = cf[:pos], cg[:pos], cgate[:pos], cpar[:pos]
        n1 = beam - int(diversity * beam) if diversity > 0 else beam
        keep = uniq[:n1]
        if diversity > 0 and len(uniq) > n1:               # exploration slots: round-robin over the cost-so-far levels, best f first
            buckets = {}
            for i in uniq[n1:]: buckets.setdefault(int(cg[i]), []).append(i)
            levels = sorted(buckets); ptr = {lv: 0 for lv in levels}
            while len(keep) < beam and levels:
                for lv in list(levels):
                    if ptr[lv] < len(buckets[lv]): keep.append(buckets[lv][ptr[lv]]); ptr[lv] += 1
                    else: levels.remove(lv)
                    if len(keep) >= beam: break
        if not keep: return False
        keep = np.array(keep); side.Y, side.S, side.g = Yc[keep], Sc[keep], cg[keep]; side.layers.append((cpar[keep], cgate[keep], side.g.copy())); hk = hs[keep]
        for h, g_ in zip(hk.tolist(), side.g.tolist()): side.seen[h] = min(side.seen.get(h, 1 << 30), g_)
        meet(sd, depth, hk, side.g)
        for i, (h, g_) in enumerate(zip(hk.tolist(), side.g.tolist())): tab[sd].setdefault(h, (depth, i, g_))
        return True

    for d in range(1, depth_side + 1):
        okF = expand(Fw, "F", d); okB = expand(Bw, "B", d)
        if verbose: print(f"  gl.{d}: koszt polaczenia {best[0] if best[1] else None}, wiazki F {len(Fw.Y)} B {len(Bw.Y)}", flush=True)
        if not (okF or okB): break
    if best[1] is None: return None
    sd, depth, i, other, od, oi = best[1]; A_, B_ = (Fw, Bw) if sd == "F" else (Bw, Fw)
    pa = path(A_, depth, i); pb = path(B_, od, oi); fpath, bpath = (pa, pb) if sd == "F" else (pb, pa)
    return best[0], [lib[k] for k in fpath + list(reversed(bpath))]


def gate_cost(name):
    if name in ("x", "t", "tdg", "s", "sdg", "z"): return 0
    if name in ("cx", "cz"): return 1
    if name in ("ccx", "ccz"): return 6
    if name in MINED: return MINED[name][1]
    if name[:3] in ("mgp", "mgq", "mgr", "mgs", "mgt", "mgu", "mgv", "mgw"): return int(name[-2]) + 2 * int(name[-1])
    return 3


def window_resynthesize_seq(n, gates, k_max=4, per_window_s=6.0, total_s=300.0, min_cost=3, beam=3000, lib_cache=None, edges=None, verbose=False):
    """Repeated windowed resynthesis of a gate list (tuples name,args,...) of monomial library gates. Every window (contiguous gates on at
    most k_max qubits) is replaced by a cheaper circuit with the same exact table (permutation and Z8 phases, global phase free) when the
    search finds one; repeats until no window improves or the time is out. Replacements are exact, so the whole circuit is unchanged
    up to a global phase. Returns the new gate list."""
    import time
    t0 = time.time(); lib_cache = {} if lib_cache is None else lib_cache; gates = [tuple(g) for g in gates]; improved = True
    EE = None if edges is None else {frozenset(e) for e in edges}
    def local_lib(k, Q=None):
        cp = None if EE is None else frozenset((x, y) for x in range(k) for y in range(x + 1, k) if frozenset((Q[x], Q[y])) in EE)
        if (k, cp) not in lib_cache: lib_cache[(k, cp)] = [g for g in library(k, coupling=cp) if not g[0].startswith("mined")]
        return lib_cache[(k, cp)]
    while improved and time.time() - t0 < total_s:
        improved = False
        for i in range(len(gates)):
            qs = set(); j = i; best_w = None
            while j < len(gates):
                nq = qs | set(gates[j][1])
                if len(nq) > k_max: break
                qs = nq; j += 1
                if len(qs) < 3: continue
                cost = sum(gate_cost(g[0]) for g in gates[i:j])
                if cost >= min_cost: best_w = (i, j, sorted(qs), cost)
                if time.time() - t0 > total_s: break
                if best_w is None or best_w[1] != j: continue
                a, b, Q, c = best_w; k = len(Q); loc = {q: x for x, q in enumerate(Q)}
                sub = [(g[0], tuple(loc[q] for q in g[1])) for g in gates[a:b]]
                perm, ph = _table_norm(emit(k, sub))
                r = bidir_search(k, list(range(1 << k)), perm, max_cost=c - 1, beam=beam, lam=0.6, depth_side=8, lib=local_lib(k, Q), target_phase=ph)
                if r:
                    new = [(g[0], tuple(Q[x] for x in g[1])) for g in r[1]]
                    gates[a:b] = new; improved = True
                    if verbose: print("window", Q, c, "->", r[0], flush=True)
                    break
            if improved: break
    return gates


_WIN = {}


def _win_task(a):
    """One window search (module level so a forked pool can run it); the library comes from the parent via _WIN."""
    k, cp, perm, ph, c, beam = a
    os.environ["QCPT_NPROC"] = "1"; os.environ["QCPT_GPU"] = "0"        # no nested pools, no shared GPU in forked workers
    from . import cppsearch
    if cp is None and cppsearch.available(): return cppsearch.window_search(k, perm, ph, c - 1, beam)         # C++ core (no coupling-map support there)
    r = bidir_search(k, list(range(1 << k)), perm, max_cost=c - 1, beam=beam, lam=0.6, depth_side=8, lib=_WIN["lib"][(k, cp)], target_phase=ph)
    return None if r is None else (r[0], [(g[0], tuple(g[1])) for g in r[1]])


def window_resynthesize_par(n, gates, k_max=4, per_window_s=6.0, total_s=300.0, min_cost=3, beam=3000, lib_cache=None, edges=None, verbose=False, failed=None):
    """Windowed resynthesis with the same replacements as window_resynthesize but (i) a window that failed once (same table, same
    bound) is never searched again, (ii) all distinct windows of a pass are searched in parallel, (iii) every improving window that
    does not overlap a better one is replaced in the same pass. Replacements are exact, as before."""
    import time, multiprocessing as mp
    t0 = time.time(); gates = [tuple(g) for g in gates]; failed = set() if failed is None else failed
    from . import cppsearch
    cache_path = os.environ.get("QCPT_WIN_CACHE"); cache_path = None if not cache_path else cache_path + ("_cpp" if cppsearch.available() else "_py") + ".pkl"   # failures persist between runs (per engine)
    if cache_path and os.path.exists(cache_path):
        import pickle
        try: failed |= pickle.load(open(cache_path, "rb"))
        except Exception: pass
    n_failed0 = len(failed)
    EE = None if edges is None else {frozenset(e) for e in edges}; lib_cache = {} if lib_cache is None else lib_cache
    def cpl(k, Q): return None if EE is None else frozenset((x, y) for x in range(k) for y in range(x + 1, k) if frozenset((Q[x], Q[y])) in EE)
    def local_lib(k, cp):
        if (k, cp) not in lib_cache: lib_cache[(k, cp)] = [g for g in library(k, coupling=cp) if not g[0].startswith("mined")]
        return lib_cache[(k, cp)]
    tabs = {}
    while time.time() - t0 < total_s:
        cands = {}
        for i in range(len(gates)):
            qs = set(); j = i
            while j < len(gates):
                nq = qs | set(gates[j][1])
                if len(nq) > k_max: break
                qs = nq; j += 1
                if len(qs) < 3: continue
                c = sum(gate_cost(g[0]) for g in gates[i:j])
                if c < min_cost: continue
                Q = sorted(qs); loc = {q: x for x, q in enumerate(Q)}; sub = tuple((g[0], tuple(loc[q] for q in g[1])) for g in gates[i:j]); k = len(Q); cp = cpl(k, Q)
                if (k, sub) not in tabs: tabs[(k, sub)] = _table_norm(emit(k, list(sub)))
                perm, ph = tabs[(k, sub)]; key = (k, cp, perm.tobytes(), ph.tobytes(), c, beam)
                if key in failed: continue
                cands.setdefault(key, (k, cp, perm, ph, c, beam)); cands[key] = cands[key]; cands.setdefault(("w",) + key, []).append((i, j, Q, sub))
        keys = [kk for kk in cands if kk[0] != "w"]
        if not keys: break
        for kk in keys: _WIN.setdefault("lib", {})[(cands[kk][0], cands[kk][1])] = local_lib(cands[kk][0], cands[kk][1])
        npr = _nproc() if len(keys) > 1 else 1
        if npr > 1:
            with mp.get_context("fork").Pool(min(npr, len(keys))) as pool: res = pool.map(_win_task, [cands[kk] for kk in keys], chunksize=1)
        else: res = [_win_task(cands[kk]) for kk in keys]
        hits = []
        for kk, r in zip(keys, res):
            if r is None: failed.add(kk); continue
            for (i, j, Q, sub) in cands[("w",) + kk]: hits.append((cands[kk][4] - r[0], i, j, Q, r[1]))
        if not hits: break
        hits.sort(key=lambda h: (-h[0], h[1])); used = []; chosen = []
        for h in hits:
            if all(h[2] <= u0 or h[1] >= u1 for u0, u1 in used): used.append((h[1], h[2])); chosen.append(h)
        for sv, i, j, Q, new in sorted(chosen, key=lambda h: -h[1]):
            gates[i:j] = [(g[0], tuple(Q[x] for x in g[1])) for g in new]
            if verbose: print("window", Q, "saves", sv, flush=True)
    if cache_path and len(failed) > n_failed0:
        import pickle; tmp = cache_path + ".tmp"; pickle.dump(failed, open(tmp, "wb")); os.replace(tmp, cache_path)
    return gates


def window_resynthesize(n, gates, **kw):
    if os.environ.get("QCPT_WIN_PAR", "1") == "0": return window_resynthesize_seq(n, gates, **kw)
    return window_resynthesize_par(n, gates, **kw)


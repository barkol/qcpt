"""Template search (meet in the middle, no distance heuristic) for circuits of the form  F1 . Ga . M . Gb . F2  on the valid inputs of a classical function.
F1, F2: <= f1, f2 free gates (cx, x); M: a sequence of <= fm free gates or one gate of `middles`; Ga, Gb: gadgets from the library.
Forward states (F1, Ga) and backward states (F2^-1, Gb^-1) are hashed modulo the global phase; every middle is tried against the sorted backward hashes."""
from __future__ import annotations
import itertools, multiprocessing as mp
import numpy as np
from . import resynth as RS

_C = {}

def _seq_tables(lib, seqs, N, inverse=False):
    P = np.empty((len(seqs), N), np.int64); F = np.zeros((len(seqs), N), np.int64)
    for k, sq in enumerate(seqs):
        perm = np.arange(N); ph = np.zeros(N, np.int64)
        for gi in (reversed(sq) if inverse else sq):
            g = lib[gi]; pt, ft = (g[5], g[6]) if inverse else (g[3], g[4])
            ph = ph + ft[perm]; perm = pt[perm]
        P[k] = perm; F[k] = ph & 7
    return P, F

def _dedupe(P, F, seqs, costs):
    seen = {}; keep = []
    for k in range(len(seqs)):
        key = (P[k].tobytes(), ((F[k] - F[k][0]) & 7).astype(np.int8).tobytes())
        if key not in seen or costs[k] < costs[seen[key]]:
            if key in seen: keep.remove(seen[key])
            seen[key] = k; keep.append(k)
    return sorted(keep)

def load_db3(path):
    """dict: 3-bit permutation (tuple) -> (cost, class_index) for every member of the 180 classes (relabelling of the bits and X flips on both sides)."""
    import json
    rows = json.load(open(path)); out = {}
    def rl(x, pi): return sum(((x >> i) & 1) << pi[i] for i in range(3))
    for ci, row in enumerate(rows):
        if row["cost"] is None: continue
        c = row["perm"]
        for pi in itertools.permutations(range(3)):
            r = [rl(x, pi) for x in range(8)]; rinv = [0] * 8
            for x in range(8): rinv[r[x]] = x
            for a in range(8):
                for b in range(8):
                    # member p with canonical form c:  c[x] = r[p[rinv[x^a]]] ^ b   =>   p[y] = rinv[c[r[y]^... ]] solved below
                    p = [0] * 8
                    for x in range(8): p[rinv[x ^ a]] = rinv[c[x] ^ b]
                    out.setdefault(tuple(p), (row["cost"], ci))
    return out

_COMP = {}

def _complete(pm_items, DB):
    """cheapest database completion of a partial 3-bit permutation (tuple of (in, out) pairs); None if there is none"""
    if pm_items in _COMP: return _COMP[pm_items]
    pm = dict(pm_items); miss_in = [v for v in range(8) if v not in pm]; miss_out = [v for v in range(8) if v not in set(pm.values())]; ent = None; perm = None
    if len(miss_in) <= 5:
        for comp in itertools.permutations(miss_out):
            pp = dict(pm); pp.update(zip(miss_in, comp)); cand = tuple(pp[k] for k in range(8))
            if cand == tuple(range(8)): continue
            e = DB.get(cand)
            if e is not None and (ent is None or e[0] < ent[0]): ent, perm = e, cand
    _COMP[pm_items] = (ent, perm); return _COMP[pm_items]

def _db_task(Q):
    C = _C; mask = sum(1 << q for q in Q); inv = np.uint8(0xFF ^ mask); FY, FS, BY, BS = C["FY"], C["FS"], C["BY"], C["BS"]; cF, cB = C["cF"], C["cB"]; best = None
    hb = np.concatenate([_hash(BY[c0:c0 + 200000] & inv, BS[c0:c0 + 200000], C["R1"], C["R2"]) for c0 in range(0, len(BY), 200000)])
    order = np.lexsort((cB, hb)); hbs = hb[order]; ok = np.where(cF + C["minB"] + 1 <= C["max_cost"])[0]; sh = [np.uint8(q) for q in Q]
    for c0 in range(0, ok.size, 100000):
        idx = ok[c0:c0 + 100000]; h = _hash(FY[idx] & inv, FS[idx], C["R1"], C["R2"])
        lo = np.searchsorted(hbs, h, "left"); hi = np.searchsorted(hbs, h, "right"); cnt = hi - lo; sel = np.where(cnt > 0)[0]
        if sel.size == 0: continue
        reps = cnt[sel]; tot = int(reps.sum()); first = np.repeat(np.cumsum(reps) - reps, reps)
        I = np.repeat(idx[sel], reps); J = order[np.repeat(lo[sel], reps) + np.arange(tot) - first]; t0 = cF[I] + cB[J]
        lim = C["max_cost"] if best is None else best[0] - 1
        k = t0 + 1 <= lim; I, J, t0 = I[k], J[k], t0[k]
        o = np.argsort(t0, kind="stable"); I, J, t0 = I[o], J[o], t0[o]
        for p0 in range(0, len(I), 50000):
            Ic, Jc, tc = I[p0:p0 + 50000], J[p0:p0 + 50000], t0[p0:p0 + 50000]
            if best is not None and tc[0] + 1 >= best[0]: break
            fy, by = FY[Ic], BY[Jc]
            vin = ((fy >> sh[0]) & 1) | (((fy >> sh[1]) & 1) << 1) | (((fy >> sh[2]) & 1) << 2); vout = ((by >> sh[0]) & 1) | (((by >> sh[1]) & 1) << 1) | (((by >> sh[2]) & 1) << 2)
            vin = vin.astype(np.int16); vout = vout.astype(np.int16)       # uint8 would turn -1 into 255 inside np.where
            bad = np.zeros(len(Ic), bool); O = np.full((len(Ic), 8), -1, np.int16)
            for v in range(8):
                mv = vin == v; vmin = np.where(mv, vout, 9).min(1); vmax = np.where(mv, vout, -1).max(1); bad |= (vmax >= 0) & (vmin != vmax); O[:, v] = vmax
            for v in range(8):
                for w in range(v + 1, 8): bad |= (O[:, v] >= 0) & (O[:, v] == O[:, w])
            for t in np.where(~bad)[0]:
                if best is not None and tc[t] + 1 >= best[0]: break
                items = tuple((v, int(O[t, v])) for v in range(8) if O[t, v] >= 0); ent, perm = _complete(items, C["DB"])
                if ent is None: continue
                totc = int(tc[t]) + ent[0]
                if totc <= C["max_cost"] and (best is None or totc < best[0]): best = (totc, int(Ic[t]), int(Jc[t]), tuple(Q), perm)
    return best

def _hash(Y, S, R1, R2):
    rel = (S - S[..., :1]) & 7
    return (Y.astype(np.uint64) * R1 + rel.astype(np.uint64) * R2).sum(-1)

def _mid_task(mi):
    C = _C; Pm, Fm, cM = C["Pm"][mi], C["Fm"][mi], C["cM"][mi]; best = None
    ok = np.where(C["cF"] + cM + C["minB"] <= C["max_cost"])[0]
    if ok.size == 0: return None
    for c0 in range(0, ok.size, 200000):
        idx = ok[c0:c0 + 200000]; Y = C["FY"][idx]; S = C["FS"][idx]
        Y2 = Pm[Y]; S2 = (S + Fm[Y]) & 7; h = _hash(Y2, S2, C["R1"], C["R2"])
        pos = np.searchsorted(C["HB"], h); pos[pos >= len(C["HB"])] = len(C["HB"]) - 1
        hit = np.where(C["HB"][pos] == h)[0]
        for t in hit:
            j = C["BORD"][pos[t]]; tot = C["cF"][idx[t]] + cM + C["cB"][j]
            if tot <= C["max_cost"] and (best is None or tot < best[0]): best = (int(tot), int(idx[t]), int(j), mi)
    return best

def template_search_numpy(n, inputs, target, max_cost, f1=1, fm=2, f2=2, ga=("mg1", "mg2", "mg3", "mg4"), gb=None, middles=(), lib=None, nproc=12, seed=0, verbose=False, db=None):
    lib = lib or RS.library(n, extra_signs=True); N = 1 << n; gb = ga if gb is None else gb
    X = np.asarray(inputs); Yt = np.asarray(target); m = len(X); rng = np.random.default_rng(seed); R1 = rng.integers(1, 2**62, size=m, dtype=np.uint64); R2 = rng.integers(1, 2**62, size=m, dtype=np.uint64)
    free = [i for i, g in enumerate(lib) if g[0] in ("cx", "x")]; cost = np.array([g[2] for g in lib])
    def seqs(k): return [sq for L in range(k + 1) for sq in itertools.product(free, repeat=L)]
    def tabs(sqs, inv=False):
        P, F = _seq_tables(lib, sqs, N, inv); c = np.array([sum(cost[i] for i in sq) for sq in sqs]); keep = _dedupe(P, F, sqs, c); return [sqs[k] for k in keep], P[keep], F[keep], c[keep]
    S1, P1, F1, c1 = tabs(seqs(f1)); S2, P2i, F2i, c2 = tabs(seqs(f2), True)
    GA = [i for i, g in enumerate(lib) if g[0] in ga]; GB = [i for i, g in enumerate(lib) if g[0] in gb]
    SM, PM, FM, cM = tabs(seqs(fm) + [(i,) for i, g in enumerate(lib) if g[0] in middles])
    if verbose: print("F1 %d, Ga %d, middles %d, Gb %d, F2 %d" % (len(S1), len(GA), len(SM), len(GB), len(S2)), flush=True)
    gaP = np.stack([lib[i][3] for i in GA]); gaF = np.stack([lib[i][4] for i in GA]).astype(np.int64); gbP = np.stack([lib[i][5] for i in GB]); gbF = np.stack([lib[i][6] for i in GB]).astype(np.int64)
    # forward states: F1 then Ga
    Y1 = P1[:, X]; Sx = F1[:, X]; ar = np.arange(len(GA))[None, :, None]
    FY = gaP[ar, Y1[:, None, :]]; FS = (Sx[:, None, :] + gaF[ar, Y1[:, None, :]]) & 7
    cF = (c1[:, None] + np.array([cost[i] for i in GA])[None, :]).ravel(); FY = FY.reshape(-1, m).astype(np.uint8); FS = FS.reshape(-1, m).astype(np.uint8)
    # backward states: F2^-1 then Gb^-1 from the target
    Yb1 = P2i[:, Yt]; Sb1 = F2i[:, Yt]; ar = np.arange(len(GB))[None, :, None]
    BY = gbP[ar, Yb1[:, None, :]]; BS = (Sb1[:, None, :] + gbF[ar, Yb1[:, None, :]]) & 7
    cB = (c2[:, None] + np.array([cost[i] for i in GB])[None, :]).ravel(); BY = BY.reshape(-1, m).astype(np.uint8); BS = BS.reshape(-1, m).astype(np.uint8)
    HB = np.concatenate([_hash(BY[c0:c0 + 200000], BS[c0:c0 + 200000], R1, R2) for c0 in range(0, len(BY), 200000)]); BORD = np.argsort(HB, kind="stable"); HBs = HB[BORD]
    _C.update(Pm=PM.astype(np.uint8), Fm=FM.astype(np.uint8), cM=cM, cF=cF, FY=FY, FS=FS, R1=R1, R2=R2, HB=HBs, BORD=BORD, cB=cB, minB=int(cB.min()), max_cost=max_cost, BY=BY, BS=BS, DB=db)
    if verbose: print("forward %d, backward %d states" % (len(FY), len(BY)), flush=True)
    with mp.get_context("fork").Pool(nproc) as pool: res = [r for r in pool.map(_mid_task, range(len(SM)), chunksize=4) if r is not None]
    dbres = []
    if db is not None:
        with mp.get_context("fork").Pool(nproc) as pool: dbres = [r for r in pool.map(_db_task, list(itertools.combinations(range(n), 3)), chunksize=1) if r is not None]
    if dbres and (not res or min(dbres)[0] < min(res)[0]):
        tot, fi, bj, Q, perm = min(dbres); i1, ia = divmod(fi, len(GA)); i2, ib = divmod(bj, len(GB))
        return tot, [lib[i] for i in S1[i1]] + [lib[GA[ia]]] + [("DBMID", Q, perm)] + [lib[GB[ib]]] + [lib[i] for i in S2[i2]]
    if not res: return None
    tot, fi, bj, mi = min(res); i1, ia = divmod(fi, len(GA)); i2, ib = divmod(bj, len(GB))
    gates = [lib[i] for i in S1[i1]] + [lib[GA[ia]]] + [lib[i] for i in SM[mi]] + [lib[GB[ib]]] + [lib[i] for i in S2[i2]]
    return tot, gates


def expand_db(gates, db_path, db):
    """Replace ("DBMID", Q, perm) markers by concrete gates: the stored circuit of the class, relabelled and conjugated by X so that it realises perm on the qubits Q."""
    import json
    from qiskit import QuantumCircuit
    rows = json.load(open(db_path)); out = []
    for g in gates:
        if g[0] != "DBMID": out.append(g); continue
        _, Q, perm = g; cost, ci = db[perm]; rep = rows[ci]["gates"]
        for pi in itertools.permutations(range(3)):
            for a in range(8):
                for b in range(8):
                    loc = [("x", (i,)) for i in range(3) if (a >> i) & 1] + [(nm, tuple(pi[x] for x in args)) for nm, args in rep] + [("x", (i,)) for i in range(3) if (b >> i) & 1]
                    try: pt, ph = RS._table_norm(RS.emit(3, loc))
                    except AssertionError: continue
                    if tuple(int(v) for v in pt) == tuple(perm) and not np.any(ph):
                        out += [(nm, tuple(Q[x] for x in args)) for nm, args in loc]; break
                else: continue
                break
            else: continue
            break
        else: raise RuntimeError("no transformation found for " + str(perm))
    return out



def _qcpt_binary():
    import os
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.environ.get("QCPT_BIN", os.path.join(root, "cpp", "qcpt"))
    if not os.path.exists(path): raise RuntimeError(f"C++ search core not built: run `make -C {os.path.join(root, 'cpp')}` (needs only g++ with C++17)")
    return path, os.path.join(root, "cpp", "data", "db3.tab")


def template_search(n, inputs, target, max_cost, f1=1, fm=2, f2=2, ga=("mg1", "mg2", "mg3", "mg4"), gb=None, middles=(), lib=None, nproc=12, seed=0, verbose=False, db=None):
    """Same contract as template_search_numpy (the reference implementation), computed by the C++ core (cpp/qcpt): returns (cost, gates) or None.
    The gate library is the one of RS.library(n, extra_signs=True); `db` only switches the three-qubit database middle on."""
    import os, subprocess
    lib = lib or RS.library(n, extra_signs=True); exe, dbfile = _qcpt_binary(); gb = ga if gb is None else gb
    prob = f"{n} {len(inputs)}\n{' '.join(str(int(v)) for v in inputs)}\n{' '.join(str(int(v)) for v in target)}\n{max_cost} {f1} {fm} {f2} {nproc} {seed}\n"
    prob += f"ga {len(ga)} {' '.join(ga)}\ngb {len(gb)} {' '.join(gb)}\nmid {len(middles)} {' '.join(middles)}\n"
    r = subprocess.run([exe, "template", dbfile if db is not None else "-"], input=prob, capture_output=True, text=True, env=dict(os.environ, **({"QCPT_VERBOSE": "1"} if verbose else {})))
    if r.returncode != 0: raise RuntimeError("qcpt failed: " + r.stderr[-500:])
    out = r.stdout.split("\n")
    if out[0] == "NONE": return None
    by_key = {(g[0], tuple(g[1])): g for g in lib}; gates = []
    for ln in out[1:]:
        t = ln.split()
        if not t or t[0] == "END": continue
        if t[0] == "G": gates.append(by_key[(t[1], tuple(int(v) for v in t[2:]))])
        else: gates.append(("DBMID", tuple(int(v) for v in t[1:4]), tuple(int(v) for v in t[4:12])))
    return int(out[0].split()[1]), gates

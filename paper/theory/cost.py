"""Minimal number of CNOTs for CCX on the subspace span(S), S <- {0,1}^3 (order abt, a=bit0? see enc),
up to one global phase, with arbitrary 1-qubit gates between the CNOTs.
For each S it computes the loss min_U 1-|sum_s <CCX s|U s>|/|S| over all placements of k CNOTs (no two equal ones
in a row) and R random starts. Saved per (S,k) to cost.jsonl (resumable). Usage: cost.py [R] [steps]"""
import itertools, json, math, os, sys, time
import numpy as np, torch
torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8'))); DT = torch.complex128
R = int(sys.argv[1]) if len(sys.argv) > 1 else 24; STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
OUT = 'cost.jsonl'; done = {(tuple(json.loads(l)['S']), json.loads(l)['k']) for l in open(OUT)} if os.path.exists(OUT) else set()
# qiskit little-endian: indeks bazy x = a + 2b + 4t (a=q0, b=q1, t=q2); etykieta 'abt' jako napis bitow a,b,t
def lab(x): return f"{x & 1}{(x >> 1) & 1}{(x >> 2) & 1}"
CCX = np.eye(8, dtype=complex)
for x in range(8):
    if x & 1 and x & 2: CCX[:, x] = 0; CCX[x ^ 4, x] = 1
def cnot(c, t):
    M = np.zeros((8, 8), complex)
    for x in range(8): M[x ^ (((x >> c) & 1) << t), x] = 1
    return M
PAIRS = [(c, t) for c in range(3) for t in range(3) if c != t]; CN = {p: torch.tensor(cnot(*p), dtype=DT) for p in PAIRS}
def u3(p):  # p (...,3) -> (...,2,2)
    th, ph, la = p[..., 0], p[..., 1], p[..., 2]; c, s = torch.cos(th / 2), torch.sin(th / 2)
    e = lambda z: torch.exp(1j * z.to(DT))
    return torch.stack([torch.stack([c.to(DT), -e(la) * s.to(DT)], -1), torch.stack([e(ph) * s.to(DT), e(ph + la) * c.to(DT)], -1)], -2)
def layer(p):  # p (B,3 qubits,3) -> (B,8,8), kron(q2,q1,q0)
    U = [u3(p[:, q]) for q in range(3)]
    K = torch.einsum('bij,bkl->bikjl', U[2], U[1]).reshape(-1, 4, 4)
    return torch.einsum('bij,bkl->bikjl', K, U[0]).reshape(-1, 8, 8)
def patterns(k):
    return [p for p in itertools.product(PAIRS, repeat=k) if all(p[i] != p[i + 1] for i in range(k - 1))]
def best_loss(S, k):
    cols = torch.tensor(S); V = torch.tensor(CCX, dtype=DT)[:, cols]
    pats = patterns(k) if k else [()]; best = 1.0; arg = None
    for chunk in range(0, len(pats), 400):
        P = pats[chunk:chunk + 400]; B = len(P) * R
        par = (torch.rand(B, k + 1, 3, 3, dtype=torch.float64) * 2 * math.pi).requires_grad_()
        opt = torch.optim.Adam([par], lr=0.05)
        CNs = [torch.stack([CN[pp[i]] for pp in P for _ in range(R)]) for i in range(k)]
        def loss_vec():
            U = layer(par[:, 0])
            for i in range(k): U = layer(par[:, i + 1]) @ CNs[i] @ U
            ov = torch.einsum('ij,bij->b', V.conj(), U[:, :, cols]) / len(S)
            return 1 - ov.abs()
        for it in range(STEPS):
            opt.zero_grad(); L = loss_vec(); L.sum().backward(); opt.step()
            if it == STEPS // 2:
                for g in opt.param_groups: g['lr'] = 0.01
        with torch.no_grad():
            L = loss_vec(); i = int(torch.argmin(L))
            if float(L[i]) < best: best = float(L[i]); arg = P[i // R]
        if best < 1e-9: break
    return best, arg
def tabela(S):
    s = {lab(x) for x in S}
    if not s & {'110', '111'}: return 0
    if not s & {'010', '011'} or not s & {'100', '101'}: return 1
    if not {'100', '101', '010', '011'} <= s: return 3
    return 6
if __name__ == '__main__':
    # S z dokladnoscia do symetrii a<->b
    def swap(x): return (x & 4) | ((x & 1) << 1) | ((x >> 1) & 1)
    reps = {}
    for m in range(1, 256):
        S = tuple(x for x in range(8) if m >> x & 1); key = min(S, tuple(sorted(swap(x) for x in S)))
        reps.setdefault(key, S)
    reps = sorted(reps.values(), key=lambda S: (tabela(S), len(S)))
    print('reprezentantow S:', len(reps), {c: sum(tabela(S) == c for S in reps) for c in (0, 1, 3, 6)}, flush=True)
    PAS, NPAS = int(os.environ.get('PAS', '0')), int(os.environ.get('NPAS', '1')); zad = []
    for S in reps:
        c = tabela(S)
        ks = {0: [], 1: [0, 1], 3: [1, 2, 3], 6: [4, 5]}[c]
        if c == 6 and S == reps[[tabela(x) for x in reps].index(6)]: ks = [4, 5, 6]  # kontrola dodatnia na jednym S
        for k in ks: zad.append((S, c, k))
    zad.sort(key=lambda z: -z[2])
    for n, (S, c, k) in enumerate(zad):
            if n % NPAS != PAS or (S, k) in done: continue
            t0 = time.time(); L, arg = best_loss(list(S), k)
            row = dict(S=list(S), labels=sorted(lab(x) for x in S), tabela=c, k=k, best_loss=L, pattern=arg, s=round(time.time() - t0, 1), R=R, steps=STEPS)
            open(OUT, 'a').write(json.dumps(row) + '\n'); print(json.dumps(row), flush=True)
    print('KONIEC')

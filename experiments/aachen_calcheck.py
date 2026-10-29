"""Check of the manufacturer calibration of ibm_aachen on the probed regions of 29 October 2026 (derived from aachen_bench.py: the qubits and couplers come from the probe, not from the calibration).
Original docstring: Benchmarking ibm_aachen (typowe testy QPU) + porownanie z kalibracja producenta. Bloki (kazdy = jedno zgloszenie SamplerV2):
  TI  T1 i echo-T2 na zbiorze niezaleznym (zadne dwa kubity sasiednie, brak przesluchow ZZ), 6 opoznien
  TS  echo-T2 jednoczesne na tym samym zbiorze + sasiadach (przesluchy ZZ)
  RO  blad odczytu (przygotowanie |0> i |1> na wszystkich kubitach zbioru)
  R1  1q randomized benchmarking (rownolegle na zbiorze niezaleznym)
  R2  2q randomized benchmarking (rownolegle na skojarzeniu krawedzi)
  GH  populacje GHZ-n (n=4,8,12)
Etapy: sprawdz | wyslij BLOK | odbierz BLOK (z analiza i porownaniem z kalibracja). Stan zgloszen w state_bench.json."""
import json, os, sys, time, itertools
import numpy as np
from scipy.optimize import curve_fit
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import random_clifford
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

ACC, BK = os.environ.get("IBM_ACCOUNT", "eu-de-aachen-156"), os.environ.get("IBM_BACKEND", "ibm_aachen")   # Nighthawk: set both, and BENCH_DIR
HERE = os.environ.get("BENCH_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "aachen_calcheck_2026-10-29"); os.makedirs(HERE, exist_ok=True)
STATE = os.path.join(HERE, "state_bench.json")
S_PER_SHOT, S_PER_JOB, SAFETY = 0.30e-3, 2.2, 1.35
DELAYS = [10, 40, 80, 150, 250, 400]            # us
NQ, NE = 14, 8                                    # kubity niezalezne (T1/T2/RB1), krawedzie rozlaczne (RB2)


def graph(bk):
    cm = bk.coupling_map; adj = {}
    for a, b in cm.get_edges(): adj.setdefault(a, set()).add(b); adj.setdefault(b, set()).add(a)
    return adj


FX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
PROBE = json.load(open(os.path.join(FX, "aachen_probe4", "probe.json"))); SEL = json.load(open(os.path.join(FX, "aachen_calcheck", "selection.json")))


def pick(bk):
    """NQ independent qubits from the whole chip (best T1+T2 of the calibration among the qubits the probe finds healthy: readout flip < 0.2, T1 and T2 > 100 us, not excluded) and NE disjoint couplers
    INSIDE the probed regions of the repeat (6, 8 and 12 qubits), couplers above 1.2% per CZ in the probe dropped, in the order of the calibrated error."""
    adj = graph(bk); tp = bk.target.qubit_properties
    qflip = {int(k): v for k, v in PROBE["qubit"].items()}; bad = set(SEL["bad_qubits"])
    badc = {tuple(sorted(map(int, k.split("-")))) for k, v in PROBE["coupler"].items() if v["err_per_cz"] > 0.012}
    good = [q for q in adj if q not in bad and qflip.get(q, 1.0) < 0.2 and (tp[q].t1 or 0) > 100e-6 and (tp[q].t2 or 0) > 100e-6]
    score = {q: (tp[q].t1 or 0) + (tp[q].t2 or 0) for q in good}
    S = []
    for q in sorted(good, key=lambda q: -score[q]):
        if all(q not in adj[s] for s in S): S.append(q)
        if len(S) == NQ: break
    reg = set().union(*[set(SEL["regions"][k]["qubits"]) for k in ("6", "8", "12")])
    cz = {tuple(sorted(k)): v.error for k, v in bk.target["cz"].items() if v is not None and v.error is not None and tuple(sorted(k)) not in badc and k[0] in reg and k[1] in reg}
    E, used = [], set()
    for (a, b), e in sorted(cz.items(), key=lambda kv: kv[1]):
        if a in used or b in used: continue
        E.append((a, b)); used |= {a, b}
        if len(E) == NE: break
    return S, E, adj


def circ_T(kind, qs, d_us, extra=()):
    allq = list(qs) + list(extra); c = QuantumCircuit(len(allq), len(qs)); c.name = f"{kind}_{d_us}"
    n = len(qs)
    if kind == "T1":
        for i in range(n): c.x(i)
        c.delay(d_us, unit="us")
    else:                                         # echo Hahn na kubitach qs; extra = sasiedzi w superpozycji (przesluchy ZZ)
        for i in range(len(allq)): c.h(i)
        c.delay(d_us / 2, unit="us")
        for i in range(len(allq)): c.x(i)
        c.delay(d_us / 2, unit="us")
        for i in range(len(allq)): c.h(i)
    c.measure(range(n), range(n)); return c, allq


def build(blk, bk):
    S, E, adj = pick(bk); circs = []; meta = {"S": S, "E": E}
    if blk == "TI":
        for d in DELAYS:
            for k in ("T1", "T2"): c, al = circ_T(k, S, d); circs.append((c, al))
    elif blk == "TS":
        nb = sorted({n for s in S for n in adj[s]} - set(S))[:20]; meta["nb"] = nb
        for d in DELAYS: c, al = circ_T("T2", S, d, extra=nb); c.name = f"T2s_{d}"; circs.append((c, al))
    elif blk == "RO":
        qs = sorted(set(S) | {x for e in E for x in e}); meta["qs"] = qs
        for v in (0, 1):
            c = QuantumCircuit(len(qs), len(qs)); c.name = f"RO_{v}"
            if v: [c.x(i) for i in range(len(qs))]
            c.measure(range(len(qs)), range(len(qs))); circs.append((c, qs))
    elif blk == "R1":
        from qiskit.quantum_info import Clifford
        rng = np.random.default_rng(1); meta["lengths"] = [1, 10, 25, 50, 100, 200, 400]; meta["seeds"] = int(os.environ.get("BENCH_SEEDS", 2))
        for m in meta["lengths"]:
            for s in range(meta["seeds"]):
                c = QuantumCircuit(len(S), len(S)); c.name = f"R1_{m}_{s}"
                for i in range(len(S)):
                    cl = Clifford(QuantumCircuit(1)); seq = []
                    for _ in range(m):
                        g = random_clifford(1, seed=int(rng.integers(1 << 30))); seq.append(g); cl = cl.compose(g)
                    for g in seq: c.compose(g.to_circuit(), qubits=[i], inplace=True); c.barrier(i)
                    c.compose(cl.adjoint().to_circuit(), qubits=[i], inplace=True)
                c.measure(range(len(S)), range(len(S))); circs.append((c, S))
    elif blk == "R2":
        from qiskit.quantum_info import Clifford
        rng = np.random.default_rng(2); meta["lengths"] = [1, 2, 4, 8, 14, 22]; meta["seeds"] = int(os.environ.get("BENCH_SEEDS", 2))
        for m in meta["lengths"]:
            for s in range(meta["seeds"]):
                c = QuantumCircuit(2 * len(E), 2 * len(E)); c.name = f"R2_{m}_{s}"
                for j in range(len(E)):
                    cl = Clifford(QuantumCircuit(2)); seq = []
                    for _ in range(m):
                        g = random_clifford(2, seed=int(rng.integers(1 << 30))); seq.append(g); cl = cl.compose(g)
                    for g in seq: c.compose(g.to_circuit(), qubits=[2 * j, 2 * j + 1], inplace=True); c.barrier(2 * j, 2 * j + 1)
                    c.compose(cl.adjoint().to_circuit(), qubits=[2 * j, 2 * j + 1], inplace=True)
                c.measure(range(2 * len(E)), range(2 * len(E))); circs.append((c, [x for e in E for x in e]))
    elif blk == "QV":
        from qiskit.circuit.library import QuantumVolume
        meta["ns"] = [3, 4, 5]; meta["ncirc"] = int(os.environ.get("QV_NCIRC", 20))
        for n_ in meta["ns"]:
            for s in range(meta["ncirc"]):
                c = QuantumVolume(n_, n_, seed=s).decompose(); c.measure_all(); c.name = f"QV_{n_}_{s}"; circs.append((c, None))
    elif blk == "GH":
        meta["ns"] = [4, 8, 12]
        for n in meta["ns"]:
            c = QuantumCircuit(n, n); c.name = f"GHZ_{n}"; c.h(0); [c.cx(i, i + 1) for i in range(n - 1)]; c.measure(range(n), range(n)); circs.append((c, None))
    out = []
    for c, lay in circs:
        t = transpile(c, bk, initial_layout=lay, optimization_level=0 if blk in ("R1", "R2") else 1, seed_transpiler=7) if lay is not None else transpile(c, bk, optimization_level=3, seed_transpiler=7)
        out.append(t)
    return out, meta, [c for c, _ in circs]


SHOTS = {"TI": 1000, "TS": 1000, "RO": 4000, "R1": 1000, "R2": 1000}                       # fewer shots and seeds than in aachen_bench.py (budget)
def est(n_circ, shots, blk): return n_circ * shots * S_PER_SHOT * (1.8 if blk in ("R1", "R2") else 1.0) + S_PER_JOB


def decay(m, A, p, B): return A * p ** m + B


if __name__ == "__main__":
    stage = sys.argv[1]; svc = QiskitRuntimeService(name=ACC); bk = svc.backend(BK)
    state = json.load(open(STATE)) if os.path.exists(STATE) else {}
    if stage == "sprawdz":
        u = svc.usage(); tot = 0.0
        for blk in SHOTS:
            ts, meta, _ = build(blk, bk); e = est(len(ts), SHOTS[blk], blk); tot += e
            print(f"blok {blk}: {len(ts):3d} obwodow x {SHOTS[blk]} strzalow, szac. {e:6.1f} s", {k: (v if k in ('S', 'E') else '') for k, v in meta.items() if k in ('S', 'E')} if blk == "TI" else "")
        print(f"RAZEM {tot:.0f} s x {SAFETY} = {tot * SAFETY:.0f} s | pozostalo {u.get('usage_remaining_seconds')} s")
    elif stage == "wyslij":
        blk = sys.argv[2]
        if blk in state: sys.exit(f"{blk} juz wyslany ({state[blk]['job_id']})")
        u = svc.usage(); ts, meta, _ = build(blk, bk); e = est(len(ts), SHOTS[blk], blk) * SAFETY
        if u.get("usage_remaining_seconds", 0) < e: sys.exit(f"za maly budzet: {u.get('usage_remaining_seconds')} < {e:.0f}")
        job = SamplerV2(mode=bk).run(ts, shots=SHOTS[blk]); state[blk] = dict(job_id=job.job_id(), shots=SHOTS[blk], meta=meta, names=[t.name for t in ts], sent=time.strftime("%Y-%m-%d %H:%M:%S"))
        json.dump(state, open(STATE, "w"), indent=1); print("wyslano", blk, job.job_id())
    elif stage == "odbierz":
        blk = sys.argv[2]; D = state[blk]; job = svc.job(D["job_id"]); print("status", job.status()); res = job.result(); S = D["meta"]["S"]; E = D["meta"]["E"]
        tp = bk.target.qubit_properties; counts = []
        for k in range(len(D["names"])):
            c = res[k].data; c = getattr(c, "c", None) or getattr(c, "meas", None) or list(vars(c).values())[0] if False else (res[k].data.c if hasattr(res[k].data, "c") else res[k].data.meas); counts.append(c.get_counts())
        out = {"block": blk, "S": S, "E": E}
        def pbit(cnt, i):                          # prawdopodobienstwo '1' na bicie i (klasyczny rejestr c[i])
            tot = sum(cnt.values()); return sum(v for k, v in cnt.items() if k[::-1][i] == "1") / tot
        if blk in ("TI", "TS"):
            kinds = {"T1": [], "T2": []}
            rows = {}
            for nm, cnt in zip(D["names"], counts): kind, d = nm.split("_"); rows.setdefault(kind.replace("s", ""), {})[int(d)] = [pbit(cnt, i) for i in range(len(S))]
            ts = np.array(DELAYS, float)
            for kind, tab in rows.items():
                Tm = []
                for i in range(len(S)):
                    y = np.array([tab[int(d)][i] for d in DELAYS])
                    try:
                        if kind == "T1": p, _ = curve_fit(lambda t, A, T: A * np.exp(-t / T), ts, y, p0=[0.95, 200], bounds=([0.5, 5], [1.0, 5000]))
                        else: p, _ = curve_fit(lambda t, B, T: 0.5 * (1 - B * np.exp(-t / T)), ts, y, p0=[0.9, 150], bounds=([0.3, 5], [1.0, 5000]))
                        Tm.append(float(p[1]))
                    except Exception: Tm.append(float("nan"))
                cal = [(tp[q].t1 if kind == "T1" else tp[q].t2) * 1e6 for q in S]; Tm = np.array(Tm); cal = np.array(cal); ok = np.isfinite(Tm)
                lab = {"T1": "T1", "T2": "T2echo" if blk == "TI" else "T2echo_sasiedzi_wzbudzeni"}[kind]
                print(f"{lab}: zmierzone mediana {np.nanmedian(Tm):.0f} us | producent {np.median(cal[ok]):.0f} us | stosunek {np.nanmedian(Tm[ok] / cal[ok]):.2f} (IQR {np.nanpercentile(Tm[ok]/cal[ok],25):.2f}-{np.nanpercentile(Tm[ok]/cal[ok],75):.2f}) n={ok.sum()}")
                out[lab] = dict(meas=Tm.tolist(), cal=cal.tolist())
        elif blk == "RO":
            qs = D["meta"]["qs"]; p1_0 = np.array([pbit(counts[0], i) for i in range(len(qs))]); p0_1 = 1 - np.array([pbit(counts[1], i) for i in range(len(qs))])
            err = (p1_0 + p0_1) / 2; cal = np.array([bk.target["measure"][(q,)].error for q in qs])
            print(f"odczyt: blad przypisania zmierzony mediana {np.median(err):.4f} | producent {np.median(cal):.4f} | stosunek {np.median(err / cal):.2f} | n={len(qs)}"); out["RO"] = dict(meas=err.tolist(), cal=cal.tolist())
        elif blk in ("R1", "R2"):
            L = D["meta"]["lengths"]; nseed = D["meta"]["seeds"]; nb = 1 if blk == "R1" else 2; units = len(S) if blk == "R1" else len(E)
            surv = np.zeros((units, len(L)))
            for li, m in enumerate(L):
                for s in range(nseed):
                    cnt = counts[li * nseed + s]; tot = sum(cnt.values())
                    for u in range(units):
                        bits = [nb * u + b for b in range(nb)]
                        surv[u, li] += sum(v for k, v in cnt.items() if all(k[::-1][b] == "0" for b in bits)) / tot / nseed
            eps = []
            for u in range(units):
                try:
                    p, _ = curve_fit(lambda m, A, p: decay(m, A, p, 1.0 / 2 ** nb), np.array(L, float), surv[u], p0=[0.9, 0.99], bounds=([0.3, 0.5], [1.0, 1.0])); eps.append((1 - p[1]) * (1 - 1 / 2 ** nb))
                except Exception: eps.append(float("nan"))
            eps = np.array(eps)
            if blk == "R1":
                cal = np.array([bk.target["sx"][(q,)].error for q in S]); gate = eps / 0.9375              # 1q Clifford ~ 0.9375 sx srednio (kompilacja Qiskit)
                print(f"1q RB: EPC mediana {np.nanmedian(eps):.2e} | blad bramki sx (EPC/0.94) {np.nanmedian(gate):.2e} | producent sx {np.median(cal):.2e} | stosunek {np.nanmedian(gate / cal):.2f}")
            else:
                cal = np.array([bk.target["cz"][tuple(e)].error for e in E]); gate = eps / 1.5                      # 2q Clifford ~ 1.5 CZ srednio
                print(f"2q RB: EPC mediana {np.nanmedian(eps):.2e} | blad CZ (EPC/1.5) {np.nanmedian(gate):.2e} | producent cz {np.median(cal):.2e} | stosunek {np.nanmedian(gate / cal):.2f}")
            out["rb"] = dict(eps=eps.tolist(), cal=cal.tolist(), surv=surv.tolist(), L=L)
        elif blk == "QV":
            from qiskit.circuit.library import QuantumVolume
            from qiskit.quantum_info import Statevector
            k = 0
            for n_ in D["meta"]["ns"]:
                hops = []
                for s in range(D["meta"]["ncirc"]):
                    c = QuantumVolume(n_, n_, seed=s); p = np.abs(Statevector(c).data) ** 2; med = np.median(p); heavy = {format(i, f"0{n_}b") for i in range(len(p)) if p[i] > med}
                    cnt = counts[k]; k += 1; tot = sum(cnt.values()); hops.append(sum(v for b_, v in cnt.items() if b_ in heavy) / tot)
                hops = np.array(hops); mu = hops.mean(); se = hops.std(ddof=1) / np.sqrt(len(hops)); lb = mu - 2 * se
                print(f"QV n={n_}: srednia HOP {mu:.3f} +- {2 * se:.3f} (2 sigma), dolna granica {lb:.3f} -> {'ZALICZONY (>2/3)' if lb > 2 / 3 else 'niezaliczony'}"); out[f"qv{n_}"] = dict(hop=mu, lb=lb)
        elif blk == "GH":
            for n, cnt in zip(D["meta"]["ns"], counts):
                tot = sum(cnt.values()); pg = (cnt.get("0" * n, 0) + cnt.get("1" * n, 0)) / tot; print(f"GHZ-{n}: populacja |0..0>+|1..1> = {pg:.3f}"); out[f"ghz{n}"] = pg
        json.dump(out, open(os.path.join(HERE, f"bench_{blk}.json"), "w")); print("zuzyto lacznie s", svc.usage().get("usage_consumed_seconds"))

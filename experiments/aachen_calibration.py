"""Read-only study of the current calibration of a backend (no QPU time): T1, T2, readout and CZ errors, bad couplers, and the best qubits for the experiments.
Writes fixtures/aachen_rerun/calibration.json (snapshot with the time of the calibration) and selection.json: best connected regions of 6..14 qubits (for the mirror
circuits, scored by readout error, idle decoherence over a typical circuit time and the CZ errors inside the region), the best independent set of qubits (T1, T2, echo T2,
readout; no two neighbours) and the best disjoint couplers (randomized benchmarking). Usage: aachen_calibration.py   (IBM_ACCOUNT, IBM_BACKEND, SPAN_US = circuit duration used for idling, default 20)"""
import json, os, sys, time, itertools
import numpy as np
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "aachen_rerun"); os.makedirs(HERE, exist_ok=True)
ACC, BK = os.environ.get("IBM_ACCOUNT", "eu-de-aachen-156"), os.environ.get("IBM_BACKEND", "ibm_aachen"); SPAN_US = float(os.environ.get("SPAN_US", "20"))

def snapshot(bk):
    t = bk.target; n = bk.num_qubits; pr = bk.properties()
    def get(f, i):
        try: return (f(i) or 0.0) * 1e6
        except Exception: return 0.0                                                       # no value in the calibration: the qubit is not usable
    q = {i: dict(t1=get(pr.t1, i), t2=get(pr.t2, i), ro=t["measure"][(i,)].error if (i,) in t["measure"] and t["measure"][(i,)] is not None and t["measure"][(i,)].error is not None else 1.0,
                 sx=t["sx"][(i,)].error if (i,) in t["sx"] and t["sx"][(i,)] is not None and t["sx"][(i,)].error is not None else 1.0) for i in range(n)}
    e = {f"{a}-{b}": v.error for (a, b), v in t["cz"].items() if v is not None and v.error is not None and a < b}
    e.update({f"{b}-{a}": v.error for (a, b), v in t["cz"].items() if v is not None and v.error is not None and a > b and f"{b}-{a}" not in e})
    return dict(backend=bk.name, taken=time.strftime("%Y-%m-%d %H:%M:%S"), calibration_time=str(getattr(pr, "last_update_date", "")), qubits=q, cz=e)

def qcost(q, d): return d["ro"] + SPAN_US / max(d["t2"], 1.0) * 0.5 + SPAN_US / max(d["t1"], 1.0) * 0.5      # readout + idle decoherence over the circuit span

def regions(S, adj, sizes, ecost):
    """Greedy growth from every seed plus swap refinement: connected region of k qubits with minimal sum of qubit costs and CZ errors of its internal couplers."""
    out = {}
    for k in sizes:
        best = (1e9, None)
        for seed in adj:
            R = [seed]; cost = qcost(0, S["q"][seed])
            while len(R) < k:
                cand = {n for r in R for n in adj[r] if n not in R}
                if not cand: break
                def inc(n): return qcost(0, S["q"][n]) + sum(ecost(n, r) for r in R if r in adj[n])
                n = min(cand, key=inc); cost += inc(n); R.append(n)
            if len(R) == k and cost < best[0]: best = (cost, R)
        out[k] = dict(cost=best[0], qubits=sorted(best[1]))
    return out

if __name__ == "__main__":
    from qiskit_ibm_runtime import QiskitRuntimeService
    bk = QiskitRuntimeService(name=ACC).backend(BK); snap = snapshot(bk); json.dump(snap, open(os.path.join(HERE, "calibration.json"), "w"), indent=1)
    Q = {int(i): d for i, d in snap["qubits"].items()}; cz = {tuple(map(int, k.split("-"))): v for k, v in snap["cz"].items()}
    bad = {i for i, d in Q.items() if d["t1"] <= 0 or d["t2"] <= 0 or d["ro"] >= 0.5}                  # qubits without T1/T2 or with a useless readout
    adj = {i: set() for i in Q if i not in bad}
    for (a, b), v in cz.items():
        if v < 0.5 and a not in bad and b not in bad: adj[a].add(b); adj[b].add(a)                                         # drop dead couplers (error ~1)
    dead = sorted(k for k, v in cz.items() if v >= 0.5)
    ecost = lambda a, b: cz.get((a, b), cz.get((b, a), 1.0))
    S = dict(q=Q); ok = [d for i, d in Q.items() if i not in bad]; t1 = np.array([d["t1"] for d in ok]); t2 = np.array([d["t2"] for d in ok]); ro = np.array([d["ro"] for d in ok]); ev = np.array([v for v in cz.values() if v < 0.5])
    print(f"{snap['backend']} {snap['taken']} (kalibracja: {snap['calibration_time']})")
    print(f"T1 [us]: mediana {np.median(t1):.0f}, p10 {np.percentile(t1,10):.0f}, p90 {np.percentile(t1,90):.0f}, max {t1.max():.0f}, <50: {int((t1<50).sum())} kubitów")
    print(f"T2 [us]: mediana {np.median(t2):.0f}, p10 {np.percentile(t2,10):.0f}, p90 {np.percentile(t2,90):.0f}, max {t2.max():.0f}, <50: {int((t2<50).sum())} kubitów")
    print(f"odczyt: mediana {np.median(ro):.2e}, p90 {np.percentile(ro,90):.2e} | CZ (bez martwych): n={len(ev)} mediana {np.median(ev):.2e}, p10 {np.percentile(ev,10):.2e}, p90 {np.percentile(ev,90):.2e}, max {ev.max():.2e}")
    print(f"martwe lub bardzo złe sprzężenia (błąd ≥ 0.5): {len(dead)} {dead[:10]}; złe (>1e-2): {sum(1 for v in ev if v > 1e-2)}")
    top = sorted((i for i in Q if i not in bad), key=lambda i: -(min(Q[i]['t1'], 400) + min(Q[i]['t2'], 400)))[:10]; print("10 najlepszych kubitów wg T1+T2 (obcięte do 400 us):", [(i, round(Q[i]['t1']), round(Q[i]['t2'])) for i in top])
    reg = regions(S, adj, (6, 8, 10, 12, 14), ecost)
    for k, r in reg.items(): print(f"region {k:>2} kubitów: koszt {r['cost']:.3f} | {r['qubits']}")
    # independent set for T1/T2/RB (as aachen_bench.pick) and disjoint couplers for 2q RB
    ind = []; blocked = set()
    for i in sorted(Q, key=lambda i: -(min(Q[i]['t1'], 400) + min(Q[i]['t2'], 400)) + 1e4 * Q[i]['ro']):
        if i in blocked or Q[i]['ro'] > 2e-2: continue
        ind.append(i); blocked |= {i} | adj[i]
        if len(ind) == 14: break
    used = set(ind) | {n for s in ind for n in adj[s]}; edges = []
    for (a, b), v in sorted(cz.items(), key=lambda kv: kv[1]):
        if v >= 0.5 or a in used or b in used: continue
        edges.append((a, b)); used |= {a, b} | adj[a] | adj[b]
        if len(edges) == 8: break
    print("zbiór niezależny (14):", sorted(ind)); print("rozłączne sprzężenia do RB 2q (8):", [(a, b, f"{cz[(a, b)]:.1e}") for a, b in edges])
    json.dump(dict(calibration_taken=snap["taken"], span_us=SPAN_US, regions={str(k): r for k, r in reg.items()}, bad_qubits=sorted(bad), independent=sorted(ind), couplers=[list(e) for e in edges], dead=[list(d) for d in dead]), open(os.path.join(HERE, "selection.json"), "w"), indent=1)

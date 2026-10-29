"""Analiza testu diagnostycznego: A (ten sam uklad), B (inny obszar), C (T1, echo Hahna)."""
import json, os, pickle
import numpy as np
from scipy.optimize import curve_fit
HERE = os.path.dirname(os.path.abspath(__file__))
J = json.load(open(os.path.join(HERE, "job.json"))); C = json.load(open(os.path.join(HERE, "counts.json")))
OLD = json.load(open(os.path.join(HERE, "counts_all.json")))  # counts of the 15 mirror circuits (job davcn5o4oijs73e7b5m0)
tags = J["tags"]; taus = np.array(J["taus_us"], float); QC = J["q_C"]
def p0(c): n = len(next(iter(c))); return c.get("0" * n, 0) / sum(c.values())
def marg(c):
    n = len(next(iter(c))); t = sum(c.values()); return [sum(v for s, v in c.items() if s[::-1][b] == "1") / t for b in range(n)]
out = {"A": [], "B": [], "C": {}}
print("A: ten sam uklad (obwody ISA z 01.10)")
for i, tg in enumerate(tags):
    c = C[i]; o = OLD[3 + i]; s = (p0(c) * (1 - p0(c)) / sum(c.values())) ** 0.5
    out["A"].append(dict(tag=tg, p0=p0(c), p0_01_10=p0(o), sigma=s, marg=marg(c), marg_01_10=marg(o)))
    print(f"  {tg:6s} P0={p0(c):.3f}±{s:.3f} (01.10: {p0(o):.3f})  marg={[round(x,2) for x in marg(c)]}  01.10={[round(x,2) for x in marg(o)]}")
print("B: inny obszar", J["layout_B"])
for i, tg in enumerate(tags):
    c = C[3 + i]; s = (p0(c) * (1 - p0(c)) / sum(c.values())) ** 0.5
    out["B"].append(dict(tag=tg, p0=p0(c), sigma=s, marg=marg(c)))
    print(f"  {tg:6s} P0={p0(c):.3f}±{s:.3f}  marg={[round(x,2) for x in marg(c)]}")
nT = len(taus); T1c = C[6:6 + nT]; Ec = C[6 + nT:6 + 2 * nT]
def pq(c, k, bit):  # P(bit k == bit)
    t = sum(c.values()); return sum(v for s, v in c.items() if s[::-1][k] == bit) / t
print("C: T1 i echo (us)")
P = pickle.load(open(os.path.join(HERE, "props_wysylka.pkl"), "rb")); qp = {q: {p["name"]: p["value"] for p in ps} for q, ps in enumerate(P["qubits"])}
for k, q in enumerate(QC):
    y1 = np.array([pq(c, k, "1") for c in T1c]); ye = np.array([pq(c, k, "0") for c in Ec])
    try:
        (a, T1, b), _ = curve_fit(lambda t, a, T, b: a * np.exp(-t / T) + b, taus, y1, p0=[0.9, 200, 0.02], bounds=([0, 1, -0.1], [1.2, 5000, 0.5]))
    except Exception as e: T1 = float("nan")
    try:
        (a2, T2, b2), _ = curve_fit(lambda t, a, T, b: a * np.exp(-t / T) + b, taus, ye, p0=[0.45, 100, 0.5], bounds=([0, 1, 0.3], [0.7, 5000, 0.7]))
    except Exception as e: T2 = float("nan")
    out["C"][q] = dict(T1_fit=float(T1), T2echo_fit=float(T2), T1_kal=qp[q]["T1"], T2_kal=qp[q]["T2"], y_T1=y1.tolist(), y_echo=ye.tolist())
    print(f"  q{q}: T1 zmierzone={T1:.0f} (kal {qp[q]['T1']:.0f}),  T2echo zmierzone={T2:.0f} (kal {qp[q]['T2']:.0f})   P1(tau)={np.round(y1,2).tolist()}  P0echo={np.round(ye,2).tolist()}")
json.dump(out, open(os.path.join(HERE, "analiza.json"), "w"), indent=1)

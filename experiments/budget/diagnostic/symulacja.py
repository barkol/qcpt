"""Dokladna symulacja macierzy gestosci wykonanych obwodow ISA (aktywne kubity fizyczne):
szeregowanie ALAP z czasami urzadzenia; po kazdej bramce depolaryzacja wg kalibracji (blad bramki
minus czesc relaksacyjna) + relaksacja T1/T2 w czasie bramki; w przerwach relaksacja T1/T2;
ZZ koherentny: faza 2*pi*zeta*dt na |11> dla sprzezonych par aktywnych, w odcinkach czasu;
opcjonalnie nadmiarowy blad lam na CZ (depolaryzacja 2q). Pomiar: bledy odczytu P(0|1), P(1|0).
Uzycie: symulacja.py [idx ...] [--bez-zz] [--zz-twirl] [--lam X]"""
import json, math, os, pickle, sys
import numpy as np
from qiskit import QuantumCircuit, qpy
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator
from qiskit_aer.noise import depolarizing_error, thermal_relaxation_error, pauli_error
HERE = os.path.dirname(os.path.abspath(__file__))
B = os.path.join(HERE, "..") + os.sep  # experiments/budget: kal_ibm_marrakesh.pkl, marrakesh_props_jobtime.pkl, lustro_isa.qpy
K = pickle.load(open(B + "kal_ibm_marrakesh.pkl", "rb")); P = pickle.load(open(B + "marrakesh_props_jobtime.pkl", "rb"))
T = K["target"]; circs = qpy.load(open(B + "lustro_isa.qpy", "rb"))
qp = {q: {p["name"]: p["value"] for p in ps} for q, ps in enumerate(P["qubits"])}
gerr = {}
for g in P["gates"]:
    e = [p["value"] for p in g["parameters"] if p["name"] == "gate_error"]
    if e: gerr[(g["gate"], tuple(g["qubits"]))] = e[0]
gen = {x["name"]: x["value"] for x in P["general"]}
def zeta(a, b):
    for k in (f"zz_{a}{b}", f"zz_{b}{a}"):
        if k in gen: return abs(gen[k]) * 1e9
    return 0.0
edges = {tuple(sorted(e)) for e in K["coupling"]}
def dur(name, qs):
    if name in ("rz", "barrier"): return 0.0
    p = T[name].get(tuple(qs)) or T[name].get(tuple(qs[::-1])); return p.duration


def schedule(c):
    ops = [(i.operation, [c.find_bit(q).index for q in i.qubits], [c.find_bit(b).index for b in i.clbits]) for i in c.data]
    cur = {}; sched = []
    for op, qs, cs in reversed(ops):
        if op.name == "barrier":
            t = max(cur.get(q, 0.0) for q in qs)
            for q in qs: cur[q] = t
            continue
        d = 0.0 if op.name == "measure" else dur(op.name, qs)
        t0 = max(cur.get(q, 0.0) for q in qs)
        for q in qs: cur[q] = t0 + d
        sched.append((op, qs, cs, t0, d))
    Tend = max(cur.values())
    return [(op, qs, cs, Tend - (t0 + d), d) for op, qs, cs, t0, d in reversed(sched)], Tend


OVR = {}
def relax(q, t):
    T1 = OVR.get((q, "T1"), qp[q]["T1"]) * 1e-6; T2 = min(OVR.get((q, "T2"), qp[q]["T2"]) * 1e-6, 2 * T1)
    return thermal_relaxation_error(T1, T2, t)


def build(c, with_zz=True, zz_twirl=False, lam=0.0, cz_phase=0.0):
    sched, Tend = schedule(c)
    act = sorted({q for op, qs, cs, t0, d in sched for q in qs})
    idx = {q: i for i, q in enumerate(act)}; n = len(act)
    pairs = [(a, b) for a, b in edges if a in idx and b in idx]
    meas = [(qs[0], cs[0]) for op, qs, cs, t0, d in sched if op.name == "measure"]
    first = {}
    for op, qs, cs, t0, d in sched:
        if op.name != "measure":
            for q in qs: first.setdefault(q, t0)
    out = QuantumCircuit(n)
    last = {q: first.get(q, 0.0) for q in act}   # czas, do ktorego kubit ma juz uwzgledniona relaksacje
    zz_t = 0.0                                   # czas, do ktorego naliczono ZZ
    def zz_until(t):
        nonlocal zz_t
        dt = t - zz_t
        if dt <= 0 or not with_zz: zz_t = max(zz_t, t); return
        for a, b in pairs:
            if a in first and b in first and min(first[a], first[b]) <= zz_t:
                if first[a] > zz_t or first[b] > zz_t: continue
                phi = 2 * math.pi * zeta(a, b) * dt
                if zz_twirl:
                    p = 0.75 * math.sin(phi / 2) ** 2
                    out.append(pauli_error([("ZZ", p / 3), ("ZI", p / 3), ("IZ", p / 3), ("II", 1 - p)]).to_instruction(), [idx[a], idx[b]])
                else:
                    out.append(UnitaryGate(np.diag([1, 1, 1, np.exp(1j * phi)])), [idx[a], idx[b]])
        zz_t = t
    for op, qs, cs, t0, d in sched:
        if op.name == "measure": continue
        zz_until(t0)
        for q in qs:
            gap = t0 - last[q]
            if gap > 1e-12: out.append(relax(q, gap).to_instruction(), [idx[q]])
        out.append(op, [idx[q] for q in qs])
        if cz_phase and op.name == "cz":
            out.append(UnitaryGate(np.diag([1, 1, 1, np.exp(1j * cz_phase)])), [idx[q] for q in qs])
        if d > 0:
            e = gerr.get((op.name, tuple(qs))) or gerr.get((op.name, tuple(qs[::-1]))) or 0.0
            if len(qs) == 2:
                # depolaryzacja tak, by razem z relaksacja dac blad kalibracji (przyblizenie: odejmujemy relaksacje)
                r = sum(1 - (1 - math.exp(-d / (qp[q]["T1"] * 1e-6))) / 3 for q in qs) / 2
                p_rel = 1 - r
                p_dep = max(0.0, e - p_rel) * 4 / 3 * (16 / 15)
                if p_dep > 0: out.append(depolarizing_error(min(p_dep, 1.0), 2).to_instruction(), [idx[q] for q in qs])
                if lam > 0: out.append(depolarizing_error(min(lam * 16 / 15, 1.0), 2).to_instruction(), [idx[q] for q in qs])
            else:
                p_dep = e * 2 * 3 / 2 / 1.0 * 0.5 * 4 / 3
                if p_dep > 0: out.append(depolarizing_error(min(p_dep, 1.0), 1).to_instruction(), [idx[qs[0]]])
            for q in qs: out.append(relax(q, d).to_instruction(), [idx[q]])
        for q in qs: last[q] = t0 + d
    tm = max(t0 for op, qs, cs, t0, d in sched if op.name == "measure")
    zz_until(tm)
    for q in act:
        gap = tm - last[q]
        if gap > 1e-12 and q in first: out.append(relax(q, gap).to_instruction(), [idx[q]])
    out.save_density_matrix()
    return out, act, idx, meas


def p0_and_marginals(c, **kw):
    out, act, idx, meas = build(c, **kw)
    rho = AerSimulator(method="density_matrix").run(out).result().data()["density_matrix"]
    probs = np.real(np.diag(np.asarray(rho)))
    n = len(act); nb = len(meas)
    # rozklad na bitach klasycznych z bledem odczytu
    dist = np.zeros(2 ** nb)
    for s in range(2 ** n):
        if probs[s] < 1e-14: continue
        bits = 0
        for q, cbit in meas:
            if (s >> idx[q]) & 1: bits |= 1 << cbit
        dist[bits] += probs[s]
    for q, cbit in meas:   # blad odczytu na bicie cbit
        p01 = qp[q].get("prob_meas1_prep0", 0.0); p10 = qp[q].get("prob_meas0_prep1", 0.0)
        new = np.zeros_like(dist)
        for b in range(len(dist)):
            if dist[b] == 0: continue
            if (b >> cbit) & 1: new[b] += dist[b] * (1 - p10); new[b ^ (1 << cbit)] += dist[b] * p10
            else: new[b] += dist[b] * (1 - p01); new[b ^ (1 << cbit)] += dist[b] * p01
        dist = new
    marg = [float(sum(dist[b] for b in range(len(dist)) if (b >> k) & 1)) for k in range(nb)]
    return float(dist[0]), marg, len(act)


if __name__ == "__main__":
    args = sys.argv[1:]; lam = 0.0
    if "--lam" in args: lam = float(args[args.index("--lam") + 1])
    ids = [int(a) for a in args if a.isdigit()] or list(range(15))
    kw = dict(with_zz="--bez-zz" not in args, zz_twirl="--zz-twirl" in args, lam=lam)
    W = json.load(open(os.path.join(HERE, "..", "..", "fixtures", "mirror_test", "wyniki_hw.json")))  # = paper/tables/data/mirror_hw.json
    out = []
    for k in ids:
        p0, marg, na = p0_and_marginals(circs[k], **kw)
        r = W["rows"][k]; out.append(dict(k=k, name=r["name"], tag=r["tag"], hw=r["p0_hw"], sim=p0, marg=marg, n_active=na))
        print(f"{r['name'][:20]:20s} {r['tag']:6s} aktywnych={na:2d} hw={r['p0_hw']:.3f} sim={p0:.3f}  marg=" + " ".join(f"{m:.2f}" for m in marg), flush=True)
    tag = ("zz_twirl" if kw["zz_twirl"] else ("zz" if kw["with_zz"] else "bez_zz")) + (f"_lam{lam}" if lam else "")
    json.dump(out, open(os.path.join(HERE, f"sym_{tag}.json"), "w"), indent=1)

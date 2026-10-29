"""Model of the cutoff ranking of the budgeted Draper adder (offline, no QPU): ideal success probability (all inputs) x exp(-C), C = calibrated cost sum(-ln(1-eps_e)) of the circuit routed on ibm_aachen,
mean over eight transpiler seeds (as ~/git/toptoffoli-fix/experiments/qft_adder_approx.py, extended to n = 4, 5, 6 and an ideal success for every n). Two routings: (a) the whole chip, (b) the probed region of 2n qubits
(induced healthy coupling map). Calibration: the live target of ibm_aachen (checked against fixtures/aachen_next/calibration.json)."""
import os, sys, json, math, time
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit.transpiler import CouplingMap
from qiskit_ibm_runtime import QiskitRuntimeService
sys.path.insert(0, "/tmp/sq4/experiments"); sys.path.insert(0, "/tmp/sq4")
FX = "/tmp/sq4/experiments/fixtures/"
bk = QiskitRuntimeService(name="eu-de-aachen-156").backend("ibm_aachen"); T = bk.target
print("calibration of the live target:", bk.properties().last_update_date, "| fixture:", json.load(open(FX + "aachen_next/calibration.json"))["calibration_time"], flush=True)
def cost(t, phys=None):
    tot = 0.0
    for i in t.data:
        q = tuple(t.find_bit(x).index for x in i.qubits); nm = i.operation.name
        if nm in ("measure", "barrier"): continue
        if phys is not None: q = tuple(phys[x] for x in q)
        if nm in T.operation_names:
            p = T[nm].get(q)
            if p is not None and p.error is not None: tot += -math.log(1 - min(p.error, 0.5))
    return tot
CAL = json.load(open(FX + "aachen_next/calibration.json"))
CZ0 = {}
for k_, v_ in CAL["cz"].items():
    a_, b_ = map(int, k_.split("-")); CZ0[(a_, b_)] = CZ0[(b_, a_)] = v_
SX0 = {int(q): d["sx"] for q, d in CAL["qubits"].items()}
def cost0(t, phys=None):
    """cost with the calibration of 29 October 03:20 (fixture): cz from the coupler table, sx and x from the sx error of the qubit (x has the same error in the calibration), rz free"""
    tot = 0.0
    for i in t.data:
        q = tuple(t.find_bit(x).index for x in i.qubits); nm = i.operation.name
        if nm in ("measure", "barrier", "rz"): continue
        if phys is not None: q = tuple(phys[x] for x in q)
        e = CZ0.get(q) if nm == "cz" else (SX0.get(q[0]) if nm in ("sx", "x") else None)
        if e is not None: tot += -math.log(1 - min(e, 0.5))
    return tot
def adder(n, m):
    qc = QuantumCircuit(2 * n)
    def qft(qs, inverse=False):
        ops = []
        for j in range(len(qs) - 1, -1, -1):
            ops.append(("h", qs[j]))
            for k in range(j - 1, -1, -1):
                d = j - k + 1
                if d <= m: ops.append(("cp", qs[j], qs[k], math.pi / 2 ** (d - 1)))
        for op in (reversed(ops) if inverse else ops):
            if op[0] == "h": qc.h(op[1])
            else: qc.cp(-op[3] if inverse else op[3], op[1], op[2])
    b = list(range(n, 2 * n)); qft(b)
    for j in range(n):
        for k in range(n):
            d = k - j + 1
            if 1 <= d <= m: qc.cp(2 * math.pi / 2 ** d, j, b[k])
    qft(b, inverse=True); return qc
def ideal_success(qc, n):
    N = 1 << n; U = Operator(qc).data; P = (np.abs(U) ** 2).reshape(N, N, N * N).sum(axis=1)       # [b_out, input] summed over the a register (qubits 0..n-1 = low bits)
    ins = np.arange(N * N); a = ins % N; b = ins // N
    return float(P[(a + b) % N, ins].mean())
SEL = json.load(open(FX + "aachen_draper/selection.json")); PROBE = json.load(open(FX + "aachen_probe5/probe.json"))["coupler"]
badc = {tuple(sorted(map(int, k.split("-")))) for k, v in PROBE.items() if v["err_per_cz"] > 0.012}
alive = {tuple(sorted(k)) for k, v in T["cz"].items() if v is not None and v.error is not None and v.error < 0.5}
BASIS = ["cz", "sx", "rz", "x"]; out = []
for n in (4, 5, 6):
    reg = sorted(SEL["regions"][str(2 * n)]["qubits"]); pos = {q: i for i, q in enumerate(reg)}
    ed = sorted(e for e in alive if e[0] in pos and e[1] in pos and e not in badc)
    CM = CouplingMap([(pos[a], pos[b]) for a, b in ed] + [(pos[b], pos[a]) for a, b in ed])
    for m in range(1, n + 1):
        qc = adder(n, m); t0 = time.time(); pid = ideal_success(qc, n); r = dict(n=n, m=m, exact=(m == n), p_ideal=pid, logical_cx=2 * sum(1 for i in qc.data if i.operation.name == "cp"))
        for mode in ("chip", "region"):
            cs, cz, c0 = [], [], []
            for seed in range(8):
                if mode == "chip": t = transpile(qc, bk, optimization_level=3, seed_transpiler=seed); cs.append(cost(t)); c0.append(cost0(t))
                else: t = transpile(qc, coupling_map=CM, basis_gates=BASIS, optimization_level=3, seed_transpiler=seed); cs.append(cost(t, phys=reg)); c0.append(cost0(t, phys=reg))
                cz.append(sum(1 for i in t.data if i.operation.num_qubits == 2))
            r[mode] = dict(cost=float(np.mean(cs)), cost_std=float(np.std(cs)), routed_cx=float(np.mean(cz)), model=pid * math.exp(-float(np.mean(cs))),
                           cost_cal0329=float(np.mean(c0)), model_cal0329=pid * math.exp(-float(np.mean(c0))))
        r["region_qubits"] = reg; out.append(r); print(json.dumps({k: v for k, v in r.items() if k != "region_qubits"}), "%.0fs" % (time.time() - t0), flush=True)
summ = {}
for n in (4, 5, 6):
    for mode in ("chip", "region"):
        rows = [r for r in out if r["n"] == n]; ex = [r for r in rows if r["exact"]][0]
        for lab, key in (("cal0320", "model_cal0329"), ("live1527", "model")):
            best = max(rows, key=lambda r: r[mode][key])
            summ[f"n{n}_{mode}_{lab}"] = dict(best_m=best["m"], best_model=best[mode][key], exact_model=ex[mode][key], by_m={r["m"]: round(r[mode][key], 4) for r in rows})
json.dump(dict(rows=out, summary=summ, calibration=str(bk.properties().last_update_date)), open("/tmp/forkB_out/qft_ranking.json", "w"), indent=1)
for k, v in summ.items(): print(k, "best m =", v["best_m"], "model %.3f" % v["best_model"], "exact %.3f" % v["exact_model"], v["by_m"])

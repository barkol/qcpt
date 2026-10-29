import os, sys
_R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # repository root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_O = os.environ.get("SIM_OUT", os.path.join(_R, "experiments", "sim_2026-10-29", "out")); os.makedirs(_O, exist_ok=True)      # outputs
"""Step 5b: density-matrix simulation (calibration model of 29 Oct 2026, exact probabilities) of both compilations of the 33 circuits (items33.pkl) and the numbers of the paragraph "Simulation, 33 circuits"."""
import json, os, sys, pickle
import numpy as np
sys.path[:0] = [_O, _R, _R + "/experiments"]
import simlib as S
OUT = S.OUT; tg = S.get_target(); items = pickle.load(open(os.path.join(OUT, "items33.pkl"), "rb")); MAXQ = 12
by = {}
for it in items: by.setdefault(it["name"], {})[it["tag"]] = it
jobs = []   # (name, label, item)
for nm, v in sorted(by.items()):
    q = v["qiskit"]; src = q.get("qct_final", "qiskit_fallback"); fin = q if src == "qiskit_fallback" else v[src]
    jobs.append((nm, "qiskit", q, "qiskit")); jobs.append((nm, "qct", fin, "qiskit_fallback" if src == "qiskit_fallback" else src))
todo = [j for j in jobs if len(j[2]["phys"]) <= MAXQ and not (j[1] == "qct" and j[3] == "qiskit_fallback")]
p, a = S.predict(tg, [(j[2]["t"], j[2].get("data", by[j[0]]["qiskit"].get("data", []))) for j in todo]); res = {(j[0], j[1]): (float(x), float(y)) for j, x, y in zip(todo, p, a)}
rows = []
for nm, label, it, src in jobs:
    row = dict(key=f"{nm}/{label}", name=nm, variant=label, source=src if label == "qct" else "qiskit", p0_ideal=it["p0_ideal"], esp_pred=it["esp_pred"], twoq=it["twoq_t"], active_qubits=len(it["phys"]), shots=0)
    if len(it["phys"]) > MAXQ: row.update(p0_sim=None, note=f"{len(it['phys'])} active qubits")
    else:
        key = (nm, "qiskit") if (label == "qct" and src == "qiskit_fallback") else (nm, label); p0, ac = res[key]
        row.update(p0_sim=p0, sigma=0.0, accept=ac, p_cond=(p0 / ac if ac > 0 else None), n_ancilla=len(it["phys"]) - len(by[nm]["qiskit"].get("data", [])))
    rows.append(row)
json.dump(rows, open(os.path.join(OUT, "sim33_new.json"), "w"), indent=1)
g = {}
for r in rows:
    if r.get("p0_sim") is not None: g.setdefault(r["name"], {})[r["variant"]] = r
names = sorted(n for n in g if "qiskit" in g[n] and "qct" in g[n]); gain = np.array([g[n]["qct"]["p0_sim"] - g[n]["qiskit"]["p0_sim"] for n in names]); sv = np.array([g[n]["qiskit"]["twoq"] - g[n]["qct"]["twoq"] for n in names], float)
src = {}
for r in rows:
    if r["variant"] == "qct": src.setdefault(r["source"], []).append(r["name"])
tq = sum(r["twoq"] for r in rows if r["variant"] == "qiskit"); tc = sum(r["twoq"] for r in rows if r["variant"] == "qct")
top = names[int(np.argmax(gain))]
N = dict(n_circuits=len(set(r["name"] for r in rows)), n_simulated=len(names), n_better_001=int((gain > 0.01).sum()), n_worse_001=int((gain < -0.01).sum()), mean_gain=float(gain.mean()), max_gain=float(gain.max()), max_gain_circuit=top, min_gain=float(gain.min()), min_gain_circuit=names[int(np.argmin(gain))],
         source_counts={k: len(v) for k, v in src.items()}, source_names={k: v for k, v in src.items() if k == "qiskit_fallback"}, total_2q_qcpt=int(tc), total_2q_qiskit=int(tq), total_2q_reduction_pct=float(100 * (tq - tc) / tq),
         slope_per_gate=float(np.sum(sv * gain) / np.sum(sv * sv)), skipped=[r["name"] + "/" + r["variant"] for r in rows if r.get("p0_sim") is None])
json.dump(N, open(os.path.join(OUT, "numbers_sim33.json"), "w"), indent=1); print(json.dumps(N, indent=1))

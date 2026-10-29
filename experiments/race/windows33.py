#!/usr/bin/env python3
"""Windowed resynthesis on all 33 circuits, starting from the race33 result (gates.json) or, where the race found nothing, from the circuit itself.
Resumable: one json per circuit in windows33/. Every improved circuit is verified against the exact circuit. Usage: windows33.py [TOTAL_S]"""
import os, sys, time, json
sys.path.insert(0, os.environ.get("QCPT_RUNS", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "runs")))
import numpy as np
import portfolio as PF
from toffoli_optimizer.core import resynth as RS
from toffoli_optimizer.core.decomposition_selector import ErrorBudgetSelector
from toffoli_optimizer.core.subspace_check import circuit_columns
os.environ["QCPT_GPU"] = "0"; tot = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0
R = os.environ.get("QCPT_RUNS", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "runs")); OUTD = os.environ.get("WIN_OUT", "windows33"); os.makedirs(R + "/" + OUTD, exist_ok=True)
cost = lambda G: sum(RS.gate_cost(g[0]) for g in G)
for ln in open(R + "/race33/circuits.txt"):
    name = ln.split()[0]; out = f"{R}/{OUTD}/{name}.json"
    if os.path.exists(out) and os.path.getsize(out) > 0: continue
    qc, pins, n, inputs = PF.load(name)
    gj = f"{R}/race33/qasm/{name}.gates.json"; src = "race"
    if os.path.exists(gj): G = [tuple([g[0], tuple(g[1])] + ([tuple(g[2])] if len(g) > 2 else [])) for g in json.load(open(gj))]
    else: src = "circuit"; G = [(i.operation.name, tuple(qc.find_bit(x).index for x in i.qubits)) for i in qc.data if i.operation.name != "barrier"]
    c0 = cost(G); t0 = time.time(); new = RS.window_resynthesize(n, G, k_max=4, total_s=tot); wall = time.time() - t0; c1 = cost(new)
    res = dict(circuit=name, source=src, cost_in=c0, cost_out=c1, wall_s=round(wall, 1), improved=c1 < c0)
    if c1 < c0 or src == "circuit":
        E = ErrorBudgetSelector().select(qc, pinned_zero=pins)["exact"]; UE = circuit_columns(E, inputs); C = circuit_columns(RS.emit(n, new), inputs)
        ph = np.vdot(UE.ravel(), C.ravel()); res["dev"] = float(np.abs(C - (ph / abs(ph)) * UE).max()); res["gates"] = [[g[0], list(g[1])] + ([list(g[2])] if len(g) > 2 else []) for g in new]
    json.dump(res, open(out, "w")); print(json.dumps({k: v for k, v in res.items() if k != "gates"}), flush=True)
open(R + "/" + OUTD + "/DONE", "w").write("done")

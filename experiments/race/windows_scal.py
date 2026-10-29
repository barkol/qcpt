#!/usr/bin/env python3
"""Windowed resynthesis on the new scaling circuits, from the race result (gates.json) or the stage-1 circuit; then freezes all 12 circuits of the four families to fixtures/nighthawk/frozen_scaling.json (the cheapest verified list of race, windows and, where neither helps, stage 1)."""
import os, sys, json
sys.path[:0] = [os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "experiments"), os.environ.get("QCPT_RUNS", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "runs"))]
import numpy as np
import portfolio as PF
from toffoli_optimizer.core import resynth as RS
from toffoli_optimizer.core.decomposition_selector import ErrorBudgetSelector
from toffoli_optimizer.core.subspace_check import circuit_columns
os.environ["QCPT_GPU"] = "0"; R = os.environ.get("QCPT_RUNS", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "runs")); OUT = R + "/scal/windows"; os.makedirs(OUT, exist_ok=True)
NEW = ["half_uncomputed_2c", "half_uncomputed_3c", "mod_incr_4b", "mod_incr_5b", "ripple_adder_3b", "ripple_adder_4b", "grover_oracle_5c"]
OLD = ["half_uncomputed_4c", "mod_incr_3b", "ripple_carry_adder_2b", "grover_oracle_mcx3", "grover_oracle_mcx4"]
cost = lambda G: sum(RS.gate_cost(g[0]) for g in G); tup = lambda G: [tuple([g[0], tuple(g[1])] + ([tuple(g[2])] if len(g) > 2 else [])) for g in G]
lst = lambda G: [[g[0], list(g[1])] + ([list(g[2])] if len(g) > 2 else []) for g in G]
def verify(qc, pins, n, inputs, G):
    E = ErrorBudgetSelector().select(qc, pinned_zero=pins)["exact"]; UE = circuit_columns(E, inputs); C = circuit_columns(RS.emit(n, G), inputs); ph = np.vdot(UE.ravel(), C.ravel()); return float(np.abs(C - (ph / abs(ph)) * UE).max())
frozen = {}; old = json.load(open(os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "experiments/fixtures/nighthawk/frozen_circuits.json")))
for nm in OLD: frozen[nm] = old[nm]
for nm in NEW:
    qc, pins, n, inputs = PF.load(nm); gj = f"{R}/race33_new/qasm/{nm}.gates.json"; cand = []
    S1 = [(i.operation.name, tuple(qc.find_bit(x).index for x in i.qubits)) for i in qc.data if i.operation.name != "barrier"]      # no race result: windows start from the circuit itself (as windows33.py)
    if os.path.exists(gj): cand.append(("race", tup(json.load(open(gj)))))
    base = cand[0][1] if cand else S1
    new = RS.window_resynthesize(n, base, k_max=4, total_s=float(os.environ.get("WIN_S", "120")))
    if cost(new) < cost(base) or not cand: cand.append(("windows", new))
    src, G = min(cand, key=lambda c: cost(c[1])); dev = verify(qc, pins, n, inputs, G)
    assert dev < 1e-9, (nm, dev)
    frozen[nm] = dict(source=src, cost=cost(G), gates=lst(G)); print(nm, "stage1-like", cost(base), "->", cost(G), src, "dev", dev, flush=True)
json.dump(frozen, open(os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "experiments/fixtures/nighthawk/frozen_scaling.json"), "w"), indent=1); print("frozen", {k: v["cost"] for k, v in frozen.items()})

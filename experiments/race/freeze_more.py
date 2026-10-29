"""Freezes more sizes of the scaling families into frozen_scaling.json: the cheapest verified circuit among the race result, windowed resynthesis (from the race result or the circuit itself) and stage 1 (stored as qasm)."""
import os, sys, json
sys.path[:0] = [os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "experiments"), os.environ.get("QCPT_RUNS", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "runs"))]
import numpy as np
import portfolio as PF
from qiskit import qasm2
from toffoli_optimizer.core import resynth as RS
from toffoli_optimizer.core.decomposition_selector import ErrorBudgetSelector
from toffoli_optimizer.core.subspace_check import circuit_columns
os.environ["QCPT_GPU"] = "0"; P = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "experiments/fixtures/nighthawk/frozen_scaling.json"); F = json.load(open(P))
cost = lambda G: sum(RS.gate_cost(g[0]) for g in G); tup = lambda G: [tuple([g[0], tuple(g[1])] + ([tuple(g[2])] if len(g) > 2 else [])) for g in G]
lst = lambda G: [[g[0], list(g[1])] + ([list(g[2])] if len(g) > 2 else []) for g in G]
def dev_of(UE, C): ph = np.vdot(UE.ravel(), C.ravel()); return float(np.abs(C - (ph / abs(ph)) * UE).max())
for nm in sys.argv[1:]:
    qc, pins, n, inputs = PF.load(nm); sel = ErrorBudgetSelector().select(qc, pinned_zero=pins); UE = circuit_columns(sel["exact"], inputs); s1 = sel["circuit"]; c1 = s1.count_ops().get("cx", 0)
    gj = os.path.join(os.environ.get("QCPT_RUNS", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "runs")), f"race33_new/qasm/{nm}.gates.json"); base = tup(json.load(open(gj))) if os.path.exists(gj) else [(i.operation.name, tuple(qc.find_bit(x).index for x in i.qubits)) for i in qc.data if i.operation.name != "barrier"]
    cands = []
    if os.path.exists(gj): cands.append(("race", base))
    w = RS.window_resynthesize(n, base, k_max=4, total_s=float(os.environ.get("WIN_S", "90")))
    if cost(w) < cost(base) or not cands: cands.append(("windows", w))
    src, G = min(cands, key=lambda c: cost(c[1])); d = dev_of(UE, circuit_columns(RS.emit(n, G), inputs)); assert d < 1e-9, (nm, d)
    entry = dict(source=src, cost=cost(G), gates=lst(G))
    if c1 < entry["cost"]:
        d1 = dev_of(UE, circuit_columns(s1, inputs)); assert d1 < 1e-9, (nm, d1); entry.update(source="stage1", cost_gates=entry["cost"], cost=c1, qasm=qasm2.dumps(s1))
    F[nm] = entry; print(nm, "race/windows", cost(G), src, "stage1", c1, "->", entry["source"], entry["cost"], flush=True)
json.dump(F, open(P, "w"), indent=1)

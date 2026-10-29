"""Reachable-context windows on one circuit, result saved as json (certified against the exact circuit) in scal/reach/NAME.json. Usage: reach_run.py NAME K BEAM TOTAL_S"""
import sys, os, json, time
sys.path[:0] = [os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "experiments"), os.environ.get("QCPT_RUNS", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "runs"))]
import numpy as np
import portfolio as PF
from toffoli_optimizer.core import resynth as RS
from toffoli_optimizer.core.window_reach import window_resynthesize_reach
from toffoli_optimizer.core.decomposition_selector import ErrorBudgetSelector
from toffoli_optimizer.core.subspace_check import circuit_columns
nm, K, B, T = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]); qc, pins, n, inputs = PF.load(nm); os.makedirs(os.path.join(os.environ.get("QCPT_RUNS", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "runs")), "scal/reach"), exist_ok=True)
out = os.path.join(os.environ.get("QCPT_RUNS", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "runs")), f"scal/reach/{nm}.json")
G = [(i.operation.name, tuple(qc.find_bit(x).index for x in i.qubits)) for i in qc.data if i.operation.name != "barrier"]
if os.path.exists(out): G = [(g[0], tuple(g[1])) for g in json.load(open(out))["gates"]]          # continue from the previous result
c0 = sum(RS.gate_cost(g[0]) for g in G); t0 = time.time(); new = window_resynthesize_reach(n, G, inputs, k_max=K, total_s=T, beam=B)
c1 = sum(RS.gate_cost(g[0]) for g in new); E = ErrorBudgetSelector().select(qc, pinned_zero=pins)["exact"]; UE = circuit_columns(E, inputs); C = circuit_columns(RS.emit(n, new), inputs); ph = np.vdot(UE.ravel(), C.ravel()); dev = float(np.abs(C - (ph / abs(ph)) * UE).max())
assert dev < 1e-9, dev
json.dump(dict(circuit=nm, k_max=K, beam=B, cost_in=c0, cost=c1, dev=dev, seconds=round(time.time() - t0, 1), gates=[[g[0], list(g[1])] for g in new]), open(out, "w")); print(nm, "K", K, "beam", B, ":", c0, "->", c1, "dev", dev, "time", round(time.time() - t0, 1), flush=True)

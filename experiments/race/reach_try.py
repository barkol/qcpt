import sys, os, json, time
sys.path[:0] = [os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "experiments"), os.environ.get("QCPT_RUNS", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "runs"))]
import numpy as np
import portfolio as PF
from toffoli_optimizer.core import resynth as RS
from toffoli_optimizer.core.window_reach import window_resynthesize_reach
from toffoli_optimizer.core.decomposition_selector import ErrorBudgetSelector
from toffoli_optimizer.core.subspace_check import circuit_columns
nm = sys.argv[1]; K = int(sys.argv[2]); T = float(sys.argv[3]); qc, pins, n, inputs = PF.load(nm)
G = [(i.operation.name, tuple(qc.find_bit(x).index for x in i.qubits)) for i in qc.data if i.operation.name != "barrier"]
c0 = sum(RS.gate_cost(g[0]) for g in G); t0 = time.time(); new = window_resynthesize_reach(n, G, inputs, k_max=K, total_s=T, beam=int(os.environ.get("WBEAM", "3000")), verbose=False)
c1 = sum(RS.gate_cost(g[0]) for g in new)
E = ErrorBudgetSelector().select(qc, pinned_zero=pins)["exact"]; UE = circuit_columns(E, inputs); C = circuit_columns(RS.emit(n, new), inputs); ph = np.vdot(UE.ravel(), C.ravel()); dev = float(np.abs(C - (ph / abs(ph)) * UE).max())
print(nm, "k_max", K, ":", c0, "->", c1, "dev", dev, "time", round(time.time() - t0, 1))

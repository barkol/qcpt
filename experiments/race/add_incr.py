"""Adds mod_incr_2b and mod_incr_6b (race result, verified against the exact circuit) to frozen_scaling.json."""
import os, sys, json
sys.path[:0] = [os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "experiments"), os.environ.get("QCPT_RUNS", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "runs"))]
import numpy as np
import portfolio as PF
from toffoli_optimizer.core import resynth as RS
from toffoli_optimizer.core.decomposition_selector import ErrorBudgetSelector
from toffoli_optimizer.core.subspace_check import circuit_columns
P = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "experiments/fixtures/nighthawk/frozen_scaling.json"); F = json.load(open(P))
for nm in ("mod_incr_2b", "mod_incr_6b"):
    qc, pins, n, inputs = PF.load(nm); g = json.load(open(os.path.join(os.environ.get("QCPT_RUNS", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "runs")), f"race33_new/qasm/{nm}.gates.json"))); G = [tuple([x[0], tuple(x[1])] + ([tuple(x[2])] if len(x) > 2 else [])) for x in g]
    E = ErrorBudgetSelector().select(qc, pinned_zero=pins)["exact"]; UE = circuit_columns(E, inputs); C = circuit_columns(RS.emit(n, G), inputs); ph = np.vdot(UE.ravel(), C.ravel()); dev = float(np.abs(C - (ph / abs(ph)) * UE).max()); assert dev < 1e-9, dev
    F[nm] = dict(source="race", cost=sum(RS.gate_cost(x[0]) for x in G), gates=[list(x) if False else [x[0], list(x[1])] + ([list(x[2])] if len(x) > 2 else []) for x in G]); print(nm, F[nm]["cost"], "dev", dev)
json.dump(F, open(P, "w"), indent=1)

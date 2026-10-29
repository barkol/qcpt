"""Where stage 1 alone is cheaper than race+windows (windows start from the exact circuit and miss the relative-phase Toffolis), store the verified stage-1 circuit (qasm) in frozen_scaling.json as the QCPT circuit."""
import os, sys, json
sys.path[:0] = [os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "experiments"), os.environ.get("QCPT_RUNS", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "runs"))]
import numpy as np
import portfolio as PF
from qiskit import qasm2
from toffoli_optimizer.core.decomposition_selector import ErrorBudgetSelector
from toffoli_optimizer.core.subspace_check import circuit_columns
P = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "experiments/fixtures/nighthawk/frozen_scaling.json"); F = json.load(open(P))
for nm in ("ripple_adder_3b", "ripple_adder_4b", "grover_oracle_5c", "half_uncomputed_2c", "half_uncomputed_3c", "mod_incr_4b", "mod_incr_5b"):
    qc, pins, n, inputs = PF.load(nm); sel = ErrorBudgetSelector().select(qc, pinned_zero=pins); s1 = sel["circuit"]; c1 = s1.count_ops().get("cx", 0)
    if c1 < F[nm]["cost"]:
        UE = circuit_columns(sel["exact"], inputs); C = circuit_columns(s1, inputs); ph = np.vdot(UE.ravel(), C.ravel()); dev = float(np.abs(C - (ph / abs(ph)) * UE).max()); assert dev < 1e-9, (nm, dev)
        F[nm].update(source="stage1", cost_gates=F[nm]["cost"], cost=c1, qasm=qasm2.dumps(s1)); print(nm, "stage 1:", c1, "CX instead of", F[nm]["cost_gates"], "dev", dev)
json.dump(F, open(P, "w"), indent=1)

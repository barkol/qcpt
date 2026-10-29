"""Row (b) re-done in SUBROUTINE semantics: conditions (C) structural, (W), (R) only,
every whole-circuit check disabled (per-site (C) verify + final certificate).
Audit: dense U_sel P_in = e^{i th} U_ex P_in, NO wire permutation."""
import sys, os, numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "experiments"))
import benchmarks as B
from qiskit.quantum_info import Operator
import toffoli_optimizer.core.decomposition_selector as DS
from toffoli_optimizer.core.equivalence_verifier import ExactEquivalenceVerifier
class Yes(ExactEquivalenceVerifier):
    def verify(self, *a, **k): return True, None, {"method": "disabled"}
DS.certify_on_input_subspace = lambda *a, **k: (True, 0.0, {"method": "disabled"})
bad = 0
for qc in B.benchmark_suite():
    r = DS.ErrorBudgetSelector(verifier=Yes()).select(qc)
    U1, U0 = Operator(r["circuit"]).data, Operator(r["exact"]).data
    tr = np.trace(U0.conj().T @ U1); ok = abs(tr) > 1e-9 and np.allclose(U1, tr/abs(tr)*U0, atol=1e-7)
    bad += (not ok); rep = r["report"]
    print(f"{qc.name[:26]:26s} C={rep['sites_applied']} W={len(rep['window_pairs_admitted'])} R={len(rep['rphase_admitted'])} drop={len(rep['approx_admitted'])} 2q {rep['two_qubit_before']}->{rep['two_qubit_after']} ok={ok}")
print("corrupted:", bad, "/ 12")

"""Attribution of the gain to the admissibility conditions (Q2 suite, 12 circuits).
Stages: exact -> (C) structural pairs -> +(W) windows -> +(R) gadgets and control drops
(ancillae pinned). For each stage the sum of 2q gates and infidelity; certificate on the input subspace.
Output: data/attribution.json. Run with PYTHONPATH=<repository root> and EXP=<experiments>."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
from pathlib import Path
HERE_P = Path(__file__).resolve().parent; PAPER = HERE_P.parent; REPO = PAPER.parent
EXP = os.environ.get("EXP", str(REPO / "experiments"))  # experiment modules of this repository
sys.path.insert(0, str(REPO)); sys.path.insert(0, EXP)
import benchmarks as B
from _clean import clean_ancillas
from qiskit.quantum_info import Operator
from toffoli_optimizer.core.decomposition_selector import ErrorBudgetSelector
from toffoli_optimizer.core.subspace_check import certify_on_input_subspace
from toffoli_optimizer.core.error_model import HardwareErrorModel
em = HardwareErrorModel()
stages = {
    "C": lambda qc: ErrorBudgetSelector(window_pairs=False).select(qc),
    "CW": lambda qc: ErrorBudgetSelector().select(qc),
    "CWR": lambda qc: ErrorBudgetSelector().select(qc, pinned_zero=clean_ancillas(qc)),
}
out = {"exact": {"twoq": 0, "infid": 0.0}}
for k in stages:
    out[k] = {"twoq": 0, "infid": 0.0, "certified": 0, "pairs_C": 0, "pairs_W": 0, "gadgets_R": 0, "drops_R": 0}
per = {}
for qc in B.benchmark_suite():
    ex = ErrorBudgetSelector().decompose_exact_only(qc); Ue = Operator(ex).data
    out["exact"]["twoq"] += em.two_qubit_count(ex); out["exact"]["infid"] += em.circuit_infidelity(ex)
    pm = sum(1 << q for q in clean_ancillas(qc))
    inputs = [x for x in range(2 ** qc.num_qubits) if not x & pm]
    per[qc.name] = {"exact": em.two_qubit_count(ex)}
    for k, f in stages.items():
        r = f(qc); rep = r["report"]; c = r["circuit"]
        ok = certify_on_input_subspace(Ue, Operator(c).data, inputs, "subroutine")[0]
        o = out[k]
        o["twoq"] += em.two_qubit_count(c); o["infid"] += em.circuit_infidelity(c); o["certified"] += int(ok)
        o["pairs_C"] += rep["sites_applied"]; o["pairs_W"] += len(rep.get("window_pairs_admitted", []))
        o["gadgets_R"] += len(rep.get("rphase_admitted", [])); o["drops_R"] += len(rep.get("approx_admitted", []))
        per[qc.name][k] = em.two_qubit_count(c)
out["per_circuit"] = per
os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
json.dump(out, open(os.path.join(HERE, "data", "attribution.json"), "w"), indent=1)
for k in ("exact", "C", "CW", "CWR"):
    print(k, out[k])

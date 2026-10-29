"""Data for the table comparing the semantics (Q2 suite, 12 circuits).
Methods: exact-only, count-greedy (RCCX everywhere), pass in the subroutine semantics (default),
pass in the program semantics (with condition U). For each: sum of 2q gates, sum of infidelity, number of circuits
incorrect as a subroutine (U_out P_in != e^{i th} U_ex P_in) and as a program
(different measurement distributions for some state on the input subspace).
Output: data/semantics.json. Run in the env ml with PYTHONPATH=<repository root>."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
from pathlib import Path
HERE_P = Path(__file__).resolve().parent; PAPER = HERE_P.parent; REPO = PAPER.parent
EXP = os.environ.get("EXP", str(REPO / "experiments"))  # experiment modules of this repository
sys.path.insert(0, str(REPO)); sys.path.insert(0, EXP)
import benchmarks as B, naive_relphase as N
from _clean import clean_ancillas
from qiskit.quantum_info import Operator
from toffoli_optimizer.core.decomposition_selector import ErrorBudgetSelector
from toffoli_optimizer.core.subspace_check import certify_on_input_subspace
from toffoli_optimizer.core.error_model import HardwareErrorModel
em = HardwareErrorModel()
methods = {"exact": lambda qc: ErrorBudgetSelector().decompose_exact_only(qc),
           "greedy": lambda qc: __import__("baselines").m_count_greedy(qc),
           "sub": lambda qc: ErrorBudgetSelector(semantics="subroutine").select(qc, pinned_zero=clean_ancillas(qc))["circuit"],
           "prog": lambda qc: ErrorBudgetSelector(semantics="program").select(qc, pinned_zero=clean_ancillas(qc))["circuit"]}
res = {m: {"twoq": 0, "infid": 0.0, "bad_sub": 0, "bad_prog": 0, "per": {}} for m in methods}
for qc in B.benchmark_suite():
    ex = ErrorBudgetSelector().decompose_exact_only(qc); Ue = Operator(ex).data
    pm = sum(1 << q for q in clean_ancillas(qc))
    inputs = [x for x in range(2 ** qc.num_qubits) if not x & pm]
    for m, f in methods.items():
        c = f(qc); Uc = Operator(c).data
        sub = certify_on_input_subspace(Ue, Uc, inputs, "subroutine")[0]
        prog = certify_on_input_subspace(Ue, Uc, inputs, "observational")[0]
        r = res[m]; r["twoq"] += em.two_qubit_count(c); r["infid"] += em.circuit_infidelity(c)
        r["bad_sub"] += (not sub); r["bad_prog"] += (not prog)
        r["per"][qc.name] = dict(twoq=em.two_qubit_count(c), sub=bool(sub), prog=bool(prog))
os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
json.dump(res, open(os.path.join(HERE, "data", "semantics.json"), "w"), indent=1)
for m, r in res.items(): print(m, r["twoq"], round(r["infid"], 4), r["bad_sub"], r["bad_prog"])

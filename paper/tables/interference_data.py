"""Circuits with interference after the Toffoli: does count-greedy also fail in the program semantics?
editor: X q0; X q1; H t; CCX; H t   (counterexample of the editor; data q0,q1 free at the input)
fig1b : CCX(q0,q1,t); H q0           (fig. 1(b); target t with data, free)
fig1b_clean: the same with t=|0> (clean ancilla): the mirrored gadget is admissible under (R)
Inputs: all basis states of the data (input subspace with all superpositions)."""
import json, os, sys
from pathlib import Path
HERE_P = Path(__file__).resolve().parent; PAPER = HERE_P.parent; REPO = PAPER.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, os.environ.get("EXP", str(REPO / "experiments")))
import naive_relphase as N
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from toffoli_optimizer.core.decomposition_selector import ErrorBudgetSelector
from toffoli_optimizer.core.subspace_check import certify_on_input_subspace
from toffoli_optimizer.core.error_model import HardwareErrorModel
em = HardwareErrorModel()
def editor():
    c = QuantumCircuit(3, name="editor"); c.x(0); c.x(1); c.h(2); c.ccx(0, 1, 2); c.h(2); return c, list(range(8))
def fig1b():
    c = QuantumCircuit(3, name="fig1b"); c.ccx(0, 1, 2); c.h(0); return c, list(range(8))
def fig1b_clean():
    c = QuantumCircuit(3, name="fig1b_clean"); c.ccx(0, 1, 2); c.h(0); return c, [x for x in range(8) if not (x >> 2) & 1]
out = {}
for build in (editor, fig1b, fig1b_clean):
    qc, inputs = build(); ex = ErrorBudgetSelector().decompose_exact_only(qc); Ue = Operator(ex).data
    row = {}
    for m, f in {"greedy": lambda q: __import__("baselines").m_count_greedy(q),
                 "sub": lambda q: ErrorBudgetSelector(semantics="subroutine").select(q, input_space=inputs)["circuit"],
                 "prog": lambda q: ErrorBudgetSelector(semantics="program").select(q, input_space=inputs)["circuit"]}.items():
        c = f(qc); Uc = Operator(c).data
        row[m] = dict(twoq=em.two_qubit_count(c), sub=bool(certify_on_input_subspace(Ue, Uc, inputs, "subroutine")[0]),
                      prog=bool(certify_on_input_subspace(Ue, Uc, inputs, "observational")[0]))
    row["exact_twoq"] = em.two_qubit_count(ex); out[qc.name] = row; print(qc.name, row)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "interference.json"), "w"), indent=1)

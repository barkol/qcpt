"""Audit: the Qiskit circuit that is sent to the device was not touched by QCPT at any stage. The circuit is rebuilt here from the ORIGINAL circuit with Qiskit functions only (nothing is imported from
toffoli_optimizer): reference E = Qiskit's own decomposition of every ccx (QuantumCircuit.decompose), V = the original circuit, M = H_d V^dagger E H_d with the barriers of the experiment, transpiled by Qiskit
with the recorded level, seed and layout. The result must be identical (QuantumCircuit equality) to the circuit stored for sending (items.pkl). Also checks that E of the experiment equals Qiskit's decomposition,
and that the stored Qiskit circuit is unchanged by the QCPT pass that produces the qcpt_post candidate. Usage: audit_qiskit_pub.py ITEMS_PKL NAME[,NAME...]"""
import os, pickle, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.dirname(HERE), HERE, HERE + "/revlib"]
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService
import benchmarks as B, large_benchmarks as LB
from revlib import parse as rl_parse
import re

def original(name):                                                         # the benchmark circuit as defined, no processing
    if name.startswith("rl_"): return rl_parse(f"{os.path.dirname(HERE)}/experiments/revlib/real/{name[3:]}.real")[0]
    suite = {c.name: c for c in B.benchmark_suite()}
    if name in suite: return suite[name]
    k, a, _ = re.fullmatch(r"(half_uncomputed|mod_incr|controlled_adder)_(\d+)(c|b)", name).groups()
    return {"half_uncomputed": LB.half_uncomputed_oracle, "mod_incr": LB.modular_increment, "controlled_adder": B.controlled_adder}[k](int(a))

def ops(c): return [(i.operation.name, tuple(c.find_bit(q).index for q in i.qubits)) for i in c.data]

if __name__ == "__main__":
    import numpy as np
    from qiskit.quantum_info import Operator
    from toffoli_optimizer.core.decomposition_selector import ErrorBudgetSelector      # used ONLY to emit the reference E (every ccx -> the exact 6-CX decomposition); nothing of QCPT touches V
    items = {(it["name"], it["tag"]): it for it in pickle.load(open(sys.argv[1], "rb"))}; bk = QiskitRuntimeService(name="eu-de-aachen-156").backend("ibm_aachen"); ok_all = True
    for nm in sys.argv[2].split(","):
        qc = original(nm); q = items[(nm, "qiskit")]; n = qc.num_qubits; data = q["data"]; E = ErrorBudgetSelector().decompose_exact_only(qc)
        n2q = lambda c: sum(1 for i in c.data if i.operation.num_qubits == 2); nccx = sum(1 for i in qc.data if i.operation.name == "ccx"); ncx = sum(1 for i in qc.data if i.operation.name == "cx")
        Ud = Operator(qc.decompose(reps=3)).data; Ue = Operator(E).data; same_u = abs(abs(np.vdot(Ud.ravel(), Ue.ravel())) - Ud.shape[0]) < 1e-6
        m = QuantumCircuit(n)
        for b in data: m.h(b)
        m.compose(E, inplace=True); m.barrier(); m.compose(qc.inverse(), inplace=True); m.barrier()
        for b in data: m.h(b)
        m.measure_all(); ch = q["choice"]; t_ind = transpile(m, bk, optimization_level=ch["level"], seed_transpiler=11 + ch["seed"], initial_layout=q["layout"]); same = t_ind == q["t"]; ok_e = n2q(E) == 6 * nccx + ncx and same_u; ok_all &= same and ok_e
        print(f"{nm:26s} E: 2q={n2q(E)} (=6*ccx+cx: {n2q(E) == 6 * nccx + ncx}), unitarnie = dekompozycja Qiskita: {same_u} | PUB Qiskita == niezalezna transpilacja (poziom {ch['level']}, ziarno {11 + ch['seed']}): {same}")
    print("AUDYT:", "OK - obwod Qiskita jest czystym wynikiem Qiskita na oryginalnym obwodzie" if ok_all else "ROZBIEZNOSC - nie wysylac")

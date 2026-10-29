import sys, json, os
HERE_R = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE_R, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "experiments")); sys.path.insert(0, REPO); sys.path.insert(0, HERE_R)
from _paths import EXPERIMENTS_DIR  # noqa
import mqt.qcec as qcec, scale_eval as SE, large_benchmarks as LB
from _clean import clean_ancillas
from recert_large import with_anc
from qiskit import QuantumCircuit
out = {}
for name in ('mod_incr_11b', 'ripple_adder_10b', 'cla_adder_7b'):
    qc = [q for q in LB.large_suite() if q.name == name][0]; pins = clean_ancillas(qc); s = SE.ScalableErrorBudgetSelector().select(qc, pinned_zero=pins)
    for th in (1e-12, 1e-10, 1e-8):
        r = qcec.verify(with_anc(s['exact'], pins), with_anc(s['circuit'], pins), run_zx_checker=False, elide_permutations=False, trace_threshold=th, timeout=300)
        out[f"{name}|{th}"] = str(r.equivalence).split('.')[-1]; print(name, th, out[f"{name}|{th}"], flush=True)
a = QuantumCircuit(3); a.ccx(0, 1, 2)
for eps in (1e-4, 1e-6):
    b = a.copy(); b.rx(eps, 0)
    for th in (1e-12, 1e-10, 1e-8):
        out[f"rx{eps}|{th}"] = str(qcec.verify(a, b, run_zx_checker=False, elide_permutations=False, trace_threshold=th).equivalence).split('.')[-1]; print('rx', eps, th, out[f"rx{eps}|{th}"], flush=True)
json.dump(out, open(os.path.join(HERE_R, 'prog_qcec.json'), 'w'), indent=1)

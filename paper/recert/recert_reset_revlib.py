"""Re-certification: reset circuits (sync_scale) and RevLib >12 lines (code v14, as in the paper).
Corrected QCEC + state-vector simulation. Results: recert_reset_revlib.jsonl"""
import json, os, sys
HERE_R = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE_R, "..", ".."))
RL = os.path.join(REPO, "experiments", "revlib")
sys.path.insert(0, os.path.join(REPO, "experiments")); sys.path.insert(0, REPO)
sys.path.insert(0, RL); sys.path.insert(0, HERE_R)
from _paths import EXPERIMENTS_DIR  # noqa
import scale_eval as SE, baselines as BL
from sync_benchmark import resetting_circuit
from _clean import clean_ancillas
from revlib import parse
from toffoli_optimizer.core.decomposition_selector import ErrorBudgetSelector
from recert_large import qcec_fixed, sv_equiv
OUT = os.path.join(HERE_R, "recert_reset_revlib.jsonl")
done = {json.loads(l)["key"] for l in open(OUT)} if os.path.exists(OUT) else set()
def emit(row):
    with open(OUT, "a") as fh: fh.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)
sel = SE.ScalableErrorBudgetSelector()
for m, L in [(3, 3), (3, 5), (3, 7), (4, 7)]:
    qc = resetting_circuit(m, L); pins = clean_ancillas(qc); key = f"reset|{qc.name}"
    if key in done: continue
    r = sel.select(qc, pinned_zero=pins); q, qs = qcec_fixed(r["exact"], r["circuit"], pins); ok, mo, sp = sv_equiv(r["exact"], r["circuit"], pins)
    emit(dict(key=key, n=qc.num_qubits, twoq=SE.EM.two_qubit_count(r["circuit"]), qcec=q, sv_ok=ok, sv_min_overlap=mo))
W = {json.loads(l)["file"]: json.loads(l) for l in open(os.path.join(RL, "wyniki.jsonl"))}
X = {json.loads(l)["file"]: json.loads(l) for l in open(os.path.join(RL, "wyniki_extra.jsonl"))}
for f in ["rd53_311.real", "sym6_316.real", "rd84_142.real", "cnt3-5_179.real", "ham7_299.real", "plus63mod4096_309.real"]:
    qc, pins, _ = parse(os.path.join(RL, "real", f))
    ex = ErrorBudgetSelector().decompose_exact_only(qc); ours = ErrorBudgetSelector().select(qc, pinned_zero=pins)["circuit"]
    variants = {"ours": ours, "qiskit_l3": BL.m_qiskit_l3(qc), "tket": BL.m_tket(qc), "greedy": BL.m_count_greedy(qc), "ours_qiskit": BL.m_qiskit_l3(ours)}
    for tag, c in variants.items():
        key = f"revlib|{f}|{tag}"
        if key in done: continue
        pub = (X.get(f, {}).get("tket_fix" if tag == "tket" else tag) or W[f].get(tag) or {}).get("cert")
        q, qs = qcec_fixed(ex, c, pins); ok, mo, sp = sv_equiv(ex, c, pins)
        emit(dict(key=key, n=qc.num_qubits, tag=tag, twoq=SE.EM.two_qubit_count(c), cert_published=pub, qcec=q, sv_ok=ok, sv_min_overlap=mo))
print("KONIEC")

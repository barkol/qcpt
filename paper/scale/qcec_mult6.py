"""Overnight: array_mult_6b (24 q) in program mode and count-greedy, subroutine certificate
QCEC without a short time limit. The result of each variant goes to nocne/array_mult_6b_<tag>.json
(resumable: an existing file is skipped). Usage: qcec_mult6.py <prog|greedy>"""
import json, os, sys, time
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "experiments")); sys.path.insert(0, REPO)
from _paths import EXPERIMENTS_DIR  # noqa: F401
import scale_eval as SE, large_benchmarks as LB
from toffoli_optimizer.core.phase_observability import is_phase_unobservable, default_affected_qubits
from _clean import clean_ancillas
tag = sys.argv[1]; OUT = os.path.join(REPO, "paper", "data", "rerun", f"array_mult_6b_{tag}.json")
if os.path.exists(OUT): print("pomijam", OUT); sys.exit(0)
qc = [q for q in LB.large_suite() if q.name == "array_mult_6b"][0]
pins = clean_ancillas(qc); sel = SE.ScalableErrorBudgetSelector(); sres = sel.select(qc, pinned_zero=pins)
rel, mir = set(sres["rel_idx"]), set(sres["mirror_idx"])
ccx = [i for i, ins in enumerate(qc.data) if ins.operation.name.lower() in ("ccx", "mcx", "mcx_gray") and len(ins.qubits) == 3]
U = {i for i in ccx if i not in rel and is_phase_unobservable(qc, i, default_affected_qubits(qc, i))}
c = sel._build(qc, rel | U, mir) if tag == "prog" else sel._build(qc, set(ccx))
print(tag, "start", time.ctime(), "2q", SE.EM.two_qubit_count(c), flush=True)
assert not pins, "the script assumes no pinned ancillae"
t0 = time.time(); r = SE.verify_scalable(sres["exact"], c, exhaustive_max_qubits=SE.EXHAUSTIVE_MAX_QUBITS, timeout_s=14400)
res = {"name": qc.name, "tag": tag, "equivalent": r.equivalent, "method": r.method, "t_s": round(time.time() - t0, 1),
       "detail": {k: str(v) for k, v in (r.detail or {}).items()}, "twoq": SE.EM.two_qubit_count(c), "finished": time.ctime()}
json.dump(res, open(OUT, "w"), indent=1); print(res, flush=True)

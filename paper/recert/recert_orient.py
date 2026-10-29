"""Re-certification of the calibration-aware orientation for circuits >12 q (code 34364bf, as in the paper)."""
import json, os, pickle, sys
# The stored results were produced with code commit 34364bf; by default this runs the code of this checkout.
HERE_R = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE_R, "..", ".."))
C = os.environ.get("CODE", REPO); sys.path.insert(0, C); sys.path.insert(0, os.path.join(C, "experiments"))
RL = os.path.join(REPO, "experiments", "revlib"); sys.path.insert(0, RL); sys.path.insert(0, HERE_R)
from _paths import EXPERIMENTS_DIR  # noqa
from revlib import parse
from qiskit import transpile
from toffoli_optimizer.core.decomposition_selector import ErrorBudgetSelector
from toffoli_optimizer.core.orientation import orient, pair_cost_from_calibration, edge_errors_from_properties
from recert_large import qcec_fixed, sv_equiv
B = os.path.join(REPO, "experiments", "budget") + os.sep
DEV = [(n, pickle.load(open(B + f"kal_{n}.pkl", "rb"))) for n in ("ibm_marrakesh", "ibm_fez", "ibm_kingston")]
DEV.append(("ibm_marrakesh@2026-10-01", {"props": pickle.load(open(B + "marrakesh_props_jobtime.pkl", "rb")), "target": DEV[0][1]["target"]}))
PUB = {json.loads(l)["key"]: json.loads(l) for l in open(B + "orientation.jsonl")}
OUT = os.path.join(HERE_R, "recert_orient.jsonl")
for f in ["rd53_311.real", "sym6_316.real", "rd84_142.real", "cnt3-5_179.real", "ham7_299.real", "plus63mod4096_309.real"]:
    qc, pins, _ = parse(os.path.join(RL, "real", f))
    s = ErrorBudgetSelector(); res = s.select(qc, pinned_zero=pins)
    for dev, d in DEV:
        pub = PUB.get(f"{dev}|RevLib|{f}", {})
        if not pub.get("oriented"): continue
        L = transpile(res["exact"], target=d["target"], optimization_level=3, seed_transpiler=11).layout.initial_index_layout()[: qc.num_qubits]
        out, info = orient(s, qc, res, pair_cost_from_calibration(edge_errors_from_properties(d["props"]), L), pinned_zero=pins)
        q, _ = qcec_fixed(res["exact"], out, pins); ok, mo, _ = sv_equiv(res["exact"], out, pins)
        row = dict(key=f"{dev}|{f}", n=qc.num_qubits, oriented=info["oriented"], oriented_pub=pub["oriented"], kept_default=info["kept_default"], qcec=q, sv_ok=ok, sv_min_overlap=mo)
        with open(OUT, "a") as fh: fh.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
print("KONIEC")

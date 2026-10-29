"""Dodatkowe kolumny RevLib: tket po poprawce adaptera (allow_swaps=False) oraz
pipeline 'ours -> Qiskit opt-3' (pass jako pre-pass). Ten sam certyfikat. Wznawialne:
wyniki_extra.jsonl, po jednym wierszu na obwod; tylko obwody z wyniki.jsonl bez bledu."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); RUN = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, RUN + "/experiments"); sys.path.insert(0, RUN); sys.path.insert(0, HERE)
from _paths import EXPERIMENTS_DIR  # noqa: F401
import baselines as BL, scale_eval as SE
from revlib import parse
from toffoli_optimizer.core.decomposition_selector import ErrorBudgetSelector
OUT = os.path.join(os.environ.get("TOPTOFFOLI_OUT", HERE), "wyniki_extra.jsonl")
done = {json.loads(l)["file"] for l in open(OUT)} if os.path.exists(OUT) else set()
base = [json.loads(l) for l in open(os.path.join(os.path.dirname(OUT), "wyniki.jsonl"))]
for b in base:
    f = b["file"]
    if "error" in b or f in done or b["gates"] > int(os.environ.get("MAXG", "1000")):
        continue
    qc, pins, info = parse(os.path.join(HERE, "real", f)); EM = SE.EM
    exact = ErrorBudgetSelector().decompose_exact_only(qc)
    ours = ErrorBudgetSelector().select(qc, pinned_zero=pins)["circuit"]
    row = {"file": f}
    for k, c in (("tket_fix", BL.m_tket(qc)), ("ours_qiskit", BL.m_qiskit_l3(ours))):
        t0 = time.time(); r = SE.certify(exact, c, pinned_zero=pins)
        row[k] = {"twoq": EM.two_qubit_count(c), "infid": EM.circuit_infidelity(c), "cert": r.equivalent,
                  "cert_method": r.method, "cert_s": round(time.time() - t0, 2)}
    with open(OUT, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f, row["tket_fix"]["twoq"], row["tket_fix"]["cert"], row["ours_qiskit"]["twoq"], row["ours_qiskit"]["cert"], flush=True)
print("KONIEC")

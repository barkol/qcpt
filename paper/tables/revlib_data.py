"""Merges the RevLib results (revlib_20261002/wyniki.jsonl + wyniki_extra.jsonl) -> data/revlib.json.
tket: version after the adapter fix (allow_swaps=False); for circuits computed before the fix it takes
tket_fix from wyniki_extra and, if missing, computes tket itself (cheap). Circuits with an error/timeout are
listed separately, they do not disappear."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.abspath(os.path.join(HERE, "..", ".."))           # repository root
R = os.path.join(RUN, "experiments", "revlib")                  # wyniki.jsonl, wyniki_extra.jsonl, nieuruchomione.json, real/
sys.path.insert(0, RUN + "/experiments"); sys.path.insert(0, RUN); sys.path.insert(0, R)
B = [json.loads(l) for l in open(os.path.join(R, "wyniki.jsonl"))]
X = {json.loads(l)["file"]: json.loads(l) for l in open(os.path.join(R, "wyniki_extra.jsonl"))} if os.path.exists(os.path.join(R, "wyniki_extra.jsonl")) else {}
TKET_FIX_OD = set(os.environ.get("TKET_OK", "").split(","))  # files already computed with the corrected tket
rows, failed = [], []
for b in B:
    if "error" in b:
        failed.append({k: b.get(k) for k in ("file", "lines", "gates", "error")}); continue
    r = {"file": b["file"], "lines": b["lines"], "gates": b["gates"], "ccx": b["ccx"], "pinned": len(b["pinned"]),
         "C": b["pairs_C"], "W": b["pairs_W"], "R": b["gadgets_R"], "drop": b["drops_R"], "fell_back": b["fell_back"]}
    for k in ("exact", "ours", "qiskit_l3", "greedy"):
        r[k] = b[k]
    x = X.get(b["file"], {})
    r["tket"] = x.get("tket_fix") or (b.get("tket") if b["file"] in TKET_FIX_OD else None)
    r["ours_qiskit"] = x.get("ours_qiskit") or b.get("ours_qiskit")
    if r["tket"] is None or r["ours_qiskit"] is None:
        import baselines as BL, scale_eval as SE
        from revlib import parse
        from toffoli_optimizer.core.decomposition_selector import ErrorBudgetSelector
        qc, pins, _ = parse(os.path.join(R, "real", b["file"])); ex = ErrorBudgetSelector().decompose_exact_only(qc)
        if r["tket"] is None:
            c = BL.m_tket(qc); ce = SE.certify(ex, c, pinned_zero=pins)
            r["tket"] = {"twoq": SE.EM.two_qubit_count(c), "infid": SE.EM.circuit_infidelity(c), "cert": ce.equivalent}
        if r["ours_qiskit"] is None:
            raise SystemExit(f"brak ours_qiskit dla {b['file']} - uruchom revlib_extra.py z MAXG")
    rows.append(r)
M = ["exact", "qiskit_l3", "tket", "greedy", "ours", "ours_qiskit"]
agg = {m: {"twoq": sum(r[m]["twoq"] for r in rows), "infid": sum(r[m]["infid"] for r in rows),
           "cert": sum(bool(r[m]["cert"]) for r in rows)} for m in M}
NR = os.path.join(R, "nieuruchomione.json")
not_run = json.load(open(NR)) if os.path.exists(NR) else []
out = {"n": len(rows), "failed": failed, "not_run": not_run, "agg": agg, "rows": rows,
       "cond": {k: sum(r[k] for r in rows) for k in ("C", "W", "R", "drop")},
       "n_improved": sum(r["ours"]["twoq"] < r["exact"]["twoq"] for r in rows),
       "n_oq_better_q": sum(r["ours_qiskit"]["twoq"] < r["qiskit_l3"]["twoq"] for r in rows),
       "n_oq_worse_q": sum(r["ours_qiskit"]["twoq"] > r["qiskit_l3"]["twoq"] for r in rows),
       "n_fell_back": sum(bool(r["fell_back"]) for r in rows),
       "max_lines": max(r["lines"] for r in rows), "max_gates": max(r["gates"] for r in rows)}
os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
json.dump(out, open(os.path.join(HERE, "data", "revlib.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))

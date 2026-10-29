"""Wznawialny sterownik scale_eval: kazdy obwod zapisywany do JSONL, po restarcie pomijany."""
import json, sys, os
from _paths import EXPERIMENTS_DIR  # noqa: F401
import scale_eval as SE
import large_benchmarks as LB
CK = os.path.join(str(SE.OUT_DIR), "scale_rows.jsonl")  # TOPTOFFOLI_OUT redirects
done = {}
if os.path.exists(CK):
    for line in open(CK):
        r = json.loads(line); done[r["name"]] = r
full = LB.large_suite()
orig = LB.large_suite
for qc in full:
    if qc.name in done:
        print("pomijam", qc.name); continue
    LB.large_suite = lambda q=qc: [q]
    _, rows = SE.run()
    with open(CK, "a") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
    print("zapisano", qc.name, flush=True)
LB.large_suite = orig
rows = [json.loads(l) for l in open(CK)]
order = {q.name: i for i, q in enumerate(full)}
rows.sort(key=lambda r: order[r["name"]])
spot = SE.phase_aware_small_spotcheck()
cross = SE.crossover_curve()
SE.write_markdown(SE.OUT_MD, rows, cross, spot)
print("wrote", SE.OUT_MD)

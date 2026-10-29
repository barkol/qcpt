import json
R = [json.loads(l) for l in open("fixtures/aachen_scaling_hw/hw_results.jsonl")]; by = {}
for r in R: by.setdefault(r["name"], {})[r["variant"]] = r
for n, v in by.items():
    q, c = v["qiskit"], v["qct"]; z = (c["p0"] - q["p0"]) / ((q["sigma"] ** 2 + c["sigma"] ** 2) ** 0.5)
    print("%-22s 2q %3d->%3d  P0 %.3f->%.3f (%+.1f sig)  Pcond %.3f->%.3f  %s" % (n, q["twoq"], c["twoq"], q["p0"], c["p0"], z, q["p_cond"], c["p_cond"], c["source"]))

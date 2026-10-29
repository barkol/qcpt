"""Re-simulation of the controlled-adder mirror circuits (ISA circuits 3, 4, 5: ref, sub, greedy)
with the T1 and Hahn-echo T2 measured on q2, q16, q3, q6 in job davpkglj371s73dnb7hg
(analiza.json) instead of the calibration values.
Writes sym_zmierzone.json ({"kalibracja": ..., "zmierzone T1/T2 (q2,q16,q3,q6)": ...}) and
sym_zmierzone_fazaCZ.json (measured T1/T2 plus a coherent CZ phase of 0.02/0.03/0.04 rad).
These are the inputs of the DGS* numbers of the paper (paper/numbers/paper_numbers.py).
Usage: sym_measured.py [--check]   (--check compares with the stored files instead of writing)"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import symulacja as S

A = json.load(open(os.path.join(HERE, "analiza.json")))["C"]
TAGS = {3: "ref", 4: "sub", 5: "greedy"}


def run(override, **kw):
    S.OVR.clear(); S.OVR.update(override)
    out = {}
    for k, tag in TAGS.items():
        p0, marg, _ = S.p0_and_marginals(S.circs[k], **kw)
        out[tag] = {"p0": p0, "marg": marg}
    return out


measured = {}
for q, v in A.items():
    measured[(int(q), "T1")] = v["T1_fit"]; measured[(int(q), "T2")] = v["T2echo_fit"]
res = {"kalibracja": run({}), "zmierzone T1/T2 (q2,q16,q3,q6)": run(measured)}
faza = {str(ph): {t: r["p0"] for t, r in run(measured, cz_phase=ph).items()} for ph in (0.02, 0.03, 0.04)}
if "--check" in sys.argv:
    import numpy as np
    for fn, new in (("sym_zmierzone.json", res), ("sym_zmierzone_fazaCZ.json", faza)):
        old = json.load(open(os.path.join(HERE, fn)))
        a = json.dumps(old, sort_keys=True); b = json.dumps(new, sort_keys=True)
        flat = lambda d: np.array([x for x in __import__("re").findall(r"-?\d+\.\d+(?:e-?\d+)?", json.dumps(d, sort_keys=True))], float)
        print(fn, "identical" if a == b else f"max abs diff {np.max(np.abs(flat(old) - flat(new))):.2e}")
else:
    json.dump(res, open(os.path.join(HERE, "sym_zmierzone.json"), "w"), indent=1)
    json.dump(faza, open(os.path.join(HERE, "sym_zmierzone_fazaCZ.json"), "w"), indent=1)
    print(json.dumps(faza))

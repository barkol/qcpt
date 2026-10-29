"""Test hipotez rozbieznosci (wszystkie 15 obwodow): (A) gorszy q16 (T2, T1), (B) koherentny blad fazy CZ,
(C) jednorodna nadmiarowa depolaryzacja CZ. Miara: suma kwadratow ln(hw/sim) oraz reszta dla sumatora."""
import json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import symulacja as S
W = json.load(open(os.path.join(S.HERE, "..", "..", "fixtures", "mirror_test", "wyniki_hw.json")))["rows"]  # = paper/tables/data/mirror_hw.json
def run(label, **kw):
    res = []
    for k in range(15):
        p0, marg, _ = S.p0_and_marginals(S.circs[k], **kw); res.append(p0)
    lr = [math.log(W[k]["p0_hw"] / res[k]) for k in range(15)]
    ss = sum(x * x for x in lr)
    out = dict(label=label, ss=ss, adder=[round(res[k], 3) for k in (3, 4, 5)], nested=[round(res[k], 3) for k in (6, 7, 8)],
               lr=[round(x, 2) for x in lr])
    print(json.dumps(out), flush=True); return out
R = [run("bazowy")]
for t2 in (20, 10, 5):
    S.OVR.clear(); S.OVR[(16, "T2")] = t2; R.append(run(f"q16 T2={t2}us"))
S.OVR.clear(); S.OVR[(16, "T2")] = 10; S.OVR[(16, "T1")] = 20; R.append(run("q16 T1=20 T2=10"))
S.OVR.clear()
for d in (0.02, 0.05, 0.08, 0.12):
    R.append(run(f"faza CZ {d}", cz_phase=d))
for lam in (0.002, 0.004):
    R.append(run(f"lam {lam}", lam=lam))
json.dump(R, open(os.path.join(S.HERE, "hipotezy.json"), "w"), indent=1)

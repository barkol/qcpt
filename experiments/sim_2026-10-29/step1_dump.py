import os, sys
_R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # repository root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_O = os.environ.get("SIM_OUT", os.path.join(_R, "experiments", "sim_2026-10-29", "out")); os.makedirs(_O, exist_ok=True)      # outputs
"""Step 1: predictions of the calibration model and of the fitted models for the 32 measured circuits (16 circuits x 2 compilations): dump + least-squares fit with a cache of the evaluations (resumable)."""
import json, os, sys, time, hashlib
import numpy as np
sys.path.insert(0, _O); sys.path.insert(0, _R); sys.path.insert(0, _R + "/experiments")
import simlib as S, data_new as D
OUT = S.OUT; tg = S.get_target(); ALL = D.load_all()
def rows_for(names, setname):
    out = []
    for nm in names:
        q, c = D.pair(ALL, nm)
        for variant, e in (("qiskit", q), ("qct", c)): out.append(dict(set=setname, name=nm, variant=variant, tag=e["res"]["tag"], twoq=e["res"]["twoq"], p0_hw=e["res"]["p0"], pcond_hw=e["res"]["p_cond"], sigma=e["res"]["sigma"], accept_hw=e["res"]["accept"], _it=e["item"], _data=D.data_qubits(ALL, nm)))
    return out
train = rows_for(D.SMALL, "train"); test = rows_for(D.LARGE, "test")
CACHE = os.path.join(OUT, "eval_cache.jsonl"); cache = {}
if os.path.exists(CACHE):
    for l in open(CACHE): r = json.loads(l); cache[r["k"]] = (r["p0"], r["acc"])
def predict(rows, K2, R2, TH, tag):
    k = f"{tag}|{K2:.6f}|{R2:.6f}|{TH:.6f}"
    if k in cache: return np.array(cache[k][0]), np.array(cache[k][1])
    p, a = S.predict(tg, [(r["_it"]["t"], r["_data"]) for r in rows], K2, R2, TH); cache[k] = (p.tolist(), a.tolist()); open(CACHE, "a").write(json.dumps(dict(k=k, p0=p.tolist(), acc=a.tolist())) + "\n"); return p, a
def stats(pred, meas):
    d = pred - meas; return dict(n=len(d), bias=float(d.mean()), rmse=float(np.sqrt((d ** 2).mean())), corr=float(np.corrcoef(pred, meas)[0, 1]))
def dump(K2, R2, TH, lab):
    rows = []
    for setname, rs in (("train", train), ("test", test)):
        p, a = predict(rs, K2, R2, TH, setname)
        for r, p0, ac in zip(rs, p, a): rows.append({k: v for k, v in r.items() if not k.startswith("_")} | dict(p0_pred=float(p0), pcond_pred=float(p0 / ac) if ac > 0 else None, accept_pred=float(ac)))
    json.dump(dict(K2=K2, R2=R2, TH=TH, rows=rows), open(os.path.join(OUT, f"pred_{lab}.json"), "w")); return rows
def fit(rs_list, x0=(2.0, 0.15, 0.03), nfev=14):
    from scipy.optimize import least_squares
    meas = np.concatenate([[r["p0_hw"] for r in rs] for rs in rs_list])
    def res(x): return np.concatenate([predict(rs, x[0], x[1], x[2], "train" if rs is train else "test")[0] for rs in rs_list]) - meas
    r = least_squares(res, list(x0), bounds=([1.0, 0.05, 0.0], [4.0, 1.0, 0.4]), diff_step=0.05, max_nfev=nfev, verbose=2); return [float(v) for v in r.x], float(r.cost)
if __name__ == "__main__":
    t0 = time.time(); print("calibration model", flush=True)
    rows = dump(1.0, 1.0, 0.0, "calib"); print("done calib %.0fs" % (time.time() - t0), flush=True)
    xa, ca = fit([train, test]); json.dump(dict(K2=xa[0], R2=xa[1], TH=xa[2], cost=ca), open(os.path.join(OUT, "noisefit_best3_all.json"), "w")); print("FIT ALL", xa, "%.0fs" % (time.time() - t0), flush=True)
    dump(xa[0], xa[1], xa[2], "fit")
    xt, ct = fit([train]); json.dump(dict(K2=xt[0], R2=xt[1], TH=xt[2], cost=ct), open(os.path.join(OUT, "noisefit_best3.json"), "w")); print("FIT TRAIN", xt, "%.0fs" % (time.time() - t0), flush=True)
    dump(xt[0], xt[1], xt[2], "fit_train"); print("DONE", "%.0fs" % (time.time() - t0), flush=True)

import os, sys
_R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # repository root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_O = os.environ.get("SIM_OUT", os.path.join(_R, "experiments", "sim_2026-10-29", "out")); os.makedirs(_O, exist_ok=True)      # outputs
"""Step 2: all numbers of section 5.4 from the new data: (A) corrected-model paragraph, (B) dynamometer fits (dyno_fam2.npz, sensitivity), (C) the nine circuits (data of figure 8)."""
import json, os, sys, math
import numpy as np
sys.path[:0] = [_O, _R, _R + "/experiments"]
import simlib as S, data_new as D
OUT = S.OUT; tg = S.get_target(); ALL = D.load_all(); N = {}
def load(lab): return json.load(open(os.path.join(OUT, f"pred_{lab}.json")))
def pairs(rows):
    g = {}
    for r in rows: g.setdefault(r["name"], {})[r["variant"]] = r
    return g
def st(d): return dict(n=len(d), bias=float(np.mean(d)), rmse=float(np.sqrt(np.mean(np.square(d)))))
def corr(a, b): return float(np.corrcoef(a, b)[0, 1])
cal, fit, fit_tr = load("calib"), load("fit"), load("fit_train")
N["fit_all"] = dict(K2=fit["K2"], R2=fit["R2"], TH=fit["TH"], T2_shorter_by=1 / fit["R2"]); N["fit_train"] = dict(K2=fit_tr["K2"], R2=fit_tr["R2"], TH=fit_tr["TH"])
def block(model, key_pred, key_hw):
    rows = model["rows"]; d = np.array([r[key_pred] - r[key_hw] for r in rows]); p = np.array([r[key_pred] for r in rows]); h = np.array([r[key_hw] for r in rows])
    small = np.array([r["set"] == "train" for r in rows]); out = dict(all=st(d) | dict(corr=corr(p, h)), small=st(d[small]), large=st(d[~small]))
    g = pairs(rows); names = sorted(g); gp = np.array([g[n]["qct"][key_pred] - g[n]["qiskit"][key_pred] for n in names]); gh = np.array([g[n]["qct"][key_hw] - g[n]["qiskit"][key_hw] for n in names])
    out["gain"] = dict(mean_pred=float(gp.mean()), mean_hw=float(gh.mean()), corr=corr(gp, gh), sign_correct=int(np.sum(np.sign(gp) == np.sign(gh))), n_pairs=len(names)); return out
N["uncond"] = dict(calib=block(cal, "p0_pred", "p0_hw"), fitted_all=block(fit, "p0_pred", "p0_hw"), fitted_train_only=block(fit_tr, "p0_pred", "p0_hw"))
N["cond"] = dict(calib=block(cal, "pcond_pred", "pcond_hw"), fitted_all=block(fit, "pcond_pred", "pcond_hw"))
# ---- (B) dynamometer
def fitfn(x, Pq, Pc, p_fixed=None):
    from scipy.optimize import least_squares
    y = np.r_[Pq, Pc]
    def res(q): p, a, c = q; return y - np.r_[np.exp(-(x / a) ** p), np.exp(-(x / c) ** p)]
    best = None
    for p0 in (1.0, 1.5, 2.0):
        for x0 in (300.0, 1000.0):
            r = least_squares(res, [p0, x0, x0], bounds=([0.5, 20, 20], [4.0, 1e5, 1e5]))
            if best is None or r.cost < best.cost: best = r
    return tuple(float(v) for v in best.x)
NFIT = {"ripple_adder": int(os.environ.get("RIPPLE_NFIT", "4"))}
def famdata(src, kq):
    R = {}
    for r in src["rows"]: R.setdefault(r["name"], {})[r["variant"]] = r
    return R
calibR, fitR = famdata(cal, None), famdata(fit, None); out = {}; sens = {}
for fam, names in D.FAM.items():
    x = np.array([calibR[n]["qiskit"]["twoq"] for n in names], float); o = np.argsort(x); names = [names[i] for i in o]; x = x[o]; xc = np.array([calibR[n]["qct"]["twoq"] for n in names], float)
    nfit = NFIT.get(fam, len(x)); ifit = np.arange(len(x))[:nfit]
    for lab, src, k in (("hw", calibR, "pcond_hw"), ("cal", calibR, "pcond_pred"), ("fit", fitR, "pcond_pred")):
        Pq = np.array([src[n]["qiskit"][k] for n in names]); Pc = np.array([src[n]["qct"][k] for n in names]); p, a, c = fitfn(x[ifit], Pq[ifit], Pc[ifit]); pre = f"{lab}_{fam}_"
        out.update({pre + "x": x, pre + "x_qct": xc, pre + "P_q": Pq, pre + "P_c": Pc, pre + "p": p, pre + "x0_q": a, pre + "x0_c": c, pre + "nfit": nfit, pre + "name": np.array(names)})
        idx = ifit; rs = []
        if len(idx) > 3:
            for drop in idx: kk = [i for i in idx if i != drop]; q = fitfn(x[kk], Pq[kk], Pc[kk]); rs.append(float(q[2] / q[1]))
        f1 = lambda P: float(np.sum(x[idx] ** 2) / -np.sum(x[idx] * np.log(np.clip(P[idx], 1e-3, .999))))
        sens[f"{lab}_{fam}"] = dict(x0_q=a, x0_c=c, ratio=c / a, p=p, leave_one_out=[min(rs), max(rs)] if rs else None, ratio_p1=f1(Pc) / f1(Pq), rms=float(np.sqrt(np.mean((np.r_[Pq[idx], Pc[idx]] - np.r_[np.exp(-(x[idx] / a) ** p), np.exp(-(x[idx] / c) ** p)]) ** 2))))
np.savez(os.path.join(OUT, "dyno_fam2.npz"), **out)
# shot-noise interval of the hardware ratio (parametric bootstrap of the conditional probabilities, 8000 shots)
rng = np.random.default_rng(1)
for fam in D.FAM:
    x = out[f"hw_{fam}_x"]; nf = int(out[f"hw_{fam}_nfit"]); Pq0, Pc0 = out[f"hw_{fam}_P_q"], out[f"hw_{fam}_P_c"]; rr = []
    for _ in range(300):
        Pq = np.clip(rng.binomial(8000, np.clip(Pq0, 0.001, 0.999)) / 8000, 0.01, 0.999); Pc = np.clip(rng.binomial(8000, np.clip(Pc0, 0.001, 0.999)) / 8000, 0.01, 0.999)
        q = fitfn(x[:nf], Pq[:nf], Pc[:nf]); rr.append(q[2] / q[1])
    lo, hi = np.percentile(rr, [2.5, 97.5]); sens[f"hw_{fam}"]["shot_noise_ci95"] = [float(lo), float(hi)]; sens[f"hw_{fam}"]["shot_noise_width"] = float(hi - lo)
N["dynamometer"] = sens; N["dynamometer_ripple_nfit"] = NFIT["ripple_adder"]
# ---- (C) the nine circuits: hardware and simulation on the same qubits (calibration model, exact probabilities)
names9 = D.NINE; items = [(n, tag) for n in names9 for tag in ("qiskit",) + tuple(t for t in [k[1] for k in ALL if k[0] == n and k[1] != "qiskit"])]
p9, a9 = S.predict(tg, [(ALL[k]["item"]["t"], D.data_qubits(ALL, k[0])) for k in items])
rows9 = []
for (nm, tag), p0s, acs in zip(items, p9, a9):
    e = ALL[(nm, tag)]; r = e["res"]; it = e["item"]
    rows9.append(dict(name=nm, tag=tag, variant="qiskit" if tag == "qiskit" else "qct", source=tag, layout=it["layout"], twoq=r["twoq"], esp_pred=it["esp_pred"], p0_sim_same_qubits=float(p0s), pcond_sim_same_qubits=float(p0s / acs) if acs > 0 else None,
                     accept_sim=float(acs), job=r.get("job"), p0=r["p0"], sigma=r["sigma"], accept=r["accept"], p_cond=r["p_cond"], shots=r["shots"], n_ancilla=r.get("n_ancilla")))
json.dump(rows9, open(os.path.join(OUT, "hw_probed9.json"), "w"), indent=1)
g = pairs(rows9); names = sorted(g); gh = np.array([g[n]["qct"]["p0"] - g[n]["qiskit"]["p0"] for n in names]); gs = np.array([g[n]["qct"]["p0_sim_same_qubits"] - g[n]["qiskit"]["p0_sim_same_qubits"] for n in names])
z = {n: (g[n]["qct"]["p0"] - g[n]["qiskit"]["p0"]) / math.hypot(g[n]["qct"]["sigma"], g[n]["qiskit"]["sigma"]) for n in names}
top = max(names, key=lambda n: z[n] * 0 + g[n]["qct"]["p0"] - g[n]["qiskit"]["p0"])
allp = [r for r in rows9]; dd = np.array([r["p0_sim_same_qubits"] - r["p0"] for r in allp]); ps = np.array([r["p0_sim_same_qubits"] for r in allp]); ph = np.array([r["p0"] for r in allp])
from scipy.stats import norm
bonf = 0.05 / 9; zcrit = float(norm.isf(bonf / 2))
N["nine"] = dict(z={n: round(z[n], 2) for n in names}, gain_hw={n: round(float(g[n]["qct"]["p0"] - g[n]["qiskit"]["p0"]), 4) for n in names}, gain_sim={n: round(float(g[n]["qct"]["p0_sim_same_qubits"] - g[n]["qiskit"]["p0_sim_same_qubits"]), 4) for n in names},
                 n_better_2sigma=int(sum(1 for n in names if z[n] > 2)), n_equal=[n for n in names if abs(z[n]) <= 2], n_better_3sigma=int(sum(1 for n in names if z[n] > 3)), mean_gain_hw=float(gh.mean()), mean_gain_sim=float(gs.mean()), factor=float(gh.mean() / gs.mean()),
                 mean_gain_without_top=float(np.mean([gh[i] for i, n in enumerate(names) if n != top])), top=top, top_P_qiskit=g[top]["qiskit"]["p0"], top_P_qcpt=g[top]["qct"]["p0"], top_z=z[top], top_2q=[g[top]["qiskit"]["twoq"], g[top]["qct"]["twoq"]],
                 bonferroni_z_crit=zcrit, not_significant_after_bonferroni=[n for n in names if z[n] < zcrit], sim_overestimate_mean=float(dd.mean()), sim_overestimate_pearson=corr(ps, ph), gain_corr_hw_vs_sim=corr(gh, gs),
                 post_select_top=dict(qiskit=g[top]["qiskit"]["p_cond"], qcpt=g[top]["qct"]["p_cond"]), accept_mean=dict(qiskit=float(np.mean([g[n]["qiskit"]["accept"] for n in names])), qcpt=float(np.mean([g[n]["qct"]["accept"] for n in names]))),
                 winning_candidate={n: g[n]["qct"]["source"] for n in names})
json.dump(N, open(os.path.join(OUT, "numbers_54.json"), "w"), indent=1)
print(json.dumps({k: N[k] for k in ("fit_all", "fit_train")}, indent=1)); print("uncond calib:", json.dumps(N["uncond"]["calib"])); print("uncond fitted_all:", json.dumps(N["uncond"]["fitted_all"])); print("uncond train-only (test):", json.dumps(N["uncond"]["fitted_train_only"]["large"]))
print("nine:", json.dumps({k: N["nine"][k] for k in ("n_better_2sigma", "n_equal", "mean_gain_hw", "mean_gain_sim", "factor", "mean_gain_without_top", "top", "top_P_qiskit", "top_P_qcpt", "top_z", "top_2q", "not_significant_after_bonferroni", "sim_overestimate_mean", "sim_overestimate_pearson", "gain_corr_hw_vs_sim", "post_select_top", "accept_mean")}, indent=0))

#!/usr/bin/env python3
"""Sensitivity of the x0 ratios (QCPT/Qiskit) in the dynamometer figure: (i) leaving out one circuit from the fit (families with 4 points), (ii) exponent p = 1 instead of free. Input data/dyno_fam2.npz; output data/dyno_sensitivity.json (quoted in section 5.4)."""
import json, os
import numpy as np
from scipy.optimize import least_squares
H = os.path.dirname(os.path.abspath(__file__)); D = np.load(os.path.join(H, "data", "dyno_fam2.npz"), allow_pickle=True)
def fit(x, Pq, Pc):
    y = np.r_[Pq, Pc]
    def res(q): p, a, c = q; return y - np.r_[np.exp(-(x / a) ** p), np.exp(-(x / c) ** p)]
    best = None
    for p0 in (1.0, 1.5, 2.0):
        for x0 in (300.0, 1000.0):
            r = least_squares(res, [p0, x0, x0], bounds=([0.5, 20, 20], [4.0, 1e5, 1e5]))
            if best is None or r.cost < best.cost: best = r
    return best.x
out = {}
for lab in ("hw", "fit"):
    for fam in ("mod_incr", "ripple_adder", "grover_ladder", "half_uncomputed"):
        x, Pq, Pc = D[f"{lab}_{fam}_x"], D[f"{lab}_{fam}_P_q"], D[f"{lab}_{fam}_P_c"]; nf = int(D[f"{lab}_{fam}_nfit"]); idx = np.arange(len(x))[:nf]; full = fit(x[idx], Pq[idx], Pc[idx]); rs = []
        if len(idx) > 3:
            for drop in idx: k = [i for i in idx if i != drop]; q = fit(x[k], Pq[k], Pc[k]); rs.append(float(q[2] / q[1]))
        f1 = lambda P: float(np.sum(x[idx] ** 2) / -np.sum(x[idx] * np.log(np.clip(P[idx], 1e-3, .999))))
        out[f"{lab}_{fam}"] = dict(ratio=float(full[2] / full[1]), p=float(full[0]), leave_one_out=[min(rs), max(rs)] if rs else None, ratio_p1=f1(Pc) / f1(Pq))
json.dump(out, open(os.path.join(H, "data", "dyno_sensitivity.json"), "w"), indent=1); print(json.dumps(out, indent=1))

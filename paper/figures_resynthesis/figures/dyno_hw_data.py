#!/usr/bin/env python3
"""Dynamometer on a common load x: for each family the circuits MEASURED on ibm_aachen (the 3 smallest in region B + 1 larger one in region B+2) and the same circuits in simulation (calibration model and the model with fitted corrections),
so x (number of 2q gates of the Qiskit circuit after transpilation) is the same in the simulation and in the experiment. P = conditional probability (ancillae = |0>). Fit P = exp(-(x/x0)^p), p shared by Qiskit and QCPT, separate x0.
Input: data_in/new/pred_calib.json, pred_fit.json (aachen_noise_fit.py dump). Output: data/dyno_fam2.npz."""
import json, os
import numpy as np
H = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(H, "..", "data_in", "new")
FAM = {"mod_incr": ["mod_incr_2b", "mod_incr_3b", "mod_incr_4b", "mod_incr_5b"], "ripple_adder": ["ripple_adder_1b", "ripple_carry_adder_2b", "ripple_adder_3b", "ripple_adder_4b"], "grover_ladder": ["grover_oracle_2c", "grover_oracle_mcx3", "grover_oracle_mcx4", "grover_oracle_5c"], "half_uncomputed": ["half_uncomputed_min", "half_uncomputed_2c", "half_uncomputed_3c", "half_uncomputed_4c"]}
NFIT = {"ripple_adder": 3}          # hardware (and its simulations): only the three smallest circuits enter the ripple-adder fit (the largest is level with Qiskit)
def fit(x, Pq, Pc):
    from scipy.optimize import least_squares
    y = np.r_[Pq, Pc]
    def res(q): p, a, c = q; return y - np.r_[np.exp(-(x / a) ** p), np.exp(-(x / c) ** p)]
    best = None
    for p0 in (1.0, 1.5, 2.0):
        for x0 in (300.0, 1000.0):
            r = least_squares(res, [p0, x0, x0], bounds=([0.5, 20, 20], [4.0, 1e5, 1e5]))
            if best is None or r.cost < best.cost: best = r
    return tuple(float(v) for v in best.x)
def rows(fn, key):
    R = {}
    for r in json.load(open(os.path.join(D, fn)))["rows"]: R.setdefault(r["name"], {})[r["variant"]] = r
    return R
calib, fitm = rows("pred_calib.json", "p"), rows("pred_fit.json", "p"); out = {}
for fam, names in FAM.items():
    names = [n for n in names if n in calib and "qiskit" in calib[n] and "qct" in calib[n]]; x = np.array([calib[n]["qiskit"]["twoq"] for n in names], float); o = np.argsort(x); names = [names[i] for i in o]; x = x[o]
    xc = np.array([calib[n]["qct"]["twoq"] for n in names], float); nfit = NFIT.get(fam, len(x)); ifit = np.arange(len(x))[:nfit]
    for lab, src, kq, kc in (("hw", calib, "pcond_hw", "pcond_hw"), ("cal", calib, "pcond_pred", "pcond_pred"), ("fit", fitm, "pcond_pred", "pcond_pred")):
        Pq = np.array([src[n]["qiskit"][kq] for n in names]); Pc = np.array([src[n]["qct"][kc] for n in names]); p, a, c = fit(x[ifit], Pq[ifit], Pc[ifit]); pre = f"{lab}_{fam}_"
        out.update({pre + "x": x, pre + "x_qct": xc, pre + "P_q": Pq, pre + "P_c": Pc, pre + "p": p, pre + "x0_q": a, pre + "x0_c": c, pre + "nfit": nfit, pre + "name": np.array(names)})
        print(f"{lab:3s} {fam:16s} x {x.astype(int)} Pq {np.round(Pq, 2)} Pc {np.round(Pc, 2)} | p {p:.2f} x0 {a:.0f} -> {c:.0f} (x{c / a:.2f})")
np.savez(os.path.join(H, "data", "dyno_fam2.npz"), **out)

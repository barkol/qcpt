#!/usr/bin/env python3
"""fig_dyno (new): dynamometer by family on a COMMON load x. In each panel: circuits measured on ibm_aachen (open diamonds, dashed curve = fit to the hardware), the same circuits in simulation from the calibration
(filled markers, solid curve) and in simulation with the model with fitted corrections (crosses). Drawing layer; data from data/dyno_fam2.npz (dyno_hw_data.py)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import figstyle as fs
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); OUTDIR = os.environ.get("FIG_OUT", HERE); fs.apply()
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7.5, "legend.fontsize": 6.4, "axes.spines.right": False})
D = np.load(os.path.join(HERE, "data", "dyno_fam2.npz"), allow_pickle=True); BLUE, VERM, GREEN = fs.BLUE, fs.ORANGE, fs.GREEN
FAM = [("mod_incr", "modular incrementer"), ("ripple_adder", "ripple adder"), ("grover_ladder", "Grover AND ladder"), ("half_uncomputed", "half-uncomputed oracle")]
fig, axs = plt.subplots(2, 2, figsize=(483.697 / 72.27, 3.35), layout="constrained")
curve = lambda x, x0, p: x * np.exp(-(x / x0) ** p)
for k, (ax, (fam, ttl)) in enumerate(zip(axs.flat, FAM)):
    g = lambda lab, key: D[f"{lab}_{fam}_{key}"]; x = g("hw", "x"); nfit = int(g("hw", "nfit")); xm = x.max() * 1.35; xs = np.linspace(0, xm, 300); fitted = np.argsort(x) < nfit
    for lab, ls, lw in (("fit", "-", 1.1), ("hw", "--", 0.9)):
        p = float(g(lab, "p")); ax.plot(xs, curve(xs, float(g(lab, "x0_q")), p), color=BLUE, ls=ls, lw=lw, zorder=2); ax.plot(xs, curve(xs, float(g(lab, "x0_c")), p), color=VERM, ls=ls, lw=lw, zorder=2)
    Pq, Pc = g("fit", "P_q"), g("fit", "P_c"); Hq, Hc = g("hw", "P_q"), g("hw", "P_c")
    for xi, a_, b_ in zip(x, Hq, Hc): ax.plot([xi, xi], [xi * a_, xi * b_], color=GREEN if b_ > a_ else VERM, lw=1.8, alpha=0.9, zorder=1)
    for sel, al, lq, lc in ((fitted, 1.0, "Qiskit, fitted model", "QCPT, fitted model"), (~fitted, 0.4, None, None)):
        if sel.any(): ax.scatter(x[sel], (x * Pq)[sel], marker="s", s=15, color=BLUE, edgecolor=fs.INK, linewidths=0.3, alpha=al, zorder=3, label=lq); ax.scatter(x[sel], (x * Pc)[sel], marker="o", s=17, color=VERM, edgecolor=fs.INK, linewidths=0.3, alpha=al, zorder=4, label=lc)
    for sel, al in ((fitted, 1.0), (~fitted, 0.4)):
        if sel.any(): ax.scatter(x[sel], (x * Hq)[sel], marker="D", s=24, facecolor="white", edgecolor=BLUE, linewidths=1.0, alpha=al, zorder=5, label="hardware (open)" if al == 1.0 else None); ax.scatter(x[sel], (x * Hc)[sel], marker="D", s=24, facecolor="white", edgecolor=VERM, linewidths=1.0, alpha=al, zorder=5)
    ymx = max(float(np.max(x * Pc)), float(np.max(curve(xs, float(g("fit", "x0_c")), float(g("fit", "p")))))); ax.set_xlim(0, xm); ax.set_ylim(0, ymx * 1.45); ax.grid(lw=0.4, color=fs.FAINT, zorder=0)
    ax.set_xlabel(r"load $x$: two-qubit gates of the Qiskit circuit"); ax.set_ylabel(r"correct gates $xP$")
    ax.text(0.04, 0.96, ttl + "\n" + r"fitted model: $x_0$ $%.0f\to%.0f$ ($\times%.2f$), $p=%.2f$" % (g("fit", "x0_q"), g("fit", "x0_c"), g("fit", "x0_c") / g("fit", "x0_q"), g("fit", "p")) + "\n" + r"hardware: $x_0$ $%.0f\to%.0f$ ($\times%.2f$), $p=%.2f$" % (g("hw", "x0_q"), g("hw", "x0_c"), g("hw", "x0_c") / g("hw", "x0_q"), g("hw", "p")),
            transform=ax.transAxes, va="top", ha="left", fontsize=6.6, bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.2), zorder=7)
    ax.text(-0.17, 1.03, "(%s)" % "abcd"[k], transform=ax.transAxes, fontweight="bold", fontsize=8.5, va="bottom")
h, l = axs.flat[1].get_legend_handles_labels(); fig.legend(h, l, loc="outside lower center", ncol=4, fontsize=6.6, frameon=False, handletextpad=0.3, columnspacing=1.2)
fig.savefig(os.path.join(OUTDIR, "fig_dyno.pdf")); fig.savefig(os.path.join(OUTDIR, "fig_dyno_preview.png"), dpi=200)

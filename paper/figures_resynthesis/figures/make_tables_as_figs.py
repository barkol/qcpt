#!/usr/bin/env python3
"""Drawing layer: figures that replace Tables 4, 5, 6, 7 and 8. Reads ONLY data/tables_as_figs.npz
(built by tables_data.py). Scale 1:1 to the article class (column 231.85 pt, text 483.70 pt).
Roles: exact-only = blue square, Qiskit/tket = grey diamond, count-greedy = yellow triangle, pass = vermilion circle;
continuous colour (change in the number of 2q gates) = Turbo map."""
import os, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import figstyle as fs
import matplotlib.pyplot as plt
from matplotlib import cm, colors

HERE = os.path.dirname(os.path.abspath(__file__)); OUTDIR = os.environ.get("FIG_OUT", HERE)
d = np.load(os.path.join(HERE, "data", "tables_as_figs.npz"))
fs.apply()
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5})
COL, WIDE = 231.848 / 72.27, 483.697 / 72.27
TURBO = plt.get_cmap("turbo"); NORM = colors.Normalize(0, 45)          # change in the number of 2q gates: 0 ... -45 % (more = red)
MK = {"exact-only": "s", "Qiskit opt-3": "D", "tket": "D", "count-greedy": "^", "QCPT": "o", "QCPT (subroutine)": "o",
      "QCPT (program)": "o", r"QCPT $\to$ Qiskit": "o"}


from paneltags import tag, place


def save(fig, name): place(fig); fig.savefig(os.path.join(OUTDIR, name)); plt.close(fig)


# ---------------------------------------------------------------- cost vs correctness (Tab. 4 + Tab. 7)
def plane(ax, names, twoq, inf, ok, n, off, prog_open=None):
    red = -100 * (twoq - twoq[0]) / twoq[0]; dinf = 100 * (inf - inf[0]) / inf[0]; yy = 100 * ok / n
    for i, nm in enumerate(names):
        hollow = nm == "QCPT (program)"
        ax.scatter(dinf[i], yy[i], s=52, marker=MK[nm], zorder=3, linewidths=0.9, edgecolor=fs.INK,
                   facecolor="none" if hollow else TURBO(NORM(red[i])))
        dx, dy, ha = off.get(nm, (6, 5, "left"))
        ax.annotate(nm.replace("QCPT (subroutine)", "QCPT"), (dinf[i], yy[i]), xytext=(dx, dy), textcoords="offset points", fontsize=7.5, ha=ha, va="center")
    ax.set_xlabel(r"change of estimated infidelity (\%)"); ax.set_ylabel("certified as subroutines (\\%)")
    ax.grid(lw=0.4, color=fs.FAINT, zorder=0); ax.set_ylim(-8, 112)
    ax.axhspan(-8, 97, color=fs.ORANGE, alpha=0.05, lw=0, zorder=0); ax.axhline(97, color="#7b1e3a", lw=1.3, zorder=2)
    ax.text(7, 14, "not equivalent as a subroutine\nfor some outputs", ha="right", va="center", fontsize=7.5, color="#8a3b00", style="italic")


fig, axs = plt.subplots(1, 2, figsize=(WIDE, 2.15), layout="constrained")
offA = {"exact-only": (9, 0, "left"), "Qiskit opt-3": (0, -13, "center"), "tket": (0, 12, "center"), "count-greedy": (10, -3, "left"),
        "QCPT (subroutine)": (0, 12, "center"), "QCPT (program)": (10, 4, "left")}
plane(axs[0], list(d["sem_names"]), d["sem_2q"], d["sem_inf"], d["sem_n"] - d["sem_sub"], int(d["sem_n"]), offA)
axs[0].set_xlim(-48, 8); tag(axs[0], "(a)")
offB = {"exact-only": (9, 0, "left"), "Qiskit opt-3": (0, -13, "center"), "tket": (0, 12, "center"), "count-greedy": (10, 0, "left"),
        "QCPT": (0, 12, "center"), r"QCPT $\to$ Qiskit": (-6, -13, "right")}
plane(axs[1], list(d["rl_names"]), d["rl_2q"], d["rl_inf"], d["rl_cert"], int(d["rl_n"]), offB)
axs[1].set_xlim(-48, 8); tag(axs[1], "(b)")
cb = fig.colorbar(cm.ScalarMappable(norm=NORM, cmap=TURBO), ax=axs, shrink=0.85, pad=0.01, aspect=22)
cb.set_label(r"two-qubit gates saved (\%)", fontsize=8); cb.ax.tick_params(labelsize=8)
save(fig, "fig_cost.pdf")

# ---------------------------------------------------------------- testy lustrzane (Tab. 5 + Tab. 6)
def mirror_panel(ax, names, ref, ps, gr, gid, shots, twoq=None):
    n = len(names); y = np.arange(n)[::-1]; se = lambda p: np.sqrt(np.clip(p * (1 - p), 1e-6, None) / shots)
    for i in range(n):
        ax.plot([ref[i], ps[i]], [y[i], y[i]], color=fs.FAINT, lw=1.6, zorder=1)
        ax.plot([gr[i], ps[i]], [y[i], y[i]], color=fs.FAINT, lw=1.6, zorder=1)
        if not np.isnan(gid[i]):
            ax.scatter(gid[i], y[i], marker="^", s=34, facecolor="none", edgecolor=fs.YELLOW, linewidths=0.9, zorder=2)
    for v, m, c, lab in ((ref, "s", fs.BLUE, "exact circuit"), (ps, "o", fs.ORANGE, "QCPT"), (gr, "^", fs.YELLOW, "count-greedy")):
        ax.errorbar(v, y, xerr=[se(p) for p in v], fmt="none", ecolor=fs.INK, elinewidth=0.5, capsize=0, zorder=2)
        ax.scatter(v, y, marker=m, s=34, color=c, edgecolor=fs.INK, linewidths=0.5, zorder=3, label=lab)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=7.5); ax.set_xlim(0, 1.045); ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel(r"return probability $P(0\cdots0)$"); ax.grid(axis="x", lw=0.4, color=fs.FAINT, zorder=0)
    if twoq is not None:
        sav = 100 * (1 - twoq[:, 1] / twoq[:, 0]); x0 = 1.13
        ax.scatter(np.full(n, x0), y, marker="s", s=170, color=TURBO(NORM(sav)), edgecolor=fs.INK, linewidths=0.4, clip_on=False, zorder=3)
        for yi, s_, a_, b_ in zip(y, sav, twoq[:, 0], twoq[:, 1]):
            ax.text(x0, yi, f"{s_:.0f}", ha="center", va="center", fontsize=6.5, color="white" if s_ < 12 or s_ > 33 else fs.INK, clip_on=False)
        ax.text(x0, n - 0.05, "2q\nsaved\n(\\%)", ha="center", va="bottom", fontsize=7, clip_on=False)


fig, axs = plt.subplots(1, 2, figsize=(WIDE, 2.75), layout="constrained", gridspec_kw={"width_ratios": [1, 1.12]})
mirror_panel(axs[0], [s for s in d["mq_names"]], d["mq_ref"], d["mq_pass"], d["mq_greedy"], d["mq_greedy_ideal"], int(d["mq_shots"]))
tag(axs[0], "(a)")
mirror_panel(axs[1], [s for s in d["ma_names"]], d["ma_ref"], d["ma_pass"], d["ma_greedy"], d["ma_greedy_ideal"], int(d["ma_shots"]), d["ma_2q"])
axs[1].set_xticks([0, 0.25, 0.5, 0.75, 1]); tag(axs[1], "(b)")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="outside lower center", ncol=3, fontsize=7.5, frameon=False, handletextpad=0.3, columnspacing=1.6)
save(fig, "fig_mirror.pdf")

# ---------------------------------------------------------------- reset circuits (Tab. 8)
if "rs_names" in d.files:                  # reset-circuit figure only if the rs_* data exist
    fig, ax = plt.subplots(figsize=(COL, 1.95), layout="constrained")
    n = len(d["rs_names"]); y = np.arange(n)[::-1]
    for i in range(n):
        ax.plot([d["rs_pass"][i], d["rs_exact"][i]], [y[i], y[i]], color=fs.FAINT, lw=2.2, zorder=1)
        red = 100 * (1 - d["rs_pass"][i] / d["rs_exact"][i])
        ax.scatter(d["rs_exact"][i], y[i], marker="s", s=44, color=fs.BLUE, edgecolor=fs.INK, linewidths=0.5, zorder=3, label="exact-only" if i == 0 else None)
        ax.scatter(d["rs_pass"][i], y[i], marker="o", s=44, color=TURBO(NORM(red)), edgecolor=fs.INK, linewidths=0.5, zorder=3, label="QCPT" if i == 0 else None)
        ax.text(d["rs_exact"][i] + 9, y[i], f"$-{red:.1f}\\%$", va="center", ha="left", fontsize=7.5)
    ax.set_yticks(y); ax.set_yticklabels([f"{s}\n$n{{=}}{int(k)}$, " + ("QCEC" if q else "exhaustive") + f", ${tm:g}$\\,s" for s, k, q, tm in zip(d["rs_names"], d["rs_n"], d["rs_qcec"], d["rs_time"])], fontsize=7)
    ax.set_xlim(0, 430); ax.set_xlabel("two-qubit gates"); ax.grid(axis="x", lw=0.4, color=fs.FAINT, zorder=0); ax.set_ylim(-0.6, n - 0.4)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2, fontsize=7.5, frameon=False)
    save(fig, "fig_reset_data.pdf")
print("fig_cost.pdf fig_mirror.pdf fig_reset_data.pdf saved")

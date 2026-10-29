#!/usr/bin/env python3
"""Drawing layer of the figures. Reads ONLY data/make_figures.npz
(built by figures_data.py; built once if missing). Style: figstyle
(LaTeX, Wong palette), colour is never the only carrier (fill patterns)."""
import os, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # paper/figstyle.py (vendored)
import figstyle as fs
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, "data", "make_figures.npz")
if not os.path.exists(NPZ):
    subprocess.run([sys.executable, os.path.join(HERE, "figures_data.py")], check=True)
d = np.load(NPZ)
fs.apply()
COL, WIDE = 3.35, 6.9
EXACT, OURS, BASE = fs.BLUE, fs.GREEN, fs.GREY

def clean(ax):
    ax.grid(axis="y", lw=0.4, color=fs.FAINT, zorder=0)

# ---------------------------------------------------------------- safety
fig, ax = plt.subplots(figsize=(COL, 1.9))
labels = ["count-greedy\nsubstitution", "analysis,\nno gate", "analysis\n+ gate"]
bad = d["safety_bad"]; n = int(d["safety_n"])
bars = ax.bar(labels, bad, color=[BASE, fs.SKY, OURS], width=0.62, zorder=3,
              hatch=["//", "..", ""], edgecolor=fs.INK, linewidth=0.5)
ax.set_ylabel(f"corrupted circuits (of {n})")
ax.set_ylim(0, n * 0.62); clean(ax)
for b_, v in zip(bars, bad):
    ax.text(b_.get_x() + b_.get_width() / 2, v + 0.12, f"{v}/{n}", ha="center", va="bottom", fontsize=7)
fig.savefig(os.path.join(HERE, "fig_safety.pdf"), bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- budget
names = [s.replace(" ", "\n", 1) if s.startswith("Qiskit") else s.replace("-", "-\n") for s in d["budget_names"]]
infid, count = d["budget_infid"], d["budget_count"]
cols = [BASE, BASE, EXACT, OURS]; hat = ["//", "\\\\", "..", ""]
fig, axs = plt.subplots(1, 2, figsize=(WIDE, 2.2))
for ax, data, ylab, ttl, fmt in [(axs[0], infid, "summed estimated infidelity", "(a)", "{:.2f}"),
                                 (axs[1], count, "summed two-qubit gate count", "(b)", "{:d}")]:
    ax.bar(range(len(names)), data, color=cols, hatch=hat, edgecolor=fs.INK, linewidth=0.5, width=0.7, zorder=3)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel(ylab); fs.panel(ax, ttl); clean(ax); ax.margins(y=0.16)
    for i, v in enumerate(data):
        ax.text(i, v * 1.015, fmt.format(int(v) if fmt == "{:d}" else v), ha="center", va="bottom", fontsize=6)
fig.tight_layout(w_pad=1.4)
fig.savefig(os.path.join(HERE, "fig_budget.pdf"), bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- device
dev = [s.replace(" ", "\n", 1) for s in d["dev_names"]]
x = np.arange(len(dev)); w = 0.38
fig, ax = plt.subplots(figsize=(COL, 2.0))
ax.bar(x - w / 2, d["dev_ex"], w, label="exact-only", color=EXACT, hatch="..", edgecolor=fs.INK, linewidth=0.5, zorder=3)
ax.bar(x + w / 2, d["dev_ours"], w, label="this work", color=OURS, edgecolor=fs.INK, linewidth=0.5, zorder=3)
ax.set_xticks(x); ax.set_xticklabels(dev, fontsize=6)
ax.set_ylabel("estimated circuit infidelity"); ax.set_ylim(0, max(d["dev_ex"]) * 1.45); clean(ax)
for xi, e, rd in zip(x, d["dev_ex"], d["dev_red"]):
    ax.text(xi, e + 0.015, rf"$-{rd:.0f}\%$", ha="center", va="bottom", fontsize=6.5)
ax.legend(loc="upper left", ncol=2)
fig.savefig(os.path.join(HERE, "fig_device.pdf"), bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- scale
if "fam_names" in d.files:
    lab = {"grover/cu": "Grover\noracle", "grover/mixed": "Grover\nhalf-unc.", "adder/cla": "CLA\nadder",
           "adder/cu": "ripple\nadder", "adder/live": "ripple,\nlive\ncarry", "modular/cu": "modular\nincr.",
           "multiplier/live": "array\nmult."}
    fam = [lab.get(f, f) + f"\n($n{{=}}{k}$)" for f, k in zip(d["fam_names"], d["fam_n"])]
    fig, axs = plt.subplots(1, 2, figsize=(WIDE, 2.2))
    axs[0].bar(range(len(fam)), d["fam_red"], color=OURS, edgecolor=fs.INK, linewidth=0.5, width=0.7, zorder=3)
    axs[0].set_xticks(range(len(fam))); axs[0].set_xticklabels(fam, fontsize=6.5)
    axs[0].set_ylabel(r"mean two-qubit reduction (\%)"); fs.panel(axs[0], "(a)", x=0.0, y=1.03, ha="left"); clean(axs[0]); axs[0].margins(y=0.18)
    for i, (v, k) in enumerate(zip(d["fam_red"], d["fam_n"])):
        axs[0].text(i, v + 0.6, f"{v:.0f}", ha="center", va="bottom", fontsize=6.8)
    ok = d["qcec_ok"]
    axs[1].scatter(d["qcec_n"][ok], d["qcec_t"][ok], s=12, color=OURS, marker="o", zorder=3, label="QCEC, equivalent")
    if "cross_n" in d.files:
        m = ~np.isnan(d["cross_exh"])
        axs[1].plot(d["cross_n"][m], d["cross_exh"][m], "s--", color=fs.ORANGE, ms=3.5, lw=1.0, label="exhaustive (crossover series)")
        axs[1].plot(d["cross_n"], d["cross_qcec"], "^:", color=fs.BLUE, ms=3.2, lw=1.0, label="QCEC (crossover series)")
        if "cross_blocks" in d.files:
            axs[1].plot(d["cross_n"], d["cross_qcec_pinned"], "v:", color=fs.SKY, ms=3.2, lw=1.0, label="QCEC, ancillas pinned")
            axs[1].plot(d["cross_n"], d["cross_blocks"], "o-", color=fs.INK, ms=3.0, lw=1.0, label="blocks of difference")
            axs[1].plot(d["cross_n"], d["cross_mc"], "D-.", color=fs.PINK, ms=2.8, lw=1.0, label="Monte Carlo ($K=24$)")
    if (~ok).any():
        axs[1].scatter(d["qcec_n"][~ok], d["qcec_t"][~ok], s=14, color=fs.ORANGE, marker="x", zorder=3, label="QCEC, not certified")
    axs[1].set_yscale("log"); axs[1].set_xlabel(r"circuit width $n$ (qubits)"); axs[1].set_ylabel("verification time (s)")
    fs.panel(axs[1], "(b)"); axs[1].grid(True, which="major", lw=0.35, color=fs.FAINT, zorder=0); axs[1].set_xticks(range(6, 25, 2))
    # legenda w pustym pasmie nad danymi: gorna granica osi ~2 dekady nad najwyzszym punktem
    ymax = max(np.nanmax(d["qcec_t"]), np.nanmax(d["cross_exh"]) if "cross_exh" in d.files else 0)
    ymax = max(ymax, np.nanmax(d["cross_mc"]) if "cross_mc" in d.files else 0, np.nanmax(d["cross_qcec_pinned"]) if "cross_qcec_pinned" in d.files else 0)
    axs[1].set_ylim(top=ymax * 10 ** (3.2 if "cross_blocks" in d.files else 2.6))
    axs[1].legend(loc="upper right", fontsize=6.0, framealpha=0.95, borderaxespad=0.4, ncol=2)
    fig.tight_layout(w_pad=1.6)
    fig.savefig(os.path.join(HERE, "fig_scale.pdf"), bbox_inches="tight"); plt.close(fig)
    print("fig_scale.pdf saved")
else:
    print("WARNING: no scale data in the npz; fig_scale.pdf NOT rebuilt")
print("fig_safety.pdf fig_budget.pdf fig_device.pdf saved")

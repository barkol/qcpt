#!/usr/bin/env python3
"""QCPT times on 33 circuits (C++ core). (a) time to the best circuit (circle, colour = 2q saving relative to stage 1, Turbo map) and the whole race (square);
(b) windowed resynthesis: Python vs C++ (same cost on all circuits). Data: data/times.npz (times_data.py)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import figstyle as fs
import matplotlib.pyplot as plt
from matplotlib import cm, colors
HERE = os.path.dirname(os.path.abspath(__file__)); OUTDIR = os.environ.get("FIG_OUT", HERE); fs.apply()
d = np.load(os.path.join(HERE, "data", "times.npz")); names = [str(n) for n in d["names"]]
sav = 1 - d["final"] / d["start"]; tb = d["t_best"]; wall = d["wall"]
order = np.argsort(np.where(np.isnan(tb), 0.0, tb))[::-1]         # from the slowest (top); circuits without improvement at the bottom
lab = lambda s: s.replace("_", r"\_")
fig, (a, b) = plt.subplots(2, 1, figsize=(fs.COL1, 4.6), gridspec_kw=dict(height_ratios=[3.9, 1.5]), layout="constrained")
fig.get_layout_engine().set(h_pad=0.05, hspace=0.04, rect=(0, 0, 1, 0.975))
norm = colors.Normalize(0, 0.7); cmap = plt.get_cmap("turbo")
y = np.arange(len(order))
for yi, k in zip(y, order):
    a.plot([min(wall[k], tb[k]) if not np.isnan(tb[k]) else wall[k], wall[k]], [yi, yi], color=fs.FAINT, lw=0.8, zorder=1)
    a.plot(wall[k], yi, "s", ms=3.0, mfc="white", mec=fs.INK, mew=0.6, zorder=3)
    if not np.isnan(tb[k]): a.plot(tb[k], yi, "o", ms=3.6, mfc=cmap(norm(sav[k])), mec=fs.INK, mew=0.4, zorder=4)
a.set_xscale("log"); a.set_yticks(y); a.set_yticklabels([r"\texttt{" + lab(names[k]) + "}" for k in order], fontsize=4.6); a.set_ylim(-0.7, len(order) - 0.3); a.invert_yaxis()
a.set_xlabel("time [s]"); a.tick_params(axis="y", length=0); a.grid(axis="x", color=fs.FAINT, lw=0.4)
a.plot([], [], "o", ms=3.6, mfc="white", mec=fs.INK, mew=0.5, label="to the best circuit"); a.plot([], [], "s", ms=3.0, mfc="white", mec=fs.INK, mew=0.6, label="whole race"); a.legend(loc="lower right", fontsize=6.2)
cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=a, orientation="horizontal", location="top", fraction=0.035, pad=0.02, aspect=40); cb.set_label("saving of 2q gates relative to stage 1", fontsize=6.5); cb.ax.tick_params(labelsize=6)
b.plot([0.01, 1000], [0.01, 1000], color=fs.GREY, lw=0.6, label=r"equal"); b.plot([0.01, 1000], [0.001, 100], color=fs.GREY, lw=0.6, ls="--", label=r"$10\times$ faster"); b.plot([0.01, 1000], [0.0001, 10], color=fs.GREY, lw=0.6, ls=":", label=r"$100\times$ faster")
b.plot(np.maximum(d["win_py"], 0.01), np.maximum(d["win_cpp"], 0.01), "o", ms=3.0, mfc=fs.BLUE, mec="none", alpha=0.85)
b.set_xscale("log"); b.set_yscale("log"); b.set_xlim(0.03, 600); b.set_ylim(0.01, 60); b.set_xlabel("Python windows [s]"); b.set_ylabel("C++ windows [s]")
b.legend(loc="upper left", fontsize=6.2)
b.text(0.97, 0.05, r"all 33 circuits: %.0f\,s $\to$ %.0f\,s" % (d["win_py"].sum(), d["win_cpp"].sum()), transform=b.transAxes, ha="right", fontsize=6.4)
fig.canvas.draw(); r = fig.canvas.get_renderer(); W, H = fig.get_size_inches() * fig.dpi        # panel tags of vertically stacked panels: at the top edge of each outline, common left edge
bbs = {"(a)": [a.get_tightbbox(r), cb.ax.get_tightbbox(r)], "(b)": [b.get_tightbbox(r)]}; x0 = min(bb.x0 for v in bbs.values() for bb in v) / W
for lab_, v in bbs.items(): fig.text(x0, max(bb.y1 for bb in v) / H + 2.0 / 72.0 / fig.get_size_inches()[1], r"\textbf{" + lab_ + "}", ha="left", va="bottom", fontsize=8)
fig.savefig(os.path.join(OUTDIR, "fig_times.pdf")); fig.savefig(os.path.join(OUTDIR, "fig_times_preview.png"), dpi=220); print("fig_times saved")

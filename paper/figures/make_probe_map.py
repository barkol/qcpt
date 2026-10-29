#!/usr/bin/env python3
"""Figure of the failed couplers of ibm_aachen (probe of 29 October 2026): (a) map of the chip, the colour and the width of a coupler give its error per CZ measured by the probe, circles mark
the regions of the paper; (b) calibration against probe for every coupler. Data: data/probe_map.npz (probe_map_data.py). Output: fig_probe_map.pdf (FIG_OUT)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import figstyle as fs
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.collections import LineCollection
HERE = os.path.dirname(os.path.abspath(__file__)); OUTDIR = os.environ.get("FIG_OUT", HERE)
d = np.load(os.path.join(HERE, "data", "probe_map.npz")); fs.apply()
x, y, ea, eb, pe, ce = d["x"], d["y"], d["ea"], d["eb"], d["probe"], d["cal"]
infl = float(d["median_probe"] / d["median_cal"])
fig = plt.figure(figsize=(fs.COL2, 3.15)); gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.16)
ax = fig.add_subplot(gs[0]); bx = fig.add_subplot(gs[1])
# (a) map
seg = [[(x[a], y[a]), (x[b], y[b])] for a, b in zip(ea, eb)]
bad = pe > 0.012
cols = np.where(bad, np.where(pe > 0.05, fs.ORANGE, fs.YELLOW), fs.FAINT)
lw = np.where(bad, 2.4, 1.1)
ax.add_collection(LineCollection([s for s, b in zip(seg, bad) if not b], colors=fs.FAINT, linewidths=1.1, zorder=1))
ax.add_collection(LineCollection([s for s, b, p in zip(seg, bad, pe) if b and p <= 0.05], colors=fs.YELLOW, linewidths=2.4, zorder=2))
ax.add_collection(LineCollection([s for s, b, p in zip(seg, bad, pe) if b and p > 0.05], colors=fs.ORANGE, linewidths=2.8, zorder=3))
ax.scatter(x, y, s=5, color=fs.GREY, zorder=4, linewidths=0)
def ring(qs, color, ls, label):
    q = np.array(qs); ax.scatter(x[q], y[q], s=52, facecolors="none", edgecolors=color, linewidths=1.0, linestyle=ls, zorder=5, label=label)
used = sorted(set(d["used_6"]) | set(d["used_8"]) | set(d["used_10"]) | set(d["used_12"]))
ring(used, fs.GREEN, "-", "regions used (6 to 12 qubits)")
for a, b in ((133, 134), (153, 154), (104, 105), (57, 67)):
    k = [i for i in range(len(ea)) if (ea[i], eb[i]) == (a, b)][0]
    ax.annotate(f"{a}--{b}\n{pe[k]*100:.1f}\\%", ((x[a] + x[b]) / 2, (y[a] + y[b]) / 2), xytext=((-40, 9) if a in (133, 153) else (6, 7)), textcoords="offset points", fontsize=5.8, color=fs.INK, zorder=6, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
ax.set_aspect("equal"); ax.axis("off"); ax.autoscale_view(); ax.margins(0.03)
ax.plot([], [], color=fs.ORANGE, lw=2.8, label=r"coupler $>5\%$ per CZ"); ax.plot([], [], color=fs.YELLOW, lw=2.4, label=r"coupler $1.2$--$5\%$"); ax.plot([], [], color=fs.FAINT, lw=1.1, label=r"healthy coupler")
ax.legend(loc="upper center", fontsize=5.8, ncol=2, borderaxespad=0.0, handlelength=1.4, labelspacing=0.25, columnspacing=1.0, bbox_to_anchor=(0.5, 0.0), frameon=False)
fs.panel(ax, "(a)")
# (b) calibration against probe
ok = ~np.isnan(pe) & ~np.isnan(ce)
bx.scatter(ce[ok & ~bad], pe[ok & ~bad], s=7, color=fs.GREY, linewidths=0, zorder=3, label="coupler")
bx.scatter(ce[ok & bad], pe[ok & bad], s=11, color=fs.ORANGE, linewidths=0, zorder=4, label=r"probe $>1.2\%$")
g = np.array([4e-4, 4e-2]); bx.plot(g, g * infl, color=fs.BLUE, lw=0.9, ls="--", zorder=2, label=rf"probe $={infl:.1f}\times$ calibration")
for a, b in ((133, 134), (153, 154)):
    k = [i for i in range(len(ea)) if (ea[i], eb[i]) == (a, b)][0]
    bx.annotate(f"{a}--{b}", (ce[k], pe[k]), xytext=((12, 6) if a == 133 else (22, -16)), textcoords="offset points", fontsize=5.8, color=fs.INK, arrowprops=dict(arrowstyle="-", lw=0.4, color=fs.GREY))
bx.set_xscale("log"); bx.set_yscale("log"); bx.set_xlim(4e-4, 4e-2); bx.set_ylim(6e-4, 0.2)
bx.set_xlabel(r"error per CZ, calibration (03:20)"); bx.set_ylabel(r"error per CZ, probe")
bx.grid(True, which="major", lw=0.35, color=fs.FAINT, zorder=0); bx.legend(loc="lower right", fontsize=5.6, markerscale=1.3)
fig.tight_layout(w_pad=0.6)
fs.panel(bx, "(b)")
fig.savefig(os.path.join(OUTDIR, "fig_probe_map.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(OUTDIR, "fig_probe_map_preview.png"), dpi=220, bbox_inches="tight"); plt.close(fig)
print("fig_probe_map saved")

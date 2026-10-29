#!/usr/bin/env python3
"""Flowchart of the QCPT compiler (stage 1: per-gate choice + certificate; stage 2: resynthesis + certificate; return to the last
certified circuit). Vector figure, unit = 1 pt, width = \\textwidth (483.7 pt)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import figstyle as fs
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__)); OUTDIR = os.environ.get("FIG_OUT", HERE)
fs.apply(); plt.rcParams.update({"font.size": 7.5})
BLUE, VERM, GRN, INK, GREY, FAINT = fs.BLUE, fs.ORANGE, fs.GREEN, fs.INK, fs.GREY, fs.FAINT
W, H = 483.697, 222.0
fig = plt.figure(figsize=(W / 72.27, H / 72.27)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")


def T(x, y, s, size=7.5, **kw):
    kw.setdefault("ha", "center"); kw.setdefault("va", "center"); return ax.text(x, y, s, fontsize=size, color=kw.pop("color", INK), zorder=kw.pop("zorder", 8), **kw)


def box(x, y, w, h, s, fc, ec=INK, size=7.5, lw=0.8):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=4", fc=fc, ec=ec, lw=lw, zorder=4)); T(x, y, s, size, linespacing=1.12)


def diamond(x, y, w, h, s, ec=GRN, size=7.2):
    ax.add_patch(Polygon([[x - w / 2, y], [x, y + h / 2], [x + w / 2, y], [x, y - h / 2]], fc="white", ec=ec, lw=1.6, zorder=4)); T(x, y, s, size, linespacing=1.1)


def arrow(pts, c=INK, lw=0.9):
    for a, b in zip(pts[:-2], pts[1:-1]): ax.plot([a[0], b[0]], [a[1], b[1]], color=c, lw=lw, zorder=3, solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=7, lw=lw, color=c, zorder=3, shrinkA=0, shrinkB=0))


def lab(x, y, s): T(x, y, s, 7, color=INK, fontweight="bold", zorder=9)


# stage backgrounds
ax.add_patch(FancyBboxPatch((3, 150), W - 6, 68, boxstyle="round,pad=0,rounding_size=6", fc="#eaf3fa", ec="none", zorder=0)); T(10, 211, "STAGE 1: per-gate certified selection", 7.5, ha="left", color=BLUE, fontweight="bold")
ax.add_patch(FancyBboxPatch((103, 62), 313, 74, boxstyle="round,pad=0,rounding_size=6", fc="#fbeee3", ec="none", zorder=0)); T(110, 129, "STAGE 2: certified resynthesis", 7.5, ha="left", color=VERM, fontweight="bold")
# row A: stage 1
xa = [48, 144, 240, 336, 432]; ya = 181; bw, bh = 84, 40
texts = ["Toffoli network,\nerror model,\nancillae $|0\\rangle$", "Reachable\nsupport of\neach Toffoli", "Candidates:\nexact, drop,\n3-CX, pair, window", "Cheapest under\ninfidelity\nbudget", "Check (R),(C),(W);\nrejected\n$\\to$ exact"]
for x, s in zip(xa, texts): box(x, ya, bw, bh, s, "white", BLUE, size=7.2)
for i in range(4): arrow([(xa[i] + bw / 2, ya), (xa[i + 1] - bw / 2, ya)])
# A5 -> B1
arrow([(xa[4], ya - bh / 2), (xa[4], 146), (48, 146), (48, 117)])
# row B
yb = 100
diamond(48, yb, 84, 34, "stage-1\ncertificate?"); diamond(150, yb, 84, 34, "classical,\n$\\leq 10$ qubits?", ec=GREY)
box(262, yb, 104, 40, "Beam search and\ntemplate search over\nMargolus gates", "white", VERM); diamond(368, yb, 84, 34, "stage-2\ncertificate?")
box(452, yb, 56, 40, "Output:\ncertified\ncircuit", "#e3f3ec", GRN, lw=1.4)
arrow([(48 + 42, yb), (150 - 42, yb)]); lab(98, yb + 6, "yes")
arrow([(150 + 42, yb), (262 - 52, yb)]); lab(200, yb + 6, "yes")
arrow([(262 + 52, yb), (368 - 42, yb)]); arrow([(368 + 42, yb), (452 - 28, yb)]); lab(412, yb + 6, "yes")
# B2 "no" -> exit (last certified circuit = stage-1 circuit), skips the resynthesis
arrow([(150, yb - 17), (150, 56), (452, 56), (452, yb - 20)], c=GREY); lab(160, yb - 26, "no")
# return to the last certified circuit (fail-closed)
box(262, 28, 130, 22, "fallback: last certified circuit", "#f1f1f1", GREY, size=7)
arrow([(48, yb - 17), (48, 28), (262 - 65, 28)], c=GREY); lab(58, 70, "no")
arrow([(368, yb - 17), (368, 28), (262 + 65, 28)], c=GREY); lab(378, 70, "no")
arrow([(262 + 65, 22), (452, 22), (452, yb - 20)], c=GREY)
fig.savefig(os.path.join(OUTDIR, "fig_flow.pdf")); fig.savefig(os.path.join(OUTDIR, "fig_flow_preview.png"), dpi=170); print("fig_flow saved")

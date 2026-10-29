#!/usr/bin/env python3
"""Concept figure: picture story of QCPT in 5 steps (first figure of the paper). Vector figure, unit = 1 pt,
width = \\textwidth of the article class (483.7 pt). Palette as in the whole paper: before/exact = blue, QCPT = vermilion,
count-greedy/warning = yellow, certificate = green. Numbers (6 CX per Toffoli, x0 260 -> 370 (measurement, 9 circuits), -31%) from dyno.npz and the tables."""
import os, sys, textwrap
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import figstyle as fs
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch, Polygon

HERE = os.path.dirname(os.path.abspath(__file__)); OUTDIR = os.environ.get("FIG_OUT", HERE)
d = np.load(os.path.join(HERE, "data", "dyno_fam2.npz")); fs.apply()
plt.rcParams.update({"font.size": 7.5})
WPT, HPT = 483.697, 232.0
BLUE, VERM, YEL, GRN, INK, GREY, FAINT = fs.BLUE, fs.ORANGE, fs.YELLOW, fs.GREEN, fs.INK, fs.GREY, fs.FAINT
fig = plt.figure(figsize=(WPT / 72.27, HPT / 72.27)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, WPT); ax.set_ylim(0, HPT); ax.axis("off")
PW = WPT / 5


def T(x, y, s, size=7.5, **kw):
    kw.setdefault("ha", "center"); kw.setdefault("va", "center"); return ax.text(x, y, s, fontsize=size, color=kw.pop("color", INK), **kw)


def wire(y, x0, x1, c=GREY): ax.plot([x0, x1], [y, y], color=c, lw=0.7, zorder=1, solid_capstyle="butt")


def ctrl(x, y, c=INK): ax.add_patch(Circle((x, y), 2.0, fc=c, ec=c, zorder=4))


def targ(x, y, c=INK, r=3.8):
    ax.add_patch(Circle((x, y), r, fc="white", ec=c, lw=0.8, zorder=4)); ax.plot([x - r, x + r], [y, y], color=c, lw=0.8, zorder=5); ax.plot([x, x], [y - r, y + r], color=c, lw=0.8, zorder=5)


def cx(x, yc, yt, c=INK): ax.plot([x, x], [yc, yt], color=c, lw=0.8, zorder=3); ctrl(x, yc, c); targ(x, yt, c)


def toff(x, y1, y2, yt, c=INK): ax.plot([x, x], [y1, yt], color=c, lw=0.8, zorder=3); ctrl(x, y1, c); ctrl(x, y2, c); targ(x, yt, c)


def chip(x, y, s, c, w=34, h=13, tc="white", size=8):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=3", fc=c, ec=INK, lw=0.5, zorder=3)); T(x, y, s, size, color=tc, fontweight="bold", zorder=6)


def arrow(x0, y0, x1, y1, c=INK, lw=0.9): ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=lw, color=c, zorder=5, shrinkA=0, shrinkB=0))


def title(k, lines):
    cx_ = k * PW + 13; ax.add_patch(Circle((cx_, 213), 8.2, fc=VERM, ec=INK, lw=0.6, zorder=3)); T(cx_, 213, str(k + 1), 8.5, color="white", fontweight="bold", zorder=6)
    T(k * PW + 26, 213, lines, 8, ha="left", fontweight="bold", linespacing=1.05)


def cap(k, s, wrap=21):
    T(k * PW + PW / 2, 31, "\n".join(textwrap.wrap(s, wrap)), 6.8, linespacing=1.15, va="center")


# ---- background: panel bands and the arrows between them
for k in range(5):
    ax.add_patch(FancyBboxPatch((k * PW + 3, 6), PW - 6, HPT - 12, boxstyle="round,pad=0,rounding_size=5", fc="#fafafa", ec=FAINT, lw=0.8, zorder=0))
    if k < 4: ax.add_patch(Polygon([[(k + 1) * PW - 5, 196], [(k + 1) * PW + 3, 202], [(k + 1) * PW - 5, 208]], fc=GREY, ec="none", zorder=6))

# ---- 1: standard decomposition
k = 0; title(k, "Exact\ndecomposition")
ys = [180, 164, 148]; x0 = k * PW
for y in ys: wire(y, x0 + 8, x0 + PW - 8)
xs = [x0 + 26, x0 + 48, x0 + 70]
for x in xs: toff(x, ys[0], ys[1], ys[2])
for x in xs:
    arrow(x, 141, x, 128)
    chip(x, 117, "6 CX", BLUE, w=19, h=12, size=6)
ax.plot([xs[0] - 10, xs[0] - 10, xs[2] + 10, xs[2] + 10], [100, 95, 95, 100], color=INK, lw=0.8)
chip(x0 + PW / 2, 80, "18 CX", BLUE, w=44, h=16, size=9)
cap(k, "Exactly decomposed, every Toffoli costs 6 CNOTs.")

# ---- 2: kontekst (osiagalny nosnik)
k = 1; title(k, "Read the\ncontext"); x0 = k * PW
cols = ["00", "01", "10", "11"]; cw, chh = 17.5, 17.5; gx = x0 + 20; gy = 150
for j, c_ in enumerate(cols): T(gx + j * cw + cw / 2, gy + 2 * chh + 7, c_, 7.5)
T(gx - 2, gy + 2 * chh + 7, "$ab$", 7.5, ha="right"); T(gx - 5, gy + 1.5 * chh, "0", 7.5); T(gx - 5, gy + 0.5 * chh, "1", 7.5); T(gx - 7, gy + 2 * chh + 7 - 0, "", 7)
for j in range(4):
    for r, tt in ((1, 0), (0, 1)):
        miss = (cols[j] == "10" and tt == 0)
        ax.add_patch(Rectangle((gx + j * cw, gy + r * chh), cw - 2, chh - 2, fc="white" if miss else BLUE, ec=INK, lw=0.5, alpha=1, hatch="////" if miss else None, zorder=3))
        if not miss: ax.add_patch(Rectangle((gx + j * cw, gy + r * chh), cw - 2, chh - 2, fc="white", ec="none", alpha=0.55, zorder=4))
T(gx - 15, gy + chh, "$t$", 7.5)
T(x0 + PW / 2, 122, "one input never reaches\nthe gate", 7, linespacing=1.1)
arrow(x0 + PW / 2, 108, x0 + PW / 2, 97); chip(x0 + PW / 2, 84, "cost 3 CX", VERM, w=48, h=14, size=8)
cap(k, "Track which inputs can reach each Toffoli; the fewer, the cheaper the gate.")

# ---- 3: cheaper gate
k = 2; title(k, "Pick a cheaper\ngate"); x0 = k * PW
ys3 = [182, 168, 154]
for y in ys3: wire(y, x0 + 6, x0 + 36); wire(y, x0 + 52, x0 + PW - 6, c=VERM)
toff(x0 + 21, *ys3); arrow(x0 + 37, 168, x0 + 51, 168)
for xx, (a_, b_) in zip((x0 + 61, x0 + 73, x0 + 85), ((1, 2), (0, 2), (1, 2))): cx(xx, ys3[a_], ys3[2], VERM)
chip(x0 + 21, 133, "6 CX", BLUE, w=28, h=12, size=7); chip(x0 + 73, 133, "3 CX", VERM, w=28, h=12, size=7)
T(x0 + PW / 2, 113, "its stray phase hits\nonly the input that\nnever occurs", 6.8, linespacing=1.1)
# budzet bledu
ax.add_patch(Rectangle((x0 + 18, 82), PW - 36, 7.5, fc="white", ec=INK, lw=0.6, zorder=3)); ax.add_patch(Rectangle((x0 + 18, 82), (PW - 36) * 0.58, 7.5, fc=VERM, ec=INK, lw=0.6, zorder=4))
T(x0 + PW / 2, 96, "error budget", 7, color=INK)
cap(k, "A 3-CNOT gate is allowed where its phase is never seen; the error budget picks the cheapest.")

# ---- 4: certificate
k = 3; title(k, "Certify"); x0 = k * PW
ax.add_patch(FancyBboxPatch((x0 + 9, 144), PW - 18, 44, boxstyle="round,pad=0,rounding_size=3", fc="white", ec=YEL, lw=1.4, zorder=3))
T(x0 + 57, 175, "state by state", 6.8, fontweight="bold"); T(x0 + 57, 162, "accepts wrong\ncircuits", 6.8, linespacing=1.05)
ax.add_patch(Polygon([[x0 + 12, 152], [x0 + 22, 152], [x0 + 17, 163]], fc=YEL, ec=INK, lw=0.5, zorder=5)); T(x0 + 17, 155, "!", 6.5, fontweight="bold", zorder=6)
ax.add_patch(FancyBboxPatch((x0 + 9, 88), PW - 18, 48, boxstyle="round,pad=0,rounding_size=3", fc="white", ec=GRN, lw=1.8, zorder=3))
T(x0 + 56, 124, "one global phase", 6.0, fontweight="bold"); T(x0 + 57, 109, "on every valid\ninput (QCPT)", 6.8, linespacing=1.05)
ax.plot([x0 + 12, x0 + 16, x0 + 24], [108, 102, 116], color=GRN, lw=2.0, zorder=6, solid_joinstyle="round")
cap(k, "QCPT accepts only circuits equal to the exact one on all valid inputs; otherwise it returns the exact circuit.")

# ---- 5: result (same micro-view, longer reach)
k = 4; title(k, "Same chip,\nlonger reach"); x0 = k * PW
px0, py0, pw_, ph_ = x0 + 14, 100, PW - 26, 88
ax.add_patch(Rectangle((px0, py0), pw_, ph_, fc="white", ec=INK, lw=0.6, zorder=2))
LMAX, YMAX = 700.0, 230.0
PEXP = float(d["hw_mod_incr_p"])
x0b, x0a = float(d["hw_mod_incr_x0_q"]), float(d["hw_mod_incr_x0_c"]); xx = np.linspace(0, 1.0, 120); load = xx * LMAX
for x0v, c, lw in ((x0b, BLUE, 1.2), (x0a, VERM, 1.8)):
    yv = load * np.exp(-(load / x0v) ** PEXP); ax.plot(px0 + xx * pw_, py0 + 3 + yv / YMAX * (ph_ - 8), color=c, lw=lw, zorder=4)
ax.fill_between(px0 + xx * pw_, py0 + 3 + load * np.exp(-(load / x0b) ** PEXP) / YMAX * (ph_ - 8), py0 + 3 + load * np.exp(-(load / x0a) ** PEXP) / YMAX * (ph_ - 8), color=VERM, alpha=0.15, lw=0, zorder=3)
T(px0 + pw_ / 2, py0 - 7, "circuit size (2q gates)", 6.8); T(px0 - 4, py0 + ph_ / 2, "correct gates", 6.8, rotation=90)
chip(x0 + PW / 2, 78, "$-31\\%$ CX", VERM, w=54, h=12, size=7); chip(x0 + PW / 2, 63, f"$\\times{x0a / x0b:.2f}$ reach", VERM, w=54, h=12, size=7)
cap(k, "Fewer CNOTs, higher return probability: the same device runs larger circuits correctly.")

fig.savefig(os.path.join(OUTDIR, "fig_concept.pdf")); print("fig_concept.pdf saved")

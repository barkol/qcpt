#!/usr/bin/env python3
"""Comparison of QCPT (human: designed and certified) with the Claude autoresearch loop (robot): table-graphic with icons.
Happy face = better result in the given criterion, sad = worse. Data: data/compare.npz (compare_data.py). Scale 1:1, column 231.85 pt."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import figstyle as fs
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, Wedge, Arc, Ellipse

HERE = os.path.dirname(os.path.abspath(__file__)); OUTDIR = os.environ.get("FIG_OUT", HERE)
d = np.load(os.path.join(HERE, "data", "compare.npz")); fs.apply(); plt.rcParams.update({"font.size": 7.5})
INK, VERM, SKY = fs.INK, fs.ORANGE, fs.SKY
HAP, SAD = "#e3f3ec", "#fcebe0"
WPT = 231.848; LAB = 78.0; CWID = (WPT - LAB) / 2
rows = [("Fewest 2q gates\n(mcx4, adder, rd53)", 31), ("Over 9 circuits\n(2q gates)", 27),
        ("Time, mean of 9:\n(1) to Claude's cost\n(2) to own best\n(3) whole run", 39), ("Cost, mean of 9:\n(1) to Claude's cost\n(2) to own best\n(3) whole run", 39), ("Energy, mean of 9:\n(1) to Claude's cost\n(2) to own best\n(3) whole run", 39),
        ("Correct as a\nsubroutine", 20), ("Reproducible", 20), ("Rewrites beyond\nthe rules", 20)]
HEAD = 46; HPT = HEAD + sum(h for _, h in rows) + 6
fig = plt.figure(figsize=(WPT / 72.27, HPT / 72.27)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, WPT); ax.set_ylim(0, HPT); ax.axis("off")


def T(x, y, s, size=7.2, **kw):
    kw.setdefault("ha", "center"); kw.setdefault("va", "center"); return ax.text(x, y, s, fontsize=size, color=kw.pop("color", INK), zorder=kw.pop("zorder", 8), **kw)


ICON_DIR = os.environ.get("ICON_DIR", os.path.join(HERE, "..", "..", "icons"))
sys.path.insert(0, os.path.join(ICON_DIR, "pylib"))
from svgpath2mpl import parse_path                      # parser SVG -> matplotlib Path (ikony Phosphor, MIT)
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D
import re


def _svg_paths(name):
    s = open(os.path.join(ICON_DIR, name + ".svg")).read()
    return [(parse_path(m.group(1)), bool(m.group(2))) for m in re.finditer(r'<path d="([^"]+)"( opacity="0.2")?', s)]


def _subpaths(path):                                     # podzial zlozonej sciezki na podsciezki (po MOVETO)
    cuts = [i for i, c in enumerate(path.codes) if c == Path.MOVETO] + [len(path.codes)]
    return [Path(path.vertices[a:b], path.codes[a:b]) for a, b in zip(cuts[:-1], cuts[1:])]


def _join(paths):
    return Path(np.concatenate([p.vertices for p in paths]), np.concatenate([p.codes for p in paths]))


def _draw(paths, x, y, sc, cy=128.0, zorder=5):
    T = Affine2D().scale(sc, -sc).translate(x - 128 * sc, y + cy * sc) + ax.transData
    for p, base in paths:
        ax.add_patch(PathPatch(p, transform=T, fc=INK, ec="none", alpha=0.13 if base else 1.0, zorder=zorder + (0 if base else 1), lw=0))


def human(x, y, r, happy):                               # okragla buzia (Phosphor smiley / smiley-sad, duotone)
    _draw(_svg_paths("smiley-duotone" if happy else "smiley-sad-duotone"), x, y, r / 104.0)
    if not happy: ax.add_patch(Ellipse((x + 0.52 * r, y - 0.2 * r), 0.17 * r, 0.3 * r, fc=SKY, ec="none", zorder=8))


def robot(x, y, r, happy):                               # robot head (Phosphor robot) with its own mouth (smile / frown)
    base, main = _svg_paths("robot-duotone")
    sub = _subpaths(main[0]); keep = _join(sub[:4])      # obrys glowy, otwor, dwoje oczu; kratke ust pomijamy
    sc = r / 104.0; _draw([(_subpaths(base[0])[0], True), (keep, False)], x, y, sc, cy=136.0)   # base without the mouth slot
    ye, yc = (y - 0.26 * r, y - 0.86 * r) if happy else (y - 0.64 * r, y - 0.06 * r)       # mouth ends / control point (smile or frown)
    mouth = Path([(x - 0.42 * r, ye), (x, yc), (x + 0.42 * r, ye)], [Path.MOVETO, Path.CURVE3, Path.CURVE3])
    ax.add_patch(PathPatch(mouth, fc="none", ec=INK, lw=1.2, capstyle="round", zorder=8))
    if not happy: ax.plot([x + 0.5 * r] * 2, [y + 0.02 * r, y - 0.2 * r], color=SKY, lw=1.4, zorder=8)


def face(kind, x, y, r, happy): (human if kind == "h" else robot)(x, y, r, happy)


xc = [LAB + CWID / 2, LAB + CWID * 1.5]
# naglowki
face("h", xc[0], HPT - 19, 10.5, True); face("r", xc[1], HPT - 19, 10.0, True)
T(xc[0], HPT - 38, "QCPT", 8.5, fontweight="bold"); T(xc[1], HPT - 38, "Claude loop", 8.5, fontweight="bold")
T(LAB / 2, HPT - 26, "criterion", 7.5, color=fs.GREY, style="italic")
y = HPT - HEAD
qg, lg = d["qcp_gates"], d["llm_gates_S"]
cells = []   # (kolumna, szczesliwy?, tekst)
for ri, (lab, h) in enumerate(rows):
    y0 = y - h; yc = y0 + h / 2
    ax.plot([0, WPT], [y0, y0], color=fs.FAINT, lw=0.5, zorder=1)
    T(4, yc, lab, 6.6 if lab.count("\n") >= 3 else 7.2, ha="left", linespacing=1.1)
    if ri == 0:
        for col in (0, 1):
            vals = qg if col == 0 else lg; oth = lg if col == 0 else qg
            ax.add_patch(Rectangle((LAB + col * CWID + 1, y0 + 1), CWID - 2, h - 2, fc="#f4f6f9", ec="none", zorder=0))
            for k in range(3):
                xk = LAB + col * CWID + CWID * (k + 0.5) / 3; hp = vals[k] <= oth[k]
                ax.add_patch(Rectangle((xk - CWID / 6 + 0.8, y0 + 1.5), CWID / 3 - 1.6, h - 3, fc=HAP if hp else SAD, ec="none", zorder=0))
                face("h" if col == 0 else "r", xk, yc + 4.5, 5.2, hp); T(xk, y0 + 5.5, f"{int(vals[k])}", 7.5, fontweight="bold")
    else:
        spec = {1: ("better: 3\nequal: 6\nworse: 0", "better: 0\nequal: 6\nworse: 3", True),
                2: (f"{d['qct_t_claude']:.1f} s\n{d['qct_t_best']:.0f} s\n{d['qct_t_full']:.0f} s", f"{d['llm_t_best']/60:.1f} min\n{d['llm_t_best']/60:.1f} min\n{d['llm_t_full']/60:.0f} min", True),
                3: ("$\\approx\\$0$\n$\\approx\\$0$\n$\\approx\\$0$", f"\\${d['llm_usd_best']:.2f}\n\\${d['llm_usd_best']:.2f}\n\\${d['llm_usd_full']:.2f}", True),
                4: (f"{d['qct_wh_claude'][0]:.2f}--{d['qct_wh_claude'][1]:.2f} Wh\n{d['qct_wh_best'][0]:.2f}--{d['qct_wh_best'][1]:.2f} Wh\n{d['qct_wh_full'][0]:.1f}--{d['qct_wh_full'][1]:.1f} Wh",
                    f"{d['llm_wh_best'][0]:.0f}--{d['llm_wh_best'][1]:.0f} Wh\n{d['llm_wh_best'][0]:.0f}--{d['llm_wh_best'][1]:.0f} Wh\n{d['llm_wh_full'][0]:.0f}--{d['llm_wh_full'][1]:.0f} Wh", True),
                5: ("certificate", f"{int(d['W_unsound'])} of {int(d['W_accepted'])} wrong", True),
                6: ("yes", "no", True), 7: ("yes", "yes", True)}[ri]
        for col in (0, 1):
            happy = True if ri == 7 else (spec[2] if col == 0 else not spec[2])
            ax.add_patch(Rectangle((LAB + col * CWID + 1, y0 + 1), CWID - 2, h - 2, fc=HAP if happy else SAD, ec="none", zorder=0))
            face("h" if col == 0 else "r", LAB + col * CWID + 12, yc, 6.0, happy)
            T(LAB + col * CWID + 12 + 9, yc, spec[col], 7.0, ha="left", linespacing=1.05)
    y = y0
ax.plot([0, WPT], [y, y], color=fs.INK, lw=0.7, zorder=1); ax.plot([0, WPT], [HPT - HEAD, HPT - HEAD], color=fs.INK, lw=0.7, zorder=1)
fig.savefig(os.path.join(OUTDIR, "fig_compare.pdf")); fig.savefig(os.path.join(OUTDIR, "fig_compare_preview.png"), dpi=220); print("fig_compare saved", round(HPT, 1))

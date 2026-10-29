# Vendored copy of the authors' figure-style module (K. Bartkiewicz), used by figures/make_figures.py.
# Distributed with this repository under its MIT licence (see ../LICENSE). Code below is unchanged.
"""Figure style: LaTeX/Computer Modern, Wong palette, column widths of the two-column layout."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform

INK="#1a1a1a"; GREY="#8c8c8c"; FAINT="#d8d8d8"
ORANGE="#D55E00"; BLUE="#0072B2"; GREEN="#009E73"; PINK="#CC79A7"; YELLOW="#E69F00"; SKY="#56B4E9"
COL1=3.404; COL2=7.083          # two-column layout: \columnwidth 246pt, \textwidth 510pt

def apply():
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}",
        "font.family":"serif", "font.serif":["Computer Modern Roman"],
        "font.size":8, "axes.labelsize":8, "axes.titlesize":8,
        "xtick.labelsize":7, "ytick.labelsize":7,
        "legend.fontsize":6.8, "legend.frameon":False, "legend.handlelength":1.3,
        "axes.linewidth":0.6, "lines.linewidth":1.3,
        "xtick.direction":"in", "ytick.direction":"in",
        "xtick.major.size":2.6, "ytick.major.size":2.6,
        "xtick.major.width":0.6, "ytick.major.width":0.6,
        "axes.spines.top":False, "axes.spines.right":False,
        "axes.edgecolor":INK, "axes.labelcolor":INK, "text.color":INK,
        "xtick.color":INK, "ytick.color":INK,
        "figure.dpi":160, "savefig.dpi":600, "savefig.pad_inches":0.02,
        "axes.axisbelow":True,
    })

class Arrow3D(FancyArrowPatch):
    """3D arrow (standard recipe): projects the end points at every draw"""
    def __init__(self, xs, ys, zs, *a, **kw):
        super().__init__((0,0),(0,0), *a, **kw); self._xyz=(xs,ys,zs)
    def do_3d_projection(self, renderer=None):
        xs,ys,zs=self._xyz
        x,y,_=proj_transform(xs,ys,zs,self.axes.M)
        self.set_positions((x[0],y[0]),(x[1],y[1]))
        return float(np.min(proj_transform(xs,ys,zs,self.axes.M)[2]))

def bare3d(ax, L, labels, pad=1.14, lw=0.7, tick=None):
    """Instead of the matplotlib cube: three axis arrows through the origin."""
    ax.set_axis_off()
    ax.set_xlim(-L,L); ax.set_ylim(-L,L); ax.set_zlim(-L,L)
    ax.set_box_aspect((1,1,1))
    for i,(lab,col) in enumerate(zip(labels,[INK]*3)):
        e=np.zeros(3); e[i]=1
        ax.add_artist(Arrow3D(*[[-L*e[k], L*e[k]] for k in range(3)],
                              mutation_scale=6, lw=lw, arrowstyle='-|>', color=col, zorder=1))
        p=L*pad*e
        ax.text(*p, lab, ha='center', va='center', fontsize=8, color=INK, zorder=6)
    if tick:
        for i in range(3):
            e=np.zeros(3); e[i]=1
            for s in (-tick,tick):
                q=s*e; d=0.028*L
                ax.plot(*[[q[k]-d*(k!=i), q[k]+d*(k!=i)] for k in range(3)], color=INK, lw=0.5, zorder=1)
    ax.scatter([0],[0],[0], s=5, color=INK, zorder=5)


# ─────────────────────────────────────────────────────────────────────────────
# Funkcje pomocnicze przeniesione z art-memristor/code/figstyle.py (linia HOM).
# Colours replaced by the Wong palette defined above.
# ─────────────────────────────────────────────────────────────────────────────
import math
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter, FixedLocator, MaxNLocator

SHADE = "0.88"          # light grey fill of highlighted areas
SINGLE = (COL1, COL1*0.72)   # one panel, column width of the two-column layout
WIDE   = (COL2, COL2*0.36)   # two panels side by side, text width of the two-column layout

def panel(ax, label, x=-0.02, y=1.02, **kw):
    """Panel marker, e.g. '(a)', placed OUTSIDE the panel area --
    just above the top-left corner of the axes -- in bold. Pass the label WITH
    parentheses, e.g. panel(ax, '(a)').

    The default (x=-0.02, y=1.02, ha='right', va='bottom') sits the marker above
    and just left of the axes box so it never overlaps the data; override x/y for
    axis-free panels (e.g. Bloch spheres: panel(ax, '(a)', x=0.0, y=1.05))."""
    kw.setdefault("fontweight", "bold")
    kw.setdefault("ha", "right")
    kw.setdefault("va", "bottom")
    kw.setdefault("fontsize", mpl.rcParams["axes.titlesize"])
    # usetex ignores fontweight -- bold must be requested in the TeX string
    if mpl.rcParams.get("text.usetex", False) and "\\textbf" not in label:
        label = r"\textbf{" + label + "}"
    ax.text(x, y, label, transform=ax.transAxes, clip_on=False, **kw)


def shade_below(ax, y0, label=None, above_label=None, color=SHADE):
    """Shade the region below y0 (e.g. a 'forbidden'/'separable' zone) and
    optionally annotate the lower (and upper) regions, HOM-style."""
    lo, hi = ax.get_ylim()
    ax.axhspan(lo, y0, color=color, lw=0, zorder=0)
    ax.set_ylim(lo, hi)
    x0, x1 = ax.get_xlim()
    xt = x0 + 0.6 * (x1 - x0)
    if above_label:
        ax.annotate(above_label, xy=(xt, y0 + 0.04 * (hi - lo)), fontsize=12)
    if label:
        ax.annotate(label, xy=(xt, y0 - 0.13 * (hi - lo)), fontsize=12)


def zero_axes(ax, x=True, y=True):
    """Thin grey reference lines through the origin (no dashes-heavy clutter)."""
    if y:
        ax.axhline(0.0, color=GREY, lw=0.8, zorder=0)
    if x:
        ax.axvline(0.0, color=GREY, lw=0.8, zorder=0)


# ── Tick conventions (house rules) ─────────────────────────────────────────
# * 4-5 labels per axis (never a dense ladder of ticks).
# * Strip trailing zeros: "1" not "1.00", "0" not "0.00", "0.5" not "0.50".
# * Axes whose data spans the unit interval use the symmetric set
#   {-1, -0.5, 0, 0.5, 1} so the +/- limits read symmetrically.
import math

from matplotlib.ticker import FuncFormatter, FixedLocator, MaxNLocator


def _strip_zeros(x, _=None):
    s = f"{x:.10g}"            # general format: 1.0->"1", 0.50->"0.5", 0.0->"0"
    return "0" if s == "-0" else s


STRIP_FMT = FuncFormatter(_strip_zeros)


def _round_ticks_in_range(lo, hi, want=5):
    """Round-step ticks (1/2/2.5/4/5 x10^k) lying INSIDE [lo,hi] with NO range
    change. Returns 4-5 ticks or None. Larger steps tried first (fewer ticks)."""
    span = hi - lo
    mag = 10 ** math.floor(math.log10(span / (want - 1)))
    # smallest step first -> the densest layout still within the 4-5 cap, which
    # favours conventional steps (20 for a 0..84 axis, not 25).
    for base in (0.5, 1, 2, 2.5, 4, 5, 10):
        step = base * mag
        first = math.ceil(lo / step - 1e-9) * step
        ticks, t = [], first
        while t <= hi + 1e-9 * step:
            ticks.append(t)
            t += step
        if 4 <= len(ticks) <= 5:
            dec = max(0, -math.floor(math.log10(step)) + 1)
            return [round(v, dec) for v in ticks]
    return None


def nice_axis(lo, hi, want=5, max_ext=0.35):
    """Round-number ticks (steps 1/2/2.5/4/5 x10^k) giving 4-5 labels.

    PREFER ticks that fit INSIDE the current range (limits unchanged) -- so a
    time axis ending at 84 keeps [0,84] with ticks 0..80 rather than padding out
    to 100.  Only when no round step fits inside do we EXTEND the range outward
    to round multiples (up to `max_ext` of the span).

    Returns (new_lo, new_hi, ticks); new_lo/new_hi equal lo/hi when not extended.
    Returns None if even extension cannot land a round step in [4,5]."""
    if not (hi > lo):
        return None
    inside = _round_ticks_in_range(lo, hi, want)
    if inside:
        return (lo, hi, inside)
    span = hi - lo
    mag = 10 ** math.floor(math.log10(span / (want - 1)))
    cands = []
    for base in (0.5, 1, 2, 2.5, 4, 5, 10):  # round-number steps (x mag)
        step = base * mag
        slo = math.floor(lo / step + 1e-9) * step
        shi = math.ceil(hi / step - 1e-9) * step
        n = int(round((shi - slo) / step)) + 1
        if 4 <= n <= 5:
            cands.append(((shi - slo) - span, slo, shi, step, n))
    cands = [c for c in cands if c[0] <= max_ext * span]
    if not cands:
        return None
    ext, slo, shi, step, n = min(cands, key=lambda c: (c[0], abs(c[4] - want)))
    dec = max(0, -math.floor(math.log10(step)) + 1)
    return (round(slo, dec), round(shi, dec),
            [round(slo + i * step, dec) for i in range(n)])


def _odd_fallback(lo, hi, want=5):
    """Last resort when round steps need too much range extension: allow
    1.5x/3x steps within the EXISTING range (no extension)."""
    span = hi - lo
    mag = 10 ** math.floor(math.log10(span / (want - 1)))
    best = None
    for mult in (1, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10):
        step = mult * mag
        first = math.ceil(lo / step - 1e-9) * step
        ticks, t = [], first
        while t <= hi + 1e-9 * step:
            ticks.append(t)
            t += step
        n = len(ticks)
        dec = max(0, -math.floor(math.log10(step)) + 1)
        rticks = [round(v, dec) for v in ticks]
        if 4 <= n <= 5:
            return rticks
        if best is None or abs(n - want) < best[0]:
            best = (abs(n - want), rticks)
    return best[1] if best else None


def _apply_nice(ax, name):
    lo0, hi0 = getattr(ax, f"get_{name}lim")()
    inverted = lo0 > hi0
    lo, hi = min(lo0, hi0), max(lo0, hi0)
    axis = getattr(ax, f"{name}axis")
    res = nice_axis(lo, hi)
    if res:
        slo, shi, ticks = res
    else:
        slo, shi, ticks = lo, hi, _odd_fallback(lo, hi)
    if not ticks:
        return
    # y-axis: keep the highest tick clear of the panel marker '(a)' sitting just
    # above the top-left corner -- if the top tick crowds the axis top, lift the
    # upper limit a touch so the topmost label has breathing room.
    if name == "y":
        rng = shi - slo
        if rng > 0 and (max(ticks) - slo) / rng > 0.90:
            shi = slo + (max(ticks) - slo) / 0.86
    getattr(ax, f"set_{name}lim")((shi, slo) if inverted else (slo, shi))
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(STRIP_FMT)


def tidy_ticks(ax, axis="both"):
    """4-5 round-number labels per axis with trailing zeros stripped (extending
    the range outward if needed). Use for non-unit axes (time, angle, ...)."""
    for name in {"x": ["x"], "y": ["y"]}.get(axis, ["x", "y"]):
        _apply_nice(ax, name)


def cap_ticks(fig):
    """Enforce the house rule of 4-5 labels per axis on EVERY axis of `fig`.

    Applied at save time so all figures comply by construction.  Skips:
      * log / symlog axes (their LogLocator decades are left alone), and
      * axes already given an explicit FixedLocator (unit_axis / set_ticks),
    so deliberately-chosen tick sets are preserved.  On the remaining linear
    axes it installs 4-5 ROUND ticks (extending the range outward if needed)
    with trailing zeros stripped."""
    for ax in fig.axes:
        for name in ("x", "y"):
            if getattr(ax, f"get_{name}scale")() != "linear":
                continue
            if isinstance(getattr(ax, f"{name}axis").get_major_locator(), FixedLocator):
                continue
            _apply_nice(ax, name)


def unit_axis(ax, axis="y", lim=1.15):
    """Symmetric unit axis: limits +/-lim, the 5-label set {-1,-0.5,0,0.5,1},
    trailing zeros stripped.  Use for axes whose data spans the unit interval."""
    ticks = [-1, -0.5, 0, 0.5, 1]
    if axis in ("x", "both"):
        ax.set_xlim(-lim, lim)
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(STRIP_FMT)
    if axis in ("y", "both"):
        ax.set_ylim(-lim, lim)
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(STRIP_FMT)

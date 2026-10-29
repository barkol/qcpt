#!/usr/bin/env python3
"""Figures of the mirror-test results: fig_sim (simulation of 33 circuits, noise model from the ibm_aachen calibration), fig_hw (measurement of 9 circuits on ibm_aachen, region B, comparison with the simulation
on the same qubits), (fig_dyno is drawn by make_dyno_fam2.py). Drawing layer; data from data_in/new/manuscript_data.json and data/dyno_new.npz (dyno_new_data.py). Wong palette for categories."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import figstyle as fs
import matplotlib.pyplot as plt
from paneltags import tag, place
HERE = os.path.dirname(os.path.abspath(__file__)); OUTDIR = os.environ.get("FIG_OUT", HERE); fs.apply()
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7.5, "legend.fontsize": 7, "axes.spines.right": False})
M = json.load(open(os.path.join(HERE, "..", "data_in", "new", "manuscript_data.json")))
WIDE = 483.697 / 72.27; BLUE, VERM, GREEN, YEL, SKY, PINK = fs.BLUE, fs.ORANGE, fs.GREEN, fs.YELLOW, fs.SKY, fs.PINK
SRC = {"qct": (VERM, "QCPT before routing"), "qct_post": (GREEN, "QCPT after routing (Qiskit's placement)"), "qct_topo": (SKY, "QCPT with its own routing"), "qiskit_fallback": (fs.GREY, "Qiskit circuit (fall-back)")}
short = lambda s: s.replace("rl_", "").replace("_", r"\_")
def pairs(rows, p0, sg):
    g = {}
    for r in rows: g.setdefault(r["name"], {})[r["variant"]] = r
    return [dict(name=n, q=v["qiskit"][p0], c=v["qct"][p0], sq=v["qiskit"].get(sg) or 0.0, sc=v["qct"].get(sg) or 0.0, xq=v["qiskit"]["twoq"], xc=v["qct"]["twoq"], src=v["qct"].get("source") or v["qct"].get("tag") or "qct") for n, v in g.items()]
sim = pairs(M["sim33"], "p0_sim", "sigma"); hw = pairs(M["hw_regionB"], "p0", "sigma"); simB = pairs([dict(r, p0=r["p0_sim_same_qubits"], sigma=0.0) for r in M["hw_regionB"]], "p0", "sigma")
def slope(pr): g = np.array([p["xq"] - p["xc"] for p in pr], float); d = np.array([p["c"] - p["q"] for p in pr]); return float(np.sum(g * d) / np.sum(g * g))

# ------------------------------------------------------------------------------------------------------ fig_sim
fig = plt.figure(figsize=(WIDE, 3.0), layout="constrained"); gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1]); a = fig.add_subplot(gs[0, 0]); b = fig.add_subplot(gs[0, 1])
S = sorted(sim, key=lambda p: -(p["c"] - p["q"])); xs = np.arange(len(S))
for i, p in enumerate(S):
    col = SRC.get(p["src"], SRC["qct"])[0]; a.plot([i, i], [p["q"], p["c"]], color=fs.FAINT, lw=1.6, zorder=1)
    a.scatter(i, p["q"], marker="s", s=14, color=BLUE, edgecolor=fs.INK, linewidths=0.3, zorder=3); a.scatter(i, p["c"], marker="o", s=17, color=col, edgecolor=fs.INK, linewidths=0.3, zorder=4)
a.set_xticks(xs); a.set_xticklabels([r"\texttt{" + short(p["name"]) + "}" for p in S], rotation=90, fontsize=5.4); a.set_xlim(-0.8, len(S) - 0.2); a.set_ylabel(r"return probability $P$ (simulation)"); a.grid(axis="y", lw=0.4, color=fs.FAINT, zorder=0)
hs = [plt.Line2D([], [], marker="s", ls="", color=BLUE, mec=fs.INK, mew=0.3, ms=4, label="Qiskit")] + [plt.Line2D([], [], marker="o", ls="", color=c, mec=fs.INK, mew=0.3, ms=4.5, label=l) for k, (c, l) in SRC.items() if any(p["src"] == k for p in S)]
a.legend(handles=hs, loc="lower left", fontsize=6.2, ncol=1, handletextpad=0.2); tag(a, "(a)")
for k, (col, lab) in SRC.items():
    sel = [p for p in sim if p["src"] == k]
    if sel: b.scatter([p["xq"] - p["xc"] for p in sel], [p["c"] - p["q"] for p in sel], s=14, color=col, edgecolor=fs.INK, linewidths=0.3, zorder=3)
b.scatter([p["xq"] - p["xc"] for p in hw], [p["c"] - p["q"] for p in hw], marker="*", s=40, color=fs.INK, zorder=5, label="hardware (9)")
for p in hw:
    if abs(p["c"] - p["q"]) > 0.15: b.annotate(r"\texttt{" + short(p["name"]) + "}", (p["xq"] - p["xc"], p["c"] - p["q"]), textcoords="offset points", xytext=(0, 6), fontsize=5.4, ha="center")
gm = np.linspace(0, max(p["xq"] - p["xc"] for p in sim) * 1.05, 20); b.plot(gm, slope(sim) * gm, color=fs.GREY, lw=0.8, ls="--", zorder=2, label=r"fit, simulation: $%.1f\times10^{-3}$ per gate" % (1e3 * slope(sim)))
b.plot(gm, slope(hw) * gm, color=fs.INK, lw=0.8, ls=":", zorder=2, label=r"fit, hardware: $%.1f\times10^{-3}$ per gate" % (1e3 * slope(hw)))
b.axhline(0, color=fs.INK, lw=0.4); b.set_xlabel(r"two-qubit gates saved relative to Qiskit"); b.set_ylabel(r"gain $P_{\mathrm{QCPT}}-P_{\mathrm{Qiskit}}$"); b.set_ylim(-0.02, 0.27); b.legend(loc="upper left", fontsize=6.0, frameon=False, handletextpad=0.2); tag(b, "(b)")
place(fig); fig.savefig(os.path.join(OUTDIR, "fig_sim.pdf")); fig.savefig(os.path.join(OUTDIR, "fig_sim_preview.png"), dpi=200); plt.close(fig)

# ------------------------------------------------------------------------------------------------------ fig_hw
fig = plt.figure(figsize=(WIDE, 2.9), layout="constrained"); gs = fig.add_gridspec(1, 2, width_ratios=[2.0, 1]); a = fig.add_subplot(gs[0, 0]); b = fig.add_subplot(gs[0, 1])
H = sorted(hw, key=lambda p: -(p["c"] - p["q"])); simd = {p["name"]: p for p in simB}
for i, p in enumerate(H):
    col = SRC.get(p["src"], SRC["qct"])[0]; a.plot([i, i], [p["q"], p["c"]], color=fs.FAINT, lw=2.0, zorder=1)
    a.errorbar(i, p["q"], yerr=p["sq"], fmt="s", ms=4.2, color=BLUE, mec=fs.INK, mew=0.3, ecolor=fs.INK, elinewidth=0.6, capsize=1.2, zorder=3); a.errorbar(i, p["c"], yerr=p["sc"], fmt="o", ms=4.6, color=col, mec=fs.INK, mew=0.3, ecolor=fs.INK, elinewidth=0.6, capsize=1.2, zorder=4)
    s_ = simd[p["name"]]; a.scatter(i + 0.22, s_["q"], marker="s", s=9, facecolor="none", edgecolor=BLUE, linewidths=0.7, zorder=2); a.scatter(i + 0.22, s_["c"], marker="o", s=10, facecolor="none", edgecolor=col, linewidths=0.7, zorder=2)
    a.text(i, 0.335, r"%d$\to$%d" % (p["xq"], p["xc"]), fontsize=5.2, ha="center", color=fs.GREY)
a.set_xticks(range(len(H))); a.set_xticklabels([r"\texttt{" + short(p["name"]) + "}" for p in H], rotation=90, fontsize=6.3); a.set_ylim(0.33, 1.0); a.set_xlim(-0.6, len(H) - 0.3); a.set_ylabel(r"return probability $P$ (hardware)"); a.grid(axis="y", lw=0.4, color=fs.FAINT, zorder=0)
hs = [plt.Line2D([], [], marker="s", ls="", color=BLUE, mec=fs.INK, mew=0.3, ms=4, label="Qiskit"), plt.Line2D([], [], marker="o", ls="", color=VERM, mec=fs.INK, mew=0.3, ms=4.5, label="QCPT (colors: source)"), plt.Line2D([], [], marker="o", ls="", mfc="none", mec=fs.INK, ms=4, label="simulation, same qubits")]
a.legend(handles=hs, loc="lower right", bbox_to_anchor=(1.0, 0.07), fontsize=6.2, handletextpad=0.2); tag(a, "(a)")
for p in hw:
    col = SRC.get(p["src"], SRC["qct"])[0]; s_ = simd[p["name"]]; b.errorbar(s_["c"] - s_["q"], p["c"] - p["q"], yerr=np.hypot(p["sq"], p["sc"]), fmt="o", ms=4.6, color=col, mec=fs.INK, mew=0.3, ecolor=fs.INK, elinewidth=0.6, capsize=1.2, zorder=3)
lim = 0.2; b.plot([0, lim], [0, lim], color=fs.GREY, lw=0.8, ls="--", zorder=1); b.axhline(0, color=fs.INK, lw=0.4); b.axvline(0, color=fs.INK, lw=0.4); b.set_xlim(-0.005, 0.09); b.set_ylim(-0.02, lim)
t_ = [p for p in hw if p["name"] == "rl_4mod5-bdd_287"][0]; s_ = simd[t_["name"]]; b.annotate(r"\texttt{4mod5-bdd\_287}", (s_["c"] - s_["q"], t_["c"] - t_["q"]), textcoords="offset points", xytext=(-4, 3), fontsize=5.6, ha="right"); b.set_xlabel(r"simulated gain (same qubits)"); b.set_ylabel(r"measured gain"); tag(b, "(b)")
place(fig); fig.savefig(os.path.join(OUTDIR, "fig_hw.pdf")); fig.savefig(os.path.join(OUTDIR, "fig_hw_preview.png"), dpi=200); plt.close(fig)

print("slopes: sim %.2e per gate, hw %.2e" % (slope(sim), slope(hw)))

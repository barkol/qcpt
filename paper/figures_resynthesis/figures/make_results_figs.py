#!/usr/bin/env python3
"""Figures of the QCPT results with resynthesis: (1) fig_resynth: number of CX per circuit (exact, without resynthesis, QCPT, Claude), (2) fig_bench: characteristics of
ibm_aachen against the manufacturer's data + GHZ and Quantum Volume. Reads ONLY data/paper_results.npz (paper_data.py)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import figstyle as fs
import matplotlib.pyplot as plt
from paneltags import tag, place

HERE = os.path.dirname(os.path.abspath(__file__)); OUTDIR = os.environ.get("FIG_OUT", HERE)
d = np.load(os.path.join(HERE, "data", "paper_results.npz"), allow_pickle=True); fs.apply()
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5})
WIDE = 483.697 / 72.27; BLUE, VERM, INK, GREY = fs.BLUE, fs.ORANGE, fs.INK, fs.GREY
idx = {n: i for i, n in enumerate(d["rs_name"])}
pretty = lambda n: n.replace("rl_", "").replace("controlled_adder", "ctrl. adder").replace("ripple_carry_adder", "ripple adder").replace("grover_oracle_mcx4", "Grover MCX-4").replace("half_uncomputed", "half-uncomp.").replace("mod_incr", "mod. incr.").replace("adder_live_carry", "adder, live carry").replace("_", " ")

# ------------------------------------------------------------------ (1) fig_resynth
A = ["controlled_adder_2b", "controlled_adder_3b", "half_uncomputed_4c", "grover_oracle_mcx4", "mod_incr_3b", "ripple_carry_adder_2b", "rl_4mod5-bdd_287", "rl_alu-bdd_288", "rl_rd53_138"]
B = ["adder_live_carry", "half_uncomputed", "rl_4mod5-v1_23", "rl_4mod5-v0_18", "rl_rd32_272", "rl_decod24-bdd_294", "rl_mod5d2_64", "rl_fredkin_6", "rl_miller_11"]
fig, axs = plt.subplots(1, 2, figsize=(WIDE, 2.25), layout="constrained", gridspec_kw={"width_ratios": [1.05, 1]})
for ax, names, ttl in ((axs[0], A, "(a)"), (axs[1], B, "(b)")):
    names = sorted(names, key=lambda c: -d["rs_exact"][idx[c]]); y = np.arange(len(names))[::-1]
    for yy, c in zip(y, names):
        i = idx[c]; ex, s1, q = d["rs_exact"][i], d["rs_stage1"][i], d["rs_qcp"][i]
        ax.plot([q, ex], [yy, yy], color=fs.FAINT, lw=2.0, zorder=1)
        ax.scatter(ex, yy, marker="s", s=26, color=BLUE, edgecolor=INK, linewidths=0.4, zorder=3)
        ax.scatter(s1, yy, marker="o", s=26, facecolor="none", edgecolor=VERM, linewidths=1.0, zorder=3)
        ax.scatter(q, yy, marker="o", s=30, color=VERM, edgecolor=INK, linewidths=0.4, zorder=4)
        for key, mk, fc in (("rs_claudeS", "D", fs.PINK), ("rs_claudeW", "D", "none")):
            v = d[key][i]
            if np.isfinite(v): ax.scatter(v, yy, marker=mk, s=22, facecolor=fc, edgecolor=INK, linewidths=0.6, zorder=5)
        ax.text(ex + 1.5, yy, f"{int(ex)}", va="center", ha="left", fontsize=7, color=BLUE)
    ax.set_yticks(y); ax.set_yticklabels([pretty(c) for c in names], fontsize=7.5); ax.set_ylim(-0.7, len(names) - 0.3)
    ax.set_xlabel("two-qubit gates (CX)"); ax.grid(axis="x", lw=0.4, color=fs.FAINT, zorder=0); ax.set_xlim(0, ax.get_xlim()[1] * 1.04); tag(ax, ttl)
hs = [plt.Line2D([], [], marker="s", ls="", color=BLUE, mec=INK, mew=0.4, ms=5, label="exact decomposition"), plt.Line2D([], [], marker="o", ls="", mfc="none", mec=VERM, mew=1.0, ms=5, label="QCPT without resynthesis"),
      plt.Line2D([], [], marker="o", ls="", color=VERM, mec=INK, mew=0.4, ms=5.5, label="QCPT"), plt.Line2D([], [], marker="D", ls="", mfc=fs.PINK, mec=INK, mew=0.6, ms=4.5, label="Claude loop (certificate)"),
      plt.Line2D([], [], marker="D", ls="", mfc="none", mec=INK, mew=0.6, ms=4.5, label="Claude loop (weak check, sound)")]
fig.legend(handles=hs, loc="outside lower center", ncol=5, frameon=False, fontsize=7.2, handletextpad=0.2, columnspacing=1.0)
place(fig); fig.savefig(os.path.join(OUTDIR, "fig_resynth.pdf")); plt.close(fig)

# ------------------------------------------------------------------ (2) fig_bench
J = lambda k: json.loads(str(d["b_" + k]))
TI, TS, RO, R1, R2, GH, QV = (J(k) for k in ("TI", "TS", "RO", "R1", "R2", "GH", "QV"))
def ratio(m, c): m, c = np.array(m, float), np.array(c, float); ok = np.isfinite(m) & np.isfinite(c); return m[ok] / c[ok]
items = [("$T_1$", ratio(TI["T1"]["meas"], TI["T1"]["cal"])), ("$T_2$ echo, isolated", ratio(TI["T2echo"]["meas"], TI["T2echo"]["cal"])),
         ("$T_2$ echo, neighbours excited", ratio(TS["T2echo_sasiedzi_wzbudzeni"]["meas"], TS["T2echo_sasiedzi_wzbudzeni"]["cal"])),
         ("readout error", ratio(RO["RO"]["meas"], RO["RO"]["cal"])), ("1q gate error (RB)", ratio(np.array(R1["rb"]["eps"]) / 0.9375, R1["rb"]["cal"])),
         ("2q gate error (RB)", ratio(np.array(R2["rb"]["eps"]) / 1.5, R2["rb"]["cal"]))]
fig, axs = plt.subplots(1, 3, figsize=(WIDE, 1.8), layout="constrained", gridspec_kw={"width_ratios": [1.9, 0.85, 0.85]})
ax = axs[0]; y = np.arange(len(items))[::-1]
ax.axvspan(0.8, 1.25, color="#e3f3ec", zorder=0); ax.axvline(1.0, color=INK, lw=0.7, zorder=1)
for yy, (lab, r) in zip(y, items):
    q1, md, q3 = np.percentile(r, [25, 50, 75]); ax.plot([q1, q3], [yy, yy], color=INK, lw=1.4, zorder=2)
    ax.scatter(md, yy, s=40, marker="o", color=VERM if lab.startswith("$T_2$ echo, neigh") else BLUE, edgecolor=INK, linewidths=0.5, zorder=3); ax.text(max(q3, md) * 1.12, yy, f"{md:.2f}", va="center", fontsize=7.5)
ax.set_xscale("log"); ax.set_xlim(0.1, 6); ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5]); ax.set_xticklabels(["0.1", "0.2", "0.5", "1", "2", "5"])
ax.set_yticks(y); ax.set_yticklabels([l for l, _ in items], fontsize=7.5); ax.set_ylim(-0.6, len(items) - 0.4); ax.set_xlabel("measured / manufacturer calibration"); tag(ax, "(a)")
ns = [4, 8, 12]; ax = axs[1]; ax.plot(ns, [GH[f"ghz{n}"] for n in ns], "o-", color=BLUE, mec=INK, mew=0.4, lw=1.2, ms=4.5); ax.set_ylim(0.6, 1.02); ax.set_xticks(ns); ax.set_xlabel("GHZ size $n$"); ax.set_ylabel("population of $|0\\ldots0\\rangle,|1\\ldots1\\rangle$"); ax.grid(lw=0.4, color=fs.FAINT, zorder=0); tag(ax, "(b)")
nq = [3, 4, 5]; ax = axs[2]; hop = np.array([QV[f"qv{n}"]["hop"] for n in nq]); lb = np.array([QV[f"qv{n}"]["lb"] for n in nq])
ax.errorbar(nq, hop, yerr=hop - lb, fmt="o", color=BLUE, mec=INK, mew=0.4, ms=4.5, capsize=2.5, lw=1.0); ax.axhline(2 / 3, color=VERM, lw=1.0, ls="--"); ax.text(3.0, 0.672, "pass threshold $2/3$", color=VERM, fontsize=7.2, va="bottom")
ax.set_xticks(nq); ax.set_ylim(0.6, 0.95); ax.set_xlabel("QV circuit width $n$"); ax.set_ylabel("heavy-output prob."); ax.grid(lw=0.4, color=fs.FAINT, zorder=0); tag(ax, "(c)")
place(fig); fig.savefig(os.path.join(OUTDIR, "fig_bench.pdf")); plt.close(fig)
print("fig_resynth.pdf fig_bench.pdf saved")

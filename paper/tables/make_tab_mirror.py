"""Mirror-test table -> tab_mirror.tex (only reads data/mirror_hw.json)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__)); D = json.load(open(os.path.join(HERE, "data", "mirror_hw.json")))
R = {(r["name"], r["tag"]): r for r in D["rows"]}
NM = [("half_uncomputed", "Half-uncomputed"), ("live_and_chain", "Live AND chain"), ("single_live_toffoli", "Single live Toffoli"),
      ("controlled_adder_2b", "Controlled adder (2b)"), ("nested_compute_uncompute", "Nested comp./unc.")]
def c(r): return f"${r['p0_hw']:.3f}$"
L = [r"\begin{table}[t]", r"\centering\small",
     r"\caption{Mirror test on \texttt{ibm\_marrakesh}: probability of returning to",
     r"$\ket{0\cdots0}$ after $H_d V^\dagger E H_d$ ($" + str(D['rows'][0]['shots']) + r"$ shots per circuit, standard error",
     r"$\le" + f"{max(r['sigma'] for r in D['rows']):.3f}" + r"$). Ref.: $V=E$. Ideal values in parentheses;",
     r"they equal $1$ for the reference and for the pass on every circuit.}",
     r"\label{tab:mirror}", r"\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{lccc}", r"\toprule",
     r"Circuit & Ref. & This work & Count-greedy \\", r"\midrule"]
for k, lab in NM:
    g = R[(k, "greedy")]
    L.append(f"{lab} & {c(R[(k,'ref')])} & {c(R[(k,'sub')])} & {c(g)} (${g['p0_ideal']:.2f}$) \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(HERE, "tab_mirror.tex"), "w").write("\n".join(L) + "\n"); print("\n".join(L))

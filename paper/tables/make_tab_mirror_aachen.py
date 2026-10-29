"""Table of the second mirror test (ibm_aachen) -> tab_mirror_aachen.tex (only reads data/mirror_aachen.json)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__)); D = json.load(open(os.path.join(HERE, "data", "mirror_aachen.json")))
R = {(r["name"], r["tag"]): r for r in D["rows"]}
NM = [("half_uncomputed", "Half-unc.\\ (2c)"), ("half_uncomputed_3c", "Half-unc.\\ (3c)"), ("half_uncomputed_4c", "Half-unc.\\ (4c)"),
      ("mod_incr_3b", "Mod.\\ incr.\\ (3b)"), ("controlled_adder_2b", "Contr.\\ adder (2b)"), ("nested_compute_uncompute", "Nested (control)")]
def c(r): return f"${r['p0_hw']:.3f}$"
L = [r"\begin{table}[t]", r"\centering\footnotesize",
     r"\caption{Mirror test on \texttt{ibm\_aachen} ($" + str(D['rows'][0]['shots']) + r"$ shots per circuit, standard error",
     r"$\le" + f"{max(r['sigma'] for r in D['rows']):.3f}" + r"$), as in Table~\ref{tab:mirror}. The column 2q gives the",
     r"two-qubit gates after transpilation for the reference, the pass and count-greedy substitution. Ideal values in parentheses;",
     r"they equal $1$ for the reference and for the pass on every circuit.}",
     r"\label{tab:mirror2}", r"\setlength{\tabcolsep}{1.6pt}", r"\begin{tabular}{lcccc}", r"\toprule",
     r"Circuit & 2q & Ref. & Pass & Count-greedy \\", r"\midrule"]
for k, lab in NM:
    g = R[(k, "greedy")]; tq = "/".join(str(R[(k, t)]["twoq_transpiled"]) for t in ("ref", "sub", "greedy"))
    L.append(f"{lab} & ${tq}$ & {c(R[(k,'ref')])} & {c(R[(k,'sub')])} & {c(g)}\\,(${g['p0_ideal']:.2f}$) \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(HERE, "tab_mirror_aachen.tex"), "w").write("\n".join(L) + "\n"); print("\n".join(L))

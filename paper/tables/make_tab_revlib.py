"""RevLib table -> tab_revlib.tex (only reads data/revlib.json)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__)); D = json.load(open(os.path.join(HERE, "data", "revlib.json")))
A = D["agg"]; n = D["n"]; ex = A["exact"]["twoq"]
NM = [("exact", "Exact only"), ("qiskit_l3", "Qiskit opt-3"), ("tket", "tket"), ("greedy", "Count-greedy"),
      ("ours", "This work"), ("ours_qiskit", r"This work $\to$ Qiskit opt-3")]
L = [r"\begin{table}[t]", r"\centering\small",
     r"\caption{RevLib suite: all $" + str(n) + r"$ NOT/CNOT/Toffoli realizations with at most $" + str(D["max_lines"]) +
     r"$ lines that complete within the time limit. Summed two-qubit count and infidelity, change of",
     r"the two-qubit count relative to exact-only, and the number of outputs that pass the",
     r"subroutine certificate (Cert.). Constant-$0$ lines are pinned as clean ancillae.}",
     r"\label{tab:revlib}", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lrrrc}", r"\toprule",
     r"Method & 2q & Infid. & $\Delta$2q & Cert. \\", r"\midrule"]
for k, lab in NM:
    a = A[k]; d = 100 * (a["twoq"] - ex) / ex
    L.append(f"{lab} & ${a['twoq']}$ & ${a['infid']:.2f}$ & " + ("--" if k == "exact" else f"${d:+.1f}\\%$") + f" & ${a['cert']}/{n}$ \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(HERE, "tab_revlib.tex"), "w").write("\n".join(L) + "\n"); print("\n".join(L))

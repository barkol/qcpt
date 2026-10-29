"""Table comparing the semantics -> tab_semantics.tex (only reads data/*.json and baseline_results.md)."""
import json, os, re
HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(HERE, "data", "semantics.json")))
I = json.load(open(os.path.join(HERE, "data", "interference.json")))
b = open(os.environ.get("BASE", os.path.join(HERE, "..", "data", "rerun", "experiments", "baseline_results.md"))).read()
def base(key):
    m = re.search(re.escape("| " + key + " |") + r"\s*([\d.]+)\s*\|\s*(\d+)\s*\|", b); return int(m.group(2)), float(m.group(1))
def inter(m):
    if m is None: return "--"
    return f"{sum(not I[c][m]['sub'] for c in I)} / {sum(not I[c][m]['prog'] for c in I)}"
rows = [("Exact only", S["exact"]["twoq"], S["exact"]["infid"], 0, 0, "0 / 0"),
        ("Qiskit opt-3", *base("Qiskit opt-3"), 0, 0, "--"),
        ("tket", *base("tket (FullPeephole)"), 0, 0, "--"),
        ("Count-greedy", S["greedy"]["twoq"], S["greedy"]["infid"], S["greedy"]["bad_sub"], S["greedy"]["bad_prog"], inter("greedy")),
        ("This work, subroutine", S["sub"]["twoq"], S["sub"]["infid"], S["sub"]["bad_sub"], S["sub"]["bad_prog"], inter("sub")),
        ("This work, program", S["prog"]["twoq"], S["prog"]["infid"], S["prog"]["bad_sub"], S["prog"]["bad_prog"], inter("prog"))]
L = [r"\begin{table}[t]", r"\centering\small",
     r"\caption{The two correctness semantics on the $12$-circuit suite. Columns give the",
     r"summed two-qubit count and infidelity, the number of circuits whose",
     r"action as a subroutine changes (Sub.), the number whose measurement",
     r"statistics change for some valid input state (Prog.), and the same two",
     r"counts on three interference circuits (Interf., Sub.\,/\,Prog.): the",
     r"counterexample of \S\ref{sec:soundness} and the circuit of",
     r"Fig.~\ref{fig:method}(b) with a data target and with a clean-ancilla target.",
     r"Qiskit and tket outputs pass the same certificate on every circuit.}",
     r"\label{tab:semantics}", r"\setlength{\tabcolsep}{3.5pt}",
     r"\begin{tabular}{lrrccc}", r"\toprule",
     r"Method & 2q & Infid. & Sub. & Prog. & Interf. \\", r"\midrule"]
for n, t, inf, s_, p_, it in rows:
    L.append(f"{n} & ${t}$ & ${inf:.2f}$ & ${s_}$ & ${p_}$ & {it} \\\\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(HERE, "tab_semantics.tex"), "w").write("\n".join(L) + "\n")
print("\n".join(L))

#!/usr/bin/env python3
"""Computation layer: values from the tables of the paper (tables/*.tex and tab:reset in main.tex) -> figures/data/tables_as_figs.npz.
The drawing layer (make_tables_as_figs.py) only reads it. Usage: python tables_data.py SRC  (SRC = directory of the paper sources)."""
import os, re, sys
import numpy as np
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"); os.makedirs(OUT, exist_ok=True)
num = lambda s: float(re.sub(r"[^0-9.\-]", "", s.replace("--", "")) or "nan")
def rows(path, start=r"\midrule", end=r"\bottomrule"):
    t = open(path, encoding="utf-8").read(); body = t[t.index(start) + len(start): t.index(end)]
    return [[c.strip() for c in l.replace("\\\\", "").split("&")] for l in body.splitlines() if "&" in l]
d = {}
# --- semantics (12 circuits): method, 2q, infid, Sub, Prog, Interf
S = rows(os.path.join(SRC, "tables/tab_semantics.tex"))
d["sem_names"] = np.array(["exact-only", "Qiskit opt-3", "tket", "count-greedy", "QCPT (subroutine)", "QCPT (program)"])
d["sem_2q"] = np.array([num(r[1]) for r in S]); d["sem_inf"] = np.array([num(r[2]) for r in S])
d["sem_sub"] = np.array([num(r[3]) for r in S]); d["sem_prog"] = np.array([num(r[4]) for r in S]); d["sem_n"] = 12
d["sem_interf"] = np.array([[num(x) for x in r[5].split("/")] if "/" in r[5] else [np.nan, np.nan] for r in S])
# --- RevLib: metoda, 2q, infid, delta, cert
V = rows(os.path.join(SRC, "tables/tab_revlib.tex"))
d["rl_names"] = np.array(["exact-only", "Qiskit opt-3", "tket", "count-greedy", "QCPT", r"QCPT $\to$ Qiskit"])
d["rl_2q"] = np.array([num(r[1]) for r in V]); d["rl_inf"] = np.array([num(r[2]) for r in V])
d["rl_cert"] = np.array([num(r[4].split("/")[0]) for r in V]); d["rl_n"] = int(num(V[0][4].split("/")[1]))
# --- testy lustrzane
def p_ideal(c):
    m = re.match(r"\$?([\d.]+)\$?(?:\\,)?\s*\(\$?([\d.]+)\$?\)", c.strip()); return (float(m.group(1)), float(m.group(2))) if m else (num(c), np.nan)
M = rows(os.path.join(SRC, "tables/tab_mirror.tex"))
d["mq_names"] = np.array([r[0].replace("\\", "") for r in M]); d["mq_ref"] = np.array([num(r[1]) for r in M]); d["mq_pass"] = np.array([num(r[2]) for r in M])
g = [p_ideal(r[3]) for r in M]; d["mq_greedy"] = np.array([a for a, _ in g]); d["mq_greedy_ideal"] = np.array([b for _, b in g]); d["mq_shots"] = 4000
A = rows(os.path.join(SRC, "tables/tab_mirror_aachen.tex"))
d["ma_names"] = np.array([r[0].replace("\\ ", " ").replace("\\", "") for r in A])
q2 = [[num(x) for x in r[1].replace("$", "").split("/")] for r in A]; d["ma_2q"] = np.array(q2)
d["ma_ref"] = np.array([num(r[2]) for r in A]); d["ma_pass"] = np.array([num(r[3]) for r in A])
g = [p_ideal(r[4]) for r in A]; d["ma_greedy"] = np.array([a for a, _ in g]); d["ma_greedy_ideal"] = np.array([b for _, b in g]); d["ma_shots"] = 8000
# --- reset circuits (tab:reset in main.tex)
_m = os.path.join(SRC, "main.tex"); t = open(_m, encoding="utf-8").read() if os.path.exists(_m) else ""
if "\\label{tab:reset}" not in t:                      # table replaced by a figure: keep the values of the previous npz
    _f = os.path.join(OUT, "tables_as_figs.npz"); old = np.load(_f) if os.path.exists(_f) else None
    if old is not None: d.update({k: old[k] for k in old.files if k.startswith("rs_")})
    np.savez(os.path.join(OUT, "tables_as_figs.npz"), **d); print("rs_* z poprzedniego npz"); sys.exit()
blk = t[t.index("\\label{tab:reset}"):]; blk = blk[:blk.index("\\bottomrule")]
RS = [[c.strip() for c in l.replace("\\\\", "").split("&")] for l in blk.splitlines() if "-st." in l]
d["rs_names"] = np.array([f"{int(num(r[0].split('-st.')[0]))} states, {int(num(r[0].split(',')[1].split('-let')[0]))} letters" for r in RS])
d["rs_n"] = np.array([num(r[1]) for r in RS]); d["rs_exact"] = np.array([num(r[2]) for r in RS]); d["rs_pass"] = np.array([num(r[3]) for r in RS])
d["rs_qcec"] = np.array(["QCEC" in r[5] for r in RS]); d["rs_time"] = np.array([num(re.search(r"\(\$?([\d.]+)", r[5]).group(1)) for r in RS])
np.savez(os.path.join(OUT, "tables_as_figs.npz"), **d)
print("saved", sorted(d)); print({k: (v.tolist() if v.size < 8 else v.shape) for k, v in d.items() if k.startswith(("sem_2q", "sem_sub", "rl_cert", "rs_", "ma_2q"))})

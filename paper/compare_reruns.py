#!/usr/bin/env python3
"""Compare the rerun with this code (data/rerun_suites/) with the data used in the paper
(data/rerun/ and experiments/revlib/). Prints per-circuit differences in two-qubit
counts, infidelities and certificate verdicts; timings are not compared."""
import json, math, os, sys
P = os.path.dirname(os.path.abspath(__file__)); V12 = os.path.join(P, "data/rerun_suites")
OLD = os.path.join(P, "data", "rerun", "experiments"); RL = os.path.join(P, "..", "experiments", "revlib")


def load(path, key):
    if not os.path.exists(path): return None
    return {json.loads(l)[key]: json.loads(l) for l in open(path)}


def same(a, b):
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def cmp(label, old, new, fields):
    if new is None: print(f"{label}: not rerun"); return
    nd = 0
    for k in sorted(set(old) | set(new)):
        if k not in new: print(f"  {label} {k}: NOT FINISHED in rerun"); nd += 1; continue
        if k not in old: print(f"  {label} {k}: only in rerun"); continue
        d = [(f, old[k].get(f), new[k].get(f)) for f in fields if not same(old[k].get(f), new[k].get(f))]
        if d: nd += 1; print(f"  {label} {k}: " + "; ".join(f"{f}: paper {a} -> rerun {b}" for f, a, b in d))
    print(f"{label}: {len(set(old) & set(new))}/{len(old)} rows compared, {nd} with differences")


cmp("scale (scale_rows.jsonl)", load(os.path.join(OLD, "scale_rows.jsonl"), "name"), load(os.path.join(V12, "scale_rows.jsonl"), "name"),
    ["n", "family", "twoq_exact", "twoq_ours", "infid_exact", "infid_ours", "red_2q", "qiskit_2q", "sites_applied", "window_pairs", "r_gadgets", "cert_method", "cert_equiv"])
new_prog = load(os.path.join(V12, "scale_program_rows.jsonl"), "name")
# array_mult_6b: QCEC does not decide within the per-call limit; as for the paper, the verdict comes from
# an unbounded QCEC run (paper/scale/qcec_mult6.py), stored as array_mult_6b_<tag>.json.
for tag in ("prog", "greedy"):
    f = os.path.join(V12, f"array_mult_6b_{tag}.json")
    if new_prog and os.path.exists(f) and "array_mult_6b" in new_prog:
        m = json.load(open(f)); new_prog["array_mult_6b"][f"sub_ok_{tag}"] = m["equivalent"]
        new_prog["array_mult_6b"][f"sub_method_{tag}"] = "qcec_nocny"  # same label as the paper's unbounded run
cmp("scale program (scale_program_rows.jsonl)", load(os.path.join(OLD, "scale_program_rows.jsonl"), "name"), new_prog,
    ["twoq_exact", "infid_exact", "twoq_sub", "twoq_prog", "infid_prog", "twoq_greedy", "infid_greedy", "U_admitted", "pairs", "r_gadgets",
     "sub_ok_prog", "sub_ok_greedy", "prog_ok_prog", "prog_ok_greedy", "sub_method_prog", "sub_method_greedy"])
old = load(os.path.join(RL, "wyniki.jsonl"), "file"); new = load(os.path.join(V12, "results.jsonl"), "file")
X = load(os.path.join(RL, "wyniki_extra.jsonl"), "file") or {}
if new is not None:
    flat = lambda r, x: {**{k: r.get(k) for k in ("pairs_C", "pairs_W", "gadgets_R", "drops_R", "fell_back", "error")},
                         **{f"{m}.{q}": (((x or {}).get("tket_fix") if m == "tket" else (x or {}).get("ours_qiskit") if m == "ours_qiskit" else None)
                                            or r.get(m) or {}).get(q)
                            for m in ("exact", "ours", "qiskit_l3", "greedy", "ours_qiskit", "tket") for q in ("twoq", "infid", "cert")}}
    o = {k: flat(v, X.get(k)) for k, v in old.items()}; n = {k: flat(v, None) for k, v in new.items()}
    for k in o:
        if "error" in (o[k] or {}) and o[k]["error"]: o[k] = {"error": "timeout"}
    for k in n:
        if n[k].get("error"): n[k] = {"error": "timeout"}
    cmp("RevLib (wyniki.jsonl; tket and ours->Qiskit from wyniki_extra.jsonl where present)", o, n, sorted({f for v in o.values() for f in v}))

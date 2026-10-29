#!/usr/bin/env python3
"""Computation layer: QCPT results (with resynthesis) from the evaluation of 33 circuits, hardware results and device characteristics -> data/paper_results.npz.
Input: data_in/eval/*.json (evaluation), data_in/hw/*.json (results from ibm_aachen)."""
import glob, json, os, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); IN = os.path.join(H, "..", "data_in"); OUT = os.path.join(H, "data"); os.makedirs(OUT, exist_ok=True)
rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(IN, "eval", "*.json")))]
g = lambda k, d=np.nan: np.array([r.get(k) if r.get(k) is not None else d for r in rows], float)
d = dict(rs_name=np.array([r["circuit"] for r in rows]), rs_n=g("n"), rs_exact=g("exact"), rs_stage1=g("v1"), rs_qcp=g("v2"), rs_claudeS=g("claude_S"), rs_claudeW=g("claude_W"),
         rs_time=g("resynth_s"), rs_rl=np.array([r["circuit"].startswith("rl_") for r in rows]))
# Q2 suite of 12 circuits: 10 computed + 2 unchanged (phase_sensitive_oracle: not fully classical; grover_native_mcx4: no ccx), both values from stage 1
q2 = ["adder_live_carry", "compute_uncompute_pair", "controlled_adder_2b", "grover_oracle_mcx3", "grover_oracle_mcx4", "half_uncomputed", "live_and_chain", "nested_compute_uncompute", "ripple_carry_adder_2b", "single_live_toffoli"]
idx = {r["circuit"]: r for r in rows}
q2_exact = sum(idx[c]["exact"] for c in q2) + 6 + 36; q2_stage1 = sum(idx[c]["v1"] for c in q2) + 6 + 36; q2_qcp = sum(idx[c]["v2"] for c in q2) + 6 + 36
d.update(q2_exact=q2_exact, q2_stage1=q2_stage1, q2_qcp=q2_qcp)
rl = [r for r in rows if r["circuit"].startswith("rl_")]
d.update(rl_exact=sum(r["exact"] for r in rl), rl_stage1=sum(r["v1"] for r in rl), rl_qcp=sum(r["v2"] for r in rl), rl_count=len(rl), rl_time_med=float(np.median([r["resynth_s"] for r in rl])))
# hardware F (earlier circuits) and characteristics
F = json.load(open(os.path.join(IN, "hw", "results_F.json")))["rows"]; d["hwF"] = np.array([json.dumps(r) for r in F])
for k in ("TI", "TS", "RO", "R1", "R2", "GH", "QV"): d["b_" + k] = np.array(json.dumps(json.load(open(os.path.join(IN, "hw", f"bench_{k}.json")))))
np.savez(os.path.join(OUT, "paper_results.npz"), **d)
print("circuits", len(rows), "| Q2(12): exact", q2_exact, "stage1", q2_stage1, "(%.1f%%)" % (100 * (1 - q2_stage1 / q2_exact)), "QCPT", q2_qcp, "(%.1f%%)" % (100 * (1 - q2_qcp / q2_exact)))
print("RevLib<=9q (%d): exact %d stage1 %d (%.1f%%) QCPT %d (%.1f%%) | median resynthesis time %.0f s" % (len(rl), d["rl_exact"], d["rl_stage1"], 100 * (1 - d["rl_stage1"] / d["rl_exact"]), d["rl_qcp"], 100 * (1 - d["rl_qcp"] / d["rl_exact"]), d["rl_time_med"]))

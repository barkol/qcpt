#!/usr/bin/env python3
"""Computation layer: comparison of QCPT with the Claude loop -> data/compare.npz. Input: data_in/pilot.json (pilot logs), data_in/new/run_bundle.json
(claude_to_best.json, race33, parity from the compute machine). Three levels: (1) up to the cost at which the Claude loop ends, (2) up to its own best circuit, (3) the whole run.
QCPT energy = CPU time (core-s) x 150-220 W / 28 threads x 1.1-1.5 (estimate from the nominal power); Claude energy = output tokens x 0.78-1.03 mWh/token (Oviedo et al. 2025, medians: 3.91 Wh at 5000 and 0.31 Wh at 300 output tokens on H100 nodes; 3.91/5000 and 0.31/300)."""
import json, os, statistics as st, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); IN = os.path.join(H, "..", "data_in")
p = json.load(open(os.path.join(IN, "pilot.json"))); B = json.load(open(os.path.join(IN, "new", "run_bundle.json")))
C, R, P = B["claude"], B["race33"], B["parity"]; nine = list(C)
S1 = json.load(open(os.path.join(IN, "new", "stage1_times.json")))                  # stage 1 incl. its certificate (experiments/race/stage1_time.py)
circ = ["grover_oracle_mcx4", "controlled_adder_2b", "rl_rd53_138"]; qcp = {"grover_oracle_mcx4": 18, "controlled_adder_2b": 17, "rl_rd53_138": 23}
S = [p[f"{c}__S"] for c in circ]; W = [p[f"{c}__W"] for c in circ]
mean = lambda xs: float(np.mean(list(xs)))
E_TOK = (3.91 / 5000, 0.31 / 300)                         # Wh per output token: medians of Oviedo et al. at 5000 and at 300 tokens (0.78 and 1.03 mWh)
def wh(core_s): return (core_s * 150 / 28 / 3600 * 1.1, core_s * 220 / 28 / 3600 * 1.5)
t_claude, cpu_claude = [], []
for k in nine:                                           # QCPT up to Claude's cost: race step with cost <= Claude; 0 if stage 1 is already no worse
    r = R[k]; best = C[k]["best"]
    if r["start"] <= best: t_claude.append(0.0); cpu_claude.append(0.0); continue
    t = next(s["wall"] for s in r["steps"] if s["cost"] <= best); t_claude.append(t)
    pc = [v["cpu_s_at_best"] for kk, v in P.items() if kk.rsplit("_", 1)[0] == k and v.get("reached") and v.get("cpu_s_at_best") is not None]
    cpu_claude.append(st.median(pc) if pc else r["cpu_s_at_best"] * t / r["t_best"])    # parity: measured; otherwise proportional to the time
t_best = [R[k]["t_best"] for k in nine]; cpu_best = [R[k]["cpu_s_at_best"] for k in nine]; t_full = [R[k]["wall_s"] for k in nine]; cpu_full = [R[k]["cpu_s"] for k in nine]
tok_best, tok_full = mean(C[k]["tout_best"] for k in nine), mean(C[k]["tout"] for k in nine)
d = dict(circ=np.array(circ), qcp_gates=np.array([qcp[c] for c in circ]), llm_gates_S=np.array([s["best"] for s in S]),
         llm_gates_W_sound=np.array([w["best_strong"] for w in W]), llm_gates_W_raw=np.array([w["best"] for w in W]),
         W_accepted=sum(w["n_acc"] for w in W), W_unsound=sum(w["n_acc_unsound"] for w in W), W_iters=np.array([w["iters"] for w in W]),
         qct_t_claude=mean(t_claude), qct_t_best=mean(t_best), qct_t_full=mean(t_full),
         qct_wh_claude=wh(mean(cpu_claude)), qct_wh_best=wh(mean(cpu_best)), qct_wh_full=wh(mean(cpu_full)),
         stage1_wall=mean(S1[k]["wall"] for k in nine), stage1_cpu=mean(S1[k]["cpu"] for k in nine), stage1_wall_max=max(S1[k]["wall"] for k in nine),
         llm_t_best=mean(C[k]["wall_best"] for k in nine), llm_t_full=mean(C[k]["wall"] for k in nine),
         llm_usd_best=mean(C[k]["usd_best"] for k in nine), llm_usd_full=mean(C[k]["usd"] for k in nine),
         llm_wh_best=(tok_best * E_TOK[0], tok_best * E_TOK[1]), llm_wh_full=(tok_full * E_TOK[0], tok_full * E_TOK[1]), llm_wh_per_token=E_TOK, llm_tok_best=tok_best, llm_tok_full=tok_full)
np.savez(os.path.join(H, "data", "compare.npz"), **d)
print({k: (np.round(v, 3).tolist() if hasattr(v, "tolist") or isinstance(v, tuple) else v) for k, v in d.items() if k != "circ"})

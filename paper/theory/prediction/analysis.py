"""Analysis of results.jsonl -> analysis.json (+ printout). Also checks whether swapping the orientation
(action ("relphase","swap") / ("relphase_m","swap")) for mismatches of type 'orientacja'
passes the whole-circuit certificate (dense, <= 12 qubits)."""
import json
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from prediction import load  # noqa

rows = {}
for ln in open(os.path.join(HERE, "results.jsonl")):
    r = json.loads(ln)
    rows[(r["suite"], r["key"])] = r          # ostatni wpis wygrywa


def classify(t, row):
    if t["cost_actual"] == t["pred_full"]:
        return None
    if row["fell_back"]:
        return "final_certificate_fallback"
    if t["action"] is None and t["rejected_site"] and not (t["pair_C"] or t["window_W"]):
        return "rejected_structural_site_locked"
    if t["action"] is None and t["c_full"] == 3 and t["c_pass"] == 6:
        return "orientation_swap_not_tried"
    return "other"


out = {"suites": {}, "mismatch_toffolis": [], "swap_check": []}
for suite in ("primary", "revlib", "scale"):
    rs = [r for (s, k), r in rows.items() if s == suite]
    ok = [r for r in rs if not r.get("skipped")]
    sk = [{"key": r["key"], "reason": r["reason"][:200]} for r in rs if r.get("skipped")]
    hist_full, hist_pass, hist_full_free = Counter(), Counter(), Counter()
    nC = nW = nT = 0
    c6_sub = []
    act_hist = Counter()
    for r in ok:
        for t in r["toffolis"]:
            nT += 1
            hist_full[t["c_full"]] += 1
            hist_pass[t["c_pass"]] += 1
            nC += t["pair_C"]
            nW += t["window_W"]
            if not (t["pair_C"] or t["window_W"]):
                hist_full_free[t["c_full"]] += 1
            act_hist[t["action_repr"]] += 1
            if t["c_full"] == 6 and not (t["pair_C"] or t["window_W"]) and t["action"] is not None:
                c6_sub.append((r["key"], t["idx"]))
    s = {
        "circuits_run": len(ok), "skipped": sk,
        "consistent_actions_vs_2q": sum(r["twoq_from_actions"] == r["twoq_actual"] for r in ok),
        "match_full": sum(r["twoq_pred_full"] == r["twoq_actual"] for r in ok),
        "match_pass": sum(r["twoq_pred_pass"] == r["twoq_actual"] for r in ok),
        "fell_back": [r["key"] for r in ok if r["fell_back"]],
        "toffolis": nT, "in_pair_C": nC, "in_window_W": nW,
        "hist_c_full": dict(sorted(hist_full.items())),
        "hist_c_pass": dict(sorted(hist_pass.items())),
        "hist_c_full_not_paired": dict(sorted(hist_full_free.items())),
        "actions": dict(act_hist),
        "c6_substituted_outside_pairs": c6_sub,
        "per_circuit_toffoli_match_pass": sum(
            all(t["cost_actual"] == t["pred_pass"] for t in r["toffolis"]) for r in ok),
        "twoq_sum_actual": sum(r["twoq_actual"] for r in ok),
        "twoq_sum_pred_full": sum(r["twoq_pred_full"] for r in ok),
        "twoq_sum_pred_pass": sum(r["twoq_pred_pass"] for r in ok),
    }
    out["suites"][suite] = s
    for r in ok:
        for t in r["toffolis"]:
            cl = classify(t, r)
            if cl or t["cost_actual"] != t["pred_pass"]:
                out["mismatch_toffolis"].append({
                    "suite": suite, "key": r["key"], "n": r["n"], "idx": t["idx"],
                    "qubits": t["qubits"], "S_bits(abt)": t["S_bits"], "S_src": t["S_src"],
                    "c_full": t["c_full"], "c_pass": t["c_pass"], "pair_C": t["pair_C"],
                    "window_W": t["window_W"], "rejected_site": t["rejected_site"],
                    "action": t["action_repr"], "cost_actual": t["cost_actual"],
                    "pred_full": t["pred_full"], "pred_pass": t["pred_pass"], "type": cl})

# --- check that the orientation swap is feasible (dense certificate) ------------------
from toffoli_optimizer.core.decomposition_selector import ErrorBudgetSelector, _U_CCX  # noqa
from toffoli_optimizer.core.subspace_check import gadget_unitary, check_on_subspace, certify_on_input_subspace  # noqa
from toffoli_optimizer.core.decomposition_selector import (  # noqa
    append_relative_phase_ccx, append_relative_phase_ccx_mirror)
from qiskit.quantum_info import Operator  # noqa

by_circ = defaultdict(list)
for m in out["mismatch_toffolis"]:
    if m["type"] == "orientation_swap_not_tried":
        by_circ[(m["suite"], m["key"])].append(m)
U_sw = gadget_unitary(lambda qc, a, b, t: append_relative_phase_ccx(qc, b, a, t))
U_swm = gadget_unitary(lambda qc, a, b, t: append_relative_phase_ccx_mirror(qc, b, a, t))
for (suite, key), ms in by_circ.items():
    qc, pins = load(suite, key)
    if qc.num_qubits > 12:
        out["swap_check"].append({"suite": suite, "key": key, "skipped": "n>12"})
        continue
    sel = ErrorBudgetSelector()
    res = sel.select(qc, pinned_zero=tuple(pins))
    acts = dict(res["actions"])
    loc_ok = []
    for m in ms:
        S = [int(b[0]) + 2 * int(b[1]) + 4 * int(b[2]) for b in m["S_bits(abt)"]]
        for kind, U in (("relphase", U_sw), ("relphase_m", U_swm)):
            if check_on_subspace(U, _U_CCX, S, 0.0)[0]:
                acts[m["idx"]] = (kind, "swap")
                loc_ok.append((m["idx"], kind))
                break
    built = sel._build(qc, acts)
    ver, dev, _ = certify_on_input_subspace(Operator(res["exact"]).data, Operator(built).data,
                                            res["cert_inputs"], mode="subroutine", tolerance=0.0)
    out["swap_check"].append({"suite": suite, "key": key, "swapped": loc_ok,
                              "certified": bool(ver),
                              "twoq_swapped": sel.error_model.two_qubit_count(built),
                              "twoq_actual": res["report"]["two_qubit_after"]})

# --- comparison with the stored results of the QCEC selector (paper/data/rerun_suites/scale_rows.jsonl)
sr = {}
for ln in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data/rerun_suites", "scale_rows.jsonl")):
    r = json.loads(ln)
    sr[r["name"]] = r
out["scale_vs_stored"] = []
for (s_, k), r in rows.items():
    if s_ == "scale" and not r.get("skipped") and k in sr:
        out["scale_vs_stored"].append({"key": k, "n": r["n"], "stock_actual": r["twoq_actual"],
                                       "pred_pass": r["twoq_pred_pass"], "pred_full": r["twoq_pred_full"],
                                       "stored_qcec_selector": sr[k]["twoq_ours"],
                                       "sites_found": r["sites_found"], "sites_applied_stock": r["sites_applied"],
                                       "stored_sites_applied": sr[k].get("sites_applied"),
                                       "stored_window_pairs": sr[k].get("window_pairs"),
                                       "stored_r_gadgets": sr[k].get("r_gadgets")})
print("scale vs stored:")
for x in out["scale_vs_stored"]:
    print(" ", x)

json.dump(out, open(os.path.join(HERE, "analysis.json"), "w"), indent=1)
for s, v in out["suites"].items():
    print(s, {k: v[k] for k in v if k not in ("actions",)})
print("mismatch types:", Counter(m["type"] for m in out["mismatch_toffolis"]))
print("swap check:", out["swap_check"])

import os, sys
_R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # repository root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_O = os.environ.get("SIM_OUT", os.path.join(_R, "experiments", "sim_2026-10-29", "out")); os.makedirs(_O, exist_ok=True)      # outputs
"""Step 5a: both compilations (Qiskit and the final QCPT = best of before routing / own router with the SABRE portfolio / after routing / Qiskit fall-back by the predicted success) of the 33 frozen circuits
for the saved calibration target of 29 Oct 2026 (03:20) with the regions of the probe (fixtures/aachen_repro/selection.json); couplers of the probe above 1.2 % are marked dead for the compiler."""
import os, sys, json, pickle, copy
R = _R; F = R + "/experiments/fixtures"
os.environ.update(FROZEN=F + "/nighthawk/frozen_circuits.json", SELECTION=F + "/aachen_repro/selection.json", RUN_TAG="forkA_sim33", TAGS="qiskit,qct,qct_topo,qct_post",
                  QCPT_TOPO_SABRE="1", TRANSPILE_SEEDS="10", LAYOUT_TRIALS="24", TOPO_WIN_S="20", POST_SECONDS="30", QCPT_GPU="0")
sys.path[:0] = [R, R + "/experiments", _O]
import simlib as S
import mirror_test_nighthawk as MT
tg = copy.deepcopy(S.get_target()); P = json.load(open(F + "/aachen_probe3/probe.json"))["coupler"]; bad = 0
for k, v in P.items():
    a, b = map(int, k.split("-")); infl = v["err_per_cz"] / 3.5
    for q in ((a, b), (b, a)):
        p = tg["cz"].get(q)
        if p is None or p.error is None: continue
        if v["err_per_cz"] > 0.012: p.error = 0.9; bad += 1
        elif infl > p.error: p.error = infl
bk = S._Bk(tg); print("couplers marked dead:", bad // 2, flush=True)
items = MT.build(bk); pickle.dump(items, open(_O + "/items33.pkl", "wb"))
by = {}
for it in items: by.setdefault(it["name"], {})[it["tag"]] = it
print("BUILT", len(by), "circuits", flush=True)
for nm, v in sorted(by.items()): print(nm, {t: (x["twoq_t"], round(x["esp_pred"], 3)) for t, x in v.items()}, v["qiskit"].get("qct_final"), flush=True)

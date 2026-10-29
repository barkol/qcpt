import os, sys
_R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # repository root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_O = os.environ.get("SIM_OUT", os.path.join(_R, "experiments", "sim_2026-10-29", "out")); os.makedirs(_O, exist_ok=True)      # outputs
"""Measured circuits of 29 Oct 2026 (probed regions): items (transpiled circuits) and hardware results of the two jobs, in the structure of the old fit (rows with set train/test)."""
import json, os, pickle
F = _R + "/experiments/fixtures/"
NINE = ["rl_4mod5-bdd_287", "rl_rd32_271", "rl_hwb4_52", "rl_4mod5-v0_18", "rl_4mod5-v1_23", "rl_one-two-three-v2_100", "adder_live_carry", "mod_incr_3b", "ripple_carry_adder_2b"]
SMALL = ["mod_incr_2b", "mod_incr_3b", "mod_incr_4b", "ripple_adder_1b", "ripple_carry_adder_2b", "ripple_adder_3b", "grover_oracle_2c", "grover_oracle_mcx3", "grover_oracle_mcx4", "half_uncomputed_min", "half_uncomputed_2c", "half_uncomputed_3c"]
LARGE = ["mod_incr_5b", "ripple_adder_4b", "grover_oracle_5c", "half_uncomputed_4c"]
FAM = {"mod_incr": ["mod_incr_2b", "mod_incr_3b", "mod_incr_4b", "mod_incr_5b"], "ripple_adder": ["ripple_adder_1b", "ripple_carry_adder_2b", "ripple_adder_3b", "ripple_adder_4b"],
       "grover_ladder": ["grover_oracle_2c", "grover_oracle_mcx3", "grover_oracle_mcx4", "grover_oracle_5c"], "half_uncomputed": ["half_uncomputed_min", "half_uncomputed_2c", "half_uncomputed_3c", "half_uncomputed_4c"]}
def load_all():
    """dict (name, tag) -> dict(item=..., res=...) for every sent circuit; the reproduction job overrides the earlier job for the control circuit."""
    out = {}
    for fx in ("aachen_next_hw2", "aachen_repro_hw"):
        items = {(i["name"], i["tag"]): i for i in pickle.load(open(F + fx + "/items_hw.pkl", "rb"))}
        for r in map(json.loads, open(F + fx + "/hw_results.jsonl")):
            k = (r["name"], r["tag"]); out[k] = dict(item=items[k], res=r, job=fx)
    # drop stale entries of the earlier job when the reproduction job measured the circuit with another QCPT tag
    for nm in {k[0] for k in out}:
        if any(out[k]["job"] == "aachen_repro_hw" for k in out if k[0] == nm):
            for k in [k for k in out if k[0] == nm and out[k]["job"] != "aachen_repro_hw"]: del out[k]
    return out
def data_qubits(all_, name):
    q = all_[(name, "qiskit")]["item"]; return q.get("data", [])
def pair(all_, name):
    q = all_[(name, "qiskit")]; c = [v for k, v in all_.items() if k[0] == name and k[1] != "qiskit"][0]; return q, c

"""Scaling families for the dynamometer figure: four circuit families (modular incrementer, ripple adder, Grover AND ladder, half-uncomputed oracle) in three sizes each, Qiskit against QCPT (same candidates and
selection rule as the 33-circuit run), built already transpiled for ibm_aachen on the selected regions, then run in the noisy simulation (Aer density matrix, noise model from the calibration of ibm_aachen).
Nothing is sent to the QPU. Circuits: fixtures/nighthawk/frozen_scaling.json (race + windows or stage 1, whichever is cheaper, certified). Output: fixtures/aachen_scaling/{items.pkl,sim_results.jsonl}. Usage: aachen_scaling.py [build|sim]"""
import json, os, pickle, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
TAG = "aachen_scaling"; FIX = os.path.join(HERE, "fixtures", TAG); os.makedirs(FIX, exist_ok=True)
FAM = {"mod_incr": ["mod_incr_2b", "mod_incr_3b", "mod_incr_4b", "mod_incr_5b", "mod_incr_6b", "mod_incr_7b"], "ripple_adder": ["ripple_adder_1b", "ripple_carry_adder_2b", "ripple_adder_3b", "ripple_adder_4b", "ripple_adder_5b"], "grover_ladder": ["grover_oracle_2c", "grover_oracle_mcx3", "grover_oracle_mcx4", "grover_oracle_5c", "grover_oracle_6c"], "half_uncomputed": ["half_uncomputed_min", "half_uncomputed_2c", "half_uncomputed_3c", "half_uncomputed_4c", "half_uncomputed_5c"]}
sel = json.load(open(os.path.join(HERE, "fixtures", "aachen_rerun", "selection.json"))); sel["regions"] = {k: sel["regions"][k] for k in ("10", "12", "14")}              # the 10-qubit region for every circuit of 4 to 10 qubits, the 12-qubit one for mod_incr_6b
json.dump(sel, open(os.path.join(FIX, "selection_r10.json"), "w"))
os.environ.update(RUN_TAG=TAG, IBM_BACKEND="ibm_aachen", IBM_ACCOUNT="eu-de-aachen-156", SHOTS=os.environ.get("SHOTS", "8000"), SELECTION=os.path.join(FIX, "selection_r10.json"), NAMES=os.environ.get("ONLY") or ",".join(sum(FAM.values(), [])),
                  FROZEN=os.path.join(HERE, "fixtures", "nighthawk", "frozen_scaling.json"), POST_SECONDS="40")
import mirror_test_nighthawk as MT
from qiskit_ibm_runtime import QiskitRuntimeService
PKL = os.path.join(FIX, "items.pkl"); OUT = os.path.join(FIX, os.environ.get("SIM_OUT", "sim_results.jsonl")); SIMSHOTS = int(os.environ.get("SIM_SHOTS", "8000"))
if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "build"; bk = QiskitRuntimeService(name=MT.ACC).backend(MT.BK)
    if stage == "build":
        items = MT.build(bk)
        if os.environ.get("ONLY") and os.path.exists(PKL): items = [it for it in pickle.load(open(PKL, "rb")) if it["name"] not in os.environ["ONLY"].split(",")] + items          # add to the existing items
        pickle.dump(items, open(PKL, "wb")); json.dump(dict(skipped=MT.TOPO_SKIPPED, post=MT.POST_NOTES), open(os.path.join(FIX, "notes.json"), "w"), indent=1)
        for it in items: print(it["name"], it["tag"], "2q", it["twoq_t"], "ESP", round(it["esp_pred"], 3), "final:", it.get("qct_final"), flush=True)
    else:
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel
        items = pickle.load(open(PKL, "rb")); by = {}
        for it in items: by.setdefault(it["name"], {})[it["tag"]] = it
        CAL_AT = os.environ.get("CAL_AT")                                   # ISO time with offset, e.g. 2026-10-26T23:15:00+01:00: the calibration that was valid then (target_history), not the current one
        if CAL_AT:
            import datetime
            t_at = datetime.datetime.fromisoformat(CAL_AT)
            class _AtTime:                                                  # the backend with the target of that time (everything else is delegated)
                def __init__(self, b, tg): self._b = b; self.target = tg
                def __getattr__(self, n_): return getattr(self._b, n_)
            nm_src = _AtTime(bk, bk.target_history(datetime=t_at)); CAL = str(bk.properties(datetime=t_at).last_update_date)
        else: nm_src = bk; CAL = str(bk.properties().last_update_date)
        print("kalibracja:", CAL, flush=True); sim = AerSimulator(method="density_matrix", noise_model=NoiseModel.from_backend(nm_src)); done = {json.loads(l)["key"] for l in open(OUT)} if os.path.exists(OUT) else set()
        for name, v in sorted(by.items()):
            q = v["qiskit"]; src = q.get("qct_final", "qiskit_fallback"); fin = q if src == "qiskit_fallback" else v[src]; qd = q["data"]
            for label, it in (("qiskit", q), ("qct", fin)):
                key = f"{name}/{label}"
                if key in done: continue
                counts = sim.run(it["t"], shots=SIMSHOTS).result().get_counts(); n = len(next(iter(counts))); p0 = counts.get("0" * n, 0) / SIMSHOTS; anc = [x for x in range(n) if x not in qd]
                acc = sum(c for b, c in counts.items() if all(b[n - 1 - x] == "0" for x in anc)) / SIMSHOTS
                row = dict(key=key, name=name, variant=label, source=("qiskit" if label == "qiskit" else src), p0_ideal=it["p0_ideal"], esp_pred=it["esp_pred"], twoq=it["twoq_t"], active_qubits=len(it["phys"]), shots=SIMSHOTS, p0_sim=p0,
                           sigma=float(np.sqrt(max(p0 * (1 - p0), 1e-12) / SIMSHOTS)), accept=acc, p_cond=(p0 / acc if acc > 0 else None), n_ancilla=len(anc), calibration=CAL)
                open(OUT, "a").write(json.dumps(row) + "\n"); print(key, row["source"], row["twoq"], round(p0, 3), flush=True)

"""Noisy simulation of the mirror tests ("fake backend"): the circuits that would be sent to ibm_aachen (the Qiskit circuit and the final QCPT circuit of every one of the 33 circuits, already transpiled
for the device, same qubits) are run on the Aer density-matrix simulator with a noise model built from the CURRENT calibration of ibm_aachen (thermal relaxation from T1 and T2, gate and readout
errors; qiskit_aer NoiseModel.from_backend). No QPU time is used. The return probability to |0...0> is read from SIM_SHOTS shots. Resumable: one json line per circuit in sim_results.jsonl.
This is a simulation, not a measurement: in the paper these results are labelled as such. Usage: aachen_sim.py (via aachen_rerun_all.py symuluj); env MAX_SIM_QUBITS (default 12), SIM_SHOTS (default 8000)."""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import mirror_test_nighthawk as MT
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService
OUT = os.path.join(MT.HERE, "sim_results.jsonl"); SHOTS = int(os.environ.get("SIM_SHOTS", "8000")); MAXQ = int(os.environ.get("MAX_SIM_QUBITS", "12"))

if __name__ == "__main__":
    import pickle
    bk = QiskitRuntimeService(name=MT.ACC).backend(MT.BK); PKL = os.path.join(MT.HERE, 'items.pkl')
    if os.path.exists(PKL) and os.environ.get('REUSE_ITEMS', '1') == '1': items = pickle.load(open(PKL, 'rb'))                 # the circuits are built deterministically; rebuilding takes ~15 min
    else: items = MT.build(bk); pickle.dump(items, open(PKL, 'wb'))
    by = {}
    for it in items: by.setdefault(it["name"], {})[it["tag"]] = it
    nm = NoiseModel.from_backend(bk); sim = AerSimulator(method="density_matrix", noise_model=nm)
    done = {json.loads(l)["key"] for l in open(OUT)} if os.path.exists(OUT) else set(); t0 = time.time()
    for name, v in sorted(by.items()):
        q = v["qiskit"]; src = q.get("qct_final", "qiskit_fallback"); fin = q if src == "qiskit_fallback" else v[src]
        q_data = q['data']
        for label, it in (("qiskit", q), ("qct", fin)):
            key = f"{name}/{label}"
            if key in done: continue
            row = dict(key=key, name=name, variant=label, source=("qiskit_fallback" if label == "qct" and src == "qiskit_fallback" else (src if label == "qct" else "qiskit")), p0_ideal=it["p0_ideal"], esp_pred=it["esp_pred"],
                       twoq=it["twoq_t"], active_qubits=len(it["phys"]), shots=SHOTS)
            if len(it["phys"]) > MAXQ: row.update(p0_sim=None, note=f"{len(it['phys'])} active qubits"); open(OUT, "a").write(json.dumps(row) + "\n"); continue
            if label == "qct" and src == "qiskit_fallback" and f"{name}/qiskit" in done:                                     # the same circuit: reuse
                prev = [json.loads(l) for l in open(OUT) if json.loads(l)["key"] == f"{name}/qiskit"][0]; row.update({k_: prev.get(k_) for k_ in ("p0_sim", "sigma", "accept", "p_cond")}, seconds=0.0)
            else:
                ts = time.time(); counts = sim.run(it["t"], shots=SHOTS).result().get_counts(); n = len(next(iter(counts))); p0 = counts.get("0" * n, 0) / SHOTS
                anc = [q for q in range(n) if q not in q_data]                                # post-selection on the ancillae being |0> at the end: acceptance and return probability given acceptance
                acc = sum(v_ for b_, v_ in counts.items() if all(b_[n - 1 - q] == '0' for q in anc)) / SHOTS
                row.update(p0_sim=p0, sigma=float(np.sqrt(max(p0 * (1 - p0), 1e-12) / SHOTS)), accept=acc, p_cond=(p0 / acc if acc > 0 else None), n_ancilla=len(anc), seconds=round(time.time() - ts, 1))
            open(OUT, "a").write(json.dumps(row) + "\n"); print({k: (round(x, 4) if isinstance(x, float) else x) for k, x in row.items() if k in ("key", "source", "twoq", "active_qubits", "p0_ideal", "p0_sim", "esp_pred", "seconds")}, flush=True)
    print("DONE", round(time.time() - t0), "s")

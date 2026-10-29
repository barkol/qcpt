"""Collects the data of the manuscript sections on simulation and hardware into one json (fixtures/aachen_results/manuscript_data.json): the simulated return probabilities of all 33 circuits, the measured
return probabilities of the nine circuits on region B (8000 shots) and of the first three on region A, the simulation of exactly the measured circuits (same qubits, noise model of the calibration at the time),
and the RB, readout and T1/T2 results of the qubits used. Usage: make_manuscript_data.py (needs the account for the job results)."""
import json, math, os, pickle, sys
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE)
import numpy as np
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService
svc = QiskitRuntimeService(name="eu-de-aachen-156"); bk = svc.backend("ibm_aachen"); F = "fixtures/"; OUT = F + "aachen_results"; os.makedirs(OUT, exist_ok=True)
def counts_of(jobjson):
    D = json.load(open(jobjson)); res = svc.job(D["job_id"]).result(); out = []
    for k, (nm, tag) in enumerate(D["order"]): out.append((nm, tag, res[k].data.meas.get_counts(), D["layout"][k]))
    return D, out
def stats(cnt, data):
    n = len(next(iter(cnt))); tot = sum(cnt.values()); anc = [b for b in range(n) if b not in data]
    p0 = cnt.get("0" * n, 0) / tot; acc = sum(v for s, v in cnt.items() if all(s[n - 1 - b] == "0" for b in anc)) / tot
    return dict(p0=p0, sigma=math.sqrt(p0 * (1 - p0) / tot), accept=acc, p_cond=(p0 / acc if acc else None), shots=tot, n_ancilla=len(anc))
nm_ = NoiseModel.from_backend(bk); sim = AerSimulator(method="density_matrix", noise_model=nm_); cal = json.load(open(F + "aachen_examples_alt/calibration_submit.json")) if os.path.exists(F + "aachen_examples_alt/calibration_submit.json") else None
M = {"sim33": [json.loads(l) for l in open(F + "aachen_rerun/sim_results.jsonl")], "hw_regionB": [], "hw_regionA": []}
for pk, jb in ((F + "aachen_examples_alt/items_alt.pkl", F + "aachen_examples_alt/job_8000.json"), (F + "aachen_six/items_alt.pkl", F + "aachen_six/job_8000.json")):
    items = {(it["name"], it["tag"]): it for it in pickle.load(open(pk, "rb"))}; D, cs = counts_of(jb)
    for nm, tag, cnt, lay in cs:
        it = items[(nm, tag)]; q = items[(nm, "qiskit")]; s = stats(cnt, q["data"])
        c = sim.run(it["t"], shots=8000).result().get_counts(); n = len(next(iter(c))); ps = c.get("0" * n, 0) / 8000
        M["hw_regionB"].append(dict(name=nm, tag=tag, variant="qiskit" if tag == "qiskit" else "qct", layout=lay, twoq=it["twoq_t"], esp_pred=it["esp_pred"], p0_sim_same_qubits=ps, job=D["job_id"], **s))
D, cs = counts_of(F + "aachen_examples/job.json"); items = {(it["name"], it["tag"]): it for it in pickle.load(open(F + "aachen_rerun/items.pkl", "rb"))}
for nm, tag, cnt, lay in cs:
    it = items[(nm, tag)]; q = items[(nm, "qiskit")]; M["hw_regionA"].append(dict(name=nm, tag=tag, variant="qiskit" if tag == "qiskit" else "qct", layout=lay, twoq=it["twoq_t"], job=D["job_id"], **stats(cnt, q["data"])))
for key, path in (("rb_B1", "aachen_examples_alt/rb/rb_results.json"), ("rb_B2", "aachen_q90_91/rb/rb_results.json"), ("t12_B1", "aachen_examples_alt_tu/bench/T1T2_used.json"), ("t12_B2", "aachen_q90_91_tu/bench/T1T2_used.json")):
    if os.path.exists(F + path): M[key] = json.load(open(F + path))
json.dump(M, open(OUT + "/manuscript_data.json", "w"), indent=1, default=float)
print("zapisano", OUT + "/manuscript_data.json", "| hw B:", len(M["hw_regionB"]), "A:", len(M["hw_regionA"]), "sim33:", len(M["sim33"]))

"""Why does the noise model built from the calibration predict the hardware poorly, and a better model. The base model is the one of the simulation (Aer NoiseModel.from_backend with the target of the calibration of 26 Oct 2026 23:00).
Two physical corrections measured on the device are added and their sizes are FITTED to the hardware data (not guessed): (1) the two-qubit gate error is larger than in the calibration (randomized benchmarking: x1.5-2.1): factor K2 on every
CZ error; (2) the echo T2 with neighbouring qubits in superposition, the regime of every mirror circuit, is 4-6x shorter than the idle T2 of the calibration (spectator dephasing, ZZ crosstalk): factor R2 on every T2 (and T2 <= 2 T1).
Training: the 24 circuits of the first job (region B). Out-of-sample: the 8 larger circuits of the second job (region B + 2 neighbours). Exact probabilities from the density matrix (no shot noise).
Usage: aachen_noise_fit.py fit|eval K2 R2    ->  fixtures/aachen_noise_fit/*.json"""
import copy, datetime, json, os, pickle, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); FIX = os.path.join(HERE, "fixtures", "aachen_noise_fit"); os.makedirs(FIX, exist_ok=True)
from qiskit.circuit import Delay
from qiskit.providers import QubitProperties
from qiskit.transpiler import InstructionProperties
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import QiskitRuntimeService
SETS = {"train": "aachen_scaling_hw", "test": "aachen_scaling_hw3"}
bk = QiskitRuntimeService(name="eu-de-aachen-156").backend("ibm_aachen"); T_AT = datetime.datetime.fromisoformat("2026-10-26T23:15:00+01:00"); TG0 = bk.target_history(datetime=T_AT)
class _At:
    def __init__(self, tg): self.target = tg
    def __getattr__(self, n): return getattr(bk, n)
def model(K2, R2, TH=0.0):
    tg = copy.deepcopy(TG0)
    if K2 != 1.0:
        for qa, pr in list(tg["cz"].items()):
            if pr is not None and pr.error is not None: tg.update_instruction_properties("cz", qa, InstructionProperties(duration=pr.duration, error=min(pr.error * K2, 0.3)))
    qp = []
    for q in tg.qubit_properties:
        t1, t2 = q.t1, q.t2; qp.append(QubitProperties(t1=t1, t2=min(t2 * R2, 2 * t1) if t2 else t2, frequency=q.frequency))
    tg.qubit_properties = qp
    nm = NoiseModel.from_backend(_At(tg))
    if TH > 0:                                       # (3) coherent residual ZZ rotation after every CZ (calibration drift, spectator ZZ): infidelity grows as the square of the number of gates
        from qiskit_aer.noise import coherent_unitary_error
        e = np.exp(-0.5j * TH); U = np.diag([e, np.conj(e), np.conj(e), e]); err = coherent_unitary_error(U)
        for qa in tg["cz"]:
            if tg["cz"][qa] is not None: nm.add_quantum_error(err, "cz", list(qa))
    return nm
def load(tag):
    D = os.path.join(HERE, "fixtures", SETS[tag]); items = {(it["name"], it["tag"]): it for it in pickle.load(open(os.path.join(D, "items_hw.pkl"), "rb"))}; rows = [json.loads(l) for l in open(os.path.join(D, "hw_results.jsonl"))]
    return [(r, items[(r["name"], r["tag"])]) for r in rows]
def prepared(it):
    t = it["t"]; meas = [(t.find_bit(i.qubits[0]).index, t.find_bit(i.clbits[0]).index) for i in t.data if i.operation.name == "measure"]; meas.sort(key=lambda m: m[1])
    c = t.remove_final_measurements(inplace=False); c.save_probabilities_dict([m[0] for m in meas]); return c, len(meas)
_NM = None; _SIM = None
def _run(args):
    c, n, data = args; pr = _SIM.run(c).result().data()["probabilities"]
    keys = {(k if isinstance(k, int) else (int(k, 16) if str(k).startswith("0x") else int(k, 2))): v for k, v in pr.items()}; anc = [i for i in range(n) if i not in data]
    return keys.get(0, 0.0), sum(v for b, v in keys.items() if all(((b >> i) & 1) == 0 for i in anc))
def predict(K2, R2, pairs, TH=0.0):
    global _NM, _SIM
    from multiprocessing import get_context
    _NM = model(K2, R2, TH); _SIM = AerSimulator(method="density_matrix", noise_model=_NM, max_parallel_threads=1); tasks = []
    for r, it in pairs: c, n = prepared(it); tasks.append((c, n, set(it_data(it, pairs))))
    with get_context("fork").Pool(24) as pool: out = pool.map(_run, tasks, chunksize=1)
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])
def it_data(it, pairs):
    for r, i2 in pairs:
        if i2["name"] == it["name"] and i2["tag"] == "qiskit": return i2["data"]
    return it.get("data", [])
def stats(pred, meas, label):
    d = pred - meas; print("%-34s n=%2d  bias %+.3f  rmse %.3f  corr %.3f" % (label, len(d), d.mean(), np.sqrt((d ** 2).mean()), np.corrcoef(pred, meas)[0, 1]))
    return dict(bias=float(d.mean()), rmse=float(np.sqrt((d ** 2).mean())), corr=float(np.corrcoef(pred, meas)[0, 1]))
if __name__ == "__main__":
    mode = sys.argv[1]; tr, te = load("train"), load("test"); m_tr = np.array([r["p0"] for r, _ in tr]); m_te = np.array([r["p0"] for r, _ in te])
    if mode == "dump":                                       # predictions of one model on all measured circuits: dump K2 R2 TH LABEL
        K2, R2, TH, lab = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]; rows = []
        for setname, pairs in (("train", tr), ("test", te)):
            p, a = predict(K2, R2, pairs, TH)
            for (r, it), p0, ac in zip(pairs, p, a): rows.append(dict(set=setname, name=r["name"], variant=r["variant"], tag=r["tag"], twoq=r["twoq"], p0_pred=float(p0), pcond_pred=float(p0 / ac) if ac > 0 else None, accept_pred=float(ac), p0_hw=r["p0"], pcond_hw=r["p_cond"], sigma=r["sigma"]))
        json.dump(dict(K2=K2, R2=R2, TH=TH, rows=rows), open(os.path.join(FIX, f"pred_{lab}.json"), "w")); m = np.array([x["p0_pred"] - x["p0_hw"] for x in rows]); print(lab, "n", len(rows), "bias %+.3f rmse %.3f" % (m.mean(), np.sqrt((m ** 2).mean()))); sys.exit(0)
    if mode == "eval":
        K2, R2 = float(sys.argv[2]), float(sys.argv[3]); TH = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0; t0 = time.time(); p_tr, a_tr = predict(K2, R2, tr, TH); p_te, a_te = predict(K2, R2, te, TH); print("czas %.0f s" % (time.time() - t0))
        s1 = stats(p_tr, m_tr, f"K2={K2} R2={R2} TH={TH} trening (24)"); s2 = stats(p_te, m_te, f"K2={K2} R2={R2} TH={TH} test (8)")
        json.dump(dict(K2=K2, R2=R2, TH=TH, train=s1, test=s2, p_train=p_tr.tolist(), p_test=p_te.tolist(), acc_train=a_tr.tolist(), acc_test=a_te.tolist()), open(os.path.join(FIX, f"eval_K{K2}_R{R2}_T{TH}.json"), "w"))
    else:
        from scipy.optimize import least_squares
        ALL = os.environ.get("FIT_ALL") == "1"; fp = tr + te if ALL else tr; fm = np.r_[m_tr, m_te] if ALL else m_tr                 # FIT_ALL=1: fit on all 32 measured circuits
        def res(x): p, _ = predict(x[0], x[1], fp, x[2]); return p - fm
        r = least_squares(res, [2.0, 0.15, 0.03], bounds=([1.0, 0.05, 0.0], [4.0, 1.0, 0.4]), diff_step=0.05, max_nfev=14, verbose=2); print("najlepsze K2=%.3f R2=%.3f THETA=%.4f koszt %.4f" % (r.x[0], r.x[1], r.x[2], r.cost)); json.dump(dict(K2=float(r.x[0]), R2=float(r.x[1]), TH=float(r.x[2])), open(os.path.join(FIX, ("best3_all.json" if ALL else "best3.json")), "w"))

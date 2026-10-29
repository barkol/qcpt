import os, sys
_R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # repository root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_O = os.environ.get("SIM_OUT", os.path.join(_R, "experiments", "sim_2026-10-29", "out")); os.makedirs(_O, exist_ok=True)      # outputs
"""Simulation library of fork A: Aer density-matrix simulation of the transpiled mirror circuits with the noise model of the calibration of 29 Oct 2026 03:20
(target saved once to target_2026-10-29.pkl), optionally with the corrections of the paper (K2 on every CZ error, R2 on every T2 with T2 <= 2 T1, a coherent ZZ rotation TH after every CZ).
Same conventions as experiments/aachen_noise_fit.py: exact probabilities (no shot noise), p0 = P(all zeros), p_cond = p0 / P(ancillae all zero)."""
import copy, os, pickle, sys, json
import numpy as np
OUT = _O; os.makedirs(OUT, exist_ok=True)
from qiskit.providers import QubitProperties
from qiskit.transpiler import InstructionProperties
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
def get_target():
    pk = os.path.join(OUT, "target_2026-10-29.pkl")
    if os.path.exists(pk): return pickle.load(open(pk, "rb"))
    from qiskit_ibm_runtime import QiskitRuntimeService
    bk = QiskitRuntimeService(name="eu-de-aachen-156").backend("ibm_aachen"); tg = copy.deepcopy(bk.target); pickle.dump(tg, open(pk, "wb")); return tg
from qiskit.providers import BackendV2, Options
class _Bk(BackendV2):
    """Minimal BackendV2 around a saved target (NoiseModel.from_backend reads only the target)."""
    def __init__(self, tg): super().__init__(name="ibm_aachen_2026-10-29"); self._t = tg
    @property
    def target(self): return self._t
    @property
    def max_circuits(self): return None
    @classmethod
    def _default_options(cls): return Options()
    def run(self, *a, **k): raise NotImplementedError
def model(tg0, K2=1.0, R2=1.0, TH=0.0):
    tg = copy.deepcopy(tg0)
    if K2 != 1.0:
        for qa, pr in list(tg["cz"].items()):
            if pr is not None and pr.error is not None: tg.update_instruction_properties("cz", qa, InstructionProperties(duration=pr.duration, error=min(pr.error * K2, 0.3)))
    if R2 != 1.0:
        qp = []
        for q in tg.qubit_properties:
            t1, t2 = q.t1, q.t2; qp.append(QubitProperties(t1=t1, t2=min(t2 * R2, 2 * t1) if t2 else t2, frequency=q.frequency))
        tg.qubit_properties = qp
    nm = NoiseModel.from_backend(_Bk(tg))
    if TH > 0:
        from qiskit_aer.noise import coherent_unitary_error
        e = np.exp(-0.5j * TH); U = np.diag([e, np.conj(e), np.conj(e), e]); err = coherent_unitary_error(U)
        for qa in tg["cz"]:
            if tg["cz"][qa] is not None: nm.add_quantum_error(err, "cz", list(qa))
    return nm
def prepared(t):
    meas = [(t.find_bit(i.qubits[0]).index, t.find_bit(i.clbits[0]).index) for i in t.data if i.operation.name == "measure"]; meas.sort(key=lambda m: m[1])
    c = t.remove_final_measurements(inplace=False); c.save_probabilities_dict([m[0] for m in meas]); return c, len(meas)
_SIM = None
def _init(nm):
    global _SIM; _SIM = AerSimulator(method="density_matrix", noise_model=nm, max_parallel_threads=1)
def _run(args):
    c, n, data = args; pr = _SIM.run(c).result().data()["probabilities"]
    keys = {(k if isinstance(k, int) else (int(k, 16) if str(k).startswith("0x") else int(k, 2))): v for k, v in pr.items()}; anc = [i for i in range(n) if i not in data]
    return keys.get(0, 0.0), sum(v for b, v in keys.items() if all(((b >> i) & 1) == 0 for i in anc))
def predict(tg0, circuits, K2=1.0, R2=1.0, TH=0.0, procs=24):
    """circuits: list of (transpiled circuit with measurements, set of data clbit indices). Returns p0 and the acceptance (ancillae all zero) arrays."""
    from multiprocessing import get_context
    nm = model(tg0, K2, R2, TH); tasks = []
    for t, data in circuits: c, n = prepared(t); tasks.append((c, n, set(data)))
    with get_context("fork").Pool(procs, initializer=_init, initargs=(nm,)) as pool: out = pool.map(_run, tasks, chunksize=1)
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])

"""Offline stand-in for ibm_aachen built from a stored calibration (properties.json): the target holds the real couplers and the calibrated errors of cz, sx, x and
readout, so that the mirror builder (mirror_test_nighthawk.build) and its predicted success probability can run without an account."""
import json
from qiskit.circuit import Parameter, Measure, Delay
from qiskit.circuit.library import CZGate, SXGate, XGate, RZGate
from qiskit.providers import BackendV2, Options
from qiskit.transpiler import Target, InstructionProperties


class OfflineAachen(BackendV2):
    def __init__(self, properties_path):
        super().__init__(name="offline_aachen")
        P = json.load(open(properties_path)); nq = len(P["qubits"]); tg = Target(num_qubits=nq)
        gerr = lambda g: next((p["value"] for p in g["parameters"] if p["name"] == "gate_error"), None)
        cz, sx, xg = {}, {}, {}
        for g in P["gates"]:
            q = tuple(g["qubits"])
            if g["gate"] == "cz": cz[q] = cz[q[::-1]] = InstructionProperties(error=gerr(g))
            elif g["gate"] == "sx": sx[q] = InstructionProperties(error=gerr(g))
            elif g["gate"] == "x": xg[q] = InstructionProperties(error=gerr(g))
        ro = {}
        for i, qp in enumerate(P["qubits"]):
            v = next((e["value"] for e in qp if e["name"] == "readout_error"), None); ro[(i,)] = InstructionProperties(error=v)
        tg.add_instruction(CZGate(), cz); tg.add_instruction(SXGate(), sx); tg.add_instruction(XGate(), xg)
        tg.add_instruction(RZGate(Parameter("theta")), {(i,): InstructionProperties(error=0.0) for i in range(nq)})
        tg.add_instruction(Measure(), ro)
        self._target = tg

    @property
    def target(self): return self._target

    @property
    def max_circuits(self): return None

    @classmethod
    def _default_options(cls): return Options(shots=1024)

    def run(self, run_input, **options): raise NotImplementedError("offline stand-in: transpile only")

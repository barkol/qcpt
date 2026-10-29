"""Mirror test of subroutine equivalence on IBM hardware (mirror-test table, Section 5.3 of the paper).
Test lustrzany (subroutine) na IBM: M = H_d . V^dag . [bariera] . E . H_d, pomiar wszystkich kubitow.
E = dekompozycja dokladna, V = wariant (pass 'sub' albo count-greedy). Jesli U_V P_in = e^{i th} U_E P_in,
to P(0...0) = 1 idealnie; fazy wzgledne na osiagalnych stanach obnizaja P0. H tylko na kubitach danych
(czyste ancille zostaja |0>). Etapy:  plan (idealnie + symulacja szumu) | wyslij | odbierz.
Wznawialne: id zadania zapisywane natychmiast do job.json; 'wyslij' odmawia, gdy job.json istnieje."""
import json, os, sys, time
from _paths import EXPERIMENTS_DIR  # noqa: F401
import numpy as np
import benchmarks as B
from _clean import clean_ancillas
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from toffoli_optimizer.core.decomposition_selector import ErrorBudgetSelector
import baselines
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "mirror_test")
os.makedirs(HERE, exist_ok=True)
NAMES = ["half_uncomputed", "controlled_adder_2b", "nested_compute_uncompute", "single_live_toffoli", "live_and_chain"]
BACKEND = os.environ.get("IBM_BACKEND", "ibm_marrakesh"); SHOTS = int(os.environ.get("SHOTS", "4000"))


def mirror(E, V, data):
    n = E.num_qubits; m = QuantumCircuit(n)
    for q in data: m.h(q)
    m.compose(E, inplace=True); m.barrier()
    m.compose(V.inverse(), inplace=True)
    for q in data: m.h(q)
    m.measure_all(); return m


def build():
    suite = {qc.name: qc for qc in B.benchmark_suite()}; out = []
    for nm in NAMES:
        qc = suite[nm]; pins = clean_ancillas(qc); data = [q for q in range(qc.num_qubits) if q not in pins]
        E = ErrorBudgetSelector().decompose_exact_only(qc)
        variants = {"ref": E, "sub": ErrorBudgetSelector().select(qc, pinned_zero=pins)["circuit"], "greedy": baselines.m_count_greedy(qc)}
        for tag, V in variants.items():
            m = mirror(E, V, data); m.name = f"{nm}__{tag}"
            sv = Statevector(m.remove_final_measurements(inplace=False)); p0 = float(abs(sv.data[0]) ** 2)
            out.append(dict(name=nm, tag=tag, circ=m, p0_ideal=p0, twoq_V=sum(1 for i in V.data if i.operation.num_qubits == 2)))
    return out


def transpile_all(items, backend):
    res = []
    for nm in NAMES:
        grp = [it for it in items if it["name"] == nm]
        ref = transpile(grp[0]["circ"], backend, optimization_level=2, seed_transpiler=11)
        lay = ref.layout.initial_index_layout()[: grp[0]["circ"].num_qubits]
        for it in grp:
            t = ref if it["tag"] == "ref" else transpile(it["circ"], backend, optimization_level=2, initial_layout=lay, seed_transpiler=11)
            it["t"] = t; it["twoq_t"] = sum(1 for i in t.data if i.operation.num_qubits == 2 and i.operation.name != "barrier"); res.append(it)
    return res


def p0_from_counts(c, n):
    tot = sum(c.values()); z = c.get("0" * n, 0); p = z / tot; return p, float(np.sqrt(p * (1 - p) / tot)), tot


if __name__ == "__main__":
    stage = sys.argv[1]
    items = build()
    if stage == "plan":
        from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
        from qiskit_aer import AerSimulator
        fb = FakeMarrakesh(); sim = AerSimulator.from_backend(fb)
        rows = []
        for it in transpile_all(items, fb):
            c = sim.run(it["t"], shots=SHOTS, seed_simulator=5).result().get_counts()
            p, s, _ = p0_from_counts(c, it["circ"].num_qubits)
            rows.append(dict(name=it["name"], tag=it["tag"], p0_ideal=round(it["p0_ideal"], 4), p0_noisy=round(p, 4), sigma=round(s, 4), twoq_V=it["twoq_V"], twoq_transpiled=it["twoq_t"]))
            print(rows[-1], flush=True)
        json.dump(rows, open(os.path.join(HERE, "plan_FakeMarrakesh.json"), "w"), indent=1)
        print("strzalow lacznie", SHOTS * len(rows), "szac. QPU s", round(SHOTS * len(rows) * 293e-6, 1))
    elif stage == "wyslij":
        J = os.path.join(HERE, "job.json")
        if os.path.exists(J): sys.exit("job.json istnieje - nie wysylam ponownie (uzyj 'odbierz')")
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        svc = QiskitRuntimeService(name="us-east-open"); u = svc.usage(); print("pozostalo s:", u.get("usage_remaining_seconds"))
        bk = svc.backend(BACKEND); tr = transpile_all(items, bk)
        sam = SamplerV2(mode=bk); job = sam.run([it["t"] for it in tr], shots=SHOTS)
        json.dump(dict(job_id=job.job_id(), backend=BACKEND, shots=SHOTS, order=[(it["name"], it["tag"]) for it in tr],
                       p0_ideal=[it["p0_ideal"] for it in tr], twoq_t=[it["twoq_t"] for it in tr], sent=time.ctime()), open(J, "w"), indent=1)
        print("wyslano", job.job_id(), flush=True)
    elif stage == "odbierz":
        from qiskit_ibm_runtime import QiskitRuntimeService
        J = json.load(open(os.path.join(HERE, "job.json"))); svc = QiskitRuntimeService(name="us-east-open")
        job = svc.job(J["job_id"]); print("status", job.status()); res = job.result(); rows = []
        for k, ((nm, tag), pid) in enumerate(zip(J["order"], J["p0_ideal"])):
            c = res[k].data.meas.get_counts(); n = len(next(iter(c)))
            p, s, tot = p0_from_counts(c, n)
            rows.append(dict(name=nm, tag=tag, p0_ideal=round(pid, 4), p0_hw=round(p, 4), sigma=round(s, 4), shots=tot, twoq_transpiled=J["twoq_t"][k]))
            print(rows[-1])
        json.dump(dict(job=J, rows=rows, usage=str(job.usage()) if hasattr(job, "usage") else None), open(os.path.join(HERE, "wyniki_hw.json"), "w"), indent=1)

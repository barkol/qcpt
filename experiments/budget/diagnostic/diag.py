"""Test diagnostyczny sumatora (KB 02.10). A: obwody ISA 3,4,5 z zadania davcn5o4oijs73e7b5m0 (ten sam
uklad). B: te same 3 warianty na ukladzie z dala od kubitow 2-7 i 16. C: T1 i echo Hahna na q2,q16,q3,q6.
Etapy: plan | wyslij | odbierz. Id zadania zapisywany natychmiast do job.json."""
import json, os, sys, time, pickle
_EXP = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # experiments/
sys.path.insert(0, _EXP)  # mirror_test_ibm.py (the mirror-test driver)
import numpy as np
from qiskit import QuantumCircuit, transpile, qpy
HERE = os.path.dirname(os.path.abspath(__file__)); SHOTS = 4000; SHOTS_C = 500
Q_C = [2, 16, 3, 6]
TAUS_US = [0, 10, 25, 50, 100, 200, 300]
AVOID = {2, 3, 4, 5, 6, 7, 16}


def circ_T1(dt, tau_us, nq_total):
    c = QuantumCircuit(nq_total, len(Q_C))
    d = int(round(tau_us * 1e-6 / dt / 16)) * 16
    for k, q in enumerate(Q_C):
        c.x(q)
        if d: c.delay(d, q, unit="dt")
    for k, q in enumerate(Q_C): c.measure(q, k)
    c.name = f"T1_{tau_us}"; return c


def circ_echo(dt, tau_us, nq_total):
    c = QuantumCircuit(nq_total, len(Q_C))
    d = int(round(tau_us * 1e-6 / dt / 32)) * 16   # polowa opoznienia, wielokrotnosc 16 dt
    for q in Q_C:
        c.sx(q)
        if d: c.delay(d, q, unit="dt")
        c.x(q)
        if d: c.delay(d, q, unit="dt")
        c.sx(q)       # sx . x . sx = idealnie |0> -> sx^2 x = X X = I? (sprawdzane w 'plan')
    for k, q in enumerate(Q_C): c.measure(q, k)
    c.name = f"echo_{tau_us}"; return c


def build(target):
    import mirror_test_ibm as L
    items = [it for it in L.build() if it["name"] == "controlled_adder_2b"]
    isa_old = qpy.load(open(os.path.join(HERE, "..", "lustro_isa.qpy"), "rb"))
    A = [isa_old[3], isa_old[4], isa_old[5]]
    lay = None
    for seed in range(1, 400):
        t = transpile(items[0]["circ"], target=target, optimization_level=2, seed_transpiler=seed)
        L0 = t.layout.initial_index_layout()[: items[0]["circ"].num_qubits]
        used = {t.find_bit(i.qubits[k]).index for i in t.data for k in range(len(i.qubits))}
        if not (used & AVOID): lay = L0; break
    if lay is None: raise SystemExit("brak ukladu z dala od 2-7,16")
    Bc = [transpile(it["circ"], target=target, optimization_level=2, initial_layout=lay, seed_transpiler=11) for it in items]
    nq = target.num_qubits; dt = target.dt
    Cc = [transpile(circ_T1(dt, t, nq), target=target, optimization_level=0) for t in TAUS_US] + \
         [transpile(circ_echo(dt, t, nq), target=target, optimization_level=0) for t in TAUS_US]
    return A, Bc, Cc, lay, [it["p0_ideal"] for it in items], [it["tag"] for it in items]


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "plan":
        from qiskit_aer import AerSimulator
        d = pickle.load(open(os.path.join(HERE, "..", "kal_ibm_marrakesh.pkl"), "rb"))
        A, Bc, Cc, lay, p0i, tags = build(d["target"])
        print("uklad B", lay, "2q:", [sum(1 for i in c.data if i.operation.num_qubits == 2) for c in Bc], "A 2q:", [sum(1 for i in c.data if i.operation.num_qubits == 2) for c in A])
        sim = AerSimulator()
        for c in (Cc[0], Cc[len(TAUS_US)]):
            small = QuantumCircuit(4, 4)
            for i in c.data:
                qs = [c.find_bit(q).index for q in i.qubits]
                if i.operation.name in ("delay", "barrier"): continue
                if all(q in Q_C for q in qs):
                    small.append(i.operation, [Q_C.index(q) for q in qs], [c.find_bit(b).index for b in i.clbits])
            print(c.name, "idealnie:", sim.run(small, shots=200).result().get_counts())
        n = SHOTS * 6 + SHOTS_C * len(Cc); print("strzalow", n, "szac. s", round(n * 293e-6 + sum(TAUS_US) * 2 * 1e-6 * SHOTS_C, 1))
    elif st == "wyslij":
        J = os.path.join(HERE, "job.json")
        if os.path.exists(J): sys.exit("job.json istnieje")
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        svc = QiskitRuntimeService(name="us-east-open"); print("pozostalo", svc.usage().get("usage_remaining_seconds"))
        bk = svc.backend("ibm_marrakesh"); A, Bc, Cc, lay, p0i, tags = build(bk.target)
        pubs = [(c, None, SHOTS) for c in A + Bc] + [(c, None, SHOTS_C) for c in Cc]
        job = SamplerV2(mode=bk).run(pubs)
        json.dump(dict(job_id=job.job_id(), layout_B=list(lay), tags=tags, p0_ideal=p0i, taus_us=TAUS_US, q_C=Q_C, sent=time.ctime()), open(J, "w"), indent=1)
        with open(os.path.join(HERE, "obwody.qpy"), "wb") as f: qpy.dump(A + Bc + Cc, f)
        pickle.dump(bk.properties().to_dict(), open(os.path.join(HERE, "props_wysylka.pkl"), "wb"))
        print("wyslano", job.job_id())

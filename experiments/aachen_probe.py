"""Probe of every qubit and every coupler of ibm_aachen with a few short circuits (about 4 s of QPU time): (1) all qubits in parallel: 40 X gates (identity) and readout, (2) all couplers in
parallel, in matchings: H H (CZ CZ)^m H H = identity on each pair. Per-qubit flip probability and per-coupler success probability and error per CZ. Usage: aachen_probe.py send|fetch"""
import json, os, sys, time
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
HERE = os.path.dirname(os.path.abspath(__file__)); FIX = os.path.join(HERE, "fixtures", os.environ.get("PROBE_TAG", "aachen_probe")); os.makedirs(FIX, exist_ok=True)
SHOTS, M, NX = 2000, 15, 40
svc = QiskitRuntimeService(name="eu-de-aachen-156"); bk = svc.backend("ibm_aachen"); N = bk.num_qubits
alive = sorted({(min(a, b), max(a, b)) for (a, b), v in bk.target["cz"].items() if v is not None and v.error is not None and v.error < 0.5})
good_q = [q for q in range(N) if (q,) in bk.target["measure"] and bk.target["measure"][(q,)].error is not None and bk.target["measure"][(q,)].error < 0.5 and bk.target["sx"][(q,)].error is not None]
def matchings(edges):
    left = list(edges); out = []
    while left:
        used = set(); cur = []
        for e in list(left):
            if e[0] not in used and e[1] not in used: cur.append(e); used |= set(e); left.remove(e)
        out.append(cur)
    return out
MATCH = matchings([e for e in alive if e[0] in good_q and e[1] in good_q])
def build():
    circs = []
    q1 = QuantumCircuit(N, len(good_q))
    for _ in range(NX): q1.x(good_q); q1.barrier(good_q)
    q1.measure(good_q, range(len(good_q))); circs.append(("q1", q1, good_q))
    for k, mt in enumerate(MATCH):
        qs = sorted({q for e in mt for q in e}); c = QuantumCircuit(N, len(qs)); pos = {q: i for i, q in enumerate(qs)}
        c.h(qs); c.barrier(qs)
        for _ in range(2 * M):
            for a, b in mt: c.cz(a, b)
            c.barrier(qs)
        c.h(qs); c.measure(qs, [pos[q] for q in qs]); circs.append((f"m{k}", c, qs))
    return circs
if __name__ == "__main__":
    stage = sys.argv[1]; circs = build(); meta = [(n, qs, ([e for e in MATCH[int(n[1:])]] if n != "q1" else None)) for n, _, qs in circs]
    est = SHOTS * len(circs) * 0.000293 * 1.35 + 0.0
    if stage == "send":
        t = [transpile(c, bk, initial_layout=list(range(N)), optimization_level=0) for _, c, _ in circs]
        for tc in t:
            ops = {i.operation.name for i in tc.data}; assert ops <= {"cz", "sx", "x", "rz", "measure", "barrier"}, ops
        rem = svc.usage()["usage_remaining_seconds"]; print(f"{len(circs)} circuits, {sum(len(m[1]) for m in meta[1:])//2} couplers in {len(MATCH)} matchings, est {est:.1f} s, remaining {rem} s")
        if os.path.exists(os.path.join(FIX, "job.json")) or rem < est * 1.5 or os.environ.get("NO_SEND"): sys.exit("not sending (job exists / budget / NO_SEND)")
        job = SamplerV2(mode=bk).run(t, shots=SHOTS); json.dump(dict(job_id=job.job_id(), sent=time.strftime("%Y-%m-%d %H:%M:%S"), meta=meta, shots=SHOTS, M=M, NX=NX), open(os.path.join(FIX, "job.json"), "w")); print("sent", job.job_id())
    else:
        jd = json.load(open(os.path.join(FIX, "job.json"))); res = svc.job(jd["job_id"]).result(); out = dict(qubit={}, coupler={})
        for (name, qs, mt), pub in zip(jd["meta"], res):
            counts = pub.data[list(pub.data.keys())[0]].get_counts(); tot = sum(counts.values()); n = len(qs); pos = {q: i for i, q in enumerate(qs)}
            if name == "q1":
                for q in qs: out["qubit"][q] = sum(c for b, c in counts.items() if b[n - 1 - pos[q]] == "1") / tot
            else:
                for a, b in mt:
                    p00 = sum(c for s, c in counts.items() if s[n - 1 - pos[a]] == "0" and s[n - 1 - pos[b]] == "0") / tot
                    out["coupler"][f"{a}-{b}"] = dict(p00=p00, err_per_cz=1 - p00 ** (1.0 / (2 * jd["M"])))
        json.dump(out, open(os.path.join(FIX, "probe.json"), "w"), indent=1)
        qv = sorted(out["qubit"].items(), key=lambda kv: -kv[1]); cv = sorted(out["coupler"].items(), key=lambda kv: -kv[1]["err_per_cz"])
        print("worst qubits (flip after 40 X + readout):", [(q, round(p, 3)) for q, p in qv[:14]]); print("median qubit flip %.4f" % np.median([p for _, p in qv]))
        print("worst couplers (error per CZ):", [(k, round(v["err_per_cz"], 4)) for k, v in cv[:14]]); print("median error per CZ %.4f" % np.median([v["err_per_cz"] for _, v in cv]))

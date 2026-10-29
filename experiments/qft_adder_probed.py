"""Hardware test of the budgeted approximate QFT (Draper) adder on ibm_aachen, n = 6 bits (12 qubits), cutoffs m = 4, 5, 6 (m = 6 exact), 6 input pairs (random.seed(5), as in qft_adder_aachen.py of the
earlier run), on a REGION OF 12 QUBITS CHOSEN FROM A PROBE (fixtures/aachen_probe5, selection fixtures/aachen_draper): the circuits are compiled for the induced coupling map of the region without the couplers that
the probe finds above 1.2% per CZ and then placed on the physical qubits, so no gate leaves the region. Stages: plan (dry run) | wyslij | odbierz;  env QKEY (Q or R), QSEED, SHOTS (default 4000), NO_SEND."""
import os, sys, json, math, time, random
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
HERE0 = os.path.dirname(os.path.abspath(__file__)); FX = os.path.join(HERE0, "fixtures")
HERE = os.path.join(FX, "aachen_draper_2026-10-29"); os.makedirs(HERE, exist_ok=True); STATE = os.path.join(HERE, "state.json")
n, MS = 6, (4, 5, 6); SHOTS = int(os.environ.get("SHOTS", "4000")); KEY = os.environ.get("QKEY", "Q"); SEED = int(os.environ.get("QSEED", "11"))
def adder(n, m):
    qc = QuantumCircuit(2 * n)
    def qft(qs, inverse=False):
        ops = []
        for j in range(len(qs) - 1, -1, -1):
            ops.append(("h", qs[j]))
            for k in range(j - 1, -1, -1):
                d = j - k + 1
                if d <= m: ops.append(("cp", qs[j], qs[k], math.pi / 2 ** (d - 1)))
        for op in (reversed(ops) if inverse else ops):
            if op[0] == "h": qc.h(op[1])
            else: qc.cp(-op[3] if inverse else op[3], op[1], op[2])
    b = list(range(n, 2 * n)); qft(b)
    for j in range(n):
        for k in range(n):
            d = k - j + 1
            if 1 <= d <= m: qc.cp(2 * math.pi / 2 ** d, j, b[k])
    qft(b, inverse=True); return qc
random.seed(5); pairs = []
while len(pairs) < 6:
    a, b = random.randrange(1, 64), random.randrange(1, 64)
    if (a, b) not in pairs: pairs.append((a, b))
svc = QiskitRuntimeService(name="eu-de-aachen-156"); bk = svc.backend("ibm_aachen"); N = bk.num_qubits
SEL = json.load(open(os.path.join(FX, "aachen_draper", "selection.json"))); REG = sorted(SEL["regions"]["12"]["qubits"]); PROBE = json.load(open(os.path.join(FX, "aachen_probe5", "probe.json")))["coupler"]
badc = {tuple(sorted(map(int, k.split("-")))) for k, v in PROBE.items() if v["err_per_cz"] > 0.012}
alive = {tuple(sorted(k)) for k, v in bk.target["cz"].items() if v is not None and v.error is not None and v.error < 0.5}
pos = {q: i for i, q in enumerate(REG)}
edges = sorted(e for e in alive if e[0] in pos and e[1] in pos and e not in badc)
CM = CouplingMap([(pos[a], pos[b]) for a, b in edges] + [(pos[b], pos[a]) for a, b in edges])
BASIS = ["cz", "sx", "rz", "x"]
def embed(t):
    full = QuantumCircuit(N, n)
    full.compose(t, qubits=REG, clbits=list(range(n)), inplace=True); return full
def build():
    base = transpile(adder(n, n), coupling_map=CM, basis_gates=BASIS, optimization_level=3, seed_transpiler=SEED)
    lay = [int(x) for x in base.layout.initial_index_layout()[:2 * n]]; out = []; meta = []
    for (a, b) in pairs:
        for m in MS:
            qc = QuantumCircuit(2 * n, n)
            for i in range(n):
                if a >> i & 1: qc.x(i)
                if b >> i & 1: qc.x(n + i)
            qc.compose(adder(n, m), inplace=True); qc.measure(range(n, 2 * n), range(n))
            t = transpile(qc, coupling_map=CM, basis_gates=BASIS, initial_layout=lay, optimization_level=3, seed_transpiler=SEED)
            out.append(embed(t)); meta.append(dict(a=a, b=b, m=m))
    return out, meta, lay
def check(circs):
    viol = []
    for c in circs:
        for ins in c.data:
            nm = ins.operation.name; q = tuple(c.find_bit(x).index for x in ins.qubits)
            if nm in ("barrier", "measure"): 
                if nm == "measure" and q[0] not in pos: viol.append((nm, q))
                continue
            if nm not in BASIS: viol.append((nm, q))
            elif len(q) == 2 and (tuple(sorted(q)) not in alive or tuple(sorted(q)) in badc or q[0] not in pos or q[1] not in pos): viol.append((nm, q))
            elif len(q) == 1 and q[0] not in pos: viol.append((nm, q))
    return viol
if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd in ("plan", "wyslij"):
        circs, meta, lay = build(); est = len(circs) * SHOTS * 0.285e-3 + 2.2; rem = svc.usage().get("usage_remaining_seconds"); viol = check(circs)
        nz = [sum(1 for i in c.data if i.operation.num_qubits == 2) for c in circs]
        print(f"key {KEY} seed {SEED} | region {REG} | induced healthy couplers {len(edges)} (dropped inside: {sorted(e for e in alive if e[0] in pos and e[1] in pos and e in badc)}) | circuits {len(circs)} x {SHOTS} shots | est {est:.1f} s (x1.3 = {est*1.3:.1f}) | remaining {rem} | violations {len(viol)} | CZ min/max {min(nz)}/{max(nz)} | layout {lay}")
        if cmd == "plan": sys.exit(0)
        st = json.load(open(STATE)) if os.path.exists(STATE) else {}
        if KEY in st: sys.exit("already sent " + st[KEY]["job_id"])
        if viol or rem is None or rem < est * 1.5 or os.environ.get("NO_SEND"): sys.exit(f"not sending (violations {viol[:3]} / budget / NO_SEND)")
        job = SamplerV2(mode=bk).run(circs, shots=SHOTS); st[KEY] = dict(job_id=job.job_id(), shots=SHOTS, est_s=round(est, 1), sent=time.strftime("%Y-%m-%d %H:%M:%S"), meta=meta, layout=lay, region=REG, seed=SEED, cz=nz)
        json.dump(st, open(STATE, "w"), indent=1); print("sent", job.job_id())
    else:
        D = json.load(open(STATE))[KEY]; job = svc.job(D["job_id"]); print("status", job.status()); res = job.result(); rows = []
        for k, mt in enumerate(D["meta"]):
            c = res[k].data.c.get_counts(); tgt = format((mt["a"] + mt["b"]) % 64, "06b"); p = c.get(tgt, 0) / D["shots"]
            rows.append(dict(**mt, p=p, sigma=math.sqrt(max(p * (1 - p), 1e-9) / D["shots"]), cz=D["cz"][k]))
        u = svc.usage(); json.dump(dict(job=D, rows=rows, usage_after=u.get("usage_consumed_seconds")), open(os.path.join(HERE, f"wyniki_{KEY}.json"), "w"), indent=1)
        for m in MS:
            r = [x for x in rows if x["m"] == m]; print("m=%d mean P %.4f (n=%d), CZ mean %.0f" % (m, np.mean([x["p"] for x in r]), len(r), np.mean([x["cz"] for x in r])))
        print("consumed", u.get("usage_consumed_seconds"), "remaining", u.get("usage_remaining_seconds"))

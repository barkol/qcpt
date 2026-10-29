"""All experiments of one backend (Nighthawk, or ibm_aachen repeated: RUN_TAG=aachen_2 IBM_BACKEND=ibm_aachen IBM_ACCOUNT=...) in one IBM Runtime Batch: the mirror test (mirror_test_nighthawk.py: Qiskit-generated and QCPT circuits) and the device
characterization (aachen_bench.py blocks TI, TS, RO, R1, R2, QV, GH). One batch means one queueing and one calibration window for everything.
Block TU measures T1 and echo T2, with the neighbouring couplers off (neighbours idle in |0>) and on (neighbours excited or in superposition), on exactly the qubits the mirror circuits use (split into groups with no two neighbours, so there is no ZZ crosstalk); the
calibration values of those qubits at submission are stored in used_qubits.json and, after the run, the values valid at the time of the job.
Stages: sprawdz (counts and QPU budget for the whole batch) | wyslij (submits the batch; every job id is stored at once in fixtures/nighthawk/job.json
(mirror test) and fixtures/<RUN_TAG>/bench/state_bench.json (blocks), refuses if either exists). Stage odbierz fetches TU (T1T2_used.json); the other results come from the existing stages:
`mirror_test_nighthawk.py odbierz` and `BENCH_DIR=... aachen_bench.py odbierz BLOCK`.
Environment: IBM_ACCOUNT (optional), IBM_BACKEND (name of the Nighthawk backend), SHOTS, TAGS, NAMES."""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
BENCH = os.path.join(HERE, "fixtures", os.environ.get("RUN_TAG", "nighthawk"), "bench"); os.makedirs(BENCH, exist_ok=True); os.environ.setdefault("BENCH_DIR", BENCH)
os.environ.setdefault("IBM_BACKEND", "ibm_nighthawk")
import mirror_test_nighthawk as MT, aachen_bench as AB
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2, Batch
import numpy as np
from qiskit import transpile
TU_SHOTS = int(os.environ.get("TU_SHOTS", 1500))
BLOCKS = [b for b in os.environ.get("BLOCKS", "RO,R1,R2,GH,QV").split(",") if b]      # TI and TS are covered by TU (T1 and echo T2, couplers off and on, on the qubits that are used)
if os.environ.get("TU_DELAYS"): AB.DELAYS = [int(x) for x in os.environ["TU_DELAYS"].split(",")]      # shorter T1 / echo T2 scans for small budgets
SEL_PATH = os.environ.get("SELECTION"); SEL = json.load(open(SEL_PATH)) if SEL_PATH and os.path.exists(SEL_PATH) else None
if SEL:                                                           # bench blocks on the best qubits of the current calibration (aachen_calibration.py), not on a fresh pick
    def _pick(bk): return list(SEL["independent"]), [tuple(e) for e in SEL["couplers"]], AB.graph(bk)
    AB.pick = _pick

def used_qubits(items):
    return sorted({q for it in items for q in it["phys"]})

def groups(bk, qs):
    adj = AB.graph(bk); out = []
    for q in qs:
        for g in out:
            if all(x not in adj[q] for x in g): g.append(q); break
        else: out.append([q])
    return out

def circ_coupled(kind, qs, nb, d_us):
    """Neighbours nb of the measured qubits qs prepared in |1> (T1) or in superposition with the same echo (T2) during the delay: the couplers between them and qs act."""
    from qiskit import QuantumCircuit
    allq = list(qs) + list(nb); c = QuantumCircuit(len(allq), len(qs)); c.name = f"{kind}c_{d_us}"
    if kind == "T1":
        for i in range(len(allq)): c.x(i)
        c.delay(d_us, unit="us")
    else:
        for i in range(len(allq)): c.h(i)
        c.delay(d_us / 2, unit="us")
        for i in range(len(allq)): c.x(i)
        c.delay(d_us / 2, unit="us")
        for i in range(len(allq)): c.h(i)
    c.measure(range(len(qs)), range(len(qs))); return c, allq

def build_tu(bk, qs):
    """Per group of used qubits with no two neighbours: T1 and echo T2 with the neighbours idle in |0> (couplers off: no ZZ/exchange drive) and with the
    neighbours excited / in superposition (couplers on). Names: KIND[c]_DELAY_GROUP, c = neighbours driven."""
    adj = AB.graph(bk); G = groups(bk, qs); ts = []
    for gi, g in enumerate(G):
        nb = sorted({n for q in g for n in adj[q]} - set(g))
        for d in AB.DELAYS:
            for k in ("T1", "T2"):
                c, al = AB.circ_T(k, g, d); c.name = f"{k}_{d}_{gi}"; ts.append(transpile(c, bk, initial_layout=g, optimization_level=1, seed_transpiler=7))
                c, al = circ_coupled(k, g, nb, d); c.name = f"{k}c_{d}_{gi}"; ts.append(transpile(c, bk, initial_layout=al, optimization_level=1, seed_transpiler=7))
    return ts, G

def calib(bk, qs, when=None):
    pr = bk.properties(datetime=when) if when is not None else bk.properties()
    return {q: dict(t1_us=(pr.t1(q) or 0) * 1e6, t2_us=(pr.t2(q) or 0) * 1e6, readout_err=pr.readout_error(q)) for q in qs}

def fit(ts, y, kind):
    from scipy.optimize import curve_fit
    try:
        if kind == "T1": p, _ = curve_fit(lambda t, A, T: A * np.exp(-t / T), ts, y, p0=[0.95, 200], bounds=([0.5, 5], [1.0, 5000]))
        else: p, _ = curve_fit(lambda t, B, T: 0.5 * (1 - B * np.exp(-t / T)), ts, y, p0=[0.9, 150], bounds=([0.3, 5], [1.0, 5000]))
        return float(p[1])
    except Exception: return float("nan")

def return_tu(svc, bk, key="TU"):
    """Fetch block TU: T1 and echo T2 of every used qubit, against the calibration valid at the time of the job."""
    SB = os.path.join(BENCH, "state_bench.json"); D = json.load(open(SB))[key]; job = svc.job(D["job_id"]); print("status", job.status()); res = job.result()
    when = None
    try: when = job.metrics()["timestamps"]["running"]
    except Exception: pass
    from datetime import datetime
    when = datetime.fromisoformat(when.replace("Z", "+00:00")) if isinstance(when, str) else when
    G = D["meta"]["groups"]; rows = {}
    for nm, r in zip(D["names"], res):
        kind, d, gi = nm.split("_"); data = r.data; cnt = (data.c if hasattr(data, "c") else data.meas).get_counts(); tot = sum(cnt.values())
        for i, q in enumerate(G[int(gi)]): rows.setdefault((kind, q), {})[int(d)] = sum(v for k, v in cnt.items() if k[::-1][i] == "1") / tot
    qs = sorted({q for g in G for q in g}); cal = calib(bk, qs, when); ts = np.array(AB.DELAYS, float); out = {}
    for q in qs:
        f = lambda kind, T: fit(ts, np.array([rows[(kind, q)][int(d)] for d in AB.DELAYS]), T)
        out[q] = dict(T1_off_us=f("T1", "T1"), T1_on_us=f("T1c", "T1"), T2echo_off_us=f("T2", "T2"), T2echo_on_us=f("T2c", "T2"),
                      T1_cal_us=cal[q]["t1_us"], T2_cal_us=cal[q]["t2_us"], readout_err_cal=cal[q]["readout_err"])
        print(q, {k: round(v, 1) if isinstance(v, float) else v for k, v in out[q].items()})
    json.dump(dict(job_id=D["job_id"], calibration_time=str(when), qubits=out), open(os.path.join(BENCH, "T1T2_used.json" if key == "TU" else "T1T2_used_end.json"), "w"), indent=1)


if __name__ == "__main__":
    stage = sys.argv[1]; svc = QiskitRuntimeService(name=MT.ACC) if MT.ACC else QiskitRuntimeService(); bk = svc.backend(MT.BK)
    if stage == "odbierz":
        for key in [k_ for k_ in ("TU", "TU2") if k_ in json.load(open(os.path.join(BENCH, "state_bench.json")))]: return_tu(svc, bk, key)
        json.dump(__import__("aachen_calibration").snapshot(bk), open(os.path.join(os.path.dirname(MT.J), "calibration_after.json"), "w"), indent=1); sys.exit()
    if os.environ.get("ITEMS_PKL"):                                                # the exact circuits that were simulated (same placement, same QCPT choice), for the named circuits only
        import pickle; _names = set(os.environ["NAMES"].split(",")); items_all = [it for it in pickle.load(open(os.environ["ITEMS_PKL"], "rb")) if it["name"] in _names]
        assert {it["name"] for it in items_all} == _names, "a named circuit is missing in ITEMS_PKL"
    else: items_all = MT.build(bk)
    items = [it for it in items_all if it.get("send", True)]; qs = used_qubits(items); tu_ts, tu_groups = build_tu(bk, qs); blocks = {b: AB.build(b, bk) for b in BLOCKS}; u = svc.usage()
    est = len(items) * MT.SHOTS * MT.SEC_PER_SHOT + AB.est(len(tu_ts), TU_SHOTS, "TU") + sum(AB.est(len(blocks[b][0]), AB.SHOTS[b], b) for b in BLOCKS)
    pubs = [it["t"] for it in items] + tu_ts + [t for b in blocks for t in blocks[b][0]]
    viol = [(t.name, v) for t in pubs for v in MT.check_isa(t, bk)]; print(f"kontrola ISA: {len(pubs)} obwodow, naruszen: {len(viol)}", viol[:3])
    MT.show(items_all, items); print(f"uzyte kubity ({len(qs)}):", qs, "| blok TU:", len(tu_ts), "obwodow w", len(tu_groups), "grupach")
    for b in BLOCKS: print(f"blok {b}: {len(blocks[b][0])} obwodow x {AB.SHOTS[b]} strzalow")
    print(f"RAZEM szac. {est:.0f} s x {AB.SAFETY} = {est * AB.SAFETY:.0f} s | pozostalo {u.get('usage_remaining_seconds')} s")
    if stage == "wyslij":
        SB = os.path.join(BENCH, "state_bench.json")
        if os.path.exists(MT.J) or os.path.exists(SB): sys.exit("job.json lub state_bench.json istnieje - nie wysylam ponownie")
        if viol: sys.exit("obwody nie sa zgodne z backendem (ISA): nie wysylam")
        if u.get("usage_remaining_seconds", 0) < est * AB.SAFETY: sys.exit("za maly budzet")
        if SEL_PATH and (time.time() - os.path.getmtime(SEL_PATH)) > float(os.environ.get("SELECTION_MAX_AGE_H", "3")) * 3600: sys.exit("selection.json jest starszy niz SELECTION_MAX_AGE_H: uruchom ponownie aachen_calibration.py")
        json.dump(__import__("aachen_calibration").snapshot(bk), open(os.path.join(os.path.dirname(MT.J), "calibration_submit.json"), "w"), indent=1)
        state = {}
        with Batch(backend=bk) as batch:
            json.dump(dict(qubits=qs, calibration_at_submit=calib(bk, qs), groups=tu_groups, sent=time.strftime("%Y-%m-%d %H:%M:%S")), open(os.path.join(os.path.dirname(MT.J), "used_qubits.json"), "w"), indent=1)
            job = SamplerV2(mode=batch).run(tu_ts, shots=TU_SHOTS)                       # T1/T2 of the used qubits at the start of the batch (drift is read off against the copy at the end)
            state["TU"] = dict(job_id=job.job_id(), shots=TU_SHOTS, meta=dict(groups=tu_groups), names=[x.name for x in tu_ts], sent=time.strftime("%Y-%m-%d %H:%M:%S")); json.dump(state, open(SB, "w"), indent=1); print("TU", job.job_id(), flush=True)
            job = SamplerV2(mode=batch).run([it["t"] for it in items], shots=MT.SHOTS)
            json.dump(dict(job_id=job.job_id(), account=MT.ACC, backend=MT.BK, shots=MT.SHOTS, batch=batch.session_id, order=[(it["name"], it["tag"]) for it in items], p0_ideal=[it["p0_ideal"] for it in items], esp_pred=[it["esp_pred"] for it in items], qct_final={it["name"]: it.get("qct_final") for it in items_all}, qcpt_candidates={f"{it['name']}/{it['tag']}": it["esp_pred"] for it in items_all}, layout=[it["layout"] for it in items],
                           twoq_t=[it["twoq_t"] for it in items], twoq_V=[it["twoq_V"] for it in items], sent=time.strftime("%Y-%m-%d %H:%M:%S")), open(MT.J, "w"), indent=1)
            print("mirror", job.job_id(), flush=True)
            for b in BLOCKS:
                ts, meta, _ = blocks[b]; job = SamplerV2(mode=batch).run(ts, shots=AB.SHOTS[b])
                state[b] = dict(job_id=job.job_id(), shots=AB.SHOTS[b], meta=meta, names=[t.name for t in ts], sent=time.strftime("%Y-%m-%d %H:%M:%S"))
                json.dump(state, open(SB, "w"), indent=1); print(b, job.job_id(), flush=True)
            if os.environ.get("TU_END", "1") == "1":
              job = SamplerV2(mode=batch).run(tu_ts, shots=TU_SHOTS)                       # the same T1/T2 circuits at the end of the batch
              state["TU2"] = dict(job_id=job.job_id(), shots=TU_SHOTS, meta=dict(groups=tu_groups), names=[x.name for x in tu_ts], sent=time.strftime("%Y-%m-%d %H:%M:%S")); json.dump(state, open(SB, "w"), indent=1); print("TU2", job.job_id(), flush=True)

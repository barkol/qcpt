"""Hardware measurement of the next five circuits (ripple_adder_4b with the SABRE portfolio and four RevLib circuits not measured before) on ibm_aachen, regions of the current calibration.
(Derived from aachen_scaling_hw.py.) Previously: hardware measurement of the 12 scaling circuits (4 families, up to 8 qubits) on ibm_aachen, region B (qubits 90, 91, 98, 111-114, 119), Qiskit and QCPT of every circuit in ONE plain job (8000 shots).
Circuits are built already transpiled for the device (same candidates and selection rule as the 33-circuit run), the QCPT circuit of each is the best candidate by predicted success probability or the Qiskit circuit.
Stages: build (nothing sent; items_hw.pkl), wyslij (submits once; refuses a second submission, an ISA violation, an insufficient budget), odbierz (fetches counts: p0, acceptance, conditional p0 -> hw_results.jsonl).
Usage: aachen_scaling_hw.py build|wyslij|odbierz   (env NO_SEND for wyslij = dry run)"""
import json, os, pickle, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
TAG = os.environ.get("HW_TAG", "aachen_next_hw2"); FIX = os.path.join(HERE, "fixtures", TAG); os.makedirs(FIX, exist_ok=True)
NAMES = os.environ["HW_NAMES"].split(",") if os.environ.get("HW_NAMES") else ["rl_4mod5-v0_18", "ripple_adder_4b", "rl_4_49_17", "rl_decod24-bdd_294", "rl_alu-bdd_288", "rl_one-two-three-v3_101"]   # the first one was measured on 26.10 in region B (drift check)   # fixed in advance (not chosen after seeing results)
fa = json.load(open(os.path.join(HERE, "fixtures", "nighthawk", "frozen_circuits.json"))); fb = json.load(open(os.path.join(HERE, "fixtures", "nighthawk", "frozen_scaling.json")))
FROZEN = os.path.join(FIX, "frozen.json"); json.dump({n: (fa[n] if n in fa else fb[n]) for n in NAMES}, open(FROZEN, "w"))
sel = json.load(open(os.path.join(HERE, "fixtures", "aachen_next2", "selection.json"))); SEL = os.path.join(FIX, "selection.json"); json.dump(sel, open(SEL, "w"))   # current calibration, failed qubits excluded (EXCLUDE=135,139,154,155,98)
REGION = sorted(set().union(*[set(r["qubits"]) for r in sel["regions"].values()]))          # all qubits the circuits may use: the regions of the current selection (failed qubits excluded)
os.environ.update(RUN_TAG=TAG, IBM_BACKEND="ibm_aachen", IBM_ACCOUNT="eu-de-aachen-156", SHOTS="8000", SELECTION=SEL, NAMES=",".join(NAMES), FROZEN=FROZEN, POST_SECONDS="30", TRANSPILE_SEEDS="10", LAYOUT_TRIALS="24")
import mirror_test_nighthawk as MT
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
PKL = os.path.join(FIX, "items_hw.pkl"); JOB = os.path.join(FIX, "job.json"); OUT = os.path.join(FIX, "hw_results.jsonl")
def sent_items(items): return [it for it in items if it.get("send", True)]
if __name__ == "__main__":
    stage = sys.argv[1]; svc = QiskitRuntimeService(name=MT.ACC); bk = svc.backend(MT.BK)
    PR = json.load(open(os.path.join(HERE, "fixtures", "aachen_probe", "probe.json")))["coupler"]; BADP = set()          # probe of the couplers (aachen_probe.py): the calibration does not show the failed ones
    for k_, v_ in PR.items():
        a_, b_ = map(int, k_.split("-")); infl = v_["err_per_cz"] / 3.5
        for q_ in ((a_, b_), (b_, a_)):
            p_ = bk.target["cz"].get(q_)
            if p_ is None or p_.error is None: continue
            if v_["err_per_cz"] > 0.012: p_.error = 0.9; BADP.add(tuple(sorted((a_, b_))))                                # treated as dead by check_isa and by the predicted success probability
            elif infl > p_.error: p_.error = infl
    if stage == "build":
        items = MT.build(bk); pickle.dump(items, open(PKL, "wb")); json.dump(dict(skipped=MT.TOPO_SKIPPED, post=MT.POST_NOTES), open(os.path.join(FIX, "notes.json"), "w"), indent=1)
        for it in sent_items(items): print(it["name"], it["tag"], "layout", it["layout"], "2q", it["twoq_t"], "ESP", round(it["esp_pred"], 3), "final:", it.get("qct_final"), flush=True)
    elif stage == "wyslij":
        items = sent_items(pickle.load(open(PKL, "rb"))); viol = [v for it in items for v in MT.check_isa(it["t"], bk)]; outside = sorted({q for it in items for q in it["phys"]} - set(REGION))
        est = MT.SHOTS * len(items) * MT.SEC_PER_SHOT; rem = svc.usage().get("usage_remaining_seconds", 0)
        print(f"obwodow {len(items)}, strzalow {MT.SHOTS * len(items)}, szac. QPU {est:.1f} s x 1.35 = {est * 1.35:.1f} s | pozostalo {rem} s | ISA {len(viol)} | kubity poza regionem B: {outside}")
        if os.path.exists(JOB) or viol or outside or rem < est * 1.35 or os.environ.get("NO_SEND"): sys.exit("nie wysylam (job istnieje / ISA / poza regionem / budzet / NO_SEND)")
        job = SamplerV2(mode=bk).run([it["t"] for it in items], shots=MT.SHOTS)
        json.dump(dict(job_id=job.job_id(), backend=MT.BK, shots=MT.SHOTS, mode="job", order=[(it["name"], it["tag"]) for it in items], layout=[it["layout"] for it in items], p0_ideal=[it["p0_ideal"] for it in items], esp_pred=[it["esp_pred"] for it in items],
                       twoq_t=[it["twoq_t"] for it in items], sent=time.strftime("%Y-%m-%d %H:%M:%S")), open(JOB, "w"), indent=1); print("wyslano", job.job_id(), flush=True)
    else:
        jd = json.load(open(JOB)); items = {(it["name"], it["tag"]): it for it in sent_items(pickle.load(open(PKL, "rb")))}; res = svc.job(jd["job_id"]).result(); rows = []
        for (nm, tag), pub in zip(jd["order"], res):
            it = items[(nm, tag)]; counts = pub.data[list(pub.data.keys())[0]].get_counts(); tot = sum(counts.values()); n = len(next(iter(counts))); qd = items[(nm, "qiskit")]["data"]            # the data qubits belong to the circuit, not to the variant
            p0 = counts.get("0" * n, 0) / tot; anc = [x for x in range(n) if x not in qd]; acc = sum(c for b, c in counts.items() if all(b[n - 1 - x] == "0" for x in anc)) / tot
            q = [i for i in items.values() if i["name"] == nm and i["tag"] == "qiskit"][0]; final = q.get("qct_final", "qiskit_fallback")
            rows.append(dict(name=nm, tag=tag, variant="qiskit" if tag == "qiskit" else "qct", source=("qiskit" if tag == "qiskit" else tag), shots=tot, p0=p0, sigma=float(np.sqrt(max(p0 * (1 - p0), 1e-12) / tot)), accept=acc, p_cond=(p0 / acc if acc else None), twoq=it["twoq_t"], layout=it["layout"], p0_ideal=it["p0_ideal"], n_ancilla=len(anc)))
        open(OUT, "w").write("".join(json.dumps(r) + "\n" for r in rows)); print(len(rows), "wierszy ->", OUT)

"""The selected examples on OTHER qubits of ibm_aachen (a control for the qubit-specific failure seen on qubits 152 and 153): the circuits are rebuilt (Qiskit circuit, QCPT candidates, final QCPT circuit) with a
placement inside a region disjoint from the first measurement, then sent as one plain job with a reduced number of shots (budget). Refuses if the new layout touches the old qubits, if the ISA check fails,
or if the budget is below the estimate with a 35 % margin. Usage: aachen_examples_alt.py NAME[,NAME...] SHOTS"""
import json, os, pickle, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
TAG = os.environ.get("ALT_TAG", "aachen_examples_alt"); FIX = os.path.join(HERE, "fixtures", TAG); os.makedirs(FIX, exist_ok=True)
OLD = {135, 139, 152, 153, 154, 155}; REGION = [90, 91, 98, 111, 112, 113, 114, 119]
sel = json.load(open(os.path.join(HERE, "fixtures", "aachen_rerun", "selection.json"))); sel["regions"] = {k: dict(sel["regions"][k], qubits=REGION) for k in ("6", "8")}
SEL = os.path.join(FIX, "selection_alt.json"); json.dump(sel, open(SEL, "w"))
os.environ.update(RUN_TAG=TAG, IBM_BACKEND="ibm_aachen", IBM_ACCOUNT="eu-de-aachen-156", SHOTS=sys.argv[2], SELECTION=SEL, NAMES=sys.argv[1], POST_SECONDS="40")
import mirror_test_nighthawk as MT
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
if __name__ == "__main__":
    svc = QiskitRuntimeService(name=MT.ACC); bk = svc.backend(MT.BK); items_all = MT.build(bk); pickle.dump(items_all, open(os.path.join(FIX, "items_alt.pkl"), "wb"))
    items = [it for it in items_all if it.get("send", True)]; touched = sorted({q for it in items for q in it["phys"]} & OLD)
    viol = [v for it in items for v in MT.check_isa(it["t"], bk)]; est = MT.SHOTS * len(items) * MT.SEC_PER_SHOT; rem = svc.usage().get("usage_remaining_seconds", 0)
    for it in items: print(it["name"], it["tag"], "uklad", it["layout"], "2q", it["twoq_t"], "ESP", round(it["esp_pred"], 3), "qct_final:", it.get("qct_final"))
    print(f"obwodow {len(items)}, strzalow {MT.SHOTS * len(items)}, szac. QPU {est:.1f} s x 1.35 = {est * 1.35:.1f} s | pozostalo {rem} s | ISA {len(viol)} | stare kubity uzyte: {touched}")
    out = os.path.join(FIX, "job.json")
    if os.environ.get("NO_SEND"): print("NO_SEND: zbudowano i zapisano items_alt.pkl, nic nie wyslano"); sys.exit(0)
    if os.path.exists(out) or touched or viol or rem < est * 1.35: sys.exit("nie wysylam (job istnieje / stare kubity / ISA / budzet)")
    job = SamplerV2(mode=bk).run([it["t"] for it in items], shots=MT.SHOTS)
    json.dump(dict(job_id=job.job_id(), backend=MT.BK, shots=MT.SHOTS, mode="job", order=[(it["name"], it["tag"]) for it in items], layout=[it["layout"] for it in items], p0_ideal=[it["p0_ideal"] for it in items],
                   esp_pred=[it["esp_pred"] for it in items], twoq_t=[it["twoq_t"] for it in items], sent=time.strftime("%Y-%m-%d %H:%M:%S")), open(out, "w"), indent=1); print("wyslano", job.job_id(), flush=True)

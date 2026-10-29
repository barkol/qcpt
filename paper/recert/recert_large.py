"""Re-certification of the large suite (12-24 q) after the QCEC errors B1/B2.
Circuits rebuilt with the v14 code (exactly as in the paper). For the variants ours / prog / greedy:
 (a) QCEC with run_zx_checker=False, elide_permutations=False, pinned qubits as ancillae,
 (b) state-vector simulation (Aer) on K random states with the pinned qubits = |0>.
Resumable: recert_large.jsonl."""
import json, os, sys, time
HERE_R = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE_R, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "experiments")); sys.path.insert(0, REPO)
from _paths import EXPERIMENTS_DIR  # noqa
import numpy as np
import scale_eval as SE, large_benchmarks as LB
from _clean import clean_ancillas
from toffoli_optimizer.core.phase_observability import is_phase_unobservable, default_affected_qubits
from qiskit import QuantumCircuit, AncillaRegister, QuantumRegister
from qiskit_aer import AerSimulator
import mqt.qcec as qcec
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "recert_large.jsonl")
SIM = AerSimulator(method="statevector")


def with_anc(c, pins):
    pins = list(pins); ps = set(pins); free = [q for q in range(c.num_qubits) if q not in ps]
    pos = {q: k for k, q in enumerate(free + pins)}
    regs = [QuantumRegister(len(free), "d")] + ([AncillaRegister(len(pins), "a")] if pins else [])
    o = QuantumCircuit(*regs)
    for ins in c.data:
        o.append(ins.operation, [o.qubits[pos[c.find_bit(q).index]] for q in ins.qubits])
    return o


def qcec_fixed(ex, ca, pins, timeout=600):
    t0 = time.time()
    r = qcec.verify(with_anc(ex, pins), with_anc(ca, pins), run_zx_checker=False, elide_permutations=False, timeout=timeout)
    eq = str(r.equivalence).split(".")[-1]
    return eq, round(time.time() - t0, 1)


def sv_equiv(ex, ca, pins, K=3, seed=7):
    n = ex.num_qubits; data = [q for q in range(n) if q not in set(pins)]; rng = np.random.default_rng(seed); ovs = []
    idx = np.zeros(2 ** len(data), dtype=np.int64)
    for k, q in enumerate(data): idx |= ((np.arange(2 ** len(data)) >> k) & 1) << q
    for _ in range(K):
        v = rng.normal(size=2 ** len(data)) + 1j * rng.normal(size=2 ** len(data)); v /= np.linalg.norm(v)
        full = np.zeros(2 ** n, dtype=complex); full[idx] = v
        outs = []
        for c in (ex, ca):
            q = QuantumCircuit(n); q.set_statevector(full); q.compose(c, inplace=True); q.save_statevector()
            outs.append(np.asarray(SIM.run(q.decompose(reps=0)).result().get_statevector()))
        ovs.append(np.vdot(outs[0], outs[1]))
    mags = [abs(o) for o in ovs]; ph = [np.angle(o) for o in ovs]
    spread = max(abs(np.angle(np.exp(1j * (p - ph[0])))) for p in ph)
    return bool(min(mags) > 1 - 1e-6 and spread < 1e-5), float(min(mags)), float(spread)


if __name__ == "__main__":
    done = {json.loads(l)["key"] for l in open(OUT)} if os.path.exists(OUT) else set()
    published = {json.loads(l)["name"]: json.loads(l) for l in open(os.path.join(REPO, "paper", "data", "rerun", "experiments", "scale_rows.jsonl"))}
    prog_pub = {json.loads(l)["name"]: json.loads(l) for l in open(os.path.join(REPO, "paper", "data", "rerun", "experiments", "scale_program_rows.jsonl"))}
    for qc in LB.large_suite():
        pins = clean_ancillas(qc); sel = SE.ScalableErrorBudgetSelector(); sres = sel.select(qc, pinned_zero=pins)
        rel, mir = set(sres["rel_idx"]), set(sres["mirror_idx"])
        ccx = [i for i, ins in enumerate(qc.data) if ins.operation.name.lower() in ("ccx", "mcx", "mcx_gray") and len(ins.qubits) == 3]
        U = {i for i in ccx if i not in rel and is_phase_unobservable(qc, i, default_affected_qubits(qc, i))}
        variants = {"ours": sres["circuit"], "prog": sel._build(qc, rel | U, mir), "greedy": sel._build(qc, set(ccx))}
        for tag, c in variants.items():
            key = f"{qc.name}|{tag}"
            if key in done: continue
            row = {"key": key, "name": qc.name, "n": qc.num_qubits, "tag": tag, "twoq": SE.EM.two_qubit_count(c)}
            if tag == "ours": row["twoq_published"] = published[qc.name]["twoq_ours"]
            else: row["sub_ok_published"] = prog_pub[qc.name][f"sub_ok_{tag}"]
            row["qcec"], row["qcec_s"] = qcec_fixed(sres["exact"], c, pins)
            t0 = time.time(); row["sv_ok"], row["sv_min_overlap"], row["sv_phase_spread"] = sv_equiv(sres["exact"], c, pins); row["sv_s"] = round(time.time() - t0, 1)
            with open(OUT, "a") as fh: fh.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
    print("KONIEC")

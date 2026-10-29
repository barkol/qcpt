"""Ostatni obwod (array_mult_6b, n=24): QCEC nie rozstrzygnal w ~16 min, przerwany.
Zapis metryk bez certyfikatu (sub_ok=None, 'qcec_timeout')."""
import json, os
from _paths import EXPERIMENTS_DIR  # noqa
import scale_eval as SE, large_benchmarks as LB
from toffoli_optimizer.core.context_analysis import find_relative_phase_safe_sites
from toffoli_optimizer.core.phase_observability import is_phase_unobservable, default_affected_qubits
CK = os.path.join(str(SE.OUT_DIR), "scale_program_rows.jsonl")  # TOPTOFFOLI_OUT redirects
done = {json.loads(l)["name"] for l in open(CK)}
for qc in LB.large_suite():
    if qc.name in done: continue
    sel = SE.ScalableErrorBudgetSelector(); sres = sel.select(qc)
    ccx = [i for i, ins in enumerate(qc.data) if ins.operation.name.lower() in ("ccx","mcx","mcx_gray") and len(ins.qubits)==3]
    assert sres["sites_applied"] == 0, "obwod ma pary; ten skrypt zaklada ich brak"
    U = {i for i in ccx if is_phase_unobservable(qc, i, default_affected_qubits(qc, i))}
    prog, greedy = sel._build(qc, U), sel._build(qc, set(ccx)); EM = SE.EM
    row = {"name": qc.name, "n": qc.num_qubits, "family": qc.family, "twoq_exact": EM.two_qubit_count(sres["exact"]),
           "infid_exact": EM.circuit_infidelity(sres["exact"]), "twoq_sub": sres["two_qubit_after"], "infid_sub": sres["infid_after"],
           "pairs": 0, "U_admitted": len(U), "ccx": len(ccx)}
    for tag, c in (("prog", prog), ("greedy", greedy)):
        row.update({f"twoq_{tag}": EM.two_qubit_count(c), f"infid_{tag}": EM.circuit_infidelity(c),
                    f"sub_ok_{tag}": None, f"sub_method_{tag}": "qcec_timeout", f"sub_t_{tag}": None, f"prog_ok_{tag}": None})
    open(CK, "a").write(json.dumps(row) + "\n"); print(row)

"""Resetting-protocol (synchronizing-word) circuits past the exhaustive limit,
certified by the decision-diagram (QCEC) backend.

Reuses scale_eval.ScalableErrorBudgetSelector (QCEC-gated: every accepted
compute/uncompute-pair substitution is certified unitary-equivalent to the
all-exact circuit by decision diagrams) on the resetting circuits of
sync_benchmark.py. This is the scalable path for the larger words whose
exhaustive verification is infeasible.

Run:
    pip install -e ".[experiments]"
    python experiments/sync_scale.py
"""
from __future__ import annotations

import sys

from _paths import EXPERIMENTS_DIR  # noqa: F401  (must be first)

from scale_eval import ScalableErrorBudgetSelector, certify   # noqa: E402
from sync_benchmark import resetting_circuit                  # noqa: E402


def main():
    sel = ScalableErrorBudgetSelector()
    print(f"{'circuit':18s} {'q':>2} {'ccx':>3} {'2q_be':>5} {'2q_af':>5} "
          f"{'red%':>5} {'infid_red%':>10} {'cert':>5} {'method':>10} {'t(s)':>7}")
    for m, L in [(3, 3), (3, 5), (3, 7), (4, 7)]:
        qc = resetting_circuit(m, L)
        nccx = sum(1 for inst in qc.data if inst.operation.name == "ccx")
        from _clean import clean_ancillas
        pins = clean_ancillas(qc)
        r = sel.select(qc, pinned_zero=pins)
        cert = certify(r["exact"], r["circuit"], pinned_zero=pins if r.get("r_gadgets") else ())
        if cert.equivalent is not True:   # fail closed
            r = dict(r, two_qubit_after=r["two_qubit_before"], infid_after=r["infid_before"])
        b, a = r["two_qubit_before"], r["two_qubit_after"]
        ib, ia = r["infid_before"], r["infid_after"]
        print(f"{qc.name:18s} {qc.num_qubits:2d} {nccx:3d} {b:5d} {a:5d} "
              f"{100*(b-a)/b:5.1f} {100*(ib-ia)/ib:10.1f} "
              f"{str(cert.equivalent):>5} {cert.method:>10} {cert.wall_time_s:7.3f}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()

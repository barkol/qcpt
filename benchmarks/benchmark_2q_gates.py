"""Benchmark the REAL objective: final 2-qubit (CX) gate count.

The pipeline's point is to lower the number of two-qubit gates (the error-budget
driver: each 2q gate carries error) by optimizing at the Toffoli level BEFORE
decomposing to a 2-qubit basis. Each Toffoli removed/shared upstream saves ~6 CX
downstream, and Toffoli-level structure (expand-to-cancel, fan-out CSE,
permutation) is largely invisible once decomposed.

Compared, both decomposed to a 2-qubit basis at optimization_level=3 and exactly
verified:
  qiskit_direct : transpile(input) -> count CX
  ours          : ToffoliCountReducer(input) -> transpile -> count CX
"""

import random

from qiskit import QuantumCircuit, transpile

from toffoli_optimizer.core.toffoli_count_reducer import ToffoliCountReducer, count_toffoli
from toffoli_optimizer.core.equivalence_verifier import ExactEquivalenceVerifier

V = ExactEquivalenceVerifier()
TWO_Q_BASIS = ["cx", "u"]


def cx_count(circ):
    c = transpile(circ, basis_gates=TWO_Q_BASIS, optimization_level=3)
    return c.count_ops().get("cx", 0), c


def verify(inp, out):
    try:
        # decomposed circuits are general unitaries; allow the layout/output perm
        ok, _, _ = V.verify(inp, out, allow_permutation=True)
        return ok
    except Exception:
        return None


def random_ccx_network(n, m, rng):
    qc = QuantumCircuit(n)
    for _ in range(m):
        a, b, t = rng.sample(range(n), 3)
        qc.ccx(a, b, t)
    return qc


def structured():
    d = {}
    c = QuantumCircuit(4); c.ccx(0, 1, 2); c.ccx(0, 1, 2); c.ccx(0, 1, 3); d["inverse_pair"] = c
    c = QuantumCircuit(4); c.ccx(0, 1, 2); c.cx(2, 3); c.ccx(0, 1, 2); d["expand_to_cancel"] = c
    c = QuantumCircuit(6)
    c.ccx(0, 1, 2); c.ccx(0, 1, 3); c.ccx(0, 1, 4); c.ccx(0, 1, 5); d["fanout_run"] = c
    c = QuantumCircuit(4); c.ccx(0, 1, 2); c.cx(2, 3); c.swap(2, 3); c.ccx(0, 1, 2); d["perm_reducible"] = c
    return d


def run():
    rng = random.Random(7)
    rows = []
    print("=" * 74)
    print(f"{'case':20} {'CCX':>4} {'qiskit CX':>10} {'ours CX':>9} {'saved':>6}  ok")
    print("-" * 74)

    def do(name, circ):
        base_cx, base_full = cx_count(circ)
        red = ToffoliCountReducer(allow_permutation=True).reduce_toffoli_count(circ)
        ours_cx, ours_full = cx_count(red)
        ok_b = verify(circ, base_full)
        ok_o = verify(circ, ours_full)
        ok = (ok_b is not False) and (ok_o is not False)
        saved = base_cx - ours_cx
        tag = "" if ok else " !!CORRECTNESS"
        print(f"{name:20} {count_toffoli(circ):>4} {base_cx:>10} {ours_cx:>9} {saved:>6}  {'OK' if ok else 'FAIL'}{tag}")
        rows.append((name, base_cx, ours_cx, ok))

    for name, c in structured().items():
        do(name, c)
    for n, m in [(5, 12), (5, 20), (6, 16), (6, 24), (7, 20), (7, 30)]:
        do(f"rand_n{n}_m{m}", random_ccx_network(n, m, rng))

    print("-" * 74)
    tb = sum(r[1] for r in rows); to = sum(r[2] for r in rows)
    wins = sum(1 for r in rows if r[2] < r[1]); ties = sum(1 for r in rows if r[2] == r[1])
    loses = sum(1 for r in rows if r[2] > r[1])
    bad = [r[0] for r in rows if r[3] is False]
    print(f"TOTAL 2q (CX): qiskit_direct={tb}  ours={to}  -> {tb - to} fewer 2q gates "
          f"({100*(tb-to)/max(tb,1):.1f}%) i.e. that much error budget saved")
    print(f"per-case: ours fewer CX in {wins}, ties {ties}, more {loses}")
    print(f"correctness: {'ALL VERIFIED OK' if not bad else 'FAIL: ' + ','.join(bad)}")
    print("\nBoth decomposed to a 2-qubit basis (cx,u) at opt-level 3 and exactly "
          "verified (up to tracked layout permutation). Fewer CX = lower two-qubit "
          "error budget -- the pipeline's actual objective.")


if __name__ == "__main__":
    run()

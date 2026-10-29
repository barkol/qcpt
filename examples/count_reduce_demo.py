"""Demo / self-test for Feature B: Toffoli-COUNT reduction (atomic CCX/MCX).

Builds a CCX network that contains
  (a) a cancelling inverse pair of identical CCX gates, separated by gates on
      disjoint qubits (so they must be commuted together to cancel), and
  (b) a shared-control run: several CCX with the same control pair, distinct
      targets -- a compute-once fan-out opportunity.

Then runs `ToffoliCountReducer.reduce_toffoli_count` and asserts:
  * the result is EXACTLY equivalent to the input (ExactEquivalenceVerifier), and
  * the CCX count strictly dropped (cancelling + fan-out both reduce it).
"""

from qiskit import QuantumCircuit

from toffoli_optimizer.core.optimizer import (
    ToffoliCountReducer,
    ExactEquivalenceVerifier,
)
from toffoli_optimizer.core.toffoli_count_reducer import count_toffoli


def build_cancelling_circuit() -> QuantumCircuit:
    """An inverse pair of CCX(0,1->2) that cancel, with a disjoint CX(3,4) and a
    CCX(0,1->5) between them (the latter shares controls but a different target,
    so it does not block the cancellation of the (0,1->2) pair via commutation)."""
    qc = QuantumCircuit(6)
    qc.ccx(0, 1, 2)      # <-- cancels with the one below
    qc.cx(3, 4)          # disjoint, commutes
    qc.ccx(0, 1, 2)      # <-- cancels with the one above
    return qc


def build_fanout_circuit() -> QuantumCircuit:
    """A shared-control run: CCX(0,1->t) for several distinct targets t. Plus a
    leading inverse pair on different controls so BOTH rewrites fire."""
    qc = QuantumCircuit(8)
    # inverse pair on controls (2,3) -> target 4, separated by a disjoint gate
    qc.ccx(2, 3, 4)
    qc.cx(0, 1)
    qc.ccx(2, 3, 4)
    # shared-control fan-out: a&b into targets 5,6,7 (qubit 0/1 ancilla-free wires
    # exist for compute-once). 3 CCX share control pair {2,3}.
    qc.ccx(2, 3, 5)
    qc.ccx(2, 3, 6)
    qc.ccx(2, 3, 7)
    return qc


def _check(circuit: QuantumCircuit, label: str, expect_strict_drop: bool):
    verifier = ExactEquivalenceVerifier()
    reducer = ToffoliCountReducer(verifier=verifier)

    before = count_toffoli(circuit)
    out = reducer.reduce_toffoli_count(circuit)
    after = count_toffoli(out)

    ok, perm, info = verifier.verify(circuit, out)
    print(f"[{label}] CCX before={before} after={after} reduced={before-after}")
    print(f"[{label}] rewrites: {reducer.report['rewrites_applied']}")
    print(f"[{label}] shareable runs: {reducer.report['shareable_equal_control_runs']}")
    print(f"[{label}] equivalent={ok} info={info}")

    assert ok, f"[{label}] reduced circuit is NOT equivalent to the input! {info}"
    assert reducer.report["all_rewrites_verified"], \
        f"[{label}] some rewrite was not verified"
    if expect_strict_drop:
        assert after < before, f"[{label}] expected CCX count to strictly drop"
    print(f"[{label}] OK\n")
    return before, after


def main():
    print("=== Feature B: Toffoli-COUNT reduction demo ===\n")

    # (a) cancelling inverse pair -> count must strictly drop
    _check(build_cancelling_circuit(), "cancelling-pair", expect_strict_drop=True)

    # (b) combined: cancelling pair + shared-control fan-out -> strictly drop
    _check(build_fanout_circuit(), "fanout+cancel", expect_strict_drop=True)

    print("All assertions passed: every applied rewrite verified EQUIVALENT and "
          "the cancelling case strictly reduced the CCX count.")


if __name__ == "__main__":
    main()

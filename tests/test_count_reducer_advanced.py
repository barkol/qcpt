"""Advanced self-tests for the two superoptimizer capabilities added to
`ToffoliCountReducer`:

  FEATURE 1 -- PERMUTATION-AWARE REWRITING
    Output-wire relabeling is free for a downstream consumer, so a count-reducing
    rewrite that verifies only up to a NON-identity output-wire permutation is
    usable PROVIDED the permutation is tracked: it must be propagated to every
    downstream gate and composed into a cumulative output-wire permutation. The
    reduced circuit, read through the reported permutation, must compute the SAME
    function as the input.

  FEATURE 2 -- EXPAND-TO-CANCEL PEEPHOLE
    A bounded-window pass that temporarily EXPANDS a block into an equivalent form
    so a downstream gate cancels, lowering the total atomic Toffoli count. It must
    catch the canonical CCX(a,b,t); CX(t,u); CCX(a,b,t) -> CX(t,u); CCX(a,b,u)
    reduction (2 CCX -> 1) that the strict annihilation/commute passes miss.

Run with:  python test_count_reducer_advanced.py
"""

from qiskit import QuantumCircuit

from toffoli_optimizer.core.optimizer import (
    ToffoliCountReducer,
    ExactEquivalenceVerifier,
)
from toffoli_optimizer.core.toffoli_count_reducer import count_toffoli


def _identity_perm(n):
    return {i: i for i in range(n)}


# ---------------------------------------------------------------------------
# FEATURE 2: expand-to-cancel  (CCX;CX(t,u);CCX -> CX;CCX(a,b,u), 2 CCX -> 1)
# ---------------------------------------------------------------------------
def test_expand_to_cancel():
    print("=== FEATURE 2: expand-to-cancel peephole ===")
    verifier = ExactEquivalenceVerifier()

    # The canonical conjugation identity. A strict adjacency / commute / inverse-
    # pair walk leaves this at 2 CCX (the two CCX are NOT an inverse pair -- the
    # intervening CX(t,u) means the second CCX is not the inverse of the first in
    # the literal sense the annihilation pass requires).
    qc = QuantumCircuit(4)
    qc.ccx(0, 1, 2)
    qc.cx(2, 3)
    qc.ccx(0, 1, 2)
    before = count_toffoli(qc)

    # OLD passes alone (expand-to-cancel disabled) must NOT reduce it.
    old = ToffoliCountReducer(verifier=verifier, enable_expand_to_cancel=False)
    old_out = old.reduce_toffoli_count(qc)
    print(f"  old passes only:   {before} -> {count_toffoli(old_out)}  "
          f"applied={old.report['rewrites_applied']}")
    assert count_toffoli(old_out) == before, \
        "old passes unexpectedly reduced the expand-to-cancel case"

    # NEW expand-to-cancel pass reduces 2 CCX -> 1 and stays equivalent.
    new = ToffoliCountReducer(verifier=verifier)  # expand-to-cancel on by default
    out = new.reduce_toffoli_count(qc)
    after = count_toffoli(out)
    print(f"  with expand pass:  {before} -> {after}  "
          f"applied={new.report['rewrites_applied']}")
    assert after == 1, f"expected 2 CCX -> 1, got {after}"
    assert before - after == 1

    # The reduced circuit is the expected CX(2,3); CCX(0,1,3).
    names = [(i.operation.name, tuple(out.find_bit(b).index for b in i.qubits))
             for i in out.data]
    assert ("ccx", (0, 1, 3)) in names, f"expected CCX(0,1,3), got {names}"

    ok, perm, info = verifier.verify(qc, out)
    print(f"  verify equivalent={ok}  perm={perm}  info={info}")
    assert ok, f"expand-to-cancel produced a non-equivalent circuit: {info}"
    assert perm == _identity_perm(4), "expand-to-cancel needed no permutation here"
    assert new.report["output_permutation"] == _identity_perm(4)
    assert new.report["all_rewrites_verified"]
    print("  OK\n")


# ---------------------------------------------------------------------------
# FEATURE 1: permutation-aware rewriting (rewrite valid only up to a wire perm)
# ---------------------------------------------------------------------------
def test_permutation_aware_rewrite():
    print("=== FEATURE 1: permutation-aware rewriting ===")
    verifier = ExactEquivalenceVerifier()

    # CCX(0,1,2); CX(2,3); SWAP(2,3); CCX(0,1,2).
    # The SWAP sitting between the two conjugating CCX means the count-reducing
    # rewrite (2 CCX -> 1 CCX, dropping the SWAP) is valid ONLY up to a non-identity
    # output-wire permutation that exchanges wires 2 and 3. A strict (swap-as-gate)
    # walk in this reducer cannot express it, so strict leaves it untouched.
    qc = QuantumCircuit(4)
    qc.ccx(0, 1, 2)
    qc.cx(2, 3)
    qc.swap(2, 3)
    qc.ccx(0, 1, 2)
    before = count_toffoli(qc)

    # STRICT (allow_permutation=False): must NOT apply the perm-only rewrite.
    strict = ToffoliCountReducer(verifier=verifier)
    strict_out = strict.reduce_toffoli_count(qc, allow_permutation=False)
    print(f"  strict:  {before} -> {count_toffoli(strict_out)}  "
          f"applied={strict.report['rewrites_applied']}  "
          f"perm={strict.report['output_permutation']}")
    assert count_toffoli(strict_out) == before, \
        "strict mode unexpectedly applied a permutation-only rewrite"
    assert strict.report["output_permutation"] == _identity_perm(4)
    # strict result must still be exactly (identity-perm) equivalent
    ok_s, perm_s, _ = verifier.verify(qc, strict_out)
    assert ok_s and perm_s == _identity_perm(4)

    # PERMUTATION-AWARE (allow_permutation=True): applies it, lowers CCX count.
    perm_reducer = ToffoliCountReducer(verifier=verifier)
    out = perm_reducer.reduce_toffoli_count(qc, allow_permutation=True)
    after = count_toffoli(out)
    reported_perm = perm_reducer.report["output_permutation"]
    print(f"  perm:    {before} -> {after}  "
          f"applied={perm_reducer.report['rewrites_applied']}  "
          f"output_permutation={reported_perm}")

    assert after < before, f"permutation mode failed to lower CCX count ({before}->{after})"
    assert after == 1
    # The reported cumulative permutation must be NON-identity.
    assert reported_perm != _identity_perm(4), \
        "expected a non-identity output permutation"
    assert reported_perm == {0: 0, 1: 1, 2: 3, 3: 2}, \
        f"unexpected permutation {reported_perm}"

    # CORRECTNESS: the reduced circuit, read through the reported permutation,
    # computes the SAME function as the input -- and the verifier's permutation
    # MUST equal the reported cumulative permutation.
    ok, vperm, info = verifier.verify(qc, out, allow_permutation=True)
    print(f"  verify (allow_permutation=True): ok={ok}  perm={vperm}  info={info}")
    assert ok, f"permutation-aware result is NOT equivalent up to a wire perm: {info}"
    assert vperm == reported_perm, \
        f"verifier perm {vperm} != reported {reported_perm}"
    # And WITHOUT permutation it is NOT equivalent (the relabeling is essential).
    ok_strict, _, _ = verifier.verify(qc, out)
    assert not ok_strict, \
        "the permutation-rewritten circuit must NOT be strictly equivalent"
    assert perm_reducer.report["all_rewrites_verified"]
    print("  OK\n")


def test_permutation_propagates_to_downstream_gates():
    print("=== FEATURE 1b: cumulative perm propagates to downstream gates ===")
    verifier = ExactEquivalenceVerifier()

    # Same perm-triggering window, but now with DOWNSTREAM gates after it. The
    # tracked permutation must be applied to every one of those gates (wire 2<->3)
    # so the whole circuit stays correct read through the cumulative permutation.
    qc = QuantumCircuit(4)
    qc.ccx(0, 1, 2)
    qc.cx(2, 3)
    qc.swap(2, 3)
    qc.ccx(0, 1, 2)
    qc.x(2)         # downstream: must become x(3)
    qc.cx(3, 2)     # downstream: must become cx(2,3)
    before = count_toffoli(qc)

    reducer = ToffoliCountReducer(verifier=verifier)
    out = reducer.reduce_toffoli_count(qc, allow_permutation=True)
    reported_perm = reducer.report["output_permutation"]
    print(f"  {before} -> {count_toffoli(out)}  "
          f"applied={reducer.report['rewrites_applied']}  perm={reported_perm}")

    assert count_toffoli(out) < before
    assert reported_perm != _identity_perm(4)

    ok, vperm, info = verifier.verify(qc, out, allow_permutation=True)
    print(f"  verify ok={ok} perm={vperm}")
    assert ok, f"downstream propagation broke equivalence: {info}"
    assert vperm == reported_perm

    # Confirm the downstream gates were relabeled (x moved 2->3, cx(3,2)->cx(2,3)).
    names = [(i.operation.name, tuple(out.find_bit(b).index for b in i.qubits))
             for i in out.data]
    assert ("x", (3,)) in names, f"downstream x(2) was not relabeled to x(3): {names}"
    print("  OK\n")


def test_existing_demo_still_passes():
    print("=== Regression: existing count_reduce_demo asserts ===")
    import importlib.util
    from pathlib import Path
    demo_path = Path(__file__).resolve().parent.parent / "examples" / "count_reduce_demo.py"
    spec = importlib.util.spec_from_file_location("count_reduce_demo", demo_path)
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    demo._check(demo.build_cancelling_circuit(), "cancelling-pair",
                expect_strict_drop=True)
    demo._check(demo.build_fanout_circuit(), "fanout+cancel",
                expect_strict_drop=True)
    print("  OK (existing passes unchanged)\n")


def main():
    test_expand_to_cancel()
    test_permutation_aware_rewrite()
    test_permutation_propagates_to_downstream_gates()
    test_existing_demo_still_passes()
    print("ALL ADVANCED TESTS PASSED.")


if __name__ == "__main__":
    main()

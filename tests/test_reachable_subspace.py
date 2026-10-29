#!/usr/bin/env python3
"""Tests for reachable-subspace + phase-observability-aware admissibility.

This is the soundness suite for the NEW admissibility path: a standalone Toffoli
(not in a compute/uncompute pair) may get the cheap relative-phase (Margolus) gadget
when its relative phase is provably UNOBSERVABLE on the states that reach it. Every
committed substitution must produce identical computational-basis MEASUREMENT
STATISTICS on every reachable input.

Run with:  python test_reachable_subspace.py

Covers:
  A. PHASE PROVABLY UNOBSERVABLE, NOT A PAIR. A Toffoli whose forward cone is a pure
     classical-reversible cone into a computational-basis read. The NEW path admits
     the relative-phase substitution; the OLD pair-only path does NOT; the output is
     sound (same measurement statistics on ALL reachable inputs); 2q-count drops.

  B. SOUNDNESS GUARD: downstream Hadamard makes the phase observable. is_phase_
     unobservable returns False -> the substitution is REJECTED -> output stays exact
     and sound. We also verify the rejected substitution WOULD have been observably
     wrong (different measurement statistics), proving the guard is load-bearing.

  C. Regression: the existing compute/uncompute-pair admissions still work and every
     output is verified sound (delegated to the existing suite + a spot check here).

  D. Unit tests for reachable_basis_states / reachable_overapprox soundness and for
     is_phase_unobservable's classical-cone vs interference distinction.
"""

import sys

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from toffoli_optimizer.core.optimizer import (
    ErrorBudgetSelector,
    HardwareErrorModel,
    ExactEquivalenceVerifier,
)
from toffoli_optimizer.core.context_analysis import find_relative_phase_safe_sites
from toffoli_optimizer.core.reachable_subspace import (
    reachable_basis_states,
    reachable_overapprox,
)
from toffoli_optimizer.core.phase_observability import (
    is_phase_unobservable,
    default_affected_qubits,
)


# --------------------------------------------------------------------------- helpers
def _probdist(circuit, x):
    """Computational-basis probability distribution of `circuit` on basis input x."""
    n = circuit.num_qubits
    sv = Statevector.from_int(x, dims=2 ** n).evolve(circuit)
    return np.abs(sv.data) ** 2


def _same_measurement_stats(c1, c2, inputs):
    """True iff c1 and c2 give the same basis-measurement distribution on every input
    in `inputs` (both circuits must have the same qubit count)."""
    for x in inputs:
        if not np.allclose(_probdist(c1, x), _probdist(c2, x), atol=1e-9):
            return False, x
    return True, None


# ===================================================================== Scenario A ===
def _classical_cone_circuit():
    """A standalone Toffoli feeding ONLY classical-reversible ops into a basis read.

        CCX(0,1,2)        # standalone -- there is NO matching uncompute
        CX(2,3)           # target used as a control -> classical-reversible
        X(3)              # classical-reversible
        (implicit computational-basis read at the end)

    The Toffoli's relative phase rides {0,1,2}; downstream those qubits only ever feed
    cx / x and a basis read, so the phase is unobservable -> NEW path admissible.
    Crucially this is NOT a compute/uncompute pair (no second CCX), so the OLD path
    finds no site.
    """
    qc = QuantumCircuit(4)
    qc.ccx(0, 1, 2)
    qc.cx(2, 3)
    qc.x(3)
    return qc


def test_A_unobservable_standalone_admitted():
    qc = _classical_cone_circuit()

    # OLD (pair-only) path: no compute/uncompute site exists.
    assert find_relative_phase_safe_sites(qc) == [], (
        "this circuit must have NO compute/uncompute pair (it is standalone)"
    )
    old = ErrorBudgetSelector(phase_aware=False).select(qc)["report"]
    assert old["sites_applied"] == 0 and not old.get("phase_aware_admitted"), (
        "pair-only path must apply NO relative-phase gadget here"
    )
    assert old["two_qubit_after"] == old["two_qubit_before"], (
        "pair-only path cannot reduce the 2q count of a standalone Toffoli"
    )

    # is_phase_unobservable proves the phase cannot be measured.
    affected = default_affected_qubits(qc, 0)
    assert is_phase_unobservable(qc, 0, affected), (
        "classical cone into a basis read -> phase must be provably unobservable"
    )

    # NEW (phase-aware) path: admits the standalone substitution.
    sel = ErrorBudgetSelector(phase_aware=True, semantics="program")
    out = sel.select(qc)
    rep = out["report"]
    print("[A] " + rep.summary())

    assert len(rep["phase_aware_admitted"]) == 1, (
        f"NEW path must admit exactly the standalone Toffoli: {rep['phase_aware_admitted']}"
    )
    assert rep["verified"], "phase-aware output must be certified sound"
    assert rep["two_qubit_after"] < rep["two_qubit_before"], (
        f"2q count must drop: {rep['two_qubit_before']} -> {rep['two_qubit_after']}"
    )
    # exact CCX -> 6 CX; Margolus -> 3 CX; plus the trailing CX(2,3). 7 -> 4.
    assert rep["two_qubit_before"] == 7 and rep["two_qubit_after"] == 4, (
        (rep["two_qubit_before"], rep["two_qubit_after"])
    )

    # SOUNDNESS: identical measurement statistics on EVERY reachable input.
    exact = sel.decompose_exact_only(qc)
    reachable = reachable_overapprox(qc, len(qc.data), input_space="all_basis")
    ok, witness = _same_measurement_stats(exact, out["circuit"], reachable)
    assert ok, f"measurement statistics differ on reachable input {witness}"

    # Independent phase-insensitive re-certification.
    v = ExactEquivalenceVerifier()
    cert, _p, info = v.verify_on_reachable_basis(exact, out["circuit"], reachable)
    assert cert, f"independent reachable-basis certification failed: {info}"

    print("[A] PASS: standalone unobservable-phase Toffoli admitted by NEW path "
          "(OLD admits 0), 7->4 two-qubit gates, measurement-statistics-sound")


# ===================================================================== Scenario B ===
def _observable_phase_circuit():
    """A Toffoli whose relative phase IS observable, via an interferometer around it.

        H(0); H(1)        # put the controls in SUPERPOSITION (so the |11> branch
                          #   coherently coexists with |00>,|01>,|10>)
        CCX(0,1,2)        # the relative phase now rides a live |11> branch
        H(0)              # recombine -> the relative phase becomes a population diff

    The downstream H(0) acts on a phase-carrying control, so is_phase_unobservable
    must flag it observable. (The upstream H's are what give the phase something to
    interfere with; without control superposition the single-gate phase is, in fact,
    unobservable -- which is exactly why scenario A's substitution was sound.)
    """
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.h(1)
    qc.ccx(0, 1, 2)   # index 2
    qc.h(0)           # downstream interference on a phase-carrying qubit
    return qc


def test_B_observable_phase_rejected_soundness_guard():
    qc = _observable_phase_circuit()
    g = 2  # index of the CCX

    # The guard: is_phase_unobservable must return False (downstream H interferes).
    affected = default_affected_qubits(qc, g)
    assert not is_phase_unobservable(qc, g, affected), (
        "downstream Hadamard on a phase-carrying qubit must be flagged observable"
    )

    sel = ErrorBudgetSelector(phase_aware=True, semantics="program")
    out = sel.select(qc)
    rep = out["report"]
    print("[B] " + rep.summary())

    # No phase-aware substitution admitted -> the CCX keeps its EXACT decomposition.
    assert not rep.get("phase_aware_admitted"), (
        "observable-phase Toffoli must NOT be substituted"
    )
    assert rep["verified"], "rejected -> exact output must verify exactly"

    # Output is exactly equivalent to the original circuit (exact-unitary).
    v = ExactEquivalenceVerifier()
    ok, _p, info = v.verify(qc, out["circuit"], allow_permutation=False)
    assert ok, f"rejected output not exactly equivalent: {info}"

    # The guard is LOAD-BEARING: had we (wrongly) substituted the gadget at the CCX,
    # the output would be OBSERVABLY different. Build that bad circuit (Margolus at
    # the CCX, exact elsewhere) and show its measurement statistics differ.
    bad = sel._build(qc, {g})         # force the relative-phase gadget at the CCX
    exact = sel.decompose_exact_only(qc)
    same, witness = _same_measurement_stats(exact, bad, range(2 ** qc.num_qubits))
    assert not same, (
        "soundness guard would be vacuous: the rejected substitution must be "
        "observationally DIFFERENT (the recombining H makes the phase observable)"
    )
    print(f"[B] forced bad substitution differs from exact on input {witness} "
          f"(phase made observable by the interferometer)")
    print("[B] PASS: observable-phase Toffoli REJECTED, output exact & sound; "
          "the forced substitution IS observably wrong (guard is load-bearing)")


# ===================================================================== Scenario C ===
def test_C_regression_pairs_still_work():
    """Existing compute/uncompute-pair admissions still fire and verify sound."""
    qc = QuantumCircuit(4)
    qc.ccx(0, 1, 2)
    qc.cx(2, 3)        # use target only as a control
    qc.ccx(0, 1, 2)    # uncompute

    sites = find_relative_phase_safe_sites(qc)
    assert len(sites) == 1, f"the compute/uncompute pair must still be detected: {sites}"

    sel = ErrorBudgetSelector(phase_aware=True, semantics="program")
    out = sel.select(qc)
    rep = out["report"]
    print("[C] " + rep.summary())

    # The PAIR path applies (exact-unitary certified). 13 -> 7 two-qubit gates.
    assert rep["sites_applied"] == 1, "compute/uncompute pair must still be applied"
    assert rep["verified"]
    assert rep["two_qubit_before"] == 13 and rep["two_qubit_after"] == 7

    # Pair admissions are exact-unitary equivalent -> the STRICT verifier accepts.
    exact = sel.decompose_exact_only(qc)
    v = ExactEquivalenceVerifier()
    ok, _p, info = v.verify(exact, out["circuit"], allow_permutation=False)
    assert ok, f"pair output must be exact-unitary equivalent: {info}"
    print("[C] PASS: compute/uncompute-pair admission still works, exact-unitary sound")


# ===================================================================== Scenario D ===
def test_D_reachable_basis_states_exact():
    """Exact reachable-set enumeration for a classical-reversible prefix."""
    # X(0); CX(0,1); <gate at index 2>. From all 4 basis inputs of a 2-qubit circuit,
    # the prefix is a permutation, so the reachable set at index 2 is all 4 states.
    qc = QuantumCircuit(3)
    qc.x(0)
    qc.cx(0, 1)
    qc.ccx(0, 1, 2)
    r = reachable_basis_states(qc, gate_index=2, input_space="all_basis")
    assert r == frozenset(range(8)), r  # bijection -> full basis preserved

    # Restricted input space: only input 0 (all qubits |0>). X(0) -> bit0=1; CX(0,1)
    # -> bit1=1. So reachable at index 2 is {0b011} = {3}.
    r2 = reachable_basis_states(qc, gate_index=2, input_space=[0])
    assert r2 == frozenset({3}), r2
    print("[D1] PASS: exact reachable_basis_states for classical prefix")


def test_D_overapprox_is_superset_and_sound():
    """reachable_overapprox returns a sound SUPERSET of the true reachable set, even
    with a superposition-creating prefix gate."""
    # H(0) taints qubit 0; the rest are clean & fixed by the restricted input.
    qc = QuantumCircuit(3)
    qc.h(0)            # taints qubit 0 (superposition)
    qc.cx(0, 1)        # entangles 1 with tainted 0 -> taints 1
    qc.ccx(0, 1, 2)    # index 2

    # With input |000>: clean qubit 2 stays 0; qubits 0,1 are tainted (free).
    over = reachable_overapprox(qc, gate_index=2, input_space=[0])
    # Tainted {0,1} free, qubit 2 fixed 0 -> {000,001,010,011} = {0,1,2,3}.
    assert over == frozenset({0, 1, 2, 3}), over
    # It is a superset of the TRUE reachable basis support. The true support after
    # H(0),CX(0,1) on |000> is {|000>, |011>} = {0,3}; check superset.
    pre = QuantumCircuit(3); pre.h(0); pre.cx(0, 1)
    support = {i for i, a in enumerate(Statevector.from_int(0, dims=8).evolve(pre).data)
               if abs(a) > 1e-9}
    assert support <= over, f"over-approx {over} not a superset of true support {support}"
    print("[D2] PASS: reachable_overapprox is a sound superset (H prefix)")


def test_D_phase_observability_distinguishes_cone_vs_interference():
    # Classical cone -> unobservable.
    cone = QuantumCircuit(4)
    cone.ccx(0, 1, 2)
    cone.cx(2, 3)
    cone.swap(2, 3)
    assert is_phase_unobservable(cone, 0, default_affected_qubits(cone, 0))

    # Downstream controlled-phase / cz on a tainted qubit -> observable.
    interf = QuantumCircuit(3)
    interf.ccx(0, 1, 2)
    interf.cz(0, 1)   # non-classical, acts on phase-carrying controls -> observable
    assert not is_phase_unobservable(interf, 0, default_affected_qubits(interf, 0))

    # A non-classical gate that does NOT touch the phase qubits is harmless.
    harmless = QuantumCircuit(4)
    harmless.ccx(0, 1, 2)
    harmless.h(3)     # qubit 3 never carried the phase
    assert is_phase_unobservable(harmless, 0, default_affected_qubits(harmless, 0))
    print("[D3] PASS: phase-observability distinguishes classical cone vs interference")


def main():
    test_A_unobservable_standalone_admitted()
    test_B_observable_phase_rejected_soundness_guard()
    test_C_regression_pairs_still_work()
    test_D_reachable_basis_states_exact()
    test_D_overapprox_is_superset_and_sound()
    test_D_phase_observability_distinguishes_cone_vs_interference()
    print("\nAll reachable-subspace + phase-observability tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

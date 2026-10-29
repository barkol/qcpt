"""Scalable whole-circuit equivalence verification for Toffoli-network rewrites.

:mod:`equivalence_verifier` gives an EXACT correctness gate, but both of its paths
are O(2^n) in circuit width:

* the classical truth-table path enumerates 2^n basis inputs, and
* the dense-unitary path materialises a 2^n x 2^n matrix.

That is fine for the small windows the selector substitutes into, and for whole
circuits up to ~8-12 qubits, but it cannot certify a WHOLE 16-24 qubit adder /
multiplier / oracle. This module adds a SCALABLE fallback that decides whole-circuit
equivalence without ever building a 2^n object, by dispatching to a decision-diagram
equivalence-checking backend (MQT QCEC), with PyZX as a secondary option.

WHAT IS (AND IS NOT) CHECKED -- READ THIS.
=========================================
There are THREE distinct soundness regimes for a rewrite, and this module keeps them
strictly separate (the selector, :mod:`decomposition_selector`, decides which one a
given rewrite falls into):

  1. EXHAUSTIVELY VERIFIED.  Small enough that ``ExactEquivalenceVerifier.verify``
     runs (truth table or dense unitary). This is the strongest, ground-truth check.
     We always prefer it when it fits.

  2. QCEC-VERIFIED UNITARY (the new scalable path).  For circuits too wide for (1)
     whose rewrite is a PLAIN UNITARY-EQUIVALENT substitution -- i.e. the EXACT
     Clifford+T Toffoli lowering and any count-reduction that preserves the unitary.
     QCEC decides unitary equivalence UP TO GLOBAL PHASE using decision diagrams /
     the ZX-calculus; it never builds the full matrix, so it scales to widths where
     (1) is hopeless. ``equivalent`` and ``equivalent_up_to_global_phase`` both count
     as equivalent here (a global phase is physically unobservable).

  3. PROGRAM-SEMANTICS (condition (U)) SUBSTITUTIONS.  A standalone relative-phase
     gadget admitted because its phase is invisible to the terminal
     computational-basis measurement differs from the exact decomposition by a
     diagonal phase, so it is deliberately NOT unitary-equivalent and QCEC reports
     ``not_equivalent`` for it. Such outputs are certified only by the dense
     observational certificate (``subspace_check.certify_on_input_subspace`` with
     mode="observational": U_sel P_in = Phi U_ex P_in for a diagonal unitary Phi),
     which is feasible up to about 12 qubits. Above that no decision-diagram
     certificate is attempted (``qcec_certify_on_subspace(..., observational=True)``
     returns False) and the selector fails closed to the all-exact circuit.

Note that the subroutine-semantics substitutions of the paper, conditions (C), (W)
and (R), are equivalences on the input subspace up to ONE global phase. They are
therefore certified by regime (1) or (2): with pinned clean ancillas, QCEC is run
with those qubits declared as ancillas initialised to |0>
(``subspace_check.qcec_certify_on_subspace``).

:func:`verify_scalable` covers regimes (1) and (2): it dispatches exhaustive for
small width and QCEC (or PyZX) for large, for substitutions where plain unitary
equivalence is the right notion. It does not certify regime (3): handed a
program-semantics rewrite it reports ``not_equivalent``, and the caller must use the
observational certificate (see :func:`decomposition_selector.ErrorBudgetSelector.select`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from .equivalence_verifier import ExactEquivalenceVerifier


# --------------------------------------------------------------------------- backend
def _qcec_available() -> bool:
    try:
        from mqt import qcec  # noqa: F401
        return True
    except Exception:
        return False


def _pyzx_available() -> bool:
    try:
        import pyzx  # noqa: F401
        return True
    except Exception:
        return False


# ----------------------------------------------------------------------------- result
@dataclass
class ScalableVerifyResult:
    """Outcome of a whole-circuit scalable equivalence check.

    Attributes
    ----------
    equivalent : Optional[bool]
        True / False if the check ran and decided; None if no backend could run
        (e.g. circuit too large for exhaustive AND no QCEC/PyZX installed).
    method : str
        Which backend decided it: ``"exhaustive"`` (truth table or dense unitary,
        ground truth), ``"qcec"`` (decision-diagram unitary equivalence up to global
        phase), ``"pyzx"`` (ZX-calculus), or ``"none"`` (no backend available).
    up_to_global_phase : bool
        For the QCEC/PyZX path: whether equivalence holds only up to a global phase
        (which is physically unobservable). Always considered equivalent.
    n_qubits : int
        Width of the larger circuit (the rewritten one).
    wall_time_s : float
        Wall-clock seconds the deciding backend took (verification cost vs size).
    detail : dict
        Backend-specific info (raw equivalence criterion, reason on mismatch, ...).
    """

    equivalent: Optional[bool]
    method: str
    up_to_global_phase: bool = False
    n_qubits: int = 0
    wall_time_s: float = 0.0
    detail: dict = field(default_factory=dict)

    def __bool__(self) -> bool:  # convenience: truthy iff proven equivalent
        return self.equivalent is True


# ----------------------------------------------------------------------- dispatcher
def verify_scalable(
    original,
    rewritten,
    *,
    exhaustive_max_qubits: int = 12,
    allow_permutation: bool = False,
    verifier: Optional[ExactEquivalenceVerifier] = None,
    prefer: str = "auto",
    timeout_s: Optional[float] = None,
) -> ScalableVerifyResult:
    """Decide whole-circuit equivalence, dispatching by width.

    Use this for rewrites where PLAIN UNITARY EQUIVALENCE is the correct notion of
    correctness: the EXACT Clifford+T Toffoli lowering and unitary-preserving
    count-reductions. (For phase-AWARE relative-phase substitutions, which are not
    unitary-equivalent by design, use the analysis path in
    :mod:`decomposition_selector`, NOT this function -- see the module docstring.)

    Dispatch
    --------
    * width(rewritten) <= ``exhaustive_max_qubits`` -> EXHAUSTIVE
      (:meth:`ExactEquivalenceVerifier.verify`: truth table for classical-reversible
      circuits, dense unitary otherwise). Ground truth.
    * larger -> SCALABLE backend: QCEC by default (decision-diagram unitary
      equivalence up to global phase), else PyZX, else give up with ``method="none"``.

    Parameters
    ----------
    original, rewritten : QuantumCircuit
        ``rewritten`` may have >= original.num_qubits; the extra (highest-index)
        qubits are clean ancilla (|0> in, |0> out). QCEC accepts mismatched widths and
        treats the extra qubits as ancilla.
    exhaustive_max_qubits : int
        Width at/below which we run the exhaustive ground-truth check. The crossover.
    allow_permutation : bool
        Only honoured on the exhaustive path (QCEC checks fixed wiring up to global
        phase). If True and the rewrite needs an output-wire permutation, only the
        exhaustive path can certify it.
    prefer : {"auto", "exhaustive", "qcec", "pyzx"}
        Force a backend. ``"auto"`` uses the width dispatch above. ``"exhaustive"``
        forces the exact check regardless of width (may be infeasible -- caller's
        risk). ``"qcec"`` / ``"pyzx"`` force the scalable backend even for small
        circuits (useful to benchmark the crossover).
    timeout_s : float, optional
        Soft timeout passed to the QCEC backend where supported.

    Returns
    -------
    ScalableVerifyResult
    """
    verifier = verifier or ExactEquivalenceVerifier()
    n = max(original.num_qubits, rewritten.num_qubits)

    if prefer == "exhaustive" or (
        prefer == "auto" and n <= exhaustive_max_qubits
    ):
        return _run_exhaustive(verifier, original, rewritten, allow_permutation, n)

    if prefer in ("auto", "qcec") and _qcec_available():
        return _run_qcec(original, rewritten, n, timeout_s)

    if prefer in ("auto", "pyzx") and _pyzx_available():
        return _run_pyzx(original, rewritten, n)

    if prefer == "qcec" and not _qcec_available():
        return ScalableVerifyResult(
            equivalent=None, method="none", n_qubits=n,
            detail={"reason": "QCEC backend requested but mqt.qcec not importable"},
        )
    if prefer == "pyzx" and not _pyzx_available():
        return ScalableVerifyResult(
            equivalent=None, method="none", n_qubits=n,
            detail={"reason": "PyZX backend requested but pyzx not importable"},
        )

    # auto, too large for exhaustive, no scalable backend installed.
    return ScalableVerifyResult(
        equivalent=None, method="none", n_qubits=n,
        detail={"reason": f"circuit too large for exhaustive (n={n} > "
                          f"{exhaustive_max_qubits}) and no QCEC/PyZX backend "
                          f"available; cannot certify at this scale"},
    )


# --------------------------------------------------------------------------- runners
def _run_exhaustive(verifier, original, rewritten, allow_permutation, n):
    t0 = time.perf_counter()
    ok, _perm, info = verifier.verify(
        original, rewritten, allow_permutation=allow_permutation
    )
    dt = time.perf_counter() - t0
    return ScalableVerifyResult(
        equivalent=bool(ok),
        method="exhaustive",
        up_to_global_phase=False,
        n_qubits=n,
        wall_time_s=dt,
        detail=info,
    )


def _run_qcec(original, rewritten, n, timeout_s):
    """Decide unitary equivalence (up to global phase) with MQT QCEC.

    QCEC builds decision diagrams for both circuits and proves their products
    cancel to the identity (up to global phase) -- it never materialises the
    2^n x 2^n matrix, so it scales to widths the dense path cannot. It handles
    mismatched widths by treating extra qubits as clean ancilla.
    """
    from mqt import qcec

    t0 = time.perf_counter()
    # IMPORTANT: disable QCEC's ZX-calculus checker. On circuits carrying the
    # non-Clifford relative phases these rewrites involve (the Margolus gadget's
    # Ry(pi/4) rotations, and the T-gates of the exact Toffoli), the ZX checker can
    # return ``not_equivalent`` / ``no_information`` even when the circuits ARE
    # unitary-equivalent -- the ZX simplifier is incomplete for general non-Clifford
    # diagrams, so it under-reduces and reports a spurious inconclusive/negative. The
    # decision-diagram construction + alternating + simulation checkers decide these
    # cases correctly (and remain sound: a DD ``equivalent`` is a real proof up to
    # global phase). We therefore turn the ZX checker OFF and rely on the DD checkers.
    # elide_permutations=False: with the default, QCEC treats a SWAP (or 3 CX) as the
    # identity and certifies a permuted circuit. No fallback to default options: if
    # they are not accepted, the check is undecided (fail closed).
    kwargs = {"run_zx_checker": False, "elide_permutations": False, "trace_threshold": 1e-12}
    if timeout_s is not None:
        kwargs["timeout"] = float(timeout_s)
    try:
        res = qcec.verify(original, rewritten, **kwargs)
    except TypeError as exc:
        return ScalableVerifyResult(
            equivalent=None, method="qcec", up_to_global_phase=False, n_qubits=n,
            wall_time_s=time.perf_counter() - t0,
            detail={"reason": f"QCEC options unsupported: {exc!r}"})
    dt = time.perf_counter() - t0

    crit = res.equivalence
    crit_name = str(crit).rsplit(".", 1)[-1]  # e.g. "equivalent_up_to_global_phase"
    up_to_gp = "global_phase" in crit_name

    # Map QCEC's verdict to our tri-state. ``no_information`` means the checker
    # neither proved nor disproved equivalence (e.g. it timed out / ran out of
    # heuristics) -- it is NOT a proof of inequivalence, so we report ``None``
    # (undecided) rather than a (potentially wrong) ``False``.
    if crit_name in ("equivalent", "equivalent_up_to_global_phase"):
        equivalent: Optional[bool] = True
    elif crit_name in ("not_equivalent", "probably_not_equivalent"):
        equivalent = False
    else:  # no_information, etc.
        equivalent = None

    return ScalableVerifyResult(
        equivalent=equivalent,
        method="qcec",
        up_to_global_phase=up_to_gp,
        n_qubits=n,
        wall_time_s=dt,
        detail={"equivalence_criterion": crit_name},
    )


def _run_pyzx(original, rewritten, n):
    """Decide equivalence via the ZX-calculus (PyZX).

    Compose ``original`` with the adjoint of ``rewritten`` and simplify; the two are
    equivalent (up to global phase) iff the composite reduces to the identity. Only
    used when QCEC is unavailable; requires equal widths (PyZX has no ancilla notion
    here), so a width mismatch falls back to "not decided".
    """
    import pyzx as zx
    from qiskit import qasm2

    t0 = time.perf_counter()
    if original.num_qubits != rewritten.num_qubits:
        return ScalableVerifyResult(
            equivalent=None, method="pyzx", n_qubits=n,
            wall_time_s=time.perf_counter() - t0,
            detail={"reason": "PyZX path requires equal widths (no ancilla model)"},
        )
    try:
        c_a = zx.Circuit.from_qasm(qasm2.dumps(original))
        c_b = zx.Circuit.from_qasm(qasm2.dumps(rewritten))
        equal = c_a.verify_equality(c_b)
    except Exception as exc:
        return ScalableVerifyResult(
            equivalent=None, method="pyzx", n_qubits=n,
            wall_time_s=time.perf_counter() - t0,
            detail={"reason": f"pyzx verify raised {exc!r}"},
        )
    dt = time.perf_counter() - t0
    return ScalableVerifyResult(
        equivalent=bool(equal),
        method="pyzx",
        up_to_global_phase=True,  # ZX equality is up to global phase
        n_qubits=n,
        wall_time_s=dt,
        detail={"pyzx_equal": bool(equal)},
    )


# ----------------------------------------------------------- attach to the verifier
def _verify_scalable_method(
    self,
    original,
    rewritten,
    *,
    exhaustive_max_qubits: Optional[int] = None,
    allow_permutation: bool = False,
    prefer: str = "auto",
    timeout_s: Optional[float] = None,
) -> ScalableVerifyResult:
    """:meth:`ExactEquivalenceVerifier.verify_scalable` -- see
    :func:`verify_scalable`.

    Dispatches exhaustive (this verifier's own exact check) for small width and a
    decision-diagram / ZX backend (QCEC, else PyZX) for large width, for rewrites
    where plain unitary equivalence is the correct notion of correctness (the exact
    Toffoli lowering and unitary-preserving count-reductions). NOT for phase-aware
    relative-phase substitutions -- those are certified by the static analysis in
    :mod:`decomposition_selector`, since they are not unitary-equivalent by design.
    """
    if exhaustive_max_qubits is None:
        # default crossover: the dense-unitary guard this verifier already uses.
        exhaustive_max_qubits = self.max_qubits_unitary
    return verify_scalable(
        original,
        rewritten,
        exhaustive_max_qubits=exhaustive_max_qubits,
        allow_permutation=allow_permutation,
        verifier=self,
        prefer=prefer,
        timeout_s=timeout_s,
    )


# Bolt the method onto ExactEquivalenceVerifier so callers can do
# ``ExactEquivalenceVerifier().verify_scalable(orig, rewritten)`` per the task spec,
# without re-importing this module. (Defined here to keep the QCEC/PyZX dependency
# and its documentation isolated to this file.)
ExactEquivalenceVerifier.verify_scalable = _verify_scalable_method

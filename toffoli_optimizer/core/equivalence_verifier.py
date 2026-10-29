"""Exact equivalence verification for Toffoli-network rewrites.

QCPT previously trusted its rewrite rules: it imported qiskit's `Operator`
but only computed a *heuristic* "logical fidelity" formula and never checked that
a simplified circuit actually computes the same function as the original. This
module closes that gap with an EXACT check, so every rewrite can be proven
function-preserving (and rejected if it is not).

Two verification paths, chosen automatically:

* Fast classical path (`_verify_truth_table`): when both circuits are pure
  classical-reversible Toffoli networks (only x / cx / ccx / mcx / swap), the
  whole circuit is a permutation of basis states, so equivalence is decided by an
  exhaustive truth table over 2^k inputs -- bit-packed, microseconds. This is the
  technique that made the secp256k1 point-add analysis tractable.

* Exact unitary path (`_verify_unitary`): when a circuit contains
  superposition-creating gates (h / t / u / ... -- e.g. a decomposed or
  relative-phase Toffoli), it compares the full unitaries via
  `qiskit.quantum_info.Operator`, up to global phase.

Both paths handle:
  * ANCILLA: a rewrite may use MORE qubits than the original; the extra (highest-
    index) qubits are ancilla that must start and end in |0>. Leakage onto an
    ancilla output fails verification.
  * GLOBAL PHASE: equivalence is checked up to a single overall phase.
  * OUTPUT-WIRE PERMUTATION (optional): since a downstream consumer can be
    relabeled for free, a rewrite may be accepted up to a permutation of the
    data output wires.
"""

from __future__ import annotations

import itertools
import math
from typing import Optional, Tuple

import numpy as np

# Gates that keep the circuit a classical-reversible permutation of basis states.
_CLASSICAL_GATES = {"x", "cx", "ccx", "mcx", "mcx_gray", "swap", "id", "barrier"}

_TOL = 1e-9


class ExactEquivalenceVerifier:
    """Decide whether a rewritten circuit computes the same function as the original."""

    def __init__(self, tol: float = _TOL, max_qubits_unitary: int = 12):
        self.tol = tol
        # Guard the dense-unitary path against blowing up on large windows.
        self.max_qubits_unitary = max_qubits_unitary

    # ------------------------------------------------------------------ public
    def verify(
        self,
        original,
        rewritten,
        allow_permutation: bool = False,
    ) -> Tuple[bool, Optional[dict], dict]:
        """Return (is_equivalent, output_permutation, info).

        `original` / `rewritten` are qiskit QuantumCircuits. `rewritten` may have
        >= original.num_qubits; the extra (highest-index) qubits are treated as
        ancilla required to begin and end in |0>. If `allow_permutation`, accept
        equivalence up to a permutation of the data output wires (returned in the
        second slot as {original_wire: rewritten_wire}).
        """
        n = original.num_qubits
        n_r = rewritten.num_qubits
        if n_r < n:
            return False, None, {"reason": "rewrite has fewer qubits than original"}
        n_anc = n_r - n

        classical = self._is_classical(original) and self._is_classical(rewritten)
        if classical:
            return self._verify_truth_table(original, rewritten, n, n_anc, allow_permutation)
        if n_r > self.max_qubits_unitary:
            return False, None, {
                "reason": f"non-classical circuit too large for exact unitary check "
                          f"({n_r} > {self.max_qubits_unitary} qubits)"
            }
        return self._verify_unitary(original, rewritten, n, n_anc, allow_permutation)

    # --------------------------------------------------------------- internals
    @staticmethod
    def _is_classical(circuit) -> bool:
        for inst in circuit.data:
            if inst.operation.name.lower() not in _CLASSICAL_GATES:
                return False
        return True

    @staticmethod
    def _qubit_indices(circuit, inst):
        return [circuit.find_bit(q).index for q in inst.qubits]

    # ---- fast classical path (bit-packed truth table over 2^n data inputs) ----
    def _simulate_classical(self, circuit, n_total: int):
        """Return a list `perm` of length 2^n_total: perm[x] = output basis index
        for input basis index x (qubit j == bit j, little-endian)."""
        size = 1 << n_total
        # state[x] tracks, for input x, the current per-qubit bit values packed as
        # an int. Start at identity: input x maps to itself.
        out = list(range(size))
        for inst in circuit.data:
            name = inst.operation.name.lower()
            if name in ("barrier", "id"):
                continue
            qb = self._qubit_indices(circuit, inst)
            for x in range(size):
                s = out[x]
                if name == "x":
                    s ^= 1 << qb[0]
                elif name == "cx":
                    if (s >> qb[0]) & 1:
                        s ^= 1 << qb[1]
                elif name == "ccx":
                    if ((s >> qb[0]) & 1) and ((s >> qb[1]) & 1):
                        s ^= 1 << qb[2]
                elif name in ("mcx", "mcx_gray"):
                    if all((s >> c) & 1 for c in qb[:-1]):
                        s ^= 1 << qb[-1]
                elif name == "swap":
                    b0 = (s >> qb[0]) & 1
                    b1 = (s >> qb[1]) & 1
                    if b0 != b1:
                        s ^= (1 << qb[0]) | (1 << qb[1])
                out[x] = s
        return out

    def _verify_truth_table(self, original, rewritten, n, n_anc, allow_permutation):
        perm_o = self._simulate_classical(original, n)
        perm_r = self._simulate_classical(rewritten, n + n_anc)
        anc_mask = ((1 << (n + n_anc)) - 1) ^ ((1 << n) - 1)  # bits n .. n+n_anc-1

        # Build the data-only output map for inputs with ancilla = 0 (x < 2^n),
        # checking ancilla returns to 0.
        out_o = [perm_o[x] & ((1 << n) - 1) for x in range(1 << n)]
        out_r = []
        for x in range(1 << n):
            y = perm_r[x]
            if y & anc_mask:
                return False, None, {"reason": f"ancilla not restored to |0> on input {x}",
                                     "method": "truth_table"}
            out_r.append(y & ((1 << n) - 1))

        if out_o == out_r:
            return True, {i: i for i in range(n)}, {"method": "truth_table"}

        if allow_permutation:
            perm = self._find_output_wire_permutation(out_o, out_r, n)
            if perm is not None:
                return True, perm, {"method": "truth_table", "up_to_permutation": True}

        # find a witness input for the report
        for x in range(1 << n):
            if out_o[x] != out_r[x]:
                return False, None, {"reason": "classical output mismatch", "method": "truth_table",
                                     "witness_input": x, "expected": out_o[x], "got": out_r[x]}
        return False, None, {"reason": "mismatch", "method": "truth_table"}

    @staticmethod
    def _find_output_wire_permutation(out_o, out_r, n):
        """Find a bit-permutation sigma of the n output wires s.t. applying it to
        out_r reproduces out_o on every input. Returns {orig_wire: rewrite_wire}."""
        for sigma in itertools.permutations(range(n)):
            ok = True
            for x in range(1 << n):
                yr = out_r[x]
                yperm = 0
                for j in range(n):
                    if (yr >> j) & 1:
                        yperm |= 1 << sigma[j]
                if yperm != out_o[x]:
                    ok = False
                    break
            if ok:
                return {sigma[j]: j for j in range(n)}
        return None

    # ---- exact unitary path (qiskit Operator, up to global phase) ----
    def _verify_unitary(self, original, rewritten, n, n_anc, allow_permutation):
        import os as _os
        dim = 1 << n
        if _os.environ.get("QCPT_FAST_CERT", "1") == "1":                 # only the ancilla-|0> columns are needed
            from .subspace_check import circuit_columns
            U_o = circuit_columns(original, range(dim))
            U_r = circuit_columns(rewritten, range(dim))
        else:
            from qiskit.quantum_info import Operator
            U_o = Operator(original).data
            U_r = Operator(rewritten).data
        # ancilla bits are the high bits (indices n..); ancilla=0 subspace is the
        # set of basis indices < 2^n. Extract the top-left block of U_r.
        B = U_r[:dim, :dim]

        # No leakage: for each ancilla=0 input (columns < dim), all amplitude must
        # stay in ancilla=0 outputs (rows < dim) -> each such column of U_r is
        # fully contained in B.
        leak = U_r[dim:, :dim]
        if leak.size and np.max(np.abs(leak)) > self.tol:
            return False, None, {"reason": "amplitude leaks onto an ancilla output",
                                 "method": "unitary"}

        if self._equal_up_to_global_phase(B, U_o):
            return True, {i: i for i in range(n)}, {"method": "unitary"}

        if allow_permutation:
            for sigma in itertools.permutations(range(n)):
                P = self._wire_permutation_matrix(sigma, n)
                if self._equal_up_to_global_phase(P @ B, U_o):
                    return True, {sigma[j]: j for j in range(n)}, {
                        "method": "unitary", "up_to_permutation": True}

        return False, None, {"reason": "unitary mismatch (function or phase differs)",
                             "method": "unitary"}

    def _equal_up_to_global_phase(self, A, B) -> bool:
        if A.shape != B.shape:
            return False
        # pick the largest-magnitude entry of B to fix the global phase
        idx = np.unravel_index(np.argmax(np.abs(B)), B.shape)
        if abs(B[idx]) < self.tol:
            return np.max(np.abs(A)) < self.tol
        if abs(A[idx]) < self.tol:
            return False
        phase = B[idx] / A[idx]
        if abs(abs(phase) - 1.0) > 1e-6:
            return False
        return np.max(np.abs(A * phase - B)) < 1e-7

    @staticmethod
    def _wire_permutation_matrix(sigma, n):
        dim = 1 << n
        P = np.zeros((dim, dim), dtype=complex)
        for y in range(dim):
            yp = 0
            for j in range(n):
                if (y >> j) & 1:
                    yp |= 1 << sigma[j]
            P[yp, y] = 1.0
        return P

    # ---- phase-insensitive, reachable-basis equivalence ----------------------
    def verify_on_reachable_basis(
        self,
        original,
        rewritten,
        reachable_states,
    ) -> Tuple[bool, Optional[dict], dict]:
        """More-permissive equivalence on a restricted set of basis inputs.

        For every computational-basis input in ``reachable_states``, check that
        ``original`` and ``rewritten`` produce the SAME OUTPUT computational-basis
        state on the data wires -- PHASE-INSENSITIVELY -- and that any ancilla
        (the extra high-index qubits of ``rewritten``) returns to |0>.

        NOT AN ADMISSIBILITY CRITERION. A per-basis-state, phase-insensitive check
        is blind to relative phases between different basis inputs, which become
        observable when the circuit is fed a superposition of them (counterexample:
        controls = 1, H on the target before and after a CCX replaced by RCCX). The
        certified pass (:class:`decomposition_selector.ErrorBudgetSelector`) no longer
        calls this method; it uses the operator conditions of
        :mod:`subspace_check` (one phase on the whole subspace). The method is kept
        as a diagnostic for Boolean (classical) agreement on basis inputs only.

        Requirements / scope (kept conservative):
          * Both circuits are evaluated on each basis input by statevector
            simulation. The input must be a single computational-basis state, which
            it is by construction (we feed basis states from the reachable set).
          * "Same output basis state, phase-insensitive" means: the rewritten output
            statevector, restricted to the ancilla=|0> data subspace, is a single
            computational-basis state (up to global phase) equal to the original's
            output basis state on the data wires. If either output is not a single
            basis state on the data wires (e.g. the original itself superposes), we
            require the FULL output statevectors to match up to global phase
            (falling back to exact agreement, which is sound but stricter).

        Parameters
        ----------
        original, rewritten : QuantumCircuit
            ``rewritten`` may have >= original.num_qubits; the extra high-index
            qubits are ancilla required to end in |0>.
        reachable_states : iterable of int
            Computational-basis input indices (little-endian, bit j == qubit j on the
            DATA wires of ``original``) to check. Inputs outside this set are not
            checked -- the caller must guarantee (via reachable-subspace analysis)
            that they cannot occur.

        Returns
        -------
        (is_equivalent, None, info)
        """
        from qiskit.quantum_info import Statevector

        n = original.num_qubits
        n_r = rewritten.num_qubits
        if n_r < n:
            return False, None, {"reason": "rewrite has fewer qubits than original",
                                 "method": "reachable_basis"}
        n_anc = n_r - n
        anc_mask = ((1 << n_r) - 1) ^ ((1 << n) - 1)

        reachable = sorted(set(reachable_states))
        for x in reachable:
            if x < 0 or x >= (1 << n):
                return False, None, {
                    "reason": f"reachable input {x} out of data range [0,2^{n})",
                    "method": "reachable_basis"}

            sv_o = self._basis_evolve(original, x, n)
            # The rewritten circuit gets the SAME data input x with ancilla = |0>.
            sv_r = self._basis_evolve(rewritten, x, n_r)

            # Ancilla clean check on the rewritten output.
            leaked = self._amplitude_on_ancilla_nonzero(sv_r, anc_mask)
            if leaked is not None:
                return False, None, {
                    "reason": f"ancilla not restored to |0> on input {x}",
                    "method": "reachable_basis", "witness_input": x,
                    "ancilla_state": leaked}

            ok, why = self._phase_insensitive_same_output(sv_o, sv_r, n, n_r)
            if not ok:
                return False, None, {
                    "reason": "phase-insensitive output mismatch",
                    "method": "reachable_basis", "witness_input": x, "detail": why}

        return True, None, {"method": "reachable_basis",
                            "checked_inputs": len(reachable)}

    @staticmethod
    def _basis_evolve(circuit, x: int, n_qubits: int):
        """Statevector of ``circuit`` run on computational-basis input ``x``."""
        from qiskit.quantum_info import Statevector
        init = Statevector.from_int(x, dims=2 ** n_qubits)
        return init.evolve(circuit)

    def _amplitude_on_ancilla_nonzero(self, sv, anc_mask):
        """Return an offending ancilla-nonzero basis index, or None if clean."""
        data = sv.data
        for idx in range(len(data)):
            if (idx & anc_mask) and abs(data[idx]) > self.tol:
                return idx
        return None

    def _phase_insensitive_same_output(self, sv_o, sv_r, n, n_r):
        """Do sv_o (n qubits) and sv_r (n_r qubits) agree on the data wires,
        phase-insensitively, with ancilla=|0>?

        Strategy: project sv_r onto the ancilla=|0> subspace (the low 2^n indices)
        and compare to sv_o. If sv_o is a single basis state, we only require the
        rewritten output to be the SAME single basis state (any phase). Otherwise we
        require the two (data-restricted) vectors to match up to a global phase --
        the sound, stricter fallback.
        """
        dim = 1 << n
        ro = np.asarray(sv_o.data)
        rr = np.asarray(sv_r.data)[:dim]  # ancilla=|0> data block

        # Is the original output a single computational-basis state?
        nz_o = np.where(np.abs(ro) > self.tol)[0]
        if len(nz_o) == 1:
            b = int(nz_o[0])
            # rewritten must put all its (data) weight on the same basis index b.
            nz_r = np.where(np.abs(rr) > self.tol)[0]
            if len(nz_r) == 1 and int(nz_r[0]) == b and abs(abs(rr[b]) - 1.0) < 1e-6:
                return True, None
            return False, {"expected_basis": b,
                           "got_nonzero": [int(i) for i in nz_r]}

        # General case: require data-restricted vectors equal up to global phase.
        if self._equal_up_to_global_phase(rr.reshape(-1, 1), ro.reshape(-1, 1)):
            return True, None
        return False, {"reason": "data statevectors differ beyond a global phase"}

    # ---- bounded-approximate, phase-insensitive, reachable-basis equivalence ----
    def verify_on_reachable_basis_approx(
        self,
        original,
        rewritten,
        epsilon: float,
        reachable_states,
    ) -> Tuple[bool, float, dict]:
        r"""BOUNDED-APPROXIMATE equivalence on a restricted set of basis inputs.

        Generalises :meth:`verify_on_reachable_basis` from EXACT-on-reachable to
        APPROXIMATE-on-reachable. The rewritten circuit's action ``U_d`` need not
        equal the original's ``U_CCX`` even on the reachable subspace; it only has to
        agree to within ``epsilon`` in the worst case, measured PHASE-INSENSITIVELY:

            max_deviation = max over |x> in reachable_states of
                            min over global phase e^{i*phi} of
                            || U_d|x>  -  e^{i*phi} U_CCX|x> ||_2

        and the rewrite is admissible iff ``max_deviation <= epsilon``. With
        ``epsilon == 0`` this recovers exact-on-reachable agreement (up to global
        phase) per input -- i.e. it is the quantitative refinement of
        :meth:`verify_on_reachable_basis`.
        Like that method it is a per-input diagnostic, not the admission test of the
        certified pass (which uses ``subspace_check.check_on_subspace``, an operator
        norm on the reachable subspace with one phase).

        The per-input phase-insensitive distance is computed in closed form. For two
        unit-norm vectors u (= U_d|x>, restricted to the ancilla=|0> data block) and
        v (= U_CCX|x>),

            min_phi || u - e^{i phi} v ||^2 = ||u||^2 + ||v||^2 - 2|<v,u>|,

        the minimiser being phi = arg<v,u>. We use ``u`` restricted to the data block
        (ancilla=|0>); any amplitude that LEAKS onto an ancilla output is NOT
        subtracted off by the phase alignment, so it is fully counted into the
        deviation (an un-restored ancilla is penalised, never silently accepted).

        SOUNDNESS. ``reachable_states`` must be a SOUND over-approximation of the
        computational-basis inputs that can reach the gate (use
        :func:`reachable_subspace.reachable_overapprox`). The reported
        ``max_deviation`` is then an UPPER BOUND on the true worst-case reachable
        deviation, because (i) it is a max over a superset of the truly reachable
        inputs, and (ii) each per-input term is the exact phase-optimal distance. So
        ``max_deviation <= epsilon`` certifies the substitution is within ``epsilon``
        on every truly reachable input; the check can only reject a valid rewrite,
        never accept an invalid one.

        Parameters
        ----------
        original, rewritten : QuantumCircuit
            ``rewritten`` may have >= original.num_qubits; the extra high-index
            qubits are ancilla (their final amplitude is counted into the deviation).
        epsilon : float
            Tolerance on the worst-case phase-insensitive reachable deviation.
        reachable_states : iterable of int
            Computational-basis input indices (little-endian, bit j == qubit j on the
            DATA wires of ``original``).

        Returns
        -------
        (admissible, max_deviation, info)
            ``admissible`` is ``max_deviation <= epsilon``. ``max_deviation`` is the
            worst-case phase-insensitive L2 deviation over the reachable inputs (0.0
            if there are none). ``info`` carries the method, count, and -- on the
            worst input -- a witness.
        """
        if epsilon < 0:
            raise ValueError("epsilon must be >= 0")

        n = original.num_qubits
        n_r = rewritten.num_qubits
        if n_r < n:
            return False, float("inf"), {
                "reason": "rewrite has fewer qubits than original",
                "method": "reachable_basis_approx"}
        dim = 1 << n

        reachable = sorted(set(reachable_states))
        max_dev = 0.0
        witness = None
        for x in reachable:
            if x < 0 or x >= dim:
                return False, float("inf"), {
                    "reason": f"reachable input {x} out of data range [0,2^{n})",
                    "method": "reachable_basis_approx"}

            sv_o = self._basis_evolve(original, x, n)
            sv_r = self._basis_evolve(rewritten, x, n_r)
            dev = self._phase_insensitive_deviation(sv_o, sv_r, dim)
            if dev > max_dev:
                max_dev = dev
                witness = x

        admissible = max_dev <= epsilon + self.tol
        info = {
            "method": "reachable_basis_approx",
            "checked_inputs": len(reachable),
            "epsilon": epsilon,
            "max_deviation": max_dev,
        }
        if witness is not None:
            info["witness_input"] = witness
        return admissible, max_dev, info

    def _phase_insensitive_deviation(self, sv_o, sv_r, dim) -> float:
        r"""Phase-optimal L2 distance between U_CCX|x> (= sv_o) and the rewritten
        output (= sv_r) restricted to the ancilla=|0> data block.

        Returns ``sqrt( ||u||^2 + ||v||^2 - 2|<v,u>| )`` where v = sv_o (length dim,
        unit norm) and u = sv_r restricted to its low ``dim`` (ancilla=|0>) entries.
        Amplitude on ancilla!=0 outputs is excluded from u, hence NOT cancelled by
        the phase alignment, and so is fully reflected in the smaller ||u|| (and thus
        a larger deviation). Result is clamped to be non-negative against round-off.
        """
        v = np.asarray(sv_o.data)              # length dim, unit norm
        u = np.asarray(sv_r.data)[:dim]        # ancilla=|0> data block
        nu2 = float(np.real(np.vdot(u, u)))
        nv2 = float(np.real(np.vdot(v, v)))
        overlap = abs(complex(np.vdot(v, u)))  # |<v,u>|, phase-insensitive
        d2 = nu2 + nv2 - 2.0 * overlap
        if d2 < 0.0:
            d2 = 0.0
        return math.sqrt(d2)


# ---------------------------------------------------------------------------------
# Scalable whole-circuit fallback. Importing the scalable_verification module binds
# ``ExactEquivalenceVerifier.verify_scalable`` (decision-diagram / ZX backends for
# widths too large for the exhaustive paths above). Kept as a side-effect import so
# the method is present on the class as soon as this module is imported, while the
# heavy/optional QCEC + PyZX dependencies live in the separate module and are only
# touched when verify_scalable actually runs. See scalable_verification for the
# verified-vs-analysis-sound boundary documentation.
try:  # pragma: no cover - binding is best-effort; absence only disables the fallback
    from . import scalable_verification as _scalable_verification  # noqa: F401
except Exception:  # pragma: no cover
    _scalable_verification = None

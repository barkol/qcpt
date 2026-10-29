"""Error-budget-aware, context-verified Toffoli decomposition selection.

A CCX/MCX circuit has to be lowered to a two-qubit basis to run on hardware. There
are two kinds of Toffoli decomposition:

* EXACT (Clifford+T): the textbook H / T / T-dagger / CX Toffoli, **6 CX**. Always
  correct anywhere.

* RELATIVE-PHASE (Margolus): a cheaper gadget with only **3 CX** that implements
  CCX correctly *up to a relative phase* on the target subspace. It is valid only
  where that phase is later uncomputed (a compute/uncompute pair) -- see
  ``context_analysis``.

``ErrorBudgetSelector`` chooses, per Toffoli, which decomposition to use so as to
minimize the :class:`HardwareErrorModel` infidelity (dominated by the two-qubit
count). A cheaper lowering is admitted only under one of the admissibility
conditions of the paper, each of which is an OPERATOR condition (one phase for a
whole subspace), never a basis-state-by-basis-state comparison. A per-basis-state,
phase-insensitive check is blind to relative phases between reachable basis states,
which superposed inputs turn into observable interference (see ``subspace_check``),
so no such check is used for admission or certification.

* (C) compute/uncompute pairs: a structurally detected pair (``context_analysis``)
  gets relative-phase gadgets on both gates only if :class:`ExactEquivalenceVerifier`
  certifies that the circuit with both gadgets is exactly unitary-equivalent to the
  all-exact decomposition (no wire permutation).
* (W) window pairs (``window_pairs``): a mirrored pair the structural detector
  rejects is admitted only if the segment between the two gates, with gadgets,
  equals the exact segment up to one global phase on the qubits it touches.
* (R) reachable subspace: a standalone Toffoli gets a relative-phase gadget, or a
  control-drop specialisation, only if  U_d P = e^{i theta} U_CCX P  with ONE phase
  theta, where P projects onto the span of the (soundly over-approximated) reachable
  basis states at the gate (``check_on_subspace``). With ``epsilon > 0`` the
  bounded variant  min_theta ||(U_d - e^{i theta} U_CCX) P||_op <= epsilon  is used
  and epsilon is charged to the error budget.
* (U) unobservable phase, ``semantics="program"`` only: a standalone gadget whose
  diagonal phase is provably invisible to the terminal computational-basis
  measurement (``is_phase_unobservable``: every later gate on the phased qubits is a
  basis permutation). This changes the unitary and is therefore never used in the
  default ``semantics="subroutine"``.

KEY GUARANTEE: the selected circuit is certified end to end against the exact-only
decomposition on the input subspace (all basis inputs, or those with the pinned
qubits in |0>): in subroutine semantics  U_sel P_in = e^{i theta} U_ex P_in  with
one phase (dense ``certify_on_input_subspace`` up to 12 qubits, MQT QCEC with the
pinned qubits as ancillas above); in program semantics equality of the output
distributions for every input state on that subspace, superpositions included.
If the certificate fails or is undecided, the selector fails closed and returns the
all-exact decomposition.
"""

from __future__ import annotations

import math
from typing import List, Optional

from qiskit import QuantumCircuit

from .equivalence_verifier import ExactEquivalenceVerifier
from .error_model import HardwareErrorModel
from .context_analysis import find_relative_phase_safe_sites, RelativePhaseSite
from .phase_observability import is_phase_unobservable, default_affected_qubits
from .reachable_subspace import reachable_overapprox, local_overapprox
from .subspace_check import (
    project_support, gadget_unitary, check_on_subspace, certify_on_input_subspace)

_CCX_NAMES = {"ccx", "mcx", "mcx_gray"}

def _ccx_matrix():
    from qiskit.quantum_info import Operator
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    return Operator(qc).data

_U_CCX = _ccx_matrix()


def _gadget_matrix(fn):
    from qiskit.quantum_info import Operator
    qc = QuantumCircuit(3)
    fn(qc, 0, 1, 2)
    return Operator(qc).data

# Whole-circuit unitary certification is attempted up to this width.
_CERT_MAX_QUBITS = 12


_CHECK_CACHE: dict = {}
_BLOCK_CACHE: dict = {}
_UD_CACHE: dict = {}


def _check_cached(U_g, U_ref, local, eps):
    """check_on_subspace with a memo: the answer depends only on the two matrices, the local support and epsilon."""
    key = (U_g.tobytes(), U_ref.tobytes(), tuple(local), eps)
    r = _CHECK_CACHE.get(key)
    if r is None:
        if len(_CHECK_CACHE) > 20000:
            _CHECK_CACHE.clear()
        r = _CHECK_CACHE[key] = check_on_subspace(U_g, U_ref, local, eps)
    return r


def _same_circuit(a, b) -> bool:
    """True if both circuits have the same gates on the same qubits in the same order (then they are trivially equivalent)."""
    if a is b:
        return True
    if a.num_qubits != b.num_qubits or len(a.data) != len(b.data) or abs(float(a.global_phase) - float(b.global_phase)) > 1e-14:
        return False
    for x, y in zip(a.data, b.data):
        ox, oy = x.operation, y.operation
        if ox.name != oy.name or ox.num_qubits != oy.num_qubits or len(ox.params) != len(oy.params):
            return False
        if [a.find_bit(q).index for q in x.qubits] != [b.find_bit(q).index for q in y.qubits]:
            return False
        for p, q in zip(ox.params, oy.params):
            try:
                if abs(float(p) - float(q)) > 1e-14:
                    return False
            except Exception:
                return False
    return True


# --------------------------------------------------------------------------- gadgets
def append_exact_ccx(qc: QuantumCircuit, a: int, b: int, t: int) -> None:
    """Append the textbook exact Clifford+T Toffoli (6 CX) onto qc[a,b -> t].

    This is qiskit's standard CCX decomposition. We emit it explicitly (rather than
    qc.ccx + transpile) so the two-qubit count is deterministic and the basis is a
    fixed {h, t, tdg, cx} set.
    """
    qc.h(t)
    qc.cx(b, t)
    qc.tdg(t)
    qc.cx(a, t)
    qc.t(t)
    qc.cx(b, t)
    qc.tdg(t)
    qc.cx(a, t)
    qc.t(b)
    qc.t(t)
    qc.cx(a, b)
    qc.h(t)
    qc.t(a)
    qc.tdg(b)
    qc.cx(a, b)


def append_control_drop(qc: QuantumCircuit, a: int, b: int, t: int, keep) -> None:
    """Append the CONTROL-DROPPED specialisation of CCX(a,b,t) onto ``qc``.

    A Toffoli equals a cheaper gate when one (or both) of its controls is constant on
    the inputs that actually reach it:

      * ``keep == ()``        -> a control is |0> on (almost) all reachable mass, so
        the |11> branch never fires: CCX acts as IDENTITY. Emit NOTHING (0 two-qubit
        gates).
      * ``keep == (a,)``      -> control ``b`` is |1> on (almost) all reachable mass,
        so CCX reduces to a single CX(a, t).
      * ``keep == (b,)``      -> control ``a`` is |1> on (almost) all reachable mass,
        so CCX reduces to a single CX(b, t).

    This is a CONTEXT-SPECIALISED candidate: it is only correct on the reachable
    subspace (and only within tolerance there), so the selector admits it solely
    through the bounded-approximate reachable check. ``keep`` lists the control(s)
    that remain after dropping; the surviving control (if any) becomes the control of
    the residual CX onto ``t``.
    """
    keep = tuple(keep)
    if keep == ():
        return  # CCX collapses to identity on the reachable subspace
    if len(keep) == 1:
        qc.cx(keep[0], t)
        return
    raise ValueError(f"control-drop keep must have 0 or 1 controls, got {keep!r}")


def append_relative_phase_ccx(qc: QuantumCircuit, a: int, b: int, t: int) -> None:
    """Append a relative-phase (Margolus) Toffoli, **3 CX**, onto qc[a,b -> t].

    Implements CCX(a,b,t) correctly UP TO a relative phase on the target. Valid only
    inside a compute/uncompute pair where that phase is uncomputed. The construction
    is the standard Margolus gate:

        Ry(-pi/4) t ; CX(b,t) ; Ry(-pi/4) t ; CX(a,t) ; Ry(pi/4) t ; CX(b,t) ; Ry(pi/4) t

    Cost: 3 CX + 4 single-qubit Ry, vs 6 CX for the exact gadget.
    """
    q = math.pi / 4
    qc.ry(-q, t)
    qc.cx(b, t)
    qc.ry(-q, t)
    qc.cx(a, t)
    qc.ry(q, t)
    qc.cx(b, t)
    qc.ry(q, t)


def append_relative_phase_ccx_mirror(qc: QuantumCircuit, a: int, b: int, t: int) -> None:
    """Mirrored Margolus gadget, **3 CX**: the four Ry angles change sign.

    Its diagonal phase -1 sits on the output |a=1, b=0, t=1> instead of
    |a=1, b=0, t=0>. That branch is unreachable when the target enters in |0>
    (an AND written into a clean ancilla) and when it enters holding a&b (the
    uncompute), so on those reachable subspaces the gadget equals CCX exactly and
    condition (R) admits it without any pairing.
    """
    q = math.pi / 4
    qc.ry(q, t)
    qc.cx(b, t)
    qc.ry(q, t)
    qc.cx(a, t)
    qc.ry(-q, t)
    qc.cx(b, t)
    qc.ry(-q, t)


# --------------------------------------------------------------------------- report
class SelectionReport(dict):
    """A plain dict subclass so the report is JSON-friendly but easy to print."""

    def summary(self) -> str:
        n_phase = len(self.get("phase_aware_admitted", []))
        n_approx = len(self.get("approx_admitted", []))
        eps = self.get("epsilon_spent_total", 0.0)
        return (
            f"sites_found={self['sites_found']} "
            f"applied={self['sites_applied']} rejected={self['sites_rejected']} "
            f"phase_aware={n_phase} approx={n_approx} eps_spent={eps:.4g} | "
            f"2q: {self['two_qubit_before']} -> {self['two_qubit_after']} | "
            f"infidelity: {self['infidelity_before']:.4g} -> {self['infidelity_after']:.4g} | "
            f"verified={self['verified']}"
        )


# --------------------------------------------------------------------------- selector
class ErrorBudgetSelector:
    """Select per-Toffoli decompositions minimizing an error-budget infidelity.

    Parameters
    ----------
    error_model : HardwareErrorModel, optional
        The cost model whose infidelity is minimized. Defaults to standard rates.
    verifier : ExactEquivalenceVerifier, optional
        The exact correctness gate. Defaults to a fresh verifier.
    allow_permutation : bool
        Passed through to the verifier when checking the final whole-circuit
        equivalence (the per-site checks use identity wiring).
    """

    def __init__(
        self,
        error_model: Optional[HardwareErrorModel] = None,
        verifier: Optional[ExactEquivalenceVerifier] = None,
        allow_permutation: bool = False,
        phase_aware: bool = True,
        epsilon: float = 0.0,
        semantics: str = "subroutine",
        window_pairs: bool = True,
        max_window_qubits: int = 12,
        elide_swaps: bool = False,
        simplify: Optional[str] = None,
        certificate: str = "exact",
        mc_vectors: int = 24,
    ):
        self.error_model = error_model or HardwareErrorModel()
        self.verifier = verifier or ExactEquivalenceVerifier()
        self._default_verifier = verifier is None                  # the C++ columns replace only the default verifier
        self.allow_permutation = allow_permutation
        # When True, admit relative-phase gadgets at standalone Toffolis whose phase
        # is provably UNOBSERVABLE on the reachable subspace (the new, more permissive
        # path), in addition to the exact-unitary compute/uncompute-pair path.
        self.phase_aware = phase_aware
        # SEMANTICS of correctness the output must satisfy.
        #  * "subroutine" (default): the emitted circuit equals the exact one as an
        #    operator on the input subspace up to ONE global phase, so it can be used
        #    coherently inside a larger algorithm (adders, oracles). Only conditions
        #    (C) compute/uncompute pairs and (R) exact-on-reachable-subspace are used.
        #  * "program": the circuit is a complete program ending in a
        #    computational-basis measurement. Additionally admits condition (U):
        #    standalone relative-phase gadgets whose phase the terminal measurement
        #    cannot see. NOT valid for subroutines (it changes the unitary).
        if semantics not in ("subroutine", "program"):
            raise ValueError(f"semantics must be 'subroutine' or 'program', got {semantics!r}")
        self.semantics = semantics
        # Condition (W): window-certified mirror pairs (exact segment check).
        self.window_pairs = window_pairs
        self.max_window_qubits = max_window_qubits
        # When True, SWAP gates are removed before the selection by relabelling the qubits that
        # the following gates act on. The returned circuit then computes the input circuit up to the
        # reported OUTPUT PERMUTATION: logical qubit i of the input circuit ends on wire
        # ``output_permutation[i]`` of the returned circuit (result["output_permutation"]). Input wires
        # are unchanged, so ``pinned_zero`` and ``input_space`` keep their meaning. Off by default.
        self.elide_swaps = elide_swaps
        # Optional general simplification AFTER the certified selection (final step): "zx" reduces the output with the
        # ZX-calculus (PyZX), "peephole" applies commutation-aware CNOT cancellation and two-qubit block resynthesis
        # (to the output and to the all-exact circuit), "zx+peephole" both. Every candidate must have a lower error budget
        # than the output of the selection AND pass the same whole-circuit certificate against the all-exact circuit; the
        # cheapest certified candidate is returned, otherwise the output of the selection is kept (fail closed).
        if simplify not in (None, "zx", "peephole", "zx+peephole"):
            raise ValueError("simplify must be None, 'zx', 'peephole' or 'zx+peephole'")
        self.simplify = simplify
        # Whole-circuit certificate. "exact" (default): deterministic (dense columns, blocks of difference, QCEC). "monte_carlo": opt-in
        # probabilistic certificate with ``mc_vectors`` random vectors on the input subspace (cppcert.mc_compare): a pair that is not
        # equal up to one global phase (Frobenius deviation > 1e-8) is accepted with probability at most
        # (0.1 e^0.9)^(mc_vectors - 1) (reported as verify_info["failure_probability"]); a counterexample is deterministic; an undecided
        # test falls through to the deterministic certificate. Much faster for wide / long circuits (O(K G 2^n) instead of O(G 4^n)).
        if certificate not in ("exact", "monte_carlo"):
            raise ValueError("certificate must be 'exact' or 'monte_carlo'")
        if mc_vectors < 2:
            raise ValueError("mc_vectors must be >= 2")
        self.certificate = certificate
        self.mc_vectors = mc_vectors
        # BOUNDED-APPROXIMATE-ON-REACHABLE knob. epsilon == 0.0 admits only candidates
        # that agree with CCX EXACTLY on the reachable subspace (e.g. a provably
        # constant control). epsilon > 0.0 additionally admits a cheaper
        # context-specialised candidate whose deviation min_theta ||(U_d - e^{i theta}
        # U_CCX) P||_op on the (soundly over-approximated) reachable subspace is
        # <= epsilon -- AND
        # only when the two-qubit-gate infidelity it SAVES exceeds the epsilon it
        # SPENDS (epsilon is charged into the circuit's error budget).
        if epsilon < 0.0:
            raise ValueError("epsilon must be >= 0")
        self.epsilon = epsilon

    # ----------------------------------------------------------------- baselines
    def decompose_exact_only(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """All-exact baseline: replace every CCX/MCX with the 6-CX exact gadget."""
        out = QuantumCircuit(circuit.num_qubits)
        for inst in circuit.data:
            self._emit_exact(out, circuit, inst)
        return out

    # -------------------------------------------------------------------- select
    @staticmethod
    def elide_swap_gates(circuit: QuantumCircuit):
        """Remove the SWAP gates of a circuit by relabelling. Returns (circuit without SWAPs, perm) where
        perm[i] is the wire of the returned circuit that carries logical qubit i at the end; the input wires are unchanged."""
        n = circuit.num_qubits
        pos = list(range(n))
        out = QuantumCircuit(n, circuit.num_clbits)
        out.global_phase = circuit.global_phase
        for inst in circuit.data:
            qs = [circuit.find_bit(q).index for q in inst.qubits]
            if inst.operation.name == "swap" and len(qs) == 2:
                pos[qs[0]], pos[qs[1]] = pos[qs[1]], pos[qs[0]]
                continue
            out.append(inst.operation, [out.qubits[pos[q]] for q in qs], [out.clbits[circuit.find_bit(c).index] for c in inst.clbits])
        return out, pos

    def select(self, circuit: QuantumCircuit, input_space="all_basis", pinned_zero=()) -> dict:
        """See ``_select_core``. With ``elide_swaps=True`` the SWAP gates are removed first and the result carries
        ``output_permutation`` (also in ``report``): the returned circuit equals the input circuit followed by the
        relabelling of the wires, i.e. logical qubit i is on wire ``output_permutation[i]``."""
        if not self.elide_swaps:
            res = self._select_core(circuit, input_space, pinned_zero)
        else:
            if circuit.num_clbits or any(g.operation.name in ("measure", "reset") for g in circuit.data):
                raise ValueError("select() takes unitary circuits: remove measurements, resets and classical bits first")
            stripped, perm = self.elide_swap_gates(circuit)
            res = self._select_core(stripped, input_space, pinned_zero)
            res["output_permutation"] = perm
            res["report"]["output_permutation"] = perm
            res["report"]["swaps_elided"] = sum(1 for g in circuit.data if g.operation.name == "swap")
        if self.simplify:
            self._simplify_stage(res, pinned_zero)
        return res

    def _simplify_stage(self, res: dict, pinned_zero) -> None:
        """Final step: cheaper candidates (ZX, peephole) are certified in order of increasing cost; the cheapest certified one wins."""
        em = self.error_model
        rep = res["report"]
        cur, exact = res["circuit"], res["exact"]
        mode = self.simplify
        info = {"mode": mode, "candidates": [], "chosen": None}
        rep["simplify"] = info
        cost = lambda c: (em.circuit_infidelity(c), em.two_qubit_count(c))
        base = cost(cur)
        cands = {}
        zx = None
        if "zx" in mode:
            from .zx_simplify import zx_simplify, zx_available
            info["zx_available"] = zx_available()
            zx = zx_simplify(cur) if info["zx_available"] else None
            if zx is not None:
                cands["zx"] = zx
        if "peephole" in mode:
            from .peephole import peephole
            for name, src in (("peephole", cur), ("peephole_exact", exact)) + ((("zx+peephole", zx),) if zx is not None else ()):
                try:
                    cands[name] = peephole(src)
                except Exception:
                    pass
        n = exact.num_qubits
        semantics = rep["semantics"]
        eps = rep.get("epsilon_spent_total", 0.0) or 0.0
        for name, cand in sorted(cands.items(), key=lambda kv: cost(kv[1])):
            row = {"name": name, "two_qubit": em.two_qubit_count(cand), "infidelity": em.circuit_infidelity(cand), "certified": None}
            info["candidates"].append(row)
            if cand.num_qubits != n or not cost(cand) < base:
                row["certified"] = "skipped (not cheaper)" if cand.num_qubits == n else "skipped (qubit count)"
                continue
            if n <= _CERT_MAX_QUBITS:
                from .subspace_check import circuit_columns
                cols = sorted(set(res["cert_inputs"]))
                ok, _, _ = certify_on_input_subspace(circuit_columns(exact, cols), circuit_columns(cand, cols),
                                                     list(range(len(cols))), mode=semantics, tolerance=eps)
                ok = bool(ok)
            else:
                from .subspace_check import qcec_certify_on_subspace
                ok, _ = qcec_certify_on_subspace(exact, cand, pinned_zero, observational=(semantics == "observational"))
            row["certified"] = ok
            if ok:
                res["circuit"] = cand
                info["chosen"] = name
                rep["two_qubit_after"] = em.two_qubit_count(cand)
                rep["infidelity_after"] = em.circuit_infidelity(cand)
                rep["certified_error_budget"] = em.circuit_infidelity(cand) + eps
                rep["fell_back_to_exact"] = False
                break

    def _verify_site(self, circuit: QuantumCircuit, exact: QuantumCircuit, actions: dict, candidate: dict, site):
        """Admission check of one compute/uncompute pair.

        The circuits built under ``actions`` (already verified equal to ``exact`` up to a phase) and under ``candidate`` differ only
        between the compute and the uncompute gate, so it is enough that this segment, on the qubits it touches, is equal up to one
        global phase in both builds: then the whole circuit is. Segments on more than ``max_window_qubits`` qubits are checked
        globally against the exact decomposition."""
        i, j = site.compute_idx, site.uncompute_idx
        touched = sorted({circuit.find_bit(q).index for inst in circuit.data[i:j + 1] for q in inst.qubits})
        if len(touched) <= self.max_window_qubits:
            pos = {q: k for k, q in enumerate(touched)}
            sub = QuantumCircuit(len(touched))
            for inst in circuit.data[i:j + 1]:
                sub.append(inst.operation, [sub.qubits[pos[circuit.find_bit(q).index]] for q in inst.qubits])
            loc = lambda acts: {k - i: a for k, a in acts.items() if i <= k <= j}
            ok, perm, info = self._verify_pair(self._build(sub, loc(actions)), self._build(sub, loc(candidate)))
            return ok, perm, {**info, "scope": "segment"}
        return self._verify_pair(exact, self._build(circuit, candidate))

    def _verify_pair(self, ref: QuantumCircuit, built: QuantumCircuit):
        """Equivalence up to one global phase on the full input space: C++ columns when decisive, otherwise the exact verifier."""
        n = ref.num_qubits
        if self._default_verifier and built.num_qubits == n and len(ref.data) > 12 and not self.allow_permutation:
            # the two builds differ only in the gadgets; a common middle that commutes with them is slid away and the gadgets are
            # compared on their own qubits (results cached by pattern): rigorous, see blockcert.py
            from .blockcert import block_certificate
            _ok, _bi = block_certificate(ref, built, max_qubits=self.max_window_qubits, cache=_BLOCK_CACHE, tol=1e-10, budget_frac=0.5)
            if _ok:
                return True, None, {"method": "block_diff", "certified_deviation": _bi.get("certified_deviation", 0.0)}
        if self._default_verifier and built.num_qubits == n <= _CERT_MAX_QUBITS:
            from .cppcert import compare_up_to_phase
            st, fro = compare_up_to_phase(ref, built, list(range(1 << n)))
            if st == "equal":
                return True, None, {"method": "cpp_columns", "certified_deviation": fro}
            if st == "different":
                return False, None, {"method": "cpp_columns", "reason": "not equal up to a global phase", "deviation": fro}
        return self.verifier.verify(ref, built, allow_permutation=False)

    def _select_core(self, circuit: QuantumCircuit, input_space="all_basis", pinned_zero=()) -> dict:
        """Choose decompositions to minimize infidelity; verify every substitution.

        Parameters
        ----------
        circuit : QuantumCircuit
        input_space : 'all_basis' | iterable of int
            The computational-basis inputs the whole circuit may be run on. Defaults
            to ``'all_basis'`` (every 2^n input). Pass a RESTRICTED set when the
            circuit is only ever fed a known sub-domain -- e.g. ancilla qubits known
            to start in |0>. A restricted input space shrinks the (soundly
            over-approximated) reachable subspace at each gate, which is what lets the
            control-drop / bounded-approximate candidates become admissible: a control
            can only be provably constant (or rarely set) once the inputs that pin or
            correlate it are known. Soundness is unaffected -- the caller asserts the
            circuit is never run outside ``input_space``; the reachable analysis then
            uses a sound over-approximation of the states reaching each gate.

        Returns a dict with keys:
          circuit  : the decomposed, verified-correct QuantumCircuit (2q basis)
          report   : SelectionReport (sites found/applied/rejected, 2q & infidelity
                     before/after, verified flag, total certified error budget,
                     approx_admitted + per-index epsilon spent)
        """
        if circuit.num_clbits or any(g.operation.name in ("measure", "reset") for g in circuit.data):
            raise ValueError("select() takes unitary circuits: remove measurements, resets and classical bits first")
        exact = self.decompose_exact_only(circuit)

        sites = find_relative_phase_safe_sites(circuit)
        applied: List[RelativePhaseSite] = []
        rejected: List[dict] = []

        # The per-index action map of committed specialised lowerings (see _build).
        # Start empty; add a site's (compute, uncompute) pair only if that
        # substitution verifies against the exact-only decomposition.
        actions: dict = {}

        for site in sites:
            candidate = dict(actions)
            candidate[site.compute_idx] = ("relphase",)
            candidate[site.uncompute_idx] = ("relphase",)
            ok, _perm, info = self._verify_site(circuit, exact, actions, candidate, site)
            if ok:
                actions = candidate
                applied.append(site)
            else:
                rejected.append({"site": site, "reason": info.get("reason"), "info": info})

        # ---- Condition (W): window-certified mirror pairs ------------------------
        # Pairs the structural detector rejects (a gate between them writes a
        # control or the target) can still cancel their phases, e.g. the mirrored
        # MAJ/UMA blocks of a Cuccaro adder. Each pair is admitted only if the
        # segment between them, with gadgets, equals the exact segment up to one
        # global phase (exact check on the qubits the segment touches).
        from .window_pairs import admit_window_pairs
        window_pairs_log: List[dict] = []
        if self.window_pairs:
            gad = {k for k, a in actions.items() if a[0] == "relphase"}
            gad2, window_pairs_log = admit_window_pairs(
                circuit, gad, append_relative_phase_ccx,
                max_window_qubits=self.max_window_qubits,
                accepted_windows=[(s_.compute_idx, s_.uncompute_idx) for s_ in applied])
            for k in gad2 - gad:
                actions[k] = ("relphase",)

        # ---- Condition (R) for relative-phase gadgets -----------------------------
        # A standalone Toffoli gets a 3-CX gadget when the gadget equals CCX on the
        # reachable subspace with one phase. The mirrored gadget does so for every
        # AND written into a clean ancilla and for its uncompute. Reachability is
        # exact (vectorised) for classical prefixes, else the sound over-approx.
        from .reach_local import local_reachable
        pinned_zero = tuple(pinned_zero)

        from .reach_local import PrefixReach
        tracker = PrefixReach(circuit, pinned_zero)

        def _local(idx, qbs):
            loc = (tracker.local(idx, qbs) if input_space == "all_basis" else None)
            if loc is None:
                loc = local_overapprox(circuit, idx, qbs, input_space=_space)
            return loc

        # Input space for the over-approximation: valid inputs with pinned qubits 0.
        _space = input_space
        if pinned_zero and input_space == "all_basis" and circuit.num_qubits <= 20:
            _pm = sum(1 << q for q in pinned_zero)
            _space = [x for x in range(1 << circuit.num_qubits) if not x & _pm]

        rphase_admitted: List[dict] = []
        locked = {k for k in actions}
        for w in window_pairs_log:
            locked.update(w["pair"])
        for s_ in sites:
            locked.update((s_.compute_idx, s_.uncompute_idx))
        for idx, inst in enumerate(circuit.data):
            if idx in locked:
                continue
            name = inst.operation.name.lower()
            qb = [circuit.find_bit(q).index for q in inst.qubits]
            if name not in _CCX_NAMES or len(qb) != 3:
                continue
            local = _local(idx, tuple(qb))
            for kind, U_g in (("relphase", _U_MARG), ("relphase_m", _U_MARG_M)):
                ok_r, dev_r, _ = _check_cached(U_g, _U_CCX, local, 0.0)
                if ok_r:
                    actions[idx] = (kind,)
                    rphase_admitted.append({"index": idx, "qubits": qb, "gadget": kind,
                                            "condition": "R", "reachable_local": len(local)})
                    break

        # ---- Condition (U), program semantics only --------------------------------
        # A standalone Toffoli (not in a compute/uncompute pair) can get the cheap
        # relative-phase gadget when its diagonal phase is provably invisible to the
        # terminal computational-basis measurement (is_phase_unobservable: the
        # forward cone of the phased qubits contains only basis permutations). This
        # changes the unitary, so it is used only with semantics="program" and the
        # final certificate is then the observational one.
        # Indices belonging to a verified compute/uncompute PAIR are soundness-locked:
        # their relative-phase gadgets cancel only as a matched pair, so the approx
        # path below must never touch them.
        paired_idx = set()
        for s in sites:
            paired_idx.add(s.compute_idx)
            paired_idx.add(s.uncompute_idx)
        for w in window_pairs_log:
            paired_idx.update(w["pair"])

        phase_admitted: List[dict] = []
        if self.phase_aware and self.semantics == "program":
            for idx, inst in enumerate(circuit.data):
                if idx in actions or idx in paired_idx:
                    continue
                name = inst.operation.name.lower()
                qb = [circuit.find_bit(q).index for q in inst.qubits]
                if name not in _CCX_NAMES or len(qb) != 3:
                    continue  # only plain 3-qubit CCX get the Margolus gadget here

                affected = default_affected_qubits(circuit, idx)
                # Gate 1: phase provably unobservable downstream?
                if not is_phase_unobservable(circuit, idx, affected):
                    rejected.append({
                        "site": ("standalone", idx),
                        "reason": "phase observable downstream",
                        "info": {"path": "phase_aware"}})
                    continue

                # Condition (U) of the paper: the forward-cone test above proves that
                # W D W^dagger is diagonal (every later gate touching the phased qubits
                # is a basis permutation), so the terminal computational-basis
                # measurement cannot see D for ANY input state, superpositions
                # included. No reachability argument is needed or used here.
                # (Earlier versions additionally ran a basis-state-wise
                # phase-insensitive check, which is blind to relative phases and was
                # applied with gate-level reachable states as circuit inputs; it is
                # removed because it neither adds nor justifies soundness.)
                actions = dict(actions)
                actions[idx] = ("relphase",)
                phase_admitted.append({
                    "index": idx, "qubits": qb, "condition": "U"})

        # ---- Condition (R) for control drops, and its bounded-approximate variant ---
        # For each still-exact standalone CCX, try the CONTROL-DROP specialisations
        # (drop-to-identity, drop control a -> CX(b,t), drop control b -> CX(a,t)).
        # Each candidate is admitted iff (1) its operator-norm deviation from CCX with
        # ONE phase on the span of the SOUND over-approximation of the reachable
        # subspace is <= epsilon (check_on_subspace), AND (2) the two-qubit-gate infidelity it SAVES strictly exceeds
        # the epsilon it SPENDS (epsilon is charged into the error budget). With
        # epsilon == 0 only EXACT-on-reachable drops (e.g. a provably-|0> control)
        # can fire. The deviation is measured against the 8x8 CCX unitary on the
        # projected reachable subspace; control_drop is never globally equivalent, so it is
        # admissible ONLY through this reachable-subspace check.
        # NOTE on scope: a standalone CCX already assigned the relative-phase gadget
        # by condition (U) above is RECONSIDERED here -- a control-drop may be
        # strictly cheaper (down to 0 or 1 two-qubit gates vs the gadget's 3) and just
        # as sound on the reachable subspace. Indices locked into a compute/uncompute
        # PAIR (``paired_idx``) are never touched: their gadgets cancel only as a pair.
        approx_admitted: List[dict] = []
        perturbed_targets: set = set()
        epsilon_per_index: dict = {}
        em = self.error_model
        # The infidelity of a circuit is 1 - (product over its gates of (1 - err)) for the model without idle / ZZ terms, so the
        # effect of replacing the lowering of ONE gate is a ratio of small factors; the whole circuit is not rebuilt for every Toffoli.
        multiplicative = (em.t1 is None and em.zz_hz <= 0
                          and type(em).circuit_infidelity is HardwareErrorModel.circuit_infidelity
                          and type(em).gate_error is HardwareErrorModel.gate_error)

        def _factor(inst_op, action, qb):
            if action is not None and action[0] == "control_drop":        # the kept controls are qubit numbers: map them to the 3-qubit sub-circuit
                action = ("control_drop", tuple(qb.index(k) for k in action[1]))
            sub = QuantumCircuit(3)
            sub.append(inst_op, [0, 1, 2])
            return 1.0 - em.circuit_infidelity(self._build(sub, {} if action is None else {0: action}))
        F_base = (1.0 - em.circuit_infidelity(self._build(circuit, actions))) if multiplicative else None
        for idx, inst in enumerate(circuit.data):
            if idx in paired_idx:
                continue
            name = inst.operation.name.lower()
            qb = [circuit.find_bit(q).index for q in inst.qubits]
            if name not in _CCX_NAMES or len(qb) != 3:
                continue
            a, b, t = qb

            # Reachable support at g, computed in the CURRENT circuit: a gate already
            # replaced by an approximate (deviation > 0) specialisation may have
            # changed the support of its target, which we model soundly by an H on
            # that target (free value) in the circuit used for reachability.
            reach_circ, idx_map = self._reach_circuit(circuit, perturbed_targets)
            local = None
            if not perturbed_targets and input_space == "all_basis":
                local = tracker.local(idx, (a, b, t))
            if local is None:
                local = local_overapprox(reach_circ, idx_map[idx], (a, b, t), input_space=_space)
                reachable = range(1 << circuit.num_qubits) if _space == "all_basis" else reachable_overapprox(reach_circ, idx_map[idx], input_space=_space)
            else:
                reachable = local
            base_inf = None if multiplicative else em.circuit_infidelity(self._build(circuit, actions))

            best = None  # (margin, candidate_actions, descriptor)
            for keep in ((), (a,), (b,)):
                local_keep = tuple({a: 0, b: 1}[k] for k in keep)
                U_d = _UD_CACHE.get(local_keep)
                if U_d is None:
                    U_d = _UD_CACHE[local_keep] = gadget_unitary(
                        lambda qc, x, y, z, lk=local_keep: append_control_drop(qc, x, y, z, lk))
                # Condition (R) / its bounded-approximate variant: one phase on the
                # reachable SUBSPACE, operator norm (never basis-state-wise).
                admissible, max_dev, vinfo = _check_cached(U_d, _U_CCX, local, self.epsilon)
                if not admissible:
                    continue
                cand_actions = dict(actions)
                cand_actions[idx] = ("control_drop", keep)
                # Budget test: infidelity SAVED by this candidate vs the current
                # selection must exceed the epsilon SPENT (= max_dev charged in).
                if multiplicative:
                    f_old = _factor(inst.operation, actions.get(idx), qb)
                    f_new = _factor(inst.operation, ("control_drop", keep), qb)
                    saved = F_base * (f_new / f_old - 1.0)
                else:
                    saved = base_inf - em.circuit_infidelity(self._build(circuit, cand_actions))
                margin = saved - max_dev
                if margin > 0 and (best is None or margin > best[0]):
                    best = (margin, cand_actions, {
                        "index": idx, "qubits": qb, "keep": list(keep),
                        "condition": "R" if max_dev == 0.0 else "R-eps",
                        "max_deviation": max_dev,
                        "infidelity_saved": saved,
                        "reachable_inputs": len(reachable),
                        "reachable_local": len(local)})

            if best is not None:
                if multiplicative:
                    F_base *= _factor(inst.operation, best[1][idx], qb) / _factor(inst.operation, actions.get(idx), qb)
                actions = best[1]
                approx_admitted.append(best[2])
                epsilon_per_index[idx] = best[2]["max_deviation"]
                if best[2]["max_deviation"] > 0.0:
                    perturbed_targets.add((idx, t))
            elif self.epsilon > 0.0:
                rejected.append({
                    "site": ("approx", idx),
                    "reason": "no control-drop within epsilon with positive budget margin",
                    "info": {"path": "approx"}})

        selected = self._build(circuit, actions)
        epsilon_spent_total = sum(epsilon_per_index.values())

        # Final whole-circuit certification, on the input subspace P_in:
        #
        #  * No (R)/(U)/control-drop admission and no pinned qubits: the selected
        #    circuit must be exactly unitary-equivalent to exact-only (strict verifier).
        #
        #  * Otherwise, subroutine semantics: U_sel P_in = e^{i theta} U_ex P_in with
        #    ONE phase (certify_on_input_subspace(mode="subroutine"), or QCEC with the
        #    pinned qubits declared as ancillas above 12 qubits).
        #
        #  * If some substitution relied on condition (U) (program semantics only):
        #    observational equivalence, i.e. U_sel P_in = Phi U_ex P_in for a diagonal
        #    unitary Phi, which is equality of the computational-basis output
        #    distributions for every input state on span(P_in), superpositions
        #    included (mode="observational").
        #
        #  * With bounded-approximate (control-drop, epsilon > 0) admissions the same
        #    statements are certified up to the operator-norm tolerance epsilon_spent,
        #    which the error budget accounts for.
        # The whole-circuit certification is on the circuit's INPUT domain -- the set
        # of computational-basis states the caller asserts the circuit is run on
        # (``input_space``; the full 2^n space by default). We feed those as inputs to
        # both ``exact`` and ``selected``. (For ``all_basis`` this is the full space;
        # for a restricted domain it is exactly the inputs the substitutions relied
        # on.) Using ``reachable_overapprox`` at gate index 0 yields a sound superset
        # of that domain.
        if input_space == "all_basis":
            cert_inputs = range(1 << circuit.num_qubits)           # the whole space; a range, not a set of 2^n integers
        else:
            cert_inputs = reachable_overapprox(circuit, 0, input_space=input_space)
        if pinned_zero:
            pm = 0
            for q in pinned_zero:
                pm |= 1 << q
            cert_inputs = [x for x in cert_inputs if not (x & pm)]
        # Semantics actually guaranteed: subroutine equivalence (one global phase on
        # the whole input subspace) unless some substitution relied on condition (U),
        # in which case observational equivalence (terminal basis measurement).
        semantics = "observational" if phase_admitted else "subroutine"
        # Certificate by blocks of difference: equal up to ONE phase on the FULL input space, hence on every input subspace; it
        # needs no dense propagation of the whole circuit (see blockcert.py). Undecided -> the dense / QCEC certificates below.
        _mc = (None, None)
        if (self.certificate == "monte_carlo" and self._default_verifier and not self.allow_permutation and semantics == "subroutine"
                and epsilon_spent_total == 0.0 and selected.num_qubits == exact.num_qubits <= 26 and len(cert_inputs) > 0
                and not _same_circuit(exact, selected)):
            from .cppcert import mc_compare, mc_failure_probability
            from .blockcert import block_certificate
            _k = self.mc_vectors

            def _mc_block(la, lb, cols, tol):
                st, _i = mc_compare(la, lb, range(1 << la.num_qubits), k=_k, eps=tol)
                return st, (tol if st == "equal" else None)
            # blocks of difference tested with random vectors on the qubits they touch: g K 2^t instead of g 4^t per block
            # (only for wide circuits: up to 16 qubits the random test of the whole circuit costs K G 2^n, which is already small)
            _okb, _ib = (False, {})
            if exact.num_qubits > 16:
                _okb, _ib = block_certificate(exact, selected, max_qubits=22, compare=_mc_block, work_cost=lambda g, t: g * _k * 2.0 ** t)
            if _okb:
                _mc = ("equal", {**_ib, "method": "monte_carlo_blocks", "vectors": _k,
                                 "failure_probability": min(1.0, max(1, _ib.get("distinct", 1)) * mc_failure_probability(_k))})
            else:
                _mc = mc_compare(exact, selected, cert_inputs, k=_k)
        _blk = None
        if (_mc[0] is None and self._default_verifier and not self.allow_permutation and semantics == "subroutine"
                and epsilon_spent_total == 0.0 and selected.num_qubits == exact.num_qubits and not _same_circuit(exact, selected)):
            import os as _os0
            if _os0.environ.get("QCPT_BLOCK_CERT", "1") != "0":
                from .blockcert import block_certificate
                _ok_b, _info_b = block_certificate(exact, selected, max_qubits=self.max_window_qubits)
                if _ok_b:
                    _blk = _info_b
        if _same_circuit(exact, selected):
            verified, perm, vinfo = True, None, {"method": "identical_circuits", "certification": "exact_unitary"}
        elif _mc[0] == "equal":
            verified, perm = True, None
            vinfo = {**_mc[1], "certification": "input_subspace_subroutine_probabilistic"}
        elif _mc[0] == "different":
            verified, perm = False, None
            vinfo = {**_mc[1], "reason": "counterexample from a random vector", "certification": "input_subspace_subroutine_probabilistic"}
        elif _blk is not None:
            verified, perm = True, None
            vinfo = {**_blk, "certification": "exact_unitary" if not (approx_admitted or phase_admitted or rphase_admitted or pinned_zero)
                     else f"input_subspace_{semantics}"}
        elif not (approx_admitted or phase_admitted or rphase_admitted or pinned_zero):
            _st, _fro = (None, None)
            if self._default_verifier and not self.allow_permutation and selected.num_qubits == exact.num_qubits <= _CERT_MAX_QUBITS:
                from .cppcert import compare_up_to_phase
                _st, _fro = compare_up_to_phase(exact, selected, list(range(1 << exact.num_qubits)))
            if _st == "equal":
                verified, perm, vinfo = True, None, {"method": "cpp_columns", "certified_deviation": _fro}
            elif _st == "different":
                verified, perm, vinfo = False, None, {"method": "cpp_columns", "reason": "not equal up to a global phase", "deviation": _fro}
            else:
                verified, perm, vinfo = self.verifier.verify(
                    exact, selected, allow_permutation=self.allow_permutation
                )
            vinfo = {**vinfo, "certification": "exact_unitary"}
        elif max(exact.num_qubits, selected.num_qubits) <= _CERT_MAX_QUBITS \
                and selected.num_qubits == exact.num_qubits:
            import os as _os
            _st, _cpp = (None, None)
            if self._default_verifier and semantics == "subroutine" and epsilon_spent_total == 0.0 and cert_inputs:
                from .cppcert import compare_up_to_phase
                _st, _cpp = compare_up_to_phase(exact, selected, sorted(set(cert_inputs)))
            if _st == "equal":                                              # certified by the C++ columns (equal up to one phase, deviation <= 1e-8)
                verified, cert_dev, vinfo = True, _cpp, {"method": "cpp_columns"}
            elif _st == "different":                                        # not equal up to one phase (deviation > 1e-5): fail closed below
                verified, cert_dev, vinfo = False, _cpp, {"method": "cpp_columns", "reason": "not equal up to a global phase"}
            elif _os.environ.get("QCPT_FAST_CERT", "1") == "1":              # only the columns of the input subspace, never the full 2^n x 2^n matrix
                from .subspace_check import circuit_columns
                _cols = sorted(set(cert_inputs))
                U_ex = circuit_columns(exact, _cols)
                U_sel = circuit_columns(selected, _cols)
                verified, cert_dev, vinfo = certify_on_input_subspace(
                    U_ex, U_sel, list(range(len(_cols))), mode=semantics,
                    tolerance=epsilon_spent_total)
            else:
                from qiskit.quantum_info import Operator
                U_ex = Operator(exact).data
                U_sel = Operator(selected).data
                verified, cert_dev, vinfo = certify_on_input_subspace(
                    U_ex, U_sel, cert_inputs, mode=semantics,
                    tolerance=epsilon_spent_total)
            verified = bool(verified)
            perm = None
            vinfo = {**vinfo, "certification": f"input_subspace_{semantics}",
                     "certified_deviation": cert_dev}
        else:
            # Too wide for a dense certificate: decision diagrams (QCEC), with the
            # pinned qubits declared as ancillas initialised to |0>, i.e. the
            # equivalence on the input subspace. Fail closed: if the certificate
            # cannot be established, the all-exact circuit is returned below.
            from .subspace_check import qcec_certify_on_subspace
            ok_q, vinfo = qcec_certify_on_subspace(exact, selected, pinned_zero,
                                                   observational=bool(phase_admitted))
            verified, perm = (True if ok_q else False), None
            vinfo = {**vinfo, "certification": "qcec_input_subspace"}

        # Fail closed: a failed whole-circuit certificate must never let an
        # uncertified circuit out. Fall back to the all-exact decomposition.
        fell_back = False
        if verified is not True:   # False or undecided (None): fail closed
            selected = exact
            fell_back = True
            applied, phase_admitted, approx_admitted, rphase_admitted = [], [], [], []
            actions = {}
            epsilon_per_index, epsilon_spent_total = {}, 0.0
            verified, perm, vinfo = True, None, {
                **vinfo, "fallback": "exact_only_after_failed_certificate"}

        # The certified error budget includes the epsilon spent on approximate
        # admissions: the on-hardware infidelity of the cheaper circuit PLUS the
        # algorithmic deviation we deliberately introduced on the reachable subspace.
        certified_budget = em.circuit_infidelity(selected) + epsilon_spent_total
        report = SelectionReport(
            sites_found=len(sites),
            sites_applied=len(applied),
            sites_rejected=len(rejected),
            applied_sites=[tuple(s) for s in applied],
            rejected_sites=rejected,
            phase_aware_admitted=phase_admitted,
            window_pairs_admitted=window_pairs_log,
            rphase_admitted=rphase_admitted,
            pinned_zero=list(pinned_zero),
            approx_admitted=approx_admitted,
            epsilon=self.epsilon,
            epsilon_per_index=epsilon_per_index,
            epsilon_spent_total=epsilon_spent_total,
            two_qubit_before=em.two_qubit_count(exact),
            two_qubit_after=em.two_qubit_count(selected),
            infidelity_before=em.circuit_infidelity(exact),
            infidelity_after=em.circuit_infidelity(selected),
            certified_error_budget=certified_budget,
            verified=verified if verified is None else bool(verified),
            semantics=semantics,
            fell_back_to_exact=fell_back,
            verify_info=vinfo,
            output_permutation=perm,
        )
        return {"circuit": selected, "report": report, "actions": dict(actions),
                "exact": exact, "cert_inputs": cert_inputs}

    # ------------------------------------------------------------- reachability
    @staticmethod
    def _reach_circuit(circuit: QuantumCircuit, perturbed_targets):
        """Circuit used for reachability, plus an index map original -> new.

        After an approximate (deviation > 0) specialisation at gate i with target t,
        the value of t after i is no longer the CCX value. We model this soundly by
        inserting an H on t right after i: the over-approximation then treats t (and
        everything it later influences) as free.
        """
        if not perturbed_targets:
            return circuit, list(range(len(circuit.data)))
        after = {}
        for i, t in perturbed_targets:
            after.setdefault(i, []).append(t)
        out = QuantumCircuit(*circuit.qregs, *circuit.cregs)
        idx_map = []
        for i, inst in enumerate(circuit.data):
            idx_map.append(len(out.data))
            out.append(inst.operation, inst.qubits, inst.clbits)
            for t in after.get(i, []):
                out.h(out.qubits[t])
        return out, idx_map

    # ------------------------------------------------------------------ builders
    def _build(self, circuit: QuantumCircuit, actions: dict) -> QuantumCircuit:
        """Decompose ``circuit`` under a per-index ``actions`` map.

        ``actions`` maps an instruction index to a chosen specialised lowering:

          * ``("relphase",)``           -> the 3-CX relative-phase (Margolus) gadget.
          * ``("control_drop", keep)``  -> the control-dropped specialisation
            (identity if ``keep == ()``, else a single CX from the kept control).

        Any index NOT in ``actions`` gets its exact lowering. For backward
        compatibility a plain ``set`` of indices is also accepted and read as "apply
        the relative-phase gadget at each of these indices".
        """
        if isinstance(actions, (set, frozenset)):
            actions = {idx: ("relphase",) for idx in actions}
        out = QuantumCircuit(circuit.num_qubits)
        for idx, inst in enumerate(circuit.data):
            action = actions.get(idx)
            if action is None:
                self._emit_exact(out, circuit, inst)
                continue
            qb = [circuit.find_bit(q).index for q in inst.qubits]
            kind = action[0]
            # Optional orientation flag ("swap"): the two controls exchange roles, so
            # the CX pair that the gadget uses twice moves to the other control.
            a_, b_ = (qb[1], qb[0]) if len(action) > 1 and action[1] == "swap" else (qb[0], qb[1])
            if kind == "relphase":
                append_relative_phase_ccx(out, a_, b_, qb[2])
            elif kind == "relphase_m":
                append_relative_phase_ccx_mirror(out, a_, b_, qb[2])
            elif kind == "control_drop":
                append_control_drop(out, qb[0], qb[1], qb[2], action[1])
            else:
                raise ValueError(f"unknown action {action!r}")
        return out

    def _emit_exact(self, out: QuantumCircuit, src: QuantumCircuit, inst) -> None:
        """Emit the exact lowering of one source instruction onto ``out``."""
        name = inst.operation.name.lower()
        qb = [src.find_bit(q).index for q in inst.qubits]
        if name in ("ccx",) or (name in ("mcx", "mcx_gray") and len(qb) == 3):
            append_exact_ccx(out, qb[0], qb[1], qb[2])
        elif name in ("mcx", "mcx_gray"):
            # Generic multi-control: fall back to qiskit's own decomposition to a
            # CX-based basis. Rare in these tests; kept correct rather than minimal.
            tmp = QuantumCircuit(out.num_qubits)
            tmp.mcx(qb[:-1], qb[-1])
            out.compose(tmp.decompose().decompose(), inplace=True)
        else:
            # Pass through any non-CCX gate verbatim (cx, x, h, ry, ...).
            out.append(inst.operation, [out.qubits[i] for i in qb])


_U_MARG = _gadget_matrix(append_relative_phase_ccx)
_U_MARG_M = _gadget_matrix(append_relative_phase_ccx_mirror)

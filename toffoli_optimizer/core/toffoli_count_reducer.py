"""Toffoli-COUNT reduction (multi-controlled gates kept ATOMIC).

This is the capability the original QCPT pipeline lacks. QCPT reduces
*depth* by decomposing each Toffoli into a fixed CX+1q gadget -- the number of
multi-controlled (CCX/MCX/CCZ) operations it emits is therefore at least the
number it started with. Here we go the other way: we treat every CCX / CCZ / MCX
as an ATOMIC, indivisible, expensive primitive and try to reduce HOW MANY of them
the circuit contains, never decomposing one into CX + 1q gates.

Every rewrite is checked with the committed `ExactEquivalenceVerifier` over the
local window it touches BEFORE it is applied; a rewrite that does not verify
equivalent is rejected and the circuit is left unchanged. Rewrites are iterated
to a fixpoint.

Implemented count-reducing rewrites
-----------------------------------
* INVERSE-PAIR ANNIHILATION
  Two multi-controlled gates with identical (control-set, target) and nothing
  touching any of their qubits in between are self-inverse and cancel; both are
  removed. Commutation through qubit-disjoint gates is used to bring a pair
  adjacent so the cancellation is exposed.

* EQUAL-CONTROL FAN-OUT / CSE (compute-once fan-out)
  A run of CCX gates that share the SAME (unordered) control pair but hit
  different targets -- CCX(a,b,t1), CCX(a,b,t2), ... -- recomputes the boolean
  ``a & b`` once per target. With one clean ancilla we can compute ``a & b`` into
  the ancilla a single time (1 CCX), fan it out with cheap CX gates to every
  target, and uncompute (1 CCX). That turns ``k`` Toffolis into ``2`` Toffolis
  (plus k CX). It is applied only when it strictly lowers the atomic count AND
  verifies equivalent.

* WINDOWED CROSS-BOUNDARY ANNIHILATION
  A small windowed search that commutes a multi-controlled gate past Clifford /
  disjoint gates inside a bounded window to expose cancellations a strictly
  greedy adjacency pass misses. Each candidate is verified before applying.

* EXPAND-TO-CANCEL PEEPHOLE (superoptimizer idea)
  A bounded-window pass that does NOT insist every intermediate rewrite shrink the
  block. It enumerates a small set of equivalence-preserving EXPANSIONS of a
  window (e.g. the fan-out conjugation identity
  ``CCX(a,b,t); CX(t,u); CCX(a,b,t) == CX(t,u); CCX(a,b,u)`` which pushes the
  controlled action onto the moved target, eliminating one CCX), simplifies the
  result with the existing annihilation machinery, and keeps it ONLY if the net
  atomic Toffoli count STRICTLY drops AND the window verifies equivalent. A strict
  adjacency/commute walk misses these because the useful intermediate is not
  smaller gate-by-gate.

* PERMUTATION-AWARE REWRITING (opt-in)
  Output-wire relabeling is free for a downstream consumer, so a count-reducing
  rewrite that verifies only up to a NON-identity output-wire permutation is still
  usable -- PROVIDED the permutation is tracked. With ``allow_permutation=True``
  such a rewrite is applied AND the implied wire relabeling is propagated to every
  downstream gate, with the local permutation composed into a cumulative
  output-wire permutation exposed in ``report['output_permutation']``. A final
  whole-circuit guard re-verifies (up to that exact cumulative permutation) so the
  returned circuit, read through the reported permutation, computes the SAME
  function as the input.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from qiskit import QuantumCircuit

from .equivalence_verifier import ExactEquivalenceVerifier

# Multi-controlled gates we treat as the atomic, scored primitives.
_ATOMIC_NAMES = {"ccx", "ccz", "mcx", "mcx_gray", "mcphase", "mcp"}


def _gate_name(inst) -> str:
    return inst.operation.name.lower()


def _is_atomic(inst) -> bool:
    return _gate_name(inst) in _ATOMIC_NAMES


def count_toffoli(circuit: QuantumCircuit) -> int:
    """Number of atomic multi-controlled gates (CCX/CCZ/MCX...) in the circuit."""
    return sum(1 for inst in circuit.data if _is_atomic(inst))


class ToffoliCountReducer:
    """Reduce the NUMBER of atomic multi-controlled gates, keeping each atomic.

    Usage::

        reducer = ToffoliCountReducer()
        out = reducer.reduce_toffoli_count(circuit)
        print(reducer.report)
    """

    def __init__(self, verifier: Optional[ExactEquivalenceVerifier] = None,
                 enable_fanout_cse: bool = True,
                 enable_expand_to_cancel: bool = True,
                 allow_permutation: bool = False,
                 window: int = 8,
                 verbose: bool = False):
        self.verifier = verifier or ExactEquivalenceVerifier()
        self.enable_fanout_cse = enable_fanout_cse
        self.enable_expand_to_cancel = enable_expand_to_cancel
        self.allow_permutation = allow_permutation
        self.window = window
        self.verbose = verbose
        self.report: dict = {}
        # Cumulative output-wire permutation {original_wire: final_wire} accrued by
        # permutation-aware rewrites during a single reduce_toffoli_count call.
        self._cum_perm: dict = {}

    # ------------------------------------------------------------------ public
    def reduce_toffoli_count(self, circuit: QuantumCircuit,
                             allow_permutation: Optional[bool] = None) -> QuantumCircuit:
        """Return an equivalent circuit with the same-or-fewer atomic gates.

        The returned circuit is guaranteed (by `ExactEquivalenceVerifier`) to be
        functionally equivalent to the input -- every individual rewrite was
        verified over its local window before being applied, and the chain of
        verified-equivalent rewrites is equivalence-preserving end to end.

        When ``allow_permutation`` is true (per-call override of the constructor
        flag), a count-reducing rewrite that verifies only up to a non-identity
        OUTPUT-WIRE permutation may be accepted; the implied wire relabeling is
        propagated to all downstream gates and composed into a cumulative
        permutation reported in ``report['output_permutation']``. The reduced
        circuit, read through that permutation, computes the same function as the
        input. When false (default) the strict current behaviour is preserved and
        the reported permutation is the identity.
        """
        if allow_permutation is None:
            allow_permutation = self.allow_permutation
        circuit = self._normalize(circuit)                    # default registers q and c (named registers are flattened)
        self._ncl = circuit.num_clbits
        before = count_toffoli(circuit)
        applied: List[str] = []
        n = circuit.num_qubits
        # cumulative output-wire permutation {original_wire: final_wire}; identity
        self._cum_perm = {i: i for i in range(n)}

        current = circuit.copy()
        changed = True
        rounds = 0
        max_rounds = 64
        while changed and rounds < max_rounds:
            changed = False
            rounds += 1

            new, did = self._pass_inverse_pair(current)
            if did:
                current = new
                applied.append("inverse_pair_annihilation")
                changed = True
                continue

            new, did = self._pass_windowed_cross_boundary(current)
            if did:
                current = new
                applied.append("windowed_cross_boundary_annihilation")
                changed = True
                continue

            if self.enable_expand_to_cancel:
                new, did, perm = self._pass_expand_to_cancel(current, allow_permutation)
                if did:
                    current = new
                    if perm is not None and not self._is_identity_perm(perm):
                        self._compose_cumulative(perm)
                        applied.append(f"expand_to_cancel(perm={perm})")
                    else:
                        applied.append("expand_to_cancel")
                    changed = True
                    continue

            if self.enable_fanout_cse:
                new, did, info = self._pass_fanout_cse(current)
                if did:
                    current = new
                    applied.append(f"fanout_cse(controls={info})")
                    changed = True
                    continue

        # Final whole-circuit safety check: the cumulative result must still be
        # equivalent to the original input -- strictly when no permutation was
        # tracked, and up to the EXACT cumulative permutation otherwise. This is
        # defence in depth on top of the per-window checks: it must reproduce the
        # cumulative permutation we accrued, or we revert to the input.
        cum_perm = dict(self._cum_perm)
        identity_perm = {i: i for i in range(n)}
        used_permutation = cum_perm != identity_perm
        equivalent = True
        try:
            ok, vperm, _info = self.verifier.verify(
                circuit, current, allow_permutation=allow_permutation)
            equivalent = bool(ok)
            # If permutation tracking was used, the verifier's reported permutation
            # MUST match the cumulative permutation we propagated. If it does not,
            # our bookkeeping is unsound for this circuit -> reject.
            if equivalent and used_permutation:
                if vperm is None or dict(vperm) != cum_perm:
                    equivalent = False
                    _info = {"reason": "cumulative permutation mismatch",
                             "reported": dict(vperm) if vperm else None,
                             "tracked": cum_perm}
        except Exception as exc:  # pragma: no cover - defensive
            equivalent = False
            _info = {"reason": f"final verify raised {exc!r}"}
        if not equivalent:
            # Should never happen if every step verified; bail to the input.
            current = circuit.copy()
            applied.append("REVERTED:final_verify_failed")
            cum_perm = dict(identity_perm)

        after = count_toffoli(current)
        # Detect shareable equal-control runs even when no CSE was applied (report only).
        shareable = self._detect_shareable_runs(circuit)
        self.report = {
            "ccx_before": before,
            "ccx_after": after,
            "reduced": before - after,
            "rounds": rounds,
            "rewrites_applied": applied,
            "all_rewrites_verified": equivalent,
            "shareable_equal_control_runs": shareable,
            "output_permutation": cum_perm,
        }
        if self.verbose:
            print(self.report)
        return current

    # ---- permutation bookkeeping helpers ----------------------------------
    @staticmethod
    def _is_identity_perm(perm: dict) -> bool:
        return all(k == v for k, v in perm.items())

    def _compose_cumulative(self, local_perm: dict) -> None:
        """Compose a local output-wire permutation into the cumulative one.

        ``local_perm`` maps {pre_rewrite_wire: post_rewrite_wire}: it tells how the
        wires of the circuit *just before* this rewrite map onto the wires of the
        circuit *after* it. The cumulative permutation maps original-input wires to
        their current location, so we re-route each cumulative image through the new
        local map: cum'[orig] = local_perm[cum[orig]].
        """
        self._cum_perm = {orig: local_perm.get(cur, cur)
                          for orig, cur in self._cum_perm.items()}

    # =============================================================== internals
    @staticmethod
    def _qidx(circuit: QuantumCircuit, inst) -> List[int]:
        return [circuit.find_bit(q).index for q in inst.qubits]

    @staticmethod
    def _qset(circuit: QuantumCircuit, inst) -> set:
        return {circuit.find_bit(q).index for q in inst.qubits}

    @staticmethod
    def _atomic_signature(circuit: QuantumCircuit, inst) -> Tuple[frozenset, int, str]:
        """(controls-set, target, name) -- the identity used for annihilation.

        For CCX/MCX the target is the last qubit and the controls are the rest.
        For CCZ/MCP (phase gates) every qubit is symmetric, so we record the full
        qubit set as 'controls' with target = -1.
        """
        name = _gate_name(inst)
        idx = [circuit.find_bit(q).index for q in inst.qubits]
        if name in ("ccz", "ccp", "mcphase", "mcp"):
            return frozenset(idx), -1, name
        # ccx / mcx style: last qubit is target
        return frozenset(idx[:-1]), idx[-1], name

    @staticmethod
    def _normalize(circuit: QuantumCircuit) -> QuantumCircuit:
        """Circuit on the default registers q and c (a copy, qubits and clbits matched by index, if it has other registers)."""
        if [r.name for r in circuit.qregs] in ([], ["q"]) and [r.name for r in circuit.cregs] in ([], ["c"]):
            return circuit
        qc = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
        qc.global_phase = circuit.global_phase
        for inst in circuit.data:
            qc.append(inst.operation, [qc.qubits[circuit.find_bit(q).index] for q in inst.qubits],
                      [qc.clbits[circuit.find_bit(c).index] for c in inst.clbits])
        return qc

    def _build_from_instrs(self, n_qubits: int, instrs) -> QuantumCircuit:
        qc = QuantumCircuit(n_qubits, getattr(self, "_ncl", 0) if any(i.clbits for i in instrs) else 0)
        for inst in instrs:
            qc.append(inst.operation, inst.qubits, inst.clbits)
        return qc

    def _build_from_tuples(self, n_qubits: int, tuples) -> QuantumCircuit:
        """Build a circuit from a list of (operation, qubit_indices, clbits)."""
        qc = QuantumCircuit(n_qubits)
        for op, qidx, cl in tuples:
            qc.append(op, [qc.qubits[k] for k in qidx], cl)
        return qc

    # ---- verification helper over a local window --------------------------
    def _verify_window(self, orig_instrs, new_instrs, n_qubits) -> bool:
        """Verify two instruction lists (over the SAME n_qubits) are equivalent.

        Used to validate a local rewrite before committing it. The window
        circuits use the full qubit register so qubit indices line up.
        """
        a = self._build_from_instrs(n_qubits, orig_instrs)
        b = self._build_from_instrs(n_qubits, new_instrs)
        try:
            ok, _perm, _info = self.verifier.verify(a, b)
        except Exception:
            return False
        return bool(ok)

    # ---- PASS 1: inverse-pair annihilation (with commutation) -------------
    def _pass_inverse_pair(self, circuit: QuantumCircuit):
        """Find the first pair of identical atomic gates that can be brought
        adjacent by commuting through qubit-disjoint gates, then cancel them."""
        data = list(circuit.data)
        n = circuit.num_qubits

        for i in range(len(data)):
            if not _is_atomic(data[i]):
                continue
            sig_i = self._atomic_signature(circuit, data[i])
            qs_i = self._qset(circuit, data[i])
            # scan forward; commute past gates disjoint from this gate's qubits
            for j in range(i + 1, len(data)):
                qs_j = self._qset(circuit, data[j])
                if qs_i.isdisjoint(qs_j):
                    # disjoint -> commutes freely, keep scanning
                    continue
                # j touches our qubits: it must be the cancelling partner
                if _is_atomic(data[j]) and self._atomic_signature(circuit, data[j]) == sig_i:
                    # verify the local window (everything from i..j) with the
                    # pair removed equals the original window.
                    window = data[i:j + 1]
                    reduced = [data[k] for k in range(i, j + 1) if k != i and k != j]
                    if self._verify_window(window, reduced, n):
                        new_data = data[:i] + reduced + data[j + 1:]
                        return self._rebuild(circuit, new_data), True
                # touches our qubits but is not the partner -> blocked, stop
                break
        return circuit, False

    # ---- PASS 2: windowed cross-boundary annihilation ---------------------
    def _pass_windowed_cross_boundary(self, circuit: QuantumCircuit):
        """Bounded-window search: for each atomic gate, look ahead up to
        `self.window` instructions for an identical atomic gate, and try to
        cancel the pair by deleting both *and* verifying the whole window still
        matches. This catches cancellations where the intervening gates commute
        in aggregate even if a strict gate-by-gate adjacency walk gets blocked.
        """
        data = list(circuit.data)
        n = circuit.num_qubits

        for i in range(len(data)):
            if not _is_atomic(data[i]):
                continue
            sig_i = self._atomic_signature(circuit, data[i])
            hi = min(len(data), i + 1 + self.window)
            for j in range(i + 1, hi):
                if not _is_atomic(data[j]):
                    continue
                if self._atomic_signature(circuit, data[j]) != sig_i:
                    continue
                window = data[i:j + 1]
                reduced = [data[k] for k in range(i, j + 1) if k != i and k != j]
                if self._verify_window(window, reduced, n):
                    new_data = data[:i] + reduced + data[j + 1:]
                    return self._rebuild(circuit, new_data), True
        return circuit, False

    # ---- verification helper returning the permutation --------------------
    def _verify_window_perm(self, a, b, n_qubits, allow_permutation):
        """Like `_verify_window` but returns (is_equivalent, perm).

        ``a`` and ``b`` are pre-built QuantumCircuits over the same n_qubits.
        ``perm`` is the verifier's {orig_wire: rewritten_wire} map on success
        (identity when the rewrite is strictly equivalent), or None on failure.
        """
        try:
            ok, perm, _info = self.verifier.verify(
                a, b, allow_permutation=allow_permutation)
        except Exception:
            return False, None
        if not ok:
            return False, None
        if perm is None:
            perm = {i: i for i in range(n_qubits)}
        return True, dict(perm)

    @staticmethod
    def _relabel_instr(circuit, inst, wire_map):
        """Return (operation, new_qubit_indices) for `inst` with wires relabeled
        through ``wire_map`` (identity for wires not present)."""
        idx = [circuit.find_bit(q).index for q in inst.qubits]
        return inst.operation, [wire_map.get(k, k) for k in idx], list(inst.clbits)

    # ---- PASS: EXPAND-TO-CANCEL peephole ----------------------------------
    def _pass_expand_to_cancel(self, circuit: QuantumCircuit, allow_permutation):
        """Bounded-window superoptimizer pass.

        For each short window of gates, enumerate a small set of equivalence-
        preserving EXPANSION rewrites (which may momentarily NOT shrink the block),
        simplify with the existing annihilation machinery, and keep the result only
        if the net atomic count over the window STRICTLY drops AND the window
        verifies equivalent (up to permutation when allowed). If the accepted
        rewrite is valid only up to a non-identity output-wire permutation, the
        permutation is applied to every downstream gate and the local permutation
        is returned so the caller can compose it cumulatively.

        Returns (circuit, did_apply, local_perm_or_None).
        """
        data = list(circuit.data)
        n = circuit.num_qubits

        # Window length is small; the canonical identities below span 3 atomic-ish
        # gates, so a window of up to 5 instructions is plenty.
        max_w = min(self.window, 6)
        for i in range(len(data)):
            for w in range(2, max_w + 1):
                j = i + w
                if j > len(data):
                    break
                window = data[i:j]
                base = count_toffoli(self._build_from_instrs(n, window))
                if base == 0:
                    continue
                for cand in self._enumerate_expansions(
                        circuit, window, n, allow_permutation):
                    simplified = self._simplify_window(cand, n)
                    if count_toffoli(self._build_from_tuples(n, simplified)) >= base:
                        continue
                    ok, perm = self._verify_window_perm(
                        self._build_from_instrs(n, window),
                        self._build_from_tuples(n, simplified),
                        n, allow_permutation)
                    if not ok:
                        continue
                    if not allow_permutation and not self._is_identity_perm(perm):
                        continue
                    # Build the new full circuit: prefix, the simplified window,
                    # then the downstream gates relabeled through the local perm.
                    new = QuantumCircuit(n)
                    for inst in data[:i]:
                        new.append(inst.operation, inst.qubits, inst.clbits)
                    for op, qidx, cl in simplified:
                        new.append(op, [new.qubits[k] for k in qidx], cl)
                    apply_perm = perm if not self._is_identity_perm(perm) else None
                    for inst in data[j:]:
                        if apply_perm is not None:
                            op, qidx, cl = self._relabel_instr(circuit, inst, apply_perm)
                            new.append(op, [new.qubits[k] for k in qidx], cl)
                        else:
                            new.append(inst.operation, inst.qubits, inst.clbits)
                    return new, True, (apply_perm if apply_perm else None)
        return circuit, False, None

    def _enumerate_expansions(self, circuit, window, n, allow_permutation=False):
        """Yield candidate rewrites of `window` as lists of (op, qidx, clbits).

        Each candidate is an equivalence-preserving transform; correctness is NOT
        assumed -- every candidate is verifier-gated by the caller. We seed the
        list with: (1) the conjugation identity that pushes a controlled action
        onto a moved target, eliminating one CCX, and (2) the identity window
        itself (so the simplifier alone can run on the relabeled form).
        """
        # Normalize the window to (name, qidx) for pattern matching.
        items = []
        for inst in window:
            name = _gate_name(inst)
            qidx = [circuit.find_bit(b).index for b in inst.qubits]
            items.append((name, qidx, inst))

        candidates = []

        from qiskit.circuit.library import CCXGate

        # Pattern A: CCX(a,b,t) ; <inner block of CX(t,*) and SWAP gates, NOT
        # touching a,b> ; CCX(a,b,t).  The conjugation equals "controlled-on-(a&b)"
        # applied to the inner block's net action on the t-line, so the two
        # conjugating CCX collapse to ONE CCX whose target is the FINAL location of
        # the t-line, plus one CCX per CX-target the t-line fanned into:
        #   CCX(a,b,t); CX(t,u); CCX(a,b,t)            -> CX(t,u); CCX(a,b,u)
        #   CCX(a,b,t); CX(t,u); SWAP(t,x); CCX(a,b,t) -> CX(t,u); SWAP(t,x);
        #                                                  CCX(a,b,<final t-line>)
        # The SWAP case is strictly equivalent only up to an output-wire
        # permutation, which the caller tracks. Every candidate is verifier-gated.
        for s in range(len(items)):
            n0, q0, _ = items[s]
            if n0 != "ccx":
                continue
            a, b, t = q0[0], q0[1], q0[2]
            ctrls = {a, b}
            for e in range(s + 1, len(items)):
                n1, q1, _ = items[e]
                if n1 != "ccx":
                    continue
                if not (q1[2] == t and {q1[0], q1[1]} == ctrls):
                    continue
                inner = items[s + 1:e]
                # Track the t-line through the inner block. Permitted inner gates:
                #   * CX(<t-line>, u):  the t-value fans the AND into u (-> CCX(a,b,u))
                #   * SWAP(<t-line>, x): the t-line moves to wire x
                # Any gate touching a or b, or not of these forms, aborts.
                ok = True
                cur_t = t
                fan_targets = []
                for (nm, qi, _ii) in inner:
                    if not set(qi).isdisjoint(ctrls):
                        ok = False
                        break
                    if nm == "cx" and qi[0] == cur_t and qi[1] != cur_t:
                        fan_targets.append(qi[1])
                    elif nm == "swap" and cur_t in qi:
                        cur_t = qi[1] if qi[0] == cur_t else qi[0]
                    else:
                        ok = False
                        break
                if not ok:
                    continue
                # A pure-CX inner (no swap) with no fan target is just the trivial
                # inverse pair (handled elsewhere); require a real action.
                if not fan_targets and cur_t == t:
                    continue
                # Build the rewrite over the FULL window: prefix unchanged, the
                # inner block kept verbatim, then one CCX(a,b,u) per fan target and
                # (optionally) one CCX(a,b,cur_t) for the swap-moved main t-line;
                # both conjugating CCX dropped. We emit two variants -- with and
                # without the main-line CCX -- and let the verifier pick the correct
                # one (the no-swap CX case needs only the fan CCX; the swap case
                # needs the moved main-line CCX).
                prefix = []
                for k in range(0, s):
                    nm, qi, ii = items[k]
                    prefix.append((ii.operation, list(qi), list(ii.clbits)))
                mid = []
                for (nm, qi, ii) in inner:
                    mid.append((ii.operation, list(qi), list(ii.clbits)))
                suffix = []
                for k in range(e + 1, len(items)):
                    nm, qi, ii = items[k]
                    suffix.append((ii.operation, list(qi), list(ii.clbits)))
                has_swap = any(g[0] == "swap" for g in inner)
                fan = [(CCXGate(), [a, b, u], []) for u in fan_targets]
                # mid with the SWAP gates DROPPED (their effect absorbed into the
                # output-wire permutation). Keeping only the CX gates of the inner
                # block, expressed on the pre-swap wire labels.
                mid_no_swap = [g for g in mid if g[0].name.lower() != "swap"]

                if not has_swap:
                    # Pure-CX conjugation: CCX(a,b,t); CX(t,u)...; CCX(a,b,t) ->
                    # CX(t,u)...; CCX(a,b,u)... -- STRICTLY equivalent (identity
                    # perm), the canonical 2-CCX -> 1-CCX peephole strict passes
                    # miss. Always offered.
                    candidates.append(prefix + mid + fan + suffix)
                elif allow_permutation:
                    # A SWAP moved the t-line between the two conjugating CCX. We can
                    # keep ONE CCX by DROPPING the SWAP gate(s) and depositing the AND
                    # on the ORIGINAL t wire, letting the swap's wire relabeling be
                    # absorbed into the output-wire permutation (verifier-confirmed,
                    # perm tracked & propagated to all downstream gates). This
                    # verifies only up to a NON-identity permutation, so it is offered
                    # ONLY when permutations are allowed -- it strictly ENLARGES the
                    # count-reducing rewrite set beyond what a strict (swap-as-gate)
                    # walk can express. Every variant is still verifier-gated.
                    main_orig = [(CCXGate(), [a, b, t], [])]
                    # variant a: keep inner CX gates verbatim + one CCX on original t
                    candidates.append(prefix + mid_no_swap + main_orig + suffix)
                    # variant b: also push the CX-fan onto the AND (covers mixed
                    # fan-out + swap windows); both conjugating CCX collapsed.
                    candidates.append(prefix + mid_no_swap + fan + main_orig + suffix)
                    candidates.append(prefix + mid_no_swap + fan + suffix)

        # Always also offer the verbatim window (identity expansion) so the
        # simplifier can attempt annihilation on its own under permutation.
        verbatim = []
        for (nm, qi, ii) in items:
            verbatim.append((ii.operation, list(qi), list(ii.clbits)))
        candidates.append(verbatim)
        return candidates

    def _simplify_window(self, cand, n):
        """Run inverse-pair annihilation to a fixpoint on a candidate window
        (list of (op, qidx, clbits)) and return the simplified list."""
        qc = QuantumCircuit(n)
        for op, qidx, cl in cand:
            qc.append(op, [qc.qubits[k] for k in qidx], cl)
        # reuse the inverse-pair pass (strict, identity-only) as a local cleaner
        changed = True
        guard = 0
        while changed and guard < 64:
            qc2, changed = self._pass_inverse_pair(qc)
            qc = qc2
            guard += 1
        out = []
        for inst in qc.data:
            qidx = [qc.find_bit(b).index for b in inst.qubits]
            out.append((inst.operation, qidx, list(inst.clbits)))
        return out

    # ---- PASS 3: equal-control fan-out / CSE ------------------------------
    def _pass_fanout_cse(self, circuit: QuantumCircuit):
        """Detect a maximal run of CCX gates sharing the same control pair with
        distinct targets, and replace it with a compute-once fan-out using a
        clean ancilla -- only if it lowers the atomic count and verifies."""
        data = list(circuit.data)
        n = circuit.num_qubits

        i = 0
        while i < len(data):
            if _gate_name(data[i]) != "ccx":
                i += 1
                continue
            idx = self._qidx(circuit, data[i])
            ctrl_pair = frozenset(idx[:2])
            ctrl_qubits = idx[:2]

            # Gather a contiguous-by-commutation run of ccx gates with the same
            # control pair and DISTINCT targets, where intervening gates do not
            # touch the controls or any collected target.
            run_positions = [i]
            targets = [idx[2]]
            j = i + 1
            while j < len(data):
                qs_j = self._qset(circuit, data[j])
                if _gate_name(data[j]) == "ccx":
                    jidx = self._qidx(circuit, data[j])
                    if frozenset(jidx[:2]) == ctrl_pair and jidx[2] not in targets \
                            and jidx[2] not in ctrl_qubits:
                        run_positions.append(j)
                        targets.append(jidx[2])
                        j += 1
                        continue
                # any gate (ccx or not) that touches the controls breaks the run;
                # a gate touching an already-collected target also breaks it.
                if not qs_j.isdisjoint(set(ctrl_qubits)) or not qs_j.isdisjoint(set(targets)):
                    break
                j += 1

            # k >= 3 CCX sharing a control pair: compute-once fan-out turns them
            # into 2 CCX + k CX, which is a strict atomic-count win for k >= 3.
            if len(run_positions) > 2:
                first = run_positions[0]
                last = run_positions[-1]
                between = [data[k] for k in range(first, last + 1)
                           if k not in run_positions]

                candidate = self._build_fanout_candidate(
                    circuit, data, first, last, ctrl_qubits, targets, between)
                if candidate is not None and \
                        count_toffoli(candidate) < count_toffoli(circuit):
                    # A FRESH ancilla wire (highest index, beyond the original
                    # register) carries a&b. The verifier treats that wire as an
                    # ancilla that must begin and end in |0>, so the full-circuit
                    # check below proves the compute/uncompute is sound -- no
                    # leakage and no dependence on the ancilla's input value.
                    try:
                        ok, _p, _i = self.verifier.verify(circuit, candidate)
                    except Exception:
                        ok = False
                    if ok:
                        return candidate, True, list(ctrl_pair)
            i = j if j > i + 1 else i + 1
        return circuit, False, None

    def _build_fanout_candidate(self, circuit, data, first, last,
                                ctrl_qubits, targets, between):
        """Build the full candidate circuit on n+1 qubits (1 fresh ancilla wire,
        highest index) implementing compute-once fan-out for the run."""
        from qiskit.circuit.library import CCXGate, CXGate

        n = circuit.num_qubits
        anc = n  # fresh ancilla = new highest-index wire (verifier treats as |0>)
        qc = QuantumCircuit(n + 1)
        q = qc.qubits

        def remap(inst):
            qidx = [circuit.find_bit(b).index for b in inst.qubits]
            qc.append(inst.operation, [q[k] for k in qidx], list(inst.clbits))

        # everything before the run, unchanged
        for k in range(0, first):
            remap(data[k])
        # compute a&b -> ancilla, fan out with CX, uncompute
        c0, c1 = q[ctrl_qubits[0]], q[ctrl_qubits[1]]
        qc.append(CCXGate(), [c0, c1, q[anc]])
        for t in targets:
            qc.append(CXGate(), [q[anc], q[t]])
        qc.append(CCXGate(), [c0, c1, q[anc]])
        # intervening (non-run) gates inside the window, preserved after the run
        # (they are disjoint from controls and targets, so this reordering is
        # function-preserving -- the verifier confirms it).
        for inst in between:
            remap(inst)
        # everything after the run, unchanged
        for k in range(last + 1, len(data)):
            remap(data[k])
        return qc

    def _detect_shareable_runs(self, circuit: QuantumCircuit):
        """Report equal-control CCX runs (length >= 2) without modifying anything."""
        data = list(circuit.data)
        runs = []
        i = 0
        while i < len(data):
            if _gate_name(data[i]) != "ccx":
                i += 1
                continue
            idx = self._qidx(circuit, data[i])
            ctrl_pair = frozenset(idx[:2])
            targets = [idx[2]]
            j = i + 1
            while j < len(data):
                if _gate_name(data[j]) == "ccx":
                    jidx = self._qidx(circuit, data[j])
                    if frozenset(jidx[:2]) == ctrl_pair and jidx[2] not in targets:
                        targets.append(jidx[2])
                        j += 1
                        continue
                qs_j = self._qset(circuit, data[j])
                if not qs_j.isdisjoint(set(ctrl_pair)) or not qs_j.isdisjoint(set(targets)):
                    break
                j += 1
            if len(targets) >= 2:
                runs.append({"controls": sorted(ctrl_pair), "targets": targets})
            i = max(j, i + 1)
        return runs

    # ---- rebuild a circuit from a (possibly reordered) instruction list ---
    def _rebuild(self, circuit: QuantumCircuit, data) -> QuantumCircuit:
        qc = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
        for inst in data:
            qc.append(inst.operation, inst.qubits, inst.clbits)
        return qc

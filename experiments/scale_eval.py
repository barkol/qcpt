#!/usr/bin/env python3
"""Scale evaluation of the error-budget Toffoli-decomposition method.

Pushes the evaluation of QCPT's ErrorBudgetSelector PAST the ~8-12 qubit
exhaustive-verification limit, onto the 12-24 qubit ``large_benchmarks.large_suite()``,
and demonstrates the scalability boundary of the paper concretely:

  * 2q-count / 2q-infidelity reduction of OURS vs exact-only and Qiskit opt-3,
    per circuit, at sizes where the WHOLE circuit no longer fits an exhaustive check;

  * correctness CERTIFICATION of every accepted rewrite, dispatched by width:
        - EXHAUSTIVELY VERIFIED (truth table / dense unitary) where it still fits;
        - QCEC-VERIFIED UNITARY (decision diagrams, up to global phase) above the
          crossover, for the unitary-equivalent compute/uncompute-pair substitutions;
        - ANALYSIS-SOUND PHASE-AWARE for relative-phase substitutions that are not
          unitary-equivalent by design (certified by phase_observability + the SOUND
          reachable over-approximation, NOT by QCEC), spot-checked exhaustively at
          small width;

  * the verification CROSSOVER: wall-clock of exhaustive vs QCEC across circuit width
    -- where exhaustive stops being feasible and QCEC takes over.

HONESTY. We never claim exhaustive verification at a scale where it is infeasible.
Each per-circuit row carries the verification METHOD used and its wall-clock. The
selector here is QCEC-GATED: every accepted relative-phase pair substitution is
checked against the all-exact decomposition with verify_scalable (exhaustive if
small, QCEC otherwise). The stock ``ErrorBudgetSelector.select`` uses an exhaustive
internal gate that does NOT scale (it blows up ~12q and rejects everything >~13q once
the dense-unitary guard trips); we report that explicitly and use the QCEC-gated
variant to demonstrate the method at scale.

Run:  python scale_eval.py        (writes scale_results.md)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import time

from _paths import EXPERIMENTS_DIR  # noqa: F401  (must be first)

from qiskit import QuantumCircuit, transpile

import large_benchmarks as LB
from toffoli_optimizer.core.error_model import HardwareErrorModel
from toffoli_optimizer.core.equivalence_verifier import ExactEquivalenceVerifier
from toffoli_optimizer.core.decomposition_selector import (
    ErrorBudgetSelector,
    append_exact_ccx,
    append_relative_phase_ccx,
)
from toffoli_optimizer.core.context_analysis import find_relative_phase_safe_sites
from toffoli_optimizer.core.phase_observability import (
    is_phase_unobservable,
    default_affected_qubits,
)

try:
    from toffoli_optimizer.core.scalable_verification import (
        verify_scalable,
        ScalableVerifyResult,
    )
    HAVE_SCALABLE = True
except Exception as _exc:  # pragma: no cover
    HAVE_SCALABLE = False
    _SCALABLE_ERR = repr(_exc)

EM = HardwareErrorModel()
VERIFIER = ExactEquivalenceVerifier()

# Crossover: at/below this width we run the exhaustive ground-truth check; above it
# we fall back to QCEC. The verifier's own dense-unitary guard is 12, but the dense
# `Operator` build is already ~1 MINUTE at n=12 on these deep Clifford+T circuits
# (measured -- see the crossover table), so for the per-circuit CERTIFICATION dispatch
# we set the practical crossover at 10: at/below 10 exhaustive is sub-second, above it
# QCEC is the feasible certificate. (The crossover micro-benchmark separately times
# exhaustive up to where it becomes infeasible, to LOCATE this boundary empirically.)
EXHAUSTIVE_MAX_QUBITS = 10

# Hard ceiling for the crossover micro-benchmark's exhaustive attempts: above this we
# do not even try the dense path (it would take minutes / exhaust memory). n=12's
# dense unitary is two 4096x4096 complex matrices built gate-by-gate over ~170 gates.
_CROSSOVER_EXHAUSTIVE_CEILING = 12

# Output directory (default: next to this script). TOPTOFFOLI_OUT redirects the results,
# e.g. for a rerun into paper/data/rerun_suites/ that must not overwrite stored data.
OUT_DIR = Path(os.environ.get("TOPTOFFOLI_OUT", str(EXPERIMENTS_DIR)))
OUT_MD = str(OUT_DIR / "scale_results.md")
# Wall-clock limit (s) of one QCEC call in certify(); 120 s is the value used for the paper.
QCEC_TIMEOUT_S = float(os.environ.get("TOPTOFFOLI_QCEC_TIMEOUT", "120"))


# ===========================================================================
# QCEC-gated scalable selector
# ===========================================================================
class ScalableErrorBudgetSelector:
    """ErrorBudgetSelector whose per-site correctness gate is verify_scalable.

    The stock selector verifies each candidate substitution against the all-exact
    decomposition with an EXHAUSTIVE check, which is O(2^n) and does not scale past
    ~12-13 qubits. This variant keeps the SAME structural admissibility logic
    (find_relative_phase_safe_sites: compute/uncompute pairs) but swaps the gate to
    ``verify_scalable``, so above the crossover the unitary-equivalent pair
    substitutions are certified by QCEC (decision diagrams, up to global phase)
    instead of an infeasible exhaustive check.

    SCOPE. This variant admits the subroutine-semantics conditions of the paper:
    (C) compute/uncompute pairs (each pair certified by verify_scalable), (W)
    window-certified mirror pairs (exact check on the segment) and (R) standalone
    Margolus or mirrored gadgets that equal CCX with one phase on the projected
    reachable subspace (exact local reachability with the pinned ancillas in |0>).
    All of these are equivalences on the input subspace up to one global phase, so
    the whole output is then certified by ``certify`` (dense on the input subspace
    up to 12 qubits, QCEC with the pinned qubits as ancillas above). Condition (U)
    (program semantics: a lone gadget whose phase is merely UNOBSERVABLE) is NOT
    unitary-equivalent and is not admitted here; it is evaluated separately
    (``scale_program.py``, analysis + small-instance spot-check).
    """

    def __init__(self, exhaustive_max_qubits: int = EXHAUSTIVE_MAX_QUBITS):
        self.em = HardwareErrorModel()
        self.exhaustive_max_qubits = exhaustive_max_qubits

    def decompose_exact_only(self, circuit: QuantumCircuit) -> QuantumCircuit:
        out = QuantumCircuit(circuit.num_qubits)
        for inst in circuit.data:
            name = inst.operation.name.lower()
            qb = [circuit.find_bit(q).index for q in inst.qubits]
            if name == "ccx" or (name in ("mcx", "mcx_gray") and len(qb) == 3):
                append_exact_ccx(out, qb[0], qb[1], qb[2])
            else:
                out.append(inst.operation, [out.qubits[i] for i in qb])
        return out

    def _build(self, circuit: QuantumCircuit, rel_idx: set, mirror_idx=frozenset()) -> QuantumCircuit:
        from toffoli_optimizer.core.decomposition_selector import append_relative_phase_ccx_mirror
        out = QuantumCircuit(circuit.num_qubits)
        for idx, inst in enumerate(circuit.data):
            qb = [circuit.find_bit(q).index for q in inst.qubits]
            if idx in mirror_idx:
                append_relative_phase_ccx_mirror(out, qb[0], qb[1], qb[2])
            elif idx in rel_idx:
                append_relative_phase_ccx(out, qb[0], qb[1], qb[2])
            else:
                name = inst.operation.name.lower()
                if name == "ccx" or (name in ("mcx", "mcx_gray") and len(qb) == 3):
                    append_exact_ccx(out, qb[0], qb[1], qb[2])
                else:
                    out.append(inst.operation, [out.qubits[i] for i in qb])
        return out

    def select(self, circuit: QuantumCircuit, pinned_zero=()) -> dict:
        exact = self.decompose_exact_only(circuit)
        sites = find_relative_phase_safe_sites(circuit)

        rel_idx: set = set()
        applied = []
        rejected = []
        per_site_verif = []  # (n, method, wall, equivalent) for crossover data

        for site in sites:
            cand_idx = rel_idx | {site.compute_idx, site.uncompute_idx}
            candidate = self._build(circuit, cand_idx)
            res = verify_scalable(
                exact, candidate,
                exhaustive_max_qubits=self.exhaustive_max_qubits,
            )
            per_site_verif.append(
                (res.n_qubits, res.method, res.wall_time_s, res.equivalent)
            )
            if res.equivalent is True:
                rel_idx = cand_idx
                applied.append(site)
            else:
                rejected.append({"site": tuple(site),
                                 "method": res.method,
                                 "equivalent": res.equivalent,
                                 "detail": res.detail})

        # Condition (W): window-certified mirror pairs (exact check on the segment).
        from toffoli_optimizer.core.window_pairs import admit_window_pairs
        rel_idx, wlog = admit_window_pairs(
            circuit, rel_idx, append_relative_phase_ccx,
            accepted_windows=[(s.compute_idx, s.uncompute_idx) for s in applied])

        # Condition (R) for gadgets: exact local reachability (classical prefix,
        # ancillas pinned to |0>); Margolus or mirrored gadget if it equals CCX on
        # the projected reachable subspace with one phase.
        from toffoli_optimizer.core.reach_local import local_reachable
        from toffoli_optimizer.core.subspace_check import check_on_subspace
        from toffoli_optimizer.core.decomposition_selector import _U_CCX, _U_MARG, _U_MARG_M
        mirror_idx = set()
        r_count = 0
        for idx, inst in enumerate(circuit.data):
            if idx in rel_idx or inst.operation.name.lower() not in ("ccx", "mcx", "mcx_gray") or len(inst.qubits) != 3:
                continue
            qb = tuple(circuit.find_bit(q).index for q in inst.qubits)
            loc = local_reachable(circuit, idx, qb, pinned_zero)
            if loc is None:
                continue
            if check_on_subspace(_U_MARG, _U_CCX, loc, 0.0)[0]:
                rel_idx = rel_idx | {idx}; r_count += 1
            elif check_on_subspace(_U_MARG_M, _U_CCX, loc, 0.0)[0]:
                rel_idx = rel_idx | {idx}; mirror_idx.add(idx); r_count += 1

        selected = self._build(circuit, rel_idx, mirror_idx)
        return {
            "rel_idx": sorted(rel_idx), "mirror_idx": sorted(mirror_idx),
            "r_gadgets": r_count,
            "window_pairs": len(wlog),
            "circuit": selected,
            "exact": exact,
            "sites_found": len(sites),
            "sites_applied": len(applied),
            "sites_rejected": len(rejected),
            "rejected": rejected,
            "per_site_verif": per_site_verif,
            "two_qubit_before": self.em.two_qubit_count(exact),
            "two_qubit_after": self.em.two_qubit_count(selected),
            "infid_before": self.em.circuit_infidelity(exact),
            "infid_after": self.em.circuit_infidelity(selected),
        }


# ===========================================================================
# whole-circuit certification of the selected output vs exact-only
# ===========================================================================
def _with_ancillas(circ: QuantumCircuit, pinned) -> QuantumCircuit:
    """Same circuit with the pinned qubits moved into an AncillaRegister (last),
    so QCEC treats them as initialised to |0> (equivalence on the input subspace)."""
    from qiskit import QuantumRegister, AncillaRegister
    pinned = list(pinned)
    free = [q for q in range(circ.num_qubits) if q not in set(pinned)]
    order = free + pinned
    pos = {q: k for k, q in enumerate(order)}
    regs = [QuantumRegister(len(free), "d")] + ([AncillaRegister(len(pinned), "anc0")] if pinned else [])
    out = QuantumCircuit(*regs)
    for inst in circ.data:
        qb = [pos[circ.find_bit(q).index] for q in inst.qubits]
        out.append(inst.operation, [out.qubits[k] for k in qb])
    return out


def certify(exact: QuantumCircuit, selected: QuantumCircuit, pinned_zero=()):
    """Certify selected == exact-only. Without pinned qubits: verify_scalable
    (unitary up to global phase). With pinned qubits (condition (R) gadgets):
    dense input-subspace certificate up to 12 qubits, QCEC with ancilla registers
    above (equivalence on the subspace where the pinned qubits start in |0>)."""
    if pinned_zero:
        import time as _t
        from types import SimpleNamespace
        n = exact.num_qubits
        t0 = _t.time()
        if n <= 12:
            from qiskit.quantum_info import Operator
            from toffoli_optimizer.core.subspace_check import certify_on_input_subspace
            pm = sum(1 << q for q in pinned_zero)
            inputs = [x for x in range(1 << n) if not x & pm]
            ok = certify_on_input_subspace(Operator(exact).data, Operator(selected).data, inputs, "subroutine")[0]
            return SimpleNamespace(equivalent=bool(ok), method="dense_subspace", wall_time_s=_t.time() - t0,
                                   detail={"equivalence_criterion": "subroutine"}, n_qubits=n)
        from toffoli_optimizer.core.subspace_check import qcec_certify_on_subspace
        ok, info = qcec_certify_on_subspace(exact, selected, pinned_zero, timeout=QCEC_TIMEOUT_S)
        return SimpleNamespace(equivalent=bool(ok), method="qcec_ancilla", wall_time_s=_t.time() - t0,
                               detail={"equivalence_criterion": info.get("qcec", info.get("reason"))}, n_qubits=n)
    """Certify selected == exact-only via verify_scalable; record method + time."""
    res = verify_scalable(
        exact, selected, exhaustive_max_qubits=EXHAUSTIVE_MAX_QUBITS, timeout_s=QCEC_TIMEOUT_S
    )
    return res


# ===========================================================================
# phase-aware analysis soundness (separate from QCEC) + small spot-check
# ===========================================================================
def analysis_sound_phase_aware(circuit: QuantumCircuit) -> dict:
    """Report which standalone Toffolis would be admitted by the phase-AWARE path,
    purely from the static analysis (phase_observability), and -- if the circuit is
    small enough -- spot-check that the stock phase-aware selector's output is
    reachable-basis equivalent exhaustively.

    This is the regime that QCEC CANNOT certify (relative-phase substitutions are not
    unitary-equivalent). We surface it honestly as analysis-sound, never as
    'verified'.
    """
    n = circuit.num_qubits
    # Which standalone CCX have a provably-unobservable relative phase?
    paired = set()
    for s in find_relative_phase_safe_sites(circuit):
        paired.add(s.compute_idx)
        paired.add(s.uncompute_idx)
    unobservable = 0
    standalone = 0
    for idx, inst in enumerate(circuit.data):
        if inst.operation.name.lower() not in ("ccx", "mcx", "mcx_gray"):
            continue
        if len([q for q in inst.qubits]) != 3:
            continue
        if idx in paired:
            continue
        standalone += 1
        aff = default_affected_qubits(circuit, idx)
        if is_phase_unobservable(circuit, idx, aff):
            unobservable += 1

    # NOTE: we do NOT run the stock exhaustive phase-aware selector on the 12-24q
    # suite members -- its internal O(2^n) gate makes it impractical there (a 12q
    # case can take minutes). The static analysis above is width-independent and
    # always runs. Exhaustive ground-truth spot-checking of the phase-aware path is
    # done separately on TINY (<=10q) generated instances, see
    # ``phase_aware_small_spotcheck``.
    return {
        "standalone_ccx": standalone,
        "phase_unobservable": unobservable,
        "small_spot_check": None,
    }


def phase_aware_small_spotcheck():
    """Run the STOCK exhaustive phase-aware selector on TINY instances of each
    family, confirming it certifies (on the reachable basis, exhaustively) -- the
    ground-truth backstop for the analysis-sound phase-aware path.

    These are deliberately <=10 qubits so the exhaustive internal gate is fast.
    Returns a list of dicts.
    """
    tiny = [
        LB.grover_oracle(3),          # 6q, compute/uncompute
        LB.carry_lookahead_adder(2),  # 7q
        LB.modular_increment(3),      # 6q
        LB.array_multiplier(2),       # 8q, live
        LB.half_uncomputed_oracle(3), # 7q, mixed
    ]
    out = []
    for qc in tiny:
        try:
            t0 = time.time()
            rep = ErrorBudgetSelector(phase_aware=True).select(qc)["report"]
            out.append({
                "name": qc.name, "n": qc.num_qubits,
                "relphase_safe": qc.relphase_safe,
                "verified": bool(rep["verified"]),
                "sites_applied": rep["sites_applied"],
                "phase_aware_admitted": len(rep.get("phase_aware_admitted", [])),
                "certification": rep["verify_info"].get("certification"),
                "wall_s": round(time.time() - t0, 3),
            })
        except Exception as exc:  # pragma: no cover
            out.append({"name": qc.name, "n": qc.num_qubits,
                        "error": repr(exc)[:100]})
    return out


# ===========================================================================
# crossover micro-benchmark: exhaustive vs QCEC wall-clock vs width
# ===========================================================================
def crossover_curve():
    """Time exhaustive vs QCEC on the SAME equivalence (a width-w ripple adder's
    exact lowering vs itself with one verified pair substituted) as width grows.

    Returns rows (n, t_exhaustive_s|None, t_qcec_s, exhaustive_feasible).
    """
    rows = []
    for nbits in range(2, 11):  # widths 6 .. 22
        qc = LB.ripple_carry_adder(nbits)
        n = qc.num_qubits
        sel = ScalableErrorBudgetSelector()
        exact = sel.decompose_exact_only(qc)
        sites = find_relative_phase_safe_sites(qc)
        idxs = set()
        for s in sites:
            idxs |= {s.compute_idx, s.uncompute_idx}
        cand = sel._build(qc, idxs)

        # QCEC time (always)
        rq = verify_scalable(exact, cand, prefer="qcec")
        t_qcec = rq.wall_time_s
        qcec_eq = rq.equivalent

        # Exhaustive time -- only attempt up to a hard ceiling so we don't hang.
        # These exact lowerings are non-classical (h/t/ry), so verify() takes the
        # dense-unitary path: a 2^n x 2^n Operator built gate-by-gate. That is
        # sub-second to ~n=10 and ~1 minute at n=12 (the point of this curve).
        t_exh = None
        exh_eq = None
        feasible = n <= _CROSSOVER_EXHAUSTIVE_CEILING
        if feasible:
            t0 = time.time()
            try:
                ok, _p, _i = VERIFIER.verify(exact, cand, allow_permutation=False)
                t_exh = time.time() - t0
                exh_eq = bool(ok)
            except Exception:
                t_exh = None
        rows.append({
            "n": n, "nbits": nbits,
            "t_exhaustive": t_exh, "exh_eq": exh_eq,
            "t_qcec": t_qcec, "qcec_eq": qcec_eq,
            "exhaustive_feasible": feasible,
        })
    return rows


# ===========================================================================
# main eval
# ===========================================================================
def m_qiskit_l3(circ: QuantumCircuit) -> QuantumCircuit:
    return transpile(circ, basis_gates=["cx", "u"], optimization_level=3)


def run():
    suite = LB.large_suite()
    rows = []
    for qc in suite:
        n = qc.num_qubits
        sel = ScalableErrorBudgetSelector()
        from _clean import clean_ancillas
        pins = clean_ancillas(qc)
        sres = sel.select(qc, pinned_zero=pins)

        # certify selected vs exact-only (on the input subspace if ancillas are pinned)
        cert = certify(sres["exact"], sres["circuit"], pinned_zero=pins if sres.get("r_gadgets") else ())
        if cert.equivalent is not True:
            # fail closed: an uncertified selection is never reported as a reduction
            sres = dict(sres, circuit=sres["exact"], two_qubit_after=sres["two_qubit_before"],
                        infid_after=sres["infid_before"], fell_back=True)

        # qiskit opt-3 for comparison (2q count / infid)
        t0 = time.time()
        ql3 = m_qiskit_l3(qc)
        ql3_2q = EM.two_qubit_count(ql3)
        ql3_infid = EM.circuit_infidelity(ql3)
        ql3_t = time.time() - t0

        # phase-aware analysis (the QCEC-uncertifiable regime)
        pa = analysis_sound_phase_aware(qc)

        red_2q = (
            (sres["two_qubit_before"] - sres["two_qubit_after"])
            / sres["two_qubit_before"] * 100
            if sres["two_qubit_before"] else 0.0
        )
        red_inf = (
            (sres["infid_before"] - sres["infid_after"])
            / sres["infid_before"] * 100
            if sres["infid_before"] else 0.0
        )

        rows.append({
            "name": qc.name, "n": n, "family": qc.family,
            "relphase_safe": qc.relphase_safe,
            "sites_found": sres["sites_found"],
            "sites_applied": sres["sites_applied"],
            "window_pairs": sres.get("window_pairs", 0),
            "r_gadgets": sres.get("r_gadgets", 0),
            "pinned": len(pins),
            "twoq_exact": sres["two_qubit_before"],
            "twoq_ours": sres["two_qubit_after"],
            "infid_exact": sres["infid_before"],
            "infid_ours": sres["infid_after"],
            "red_2q": red_2q, "red_inf": red_inf,
            "qiskit_2q": ql3_2q, "qiskit_infid": ql3_infid,
            "cert_method": cert.method,
            "cert_equiv": cert.equivalent,
            "cert_crit": cert.detail.get("equivalence_criterion", ""),
            "cert_time": cert.wall_time_s,
            "phase_aware": pa,
        })
        print(f"{qc.name:24s} n={n:2d} sites {sres['sites_applied']}/{sres['sites_found']} "
              f"2q {sres['two_qubit_before']}->{sres['two_qubit_after']} "
              f"({red_2q:+.1f}%) cert={cert.method}:{cert.equivalent} "
              f"({cert.wall_time_s:.3f}s)")

    return suite, rows


# ===========================================================================
# reporting
# ===========================================================================
def aggregate(rows, subset=None):
    sel = [r for r in rows if subset is None or subset(r)]
    if not sel:
        return None
    twoq_exact = sum(r["twoq_exact"] for r in sel)
    twoq_ours = sum(r["twoq_ours"] for r in sel)
    infid_exact = sum(r["infid_exact"] for r in sel)
    infid_ours = sum(r["infid_ours"] for r in sel)
    qiskit_2q = sum(r["qiskit_2q"] for r in sel)
    return {
        "n": len(sel),
        "twoq_exact": twoq_exact, "twoq_ours": twoq_ours,
        "infid_exact": infid_exact, "infid_ours": infid_ours,
        "qiskit_2q": qiskit_2q,
        "red_2q_vs_exact": (twoq_exact - twoq_ours) / twoq_exact * 100 if twoq_exact else 0.0,
        "red_inf_vs_exact": (infid_exact - infid_ours) / infid_exact * 100 if infid_exact else 0.0,
        "red_2q_vs_qiskit": (qiskit_2q - twoq_ours) / qiskit_2q * 100 if qiskit_2q else 0.0,
    }


def write_markdown(path, rows, cross, spotcheck):
    L = []
    L.append("# Scale evaluation — error-budget Toffoli decomposition past the "
             "exhaustive-verification limit\n")
    L.append(
        "Large CCX/MCX benchmarks (12-24 qubits, `large_benchmarks.large_suite()`), "
        "decomposed by the certified `ScalableErrorBudgetSelector` (structural pairs (C), "
        "certified windows (W), mirrored gadgets under (R) with clean ancillas pinned to |0>) "
        f"and scored with QCPT's `HardwareErrorModel` (p2q={EM.p2q}, p1q={EM.p1q}). "
        "Every output is certified against the all-exact decomposition on the input "
        "subspace (subroutine equivalence): densely when that is cheap, otherwise with "
        "QCEC (pinned qubits declared as ancillas). An output that cannot be certified is "
        "replaced by the all-exact circuit and reported as such.\n")

    # ---- soundness regime legend
    L.append("## Verification regimes (kept strictly separate)\n")
    L.append(
        "- **exhaustively verified** — `ExactEquivalenceVerifier.verify` ran "
        "(ground truth). Feasible only up to ~12-13 qubits here.\n"
        "- **QCEC-verified on the input subspace (up to global phase)** — decision-diagram "
        "proof at widths where exhaustive is infeasible.\n"
        "- **program mode (condition U)** — relative-phase substitutions whose phase is "
        "only *unobservable* at the final measurement are NOT subroutine-equivalent and are "
        "not used here; see `scale_program.py`.\n")

    # ---- main per-circuit table
    L.append("## Per-circuit results\n")
    L.append("| Circuit | n | family | relphase_safe | sites (appl/found) | "
             "2q exact | 2q ours | 2q reduction | 2q qiskit-O3 | "
             "infid exact | infid ours | cert method | cert | cert time (s) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        eq = ("equiv" if r["cert_equiv"] is True
              else ("NEQ" if r["cert_equiv"] is False else "undecided"))
        L.append(
            f"| {r['name']} | {r['n']} | {r['family']} | {r['relphase_safe']} | "
            f"{r['sites_applied']}/{r['sites_found']} | {r['twoq_exact']} | "
            f"{r['twoq_ours']} | {r['red_2q']:+.1f}% | {r['qiskit_2q']} | "
            f"{r['infid_exact']:.4f} | {r['infid_ours']:.4f} | "
            f"{r['cert_method']} | {eq} | {r['cert_time']:.4f} |")
    L.append("")

    # ---- aggregates
    L.append("## Aggregate reductions\n")
    allg = aggregate(rows)
    safe = aggregate(rows, lambda r: r["relphase_safe"])
    live = aggregate(rows, lambda r: not r["relphase_safe"])
    big = aggregate(rows, lambda r: r["n"] > EXHAUSTIVE_MAX_QUBITS)
    L.append("| Subset | #circ | 2q exact | 2q ours | 2q red vs exact | "
             "2q red vs qiskit-O3 | infid red vs exact |")
    L.append("|---|---|---|---|---|---|---|")
    for label, a in [("ALL", allg),
                     ("compute/uncompute (relphase-safe)", safe),
                     ("LIVE (relphase-unsafe)", live),
                     (f"width > {EXHAUSTIVE_MAX_QUBITS} (QCEC-certified only)", big)]:
        if a is None:
            continue
        L.append(
            f"| {label} | {a['n']} | {a['twoq_exact']} | {a['twoq_ours']} | "
            f"{a['red_2q_vs_exact']:.1f}% | {a['red_2q_vs_qiskit']:.1f}% | "
            f"{a['red_inf_vs_exact']:.1f}% |")
    L.append("")
    # which families actually yield big reductions, for an honest narrative.
    # Group by the coarse prefix (grover / adder / modular / multiplier) so each
    # workload class appears once.
    fam_red = {}
    for r in rows:
        prefix = r["family"].split("/")[0]
        fam_red.setdefault(prefix, []).append(r["red_2q"])
    fam_summary = ", ".join(
        f"{fam} ~{sum(v)/len(v):.0f}%"
        for fam, v in sorted(fam_red.items(), key=lambda kv: -sum(kv[1]) / len(kv[1]))
    )
    L.append(
        f"> On this suite the reduction over exact-only is **{allg['red_2q_vs_exact']:.1f}% 2q / "
        f"{allg['red_inf_vs_exact']:.1f}% infidelity** across all {allg['n']} circuits, and "
        f"**{safe['red_2q_vs_exact']:.1f}% 2q** on the compute/uncompute-structured subset. "
        f"Mean 2q reduction by family: {fam_summary}. The size tracks the share of Toffolis "
        f"that are uncomputed or write into clean ancillas; Toffolis accumulating into "
        f"undeclared output registers (array multipliers, live-carry adders) stay exact.\n")

    # ---- phase-aware regime
    L.append("## Phase-aware regime (QCEC-uncertifiable; analysis-sound)\n")
    L.append(
        "The phase-AWARE standalone path (a lone relative-phase Toffoli whose phase "
        "is merely *unobservable*) is NOT unitary-equivalent to exact CCX, so QCEC "
        "reports `not_equivalent` for it by construction. Its admissibility is decided "
        "by the width-independent static analysis below. The certified selector "
        "evaluated above does NOT use this path, so these numbers do not inflate the "
        "reductions; they count the Toffolis that program mode could additionally "
        "touch.\n")
    L.append("| Circuit | n | standalone CCX | phase-unobservable (analysis) |")
    L.append("|---|---|---|---|")
    for r in rows:
        pa = r["phase_aware"]
        L.append(f"| {r['name']} | {r['n']} | {pa['standalone_ccx']} | "
                 f"{pa['phase_unobservable']} |")
    L.append("")

    # ---- small exhaustive spot-check of the phase-aware path
    L.append("### Exhaustive ground-truth spot-check (tiny instances)\n")
    L.append(
        "The stock exhaustive phase-aware `ErrorBudgetSelector` run on TINY (<=10q) "
        "instances of each family, where its O(2^n) internal gate is fast. This is "
        "the ground-truth backstop: it certifies on the reachable basis "
        "(phase-insensitive, ancilla clean) EXHAUSTIVELY at these widths.\n")
    L.append("| Circuit | n | relphase_safe | verified | pair sites | "
             "phase-aware admitted | certification | time (s) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for s in spotcheck:
        if "error" in s:
            L.append(f"| {s['name']} | {s['n']} | — | ERROR | — | — | {s['error']} | — |")
        else:
            L.append(
                f"| {s['name']} | {s['n']} | {s['relphase_safe']} | {s['verified']} | "
                f"{s['sites_applied']} | {s['phase_aware_admitted']} | "
                f"{s['certification']} | {s['wall_s']} |")
    L.append("")
    L.append(
        "> Where a standalone phase is provably unobservable (`is_phase_unobservable`) "
        "AND agrees with exact CCX on the reachable basis (`reachable_overapprox`, a "
        "SOUND superset), the substitution is analysis-sound. The over-approximation "
        "can only OVER-reject, never wrongly accept. The `live`/`mixed` tiny instances "
        "correctly admit nothing unsafe (the selector keeps live Toffolis exact and "
        "stays `verified`).\n")

    # ---- crossover
    L.append("## Verification crossover — exhaustive vs QCEC wall-clock vs width\n")
    L.append(
        "Same equivalence (a width-w ripple adder's exact lowering vs the same with "
        "every verified compute/uncompute pair substituted), timed both ways as width "
        "grows. This is the paper's scalability boundary, demonstrated.\n")
    L.append("| n (qubits) | exhaustive time (s) | exhaustive verdict | "
             "QCEC time (s) | QCEC verdict | exhaustive feasible? |")
    L.append("|---|---|---|---|---|---|")
    for c in cross:
        te = "—" if c["t_exhaustive"] is None else f"{c['t_exhaustive']:.4f}"
        ev = "—" if c["exh_eq"] is None else ("equiv" if c["exh_eq"] else "NEQ")
        qv = ("equiv" if c["qcec_eq"] is True
              else ("NEQ" if c["qcec_eq"] is False else "undecided"))
        L.append(f"| {c['n']} | {te} | {ev} | {c['t_qcec']:.4f} | {qv} | "
                 f"{'yes' if c['exhaustive_feasible'] else 'NO (infeasible)'} |")
    L.append("")

    # crossover narrative (data-driven)
    feas = [c for c in cross if c["t_exhaustive"] is not None]
    last_feas = feas[-1] if feas else None
    n_last = last_feas["n"] if last_feas else None
    t_last = last_feas["t_exhaustive"] if last_feas else None
    tq_last = last_feas["t_qcec"] if last_feas else None
    speedup = (t_last / tq_last) if (t_last and tq_last) else None
    max_qcec = max((c["t_qcec"] for c in cross), default=0.0)
    L.append(
        f"> Exhaustive verification time grows ~2^n: it is sub-second up to ~n=10, "
        f"then explodes -- the dense unitary is a 2^n x 2^n matrix built gate-by-gate "
        f"(at the largest width we even attempt, **n={n_last}**, exhaustive takes "
        f"**{t_last:.1f} s** vs QCEC's **{tq_last:.3f} s**, a "
        f"**~{speedup:.0f}x** gap). QCEC stays under **{max_qcec:.2f} s** across the "
        f"WHOLE 6-22 qubit range. The crossover -- where exhaustive stops being "
        f"practical and QCEC takes over -- is therefore around "
        f"**n = 11-12 qubits**.\n")

    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


def main():
    if not HAVE_SCALABLE:
        print(f"FATAL: scalable_verification not importable: {_SCALABLE_ERR}")
        print("Is the feat/scalable-verification QCPT worktree on sys.path?")
        sys.exit(1)

    print("=== scale eval: large benchmarks ===")
    suite, rows = run()

    print("\n=== phase-aware exhaustive spot-check (tiny instances) ===")
    spotcheck = phase_aware_small_spotcheck()
    for s in spotcheck:
        if "error" in s:
            print(f"  {s['name']:20s} n={s['n']}: ERROR {s['error']}")
        else:
            print(f"  {s['name']:20s} n={s['n']}: verified={s['verified']} "
                  f"pairs={s['sites_applied']} phase-aware+{s['phase_aware_admitted']} "
                  f"{s['certification']} ({s['wall_s']}s)")

    print("\n=== crossover micro-benchmark (exhaustive vs QCEC vs width) ===")
    cross = crossover_curve()
    for c in cross:
        te = "infeasible" if c["t_exhaustive"] is None else f"{c['t_exhaustive']:.4f}s"
        print(f"  n={c['n']:2d}: exhaustive {te:>12}  qcec {c['t_qcec']:.4f}s  "
              f"(feasible={c['exhaustive_feasible']})")

    write_markdown(OUT_MD, rows, cross, spotcheck)
    print(f"\nwrote {OUT_MD}")

    # headline
    allg = aggregate(rows)
    safe = aggregate(rows, lambda r: r["relphase_safe"])
    big = aggregate(rows, lambda r: r["n"] > EXHAUSTIVE_MAX_QUBITS)
    print("\n=== headline ===")
    print(f"  ALL  ({allg['n']}): 2q {allg['red_2q_vs_exact']:.1f}% / "
          f"infid {allg['red_inf_vs_exact']:.1f}% vs exact-only")
    print(f"  SAFE ({safe['n']}): 2q {safe['red_2q_vs_exact']:.1f}% vs exact-only")
    print(f"  width>{EXHAUSTIVE_MAX_QUBITS} ({big['n']}): 2q "
          f"{big['red_2q_vs_exact']:.1f}% vs exact-only (QCEC-certified)")
    n_qcec = sum(1 for r in rows if r["cert_method"] == "qcec")
    n_exh = sum(1 for r in rows if r["cert_method"] == "exhaustive")
    n_bad = sum(1 for r in rows if r["cert_equiv"] is not True)
    print(f"  certification: {n_exh} exhaustive, {n_qcec} QCEC; "
          f"{n_bad} not-equivalent/undecided")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Baseline error-budget numbers for the paper's Table II.

Runs REAL optimizers / decomposers over the shared benchmark suite
(``benchmarks.benchmark_suite()``, 12 CCX/MCX circuits) and scores every
method's output with the SAME metric used by our gated selector:
QCPT's :class:`HardwareErrorModel` (two-qubit infidelity + two-qubit
count) and :class:`ExactEquivalenceVerifier` for correctness.

So every row is apples-to-apples: each method takes the input CCX/MCX network,
lowers/optimizes it to a two-qubit basis (cx + 1q), and we report
{2q-infidelity, 2q-count, depth, ancilla added, output-verified?} measured by
the *identical* tooling.

Methods
-------
  qiskit_l3   : transpile(circ, basis_gates=['cx','u'], optimization_level=3)
  tket        : pytket FullPeepholeOptimise + rebase to {CX, TK1}   (if installed)
  qcontext    : NOT RUN -- no public implementation (arXiv:2302.02003); see note.
  exact_only  : ErrorBudgetSelector().decompose_exact_only  (our all-exact baseline)
  ours        : ErrorBudgetSelector().select                (gated, verified)

Run:  python baselines.py     (writes baseline_results.md)
"""

from __future__ import annotations

from _paths import EXPERIMENTS_DIR  # noqa: F401  (must be first)

from qiskit import QuantumCircuit, transpile

import benchmarks
from toffoli_optimizer.core.error_model import HardwareErrorModel
from toffoli_optimizer.core.equivalence_verifier import ExactEquivalenceVerifier
from toffoli_optimizer.core.decomposition_selector import ErrorBudgetSelector

EM = HardwareErrorModel()
VERIFIER = ExactEquivalenceVerifier()

# Optional dependency: pytket -----------------------------------------------------
try:
    from pytket import OpType
    from pytket.passes import FullPeepholeOptimise, AutoRebase
    from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
    HAVE_TKET = True
except Exception as exc:  # pragma: no cover
    HAVE_TKET = False
    _TKET_ERR = repr(exc)


# --------------------------------------------------------------------------- score
def _depth_2q(circ: QuantumCircuit) -> int:
    """Two-qubit-gate depth (critical path counting only entangling gates)."""
    return circ.depth(lambda inst: inst.operation.num_qubits >= 2)


def score(name, out_circ, ref_circ, allow_perm):
    """Measure a lowered circuit with the shared HardwareErrorModel + verifier.

    ``ref_circ`` is the original CCX/MCX network the method was asked to realise;
    correctness is verified against it (up to ancilla and, if allow_perm, output
    permutation).
    """
    n_anc = out_circ.num_qubits - ref_circ.num_qubits
    # Correctness = subroutine equivalence on the input subspace (clean ancillas
    # pinned to |0>), the semantics of the paper; identical for every method.
    try:
        from _clean import clean_ancillas
        from qiskit.quantum_info import Operator
        from toffoli_optimizer.core.subspace_check import certify_on_input_subspace
        if n_anc:
            raise ValueError("method added ancillas")
        pm = sum(1 << q for q in clean_ancillas(ref_circ))
        inputs = [x for x in range(2 ** ref_circ.num_qubits) if not x & pm]
        exact = ErrorBudgetSelector().decompose_exact_only(ref_circ)
        ok = bool(certify_on_input_subspace(Operator(exact).data, Operator(out_circ).data, inputs, "subroutine")[0])
        if not ok and allow_perm:
            ok, perm, info = VERIFIER.verify(ref_circ, out_circ, allow_permutation=True)
        info = {"reason": "input-subspace certificate"}
    except Exception as exc:
        ok, info = None, {"reason": f"verify raised {exc!r}"}
    return {
        "method": name,
        "infid": EM.circuit_infidelity(out_circ),
        "twoq": EM.two_qubit_count(out_circ),
        "depth": out_circ.depth(),
        "depth2q": _depth_2q(out_circ),
        "ancilla": n_anc,
        "verified": ok,
        "verify_reason": None if ok else info.get("reason"),
    }


# ------------------------------------------------------------------------- methods
def m_qiskit_l3(circ: QuantumCircuit) -> QuantumCircuit:
    """Qiskit transpiler at optimization_level 3, lowered to {cx, u}."""
    return transpile(circ, basis_gates=["cx", "u"], optimization_level=3)


def m_tket(circ: QuantumCircuit) -> QuantumCircuit:
    """pytket FullPeepholeOptimise, then rebase to {CX, TK1} (a 2q basis).

    ``tk_to_qiskit`` reorders the qubit registers (alphabetically), which would
    otherwise present a different qubit->index wiring than the input and spuriously
    fail equivalence. We remap the output qubits back onto the input's register
    order (by (register-name, index)), so the verifier sees the same wiring. This
    is a relabeling only; it does not touch the gate sequence or the 2q count.
    """
    tkc = qiskit_to_tk(circ)
    # allow_swaps=False: with the default, FullPeepholeOptimise may end in implicit
    # wire swaps that tk_to_qiskit drops, so the output would differ from the input
    # by a qubit permutation. A subroutine cannot permute its outputs for free.
    FullPeepholeOptimise(allow_swaps=False).apply(tkc)
    AutoRebase({OpType.CX, OpType.TK1}).apply(tkc)
    out = tk_to_qiskit(tkc)

    name2idx = {}
    for i, q in enumerate(circ.qubits):
        reg = circ.find_bit(q).registers[0]
        name2idx[(reg[0].name, reg[1])] = i
    perm = []
    for q in out.qubits:
        reg = out.find_bit(q).registers[0]
        perm.append(name2idx[(reg[0].name, reg[1])])

    remapped = QuantumCircuit(circ.num_qubits)
    for inst in out.data:
        qb = [perm[out.find_bit(q).index] for q in inst.qubits]
        remapped.append(inst.operation, [remapped.qubits[i] for i in qb])
    return remapped


def m_exact_only(circ: QuantumCircuit) -> QuantumCircuit:
    return ErrorBudgetSelector().decompose_exact_only(circ)


def m_ours(circ: QuantumCircuit) -> QuantumCircuit:
    from _clean import clean_ancillas
    return ErrorBudgetSelector().select(circ, pinned_zero=clean_ancillas(circ))["circuit"]


def m_count_greedy(circ: QuantumCircuit) -> QuantumCircuit:
    """Count-greedy baseline: the same 3-CX relative-phase gadget as the pass at
    EVERY Toffoli, no context check, no post-optimisation. Identical gadget and
    identical cost accounting as the pass, so the comparison isolates the
    admissibility decision. Unsound in subroutine semantics."""
    from toffoli_optimizer.core.decomposition_selector import (
        ErrorBudgetSelector, append_relative_phase_ccx)
    out = QuantumCircuit(circ.num_qubits)
    for inst in circ.data:
        qb = [circ.find_bit(q).index for q in inst.qubits]
        if inst.operation.name.lower() == "ccx" or (inst.operation.name.lower() in ("mcx", "mcx_gray") and len(qb) == 3):
            append_relative_phase_ccx(out, *qb)
        else:
            tmp = QuantumCircuit(circ.num_qubits); tmp.append(inst.operation, [tmp.qubits[i] for i in qb])
            out.compose(ErrorBudgetSelector().decompose_exact_only(tmp), inplace=True)
    return out


# ----------------------------------------------------------------------------- run
def aggregate(rows):
    """Sum/mean a per-circuit list of score dicts into one method total."""
    n = len(rows)
    return {
        "infid": sum(r["infid"] for r in rows),
        "twoq": sum(r["twoq"] for r in rows),
        "depth_total": sum(r["depth"] for r in rows),
        "depth_mean": sum(r["depth"] for r in rows) / n if n else 0.0,
        "depth2q_total": sum(r["depth2q"] for r in rows),
        "ancilla": sum(r["ancilla"] for r in rows),
        "all_verified": all(r["verified"] for r in rows),
        "n_verified": sum(1 for r in rows if r["verified"]),
        "n": n,
    }


def run():
    suite = benchmarks.benchmark_suite()
    print(f"benchmark suite: {len(suite)} circuits\n")

    methods = [("qiskit_l3", m_qiskit_l3)]
    if HAVE_TKET:
        methods.append(("tket", m_tket))
    methods.append(("exact_only", m_exact_only))
    methods.append(("ours", m_ours))

    # Per-method aggregate rows.
    agg = {}
    per_circuit = {}  # method -> list of score dicts
    for mname, fn in methods:
        rows = []
        for circ in suite:
            out = fn(circ)
            # Decomposed/transpiled outputs are non-classical (u/ry/tk1); allow an
            # output-wire permutation since transpilers may relabel via layout.
            rows.append(score(mname, out, circ, allow_perm=True))
        per_circuit[mname] = rows
        agg[mname] = aggregate(rows)

    return suite, methods, agg, per_circuit


# ----------------------------------------------------------------------- reporting
def fmt_table(agg, methods):
    hdr = f"{'method':12} {'2q-infid':>10} {'2q-count':>8} {'depth(tot)':>10} {'depth(mean)':>11} {'2q-depth':>8} {'ancilla':>7} {'verified':>10}"
    lines = [hdr, "-" * len(hdr)]
    order = [m for m, _ in methods]
    for m in order:
        a = agg[m]
        v = f"{a['n_verified']}/{a['n']}"
        lines.append(
            f"{m:12} {a['infid']:>10.4f} {a['twoq']:>8d} {a['depth_total']:>10d} "
            f"{a['depth_mean']:>11.2f} {a['depth2q_total']:>8d} {a['ancilla']:>7d} {v:>10}"
        )
    return "\n".join(lines)


def reductions(agg, methods):
    """Relative reduction of 'ours' vs each other method (2q-infid and 2q-count)."""
    base = agg["ours"]
    out = []
    for m, _ in methods:
        if m == "ours":
            continue
        a = agg[m]
        d_inf = (a["infid"] - base["infid"]) / a["infid"] * 100 if a["infid"] else 0.0
        d_2q = (a["twoq"] - base["twoq"]) / a["twoq"] * 100 if a["twoq"] else 0.0
        out.append((m, a["infid"], base["infid"], d_inf, a["twoq"], base["twoq"], d_2q))
    return out


def write_markdown(path, suite, methods, agg, per_circuit):
    order = [m for m, _ in methods]
    L = []
    L.append("# Table II baselines — error-budget comparison\n")
    L.append(
        "Every method takes the same CCX/MCX benchmark network "
        f"({len(suite)} circuits from `benchmarks.benchmark_suite()`), lowers it to a "
        "two-qubit basis (CX + 1q), and is scored with the IDENTICAL metric used by "
        "our gated selector: QCPT's `HardwareErrorModel` "
        f"(p2q={EM.p2q}, p1q={EM.p1q}) for two-qubit infidelity / two-qubit count, "
        "and `ExactEquivalenceVerifier` (up to ancilla + output permutation) for "
        "correctness. Infidelity and 2q-count are SUMMED over the suite; depth is "
        "reported as total and mean; ancilla is the total extra qubits added.\n")

    # main table
    L.append("## Per-method totals\n")
    L.append("| Method | 2q-infidelity (sum) | 2q-count (sum) | Depth (total) | Depth (mean) | 2q-depth (total) | Ancilla added | Verified |")
    L.append("|---|---|---|---|---|---|---|---|")
    label = {
        "qiskit_l3": "Qiskit opt-3",
        "tket": "tket (FullPeephole)",
        "exact_only": "Exact-only (ours, all-exact)",
        "ours": "**Ours (gated)**",
    }
    for m in order:
        a = agg[m]
        L.append(
            f"| {label.get(m, m)} | {a['infid']:.4f} | {a['twoq']} | {a['depth_total']} | "
            f"{a['depth_mean']:.2f} | {a['depth2q_total']} | {a['ancilla']} | "
            f"{a['n_verified']}/{a['n']} |")

    # QContext row (not run, justified)
    L.append("| QContext (arXiv:2302.02003) | — | — | — | — | — | — | not run |")
    L.append("")
    L.append(
        "> **QContext** has no usable public implementation. We cite its reported "
        "approach: it optimizes two-qubit **gate count** via context-aware "
        "synthesis but provides neither a hardware-fidelity (error-budget) model "
        "nor an exact equivalence certificate, so it is not directly comparable on "
        "the infidelity / verified columns. We therefore leave its row not-run "
        "rather than fabricate numbers.\n")

    # reductions
    L.append("## Relative reduction of Ours (gated) vs each baseline\n")
    L.append("| vs Method | their 2q-infid | our 2q-infid | infid reduction | their 2q | our 2q | 2q reduction |")
    L.append("|---|---|---|---|---|---|---|")
    for m, ai, bi, di, aq, bq, dq in reductions(agg, methods):
        L.append(f"| {label.get(m, m)} | {ai:.4f} | {bi:.4f} | {di:.1f}% | {aq} | {bq} | {dq:.1f}% |")
    L.append("")

    # per-circuit appendix
    L.append("## Per-circuit detail (2q-count / verified)\n")
    head = "| Circuit | " + " | ".join(label.get(m, m) for m in order) + " |"
    L.append(head)
    L.append("|" + "---|" * (len(order) + 1))
    names = [c.name for c in suite]
    for i, nm in enumerate(names):
        cells = []
        for m in order:
            r = per_circuit[m][i]
            vmark = "ok" if r["verified"] else ("X" if r["verified"] is False else "?")
            cells.append(f"{r['twoq']} ({vmark})")
        L.append(f"| {nm} | " + " | ".join(cells) + " |")
    L.append("")

    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


def main():
    suite, methods, agg, per_circuit = run()

    print(fmt_table(agg, methods))
    print()

    # cross-checks against the known numbers
    print("cross-checks:")
    print(f"  exact_only: 2q={agg['exact_only']['twoq']} (expected 281), "
          f"infid={agg['exact_only']['infid']:.2f} (expected 2.74)")
    print(f"  ours:       2q={agg['ours']['twoq']} (paper: 194), "
          f"infid={agg['ours']['infid']:.2f} (paper: 1.97)")
    print()

    print("ours (gated) vs each baseline:")
    for m, ai, bi, di, aq, bq, dq in reductions(agg, methods):
        print(f"  vs {m:11}: 2q-infid {ai:.4f} -> {bi:.4f} ({di:+.1f}%), "
              f"2q {aq} -> {bq} ({dq:+.1f}%)")

    if not HAVE_TKET:
        print(f"\nNOTE: pytket NOT available ({_TKET_ERR}); tket row skipped.")

    out_md = str(EXPERIMENTS_DIR / "baseline_results.md")
    write_markdown(out_md, suite, methods, agg, per_circuit)
    print(f"\nwrote {out_md}")


if __name__ == "__main__":
    main()

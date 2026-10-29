"""Safety-ablation row (b): OUR admissibility flagging WITHOUT the verification gate.

The safety ablation of the paper (Table~\\ref{tab:safety}) has three configs:

  (a) count-greedy relative-phase substitution (QContext/Maslov-style): replace
      EVERY CCX with the cheap relative-phase gadget, no analysis, no check.
      -> 6/12 silent errors (see safety_experiment.py / naive_relphase.py).

  (b) OURS WITHOUT the verification gate  <-- this script.
      Apply OUR context analysis -- the selector's structural compute/uncompute
      pairing (find_relative_phase_safe_sites) PLUS the phase-observability
      admissibility test (is_phase_unobservable on the reachable subspace) -- to
      FLAG which standalone Toffolis / pairs are admissible, then commit the
      relative-phase gadget at every flagged site BUT SKIP the per-instance
      ExactEquivalenceVerifier gate. Audit each result afterwards with the exact
      verifier (allow_permutation) and count silent errors.

  (c) OURS WITH the verification gate (the full ErrorBudgetSelector): 0/12.

The point of (b) is to isolate the contribution of the verification gate from the
contribution of the context analysis. Our context analysis is conservative-and-
sound BY CONSTRUCTION (every predicate over-approximates reachability and rejects
on the first chance of interference), so we expect (b) to be small. If M==0, the
analysis alone already never over-admits on this suite; if M>0, the analysis alone
can over-admit, and only the gate catches it. Either way (a)'s 6/12 shows that
AGGRESSIVE pattern-matched substitution is unsafe, and the gate is what licenses
our aggressive admissibility to be *trusted* as sound rather than relied on.

Uses QCPT's committed modules read-only; does not modify the repo.
"""

from __future__ import annotations

import os

from _paths import EXPERIMENTS_DIR  # noqa: F401  (must be first)

from qiskit import QuantumCircuit
from qiskit.circuit.library import RCCXGate

from toffoli_optimizer.core.equivalence_verifier import ExactEquivalenceVerifier
from toffoli_optimizer.core.context_analysis import find_relative_phase_safe_sites
from toffoli_optimizer.core.phase_observability import (
    is_phase_unobservable,
    default_affected_qubits,
)

from benchmarks import benchmark_suite

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESULTS_MD = os.path.join(_HERE, "ablation_b_results.md")

_CCX_NAMES = {"ccx", "mcx", "mcx_gray"}


def _ccx_index_qubits(circuit, idx):
    inst = circuit.data[idx]
    return [circuit.find_bit(q).index for q in inst.qubits]


def admissible_relphase_indices(circuit: QuantumCircuit) -> set:
    """OUR context-analysis admissibility FLAGGING, with NO per-instance verify.

    Replicates exactly the admissibility flagging the ErrorBudgetSelector uses,
    but stops at the structural / phase-observability decision -- it never calls
    the ExactEquivalenceVerifier per site. Returns the set of circuit.data
    indices the analysis would FLAG as relative-phase-admissible.

    Two flagging paths, mirroring decomposition_selector.select:
      * pairs: every (compute_idx, uncompute_idx) from find_relative_phase_safe_sites;
      * standalone: any remaining plain 3-qubit CCX whose introduced relative
        phase is_phase_unobservable downstream (the forward-cone / reachable test).
    """
    flagged: set = set()

    # Path 1: structural compute/uncompute pairs (context_analysis).
    sites = find_relative_phase_safe_sites(circuit)
    paired_idx = set()
    for s in sites:
        flagged.add(s.compute_idx)
        flagged.add(s.uncompute_idx)
        paired_idx.add(s.compute_idx)
        paired_idx.add(s.uncompute_idx)

    # Path 2: phase-observability-aware standalone admissibility.
    for idx, inst in enumerate(circuit.data):
        if idx in flagged:
            continue
        name = inst.operation.name.lower()
        qb = _ccx_index_qubits(circuit, idx)
        if name not in _CCX_NAMES or len(qb) != 3:
            continue
        affected = default_affected_qubits(circuit, idx)
        if is_phase_unobservable(circuit, idx, affected):
            flagged.add(idx)

    return flagged


def build_relphase(circuit: QuantumCircuit, indices: set) -> QuantumCircuit:
    """Copy of ``circuit`` with the CCX at each index in ``indices`` replaced by an
    RCCX (relative-phase / Margolus) gate on the same qubits; all else verbatim.

    NB: we substitute WITHOUT any verification -- this is config (b)."""
    new = QuantumCircuit(*circuit.qregs, *circuit.cregs, name=circuit.name)
    for i, inst in enumerate(circuit.data):
        if i in indices and inst.operation.name.lower() in _CCX_NAMES:
            new.append(RCCXGate(), inst.qubits)
        else:
            new.append(inst.operation, inst.qubits, inst.clbits)
    return new


def run():
    verifier = ExactEquivalenceVerifier()
    suite = benchmark_suite()

    rows = []
    silent_errors = 0
    total_flagged = 0

    for original in suite:
        flagged = admissible_relphase_indices(original)
        candidate = build_relphase(original, flagged)

        # AUDIT (not a gate): is the un-gated, analysis-only result still correct?
        ok, _perm, info = verifier.verify(original, candidate, allow_permutation=True)
        if not ok:
            silent_errors += 1
        total_flagged += len(flagged)

        rows.append({
            "name": original.name,
            "qubits": original.num_qubits,
            "safe_truth": getattr(original, "relphase_safe", None),
            "flagged": len(flagged),
            "correct": ok,
            "reason": info.get("reason", ""),
        })

    summary = {
        "n_circuits": len(suite),
        "silent_errors": silent_errors,
        "total_flagged": total_flagged,
    }
    return rows, summary


def write_markdown(rows, summary):
    lines = []
    lines.append("# Safety-ablation row (b): our context analysis WITHOUT the verification gate\n")
    lines.append(
        "Apply the selector's relative-phase **admissibility flagging** "
        "(structural compute/uncompute pairing `find_relative_phase_safe_sites` + "
        "the phase-observability test `is_phase_unobservable` on the reachable "
        "subspace), commit the relative-phase gadget at every flagged site, but "
        "**skip** the per-instance `ExactEquivalenceVerifier` gate. Each result is "
        "then audited with the exact verifier (`allow_permutation=True`); a failure "
        "is a silent error the analysis alone admitted.\n")
    lines.append("\n## Per-circuit results\n")
    lines.append("| circuit | qubits | relphase-safe (truth) | sites flagged | "
                 "audited correct? |")
    lines.append("|---|---:|:--:|---:|:--:|")
    for r in rows:
        truth = {True: "safe", False: "UNSAFE", None: "?"}[r["safe_truth"]]
        ok = "yes" if r["correct"] else "**NO**"
        lines.append(
            f"| `{r['name']}` | {r['qubits']} | {truth} | {r['flagged']} | {ok} |")

    n = summary["n_circuits"]
    se = summary["silent_errors"]
    tf = summary["total_flagged"]
    lines.append("\n## Aggregate (row (b))\n")
    lines.append(
        f"**Our analysis-only (un-gated) pass introduces {se} of {n} silent "
        f"errors, flagging {tf} relative-phase sites across the suite.**\n")
    lines.append(
        "\nContrast with (a) count-greedy = 6/12 and (c) gated = 0/12. The gap "
        "between (a) and (b) is the contribution of the context analysis; the gap "
        "between (b) and (c) is the contribution of the verification gate.\n")

    with open(_RESULTS_MD, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    rows, summary = run()

    hdr = f"{'circuit':28s} {'q':>2s} {'truth':>6s} {'flagged':>7s} {'audit':>6s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        truth = {True: "safe", False: "UNSAFE", None: "?"}[r["safe_truth"]]
        print(f"{r['name']:28s} {r['qubits']:>2d} {truth:>6s} {r['flagged']:>7d} "
              f"{('ok' if r['correct'] else 'WRONG'):>6s}")
    print("-" * len(hdr))
    print(f"\nROW (b): our analysis-only (un-gated) pass introduces "
          f"{summary['silent_errors']} of {summary['n_circuits']} silent errors "
          f"(flagged {summary['total_flagged']} relative-phase sites).")

    write_markdown(rows, summary)
    print(f"\nTable written to: {_RESULTS_MD}")
    return summary


if __name__ == "__main__":
    main()

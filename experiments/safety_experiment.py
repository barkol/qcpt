"""SAFETY experiment: pattern-matched / count-greedy relative-phase Toffoli
substitution produces SILENTLY INCORRECT circuits; a per-instance verification
gate makes aggressive selection sound.

For every benchmark circuit we run two passes:

  * NAIVE  : substitute every CCX with a relative-phase (Margolus / RCCX)
             Toffoli, no checking. Then ask the EXACT verifier whether the
             result still computes the original function. Each FAILURE is a
             SILENT ERROR -- the count-greedy optimizer would have shipped a
             wrong circuit.

  * GATED  : substitute per gate, but keep each substitution only if the
             per-window exact verification passes. The result is verified again
             at the end (must always pass), and we record how many valid
             relative-phase decompositions survived.

Headline: naive silently corrupts X of N circuits; the verification gate yields
0 errors while still applying R valid relative-phase decompositions.

Uses QCPT's STABLE committed ExactEquivalenceVerifier. Does not modify it.
"""

from __future__ import annotations

from _paths import EXPERIMENTS_DIR  # noqa: F401  (must be first)

from toffoli_optimizer.core.equivalence_verifier import ExactEquivalenceVerifier
from benchmarks import benchmark_suite
from naive_relphase import naive_relphase, gated_relphase

_RESULTS_MD = str(EXPERIMENTS_DIR / "safety_results.md")


def _count_ccx(circuit):
    return sum(1 for inst in circuit.data if inst.operation.name == "ccx")


def run():
    verifier = ExactEquivalenceVerifier()
    suite = benchmark_suite()

    rows = []
    naive_silent_errors = 0
    gated_silent_errors = 0
    total_relphase_gated = 0

    for original in suite:
        name = original.name
        nq = original.num_qubits
        nccx = _count_ccx(original)
        ground_truth_safe = getattr(original, "relphase_safe", None)

        # --- naive pass ---
        naive_circ = naive_relphase(original)
        naive_ok, _p, naive_info = verifier.verify(
            original, naive_circ, allow_permutation=True)
        if not naive_ok:
            naive_silent_errors += 1

        # --- gated pass ---
        gated_circ = gated_relphase(original, verifier)
        gated_ok, _p2, gated_info = verifier.verify(
            original, gated_circ, allow_permutation=True)
        if not gated_ok:
            gated_silent_errors += 1
        applied = getattr(gated_circ, "relphase_applied", 0)
        total_relphase_gated += applied

        rows.append({
            "name": name,
            "qubits": nq,
            "ccx": nccx,
            "safe_truth": ground_truth_safe,
            "naive_correct": naive_ok,
            "gated_correct": gated_ok,
            "relphase_gated": applied,
            "naive_reason": naive_info.get("reason", ""),
        })

    n = len(suite)
    summary = {
        "n_circuits": n,
        "naive_silent_errors": naive_silent_errors,
        "gated_silent_errors": gated_silent_errors,
        "total_relphase_gated": total_relphase_gated,
    }
    return rows, summary


def _fmt_bool(b):
    return "yes" if b else "**NO**"


def write_markdown(rows, summary):
    lines = []
    lines.append("# Safety experiment: verified vs. count-greedy relative-phase substitution\n")
    lines.append(
        "Count-greedy / pattern-matched relative-phase (Margolus / RCCX) Toffoli "
        "substitution -- the kind a template optimizer performs when it trusts its "
        "rewrite library -- is applied two ways over the benchmark suite:\n")
    lines.append(
        "* **naive**: replace every CCX with a relative-phase Toffoli, no checking.\n"
        "* **gated**: same per-gate substitution, kept only if a per-window exact "
        "verification (QCPT `ExactEquivalenceVerifier`) proves the circuit "
        "still computes the original function.\n")
    lines.append("\n## Per-circuit results\n")
    lines.append("| circuit | qubits | CCX | relphase-safe (truth) | naive correct? | "
                 "gated correct? | relphase applied (gated) |")
    lines.append("|---|---:|---:|:--:|:--:|:--:|---:|")
    for r in rows:
        truth = {True: "safe", False: "UNSAFE", None: "?"}[r["safe_truth"]]
        lines.append(
            f"| `{r['name']}` | {r['qubits']} | {r['ccx']} | {truth} | "
            f"{_fmt_bool(r['naive_correct'])} | {_fmt_bool(r['gated_correct'])} | "
            f"{r['relphase_gated']} |")

    n = summary["n_circuits"]
    nse = summary["naive_silent_errors"]
    gse = summary["gated_silent_errors"]
    rg = summary["total_relphase_gated"]
    lines.append("\n## Aggregate (headline)\n")
    lines.append(
        f"**Naive substitution silently corrupts {nse} of {n} circuits; the "
        f"verification gate yields {gse} errors while still applying {rg} valid "
        f"relative-phase decompositions.**\n")
    lines.append(
        "\nEvery naive failure is a circuit that the count-greedy optimizer would "
        "have shipped while computing the WRONG function -- detected here only "
        "because we ran an exact equivalence check after the fact. The gated pass "
        "keeps exactly the relative-phase decompositions that are provably valid "
        "(the compute/uncompute ones) and falls back to exact Toffolis everywhere "
        "else, so it is sound by construction.\n")

    with open(_RESULTS_MD, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    rows, summary = run()

    # console table
    hdr = f"{'circuit':28s} {'q':>2s} {'ccx':>3s} {'truth':>6s} {'naive':>6s} {'gated':>6s} {'rp':>3s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        truth = {True: "safe", False: "UNSAFE", None: "?"}[r["safe_truth"]]
        print(f"{r['name']:28s} {r['qubits']:>2d} {r['ccx']:>3d} {truth:>6s} "
              f"{('ok' if r['naive_correct'] else 'WRONG'):>6s} "
              f"{('ok' if r['gated_correct'] else 'WRONG'):>6s} "
              f"{r['relphase_gated']:>3d}")
    print("-" * len(hdr))
    n = summary["n_circuits"]
    print(f"\nHEADLINE: naive substitution silently corrupts "
          f"{summary['naive_silent_errors']} of {n} circuits; "
          f"the verification gate yields {summary['gated_silent_errors']} errors "
          f"while still applying {summary['total_relphase_gated']} valid "
          f"relative-phase decompositions.")

    write_markdown(rows, summary)
    print(f"\nTable written to: {_RESULTS_MD}")

    # Make the paper's safety result a hard assertion.
    assert summary["gated_silent_errors"] == 0, "GATED pass produced a silent error!"
    assert summary["naive_silent_errors"] > 0, "naive pass produced no silent errors (benchmark too weak)"
    return summary


if __name__ == "__main__":
    main()

"""Error-ratio sensitivity sweep + real-device operating points.

This experiment probes how the budget-aware Toffoli decomposition selector
(:class:`ErrorBudgetSelector`) behaves as the hardware error model's
two-qubit/single-qubit error *ratio* ``r = p2q / p1q`` is varied, and at the real
(p2q, p1q) operating points of leading quantum platforms.

Two things are measured for each (circuit, error model):

  * the SELECTED DECOMPOSITION MIX -- how many Toffolis were lowered with the
    exact (6-CX) gadget vs the cheap relative-phase / Margolus (3-CX) gadget,
    split by the admission path (compute/uncompute pair vs standalone
    phase-observability), and
  * the resulting two-qubit-gate COUNT and total circuit INFIDELITY, against the
    exact-only baseline.

SCIENTIFIC QUESTION (deliverable 1): does the budget-optimal selection CHANGE with
the ratio r? The intuition is that when 2q gates dominate (high r) the cheap
relative-phase decomposition should win big, while at low r the exact gadget might
be preferred. We measure this directly and report the answer honestly -- including
the case where the selection is invariant because the selector's *admission*
criterion is correctness-driven (structural + reachable-subspace verification) and
does NOT consult the error model.

REAL DEVICES (deliverable 2): we additionally evaluate the selector at each
platform's published (p2q, p1q) and report the budget improvement delivered on
that real hardware. Device numbers + citations live in ``DEVICES`` below and are
reproduced (with sources/dates) in ``sensitivity_results.md``.

Run:
    (in a Python environment with the requirements of pyproject.toml)
    pip install -e ".[experiments]"
    python experiments/sensitivity_sweep.py

Outputs:
    experiments/fixtures/sensitivity_data.csv (machine-readable, for figures)
    stdout tables (also captured into sensitivity_results.md by hand)
"""

from __future__ import annotations

import csv
from dataclasses import dataclass

from _paths import EXPERIMENTS_DIR  # noqa: F401  (must be first)

from qiskit import QuantumCircuit  # noqa: E402

from toffoli_optimizer.core.optimizer import (  # noqa: E402
    ErrorBudgetSelector,
    HardwareErrorModel,
)

from benchmarks import benchmark_suite  # noqa: E402


# --------------------------------------------------------------------------- #
# Real-device operating points. Every number is a published figure with a       #
# citation + date in sensitivity_results.md. p2q / p1q are gate ERROR rates     #
# (infidelities), not fidelities.                                               #
# --------------------------------------------------------------------------- #
@dataclass
class Device:
    key: str
    label: str
    p2q: float      # two-qubit gate error (infidelity)
    p1q: float      # single-qubit gate error (infidelity)
    note: str


DEVICES = [
    # IBM Heron r2 (ibm_fez): CZ error 2.85e-3 (Pelofske/IBM survey, arXiv:2410.00916,
    # Oct 2024). SX (1q) error ~2.5e-4 is IBM's commonly-quoted typical for Heron r2;
    # treated as approximate.
    Device("ibm_heron_r2", "IBM Heron r2 (ibm_marrakesh)", 1.8e-3, 2.5e-4,
           "CZ ~1.8e-3 (ibm_marrakesh, arXiv:2510.13577); 1q ~2.5e-4 (IBM typical)"),
    # Google Willow (105q, Device 1): CZ 0.33%, 1q 0.035% simultaneous benchmarking
    # (Nature 2024 / Willow spec sheet, Dec 2024).
    Device("google_willow", "Google Willow (105q)", 3.3e-3, 3.5e-4,
           "CZ 0.33%, 1q 0.035% simultaneous (Nature 2024)"),
    # Quantinuum H2 (System Model H2): 2q 99.87% -> 1.3e-3; 1q 99.997% -> 3e-5
    # (Quantinuum, 2024). All-to-all connectivity.
    Device("quantinuum_h2", "Quantinuum H2", 1.4e-3, 3.0e-5,
           "2q ~1.4e-3 (arXiv:2404.02280); 1q 99.997% -> 3e-5 (Quantinuum H2 spec, 2024) -> r~47"),
    # IonQ Forte: 2q 99.6% -> 4e-3; 1q 99.98% -> 2e-4 (IonQ Forte spec).
    Device("ionq_forte", "IonQ Forte", 4.6e-3, 2.0e-4,
           "2q median 4.6e-3 (DRB, Chen et al., Quantum 8, 1516, 2024); 1q 2.0e-4"),
    # IonQ Aria: 2q 99.6% -> 4e-3; 1q 99.95% -> 5e-4 (IonQ Aria spec).
    Device("ionq_aria", "IonQ Aria", 4.0e-3, 5.0e-4,
           "2q 99.6% -> 4e-3; 1q 99.95% -> 5e-4 (IonQ Aria spec)"),
]


# Representative benchmark subset (covers the relphase-safe and live regimes, and
# both the pair path and the standalone phase-aware path).
SUBSET_NAMES = [
    "ripple_carry_adder_2b",   # compute/uncompute pairs + phase-aware standalone
    "controlled_adder_2b",     # all admitted via the phase-aware standalone path
    "grover_oracle_mcx3",      # AND-ladder compute/uncompute pairs
    "grover_oracle_mcx4",      # larger AND-ladder
    "nested_compute_uncompute",
    "live_and_chain",          # live -> no reduction expected
    "single_live_toffoli",     # lone live Toffoli -> no reduction
    "half_uncomputed",         # mixed: pair admitted, live kept exact
]


def _subset():
    by_name = {qc.name: qc for qc in benchmark_suite()}
    out = []
    for n in SUBSET_NAMES:
        if n in by_name:
            out.append(by_name[n])
    return out


def _count_ccx(qc: QuantumCircuit) -> int:
    return sum(1 for inst in qc.data if inst.operation.name.lower() in ("ccx", "mcx"))


def _mix_from_report(qc: QuantumCircuit, report) -> dict:
    """Decode the selected decomposition mix from a SelectionReport.

    relphase_total = (#Toffolis lowered with a 3-CX gadget) =
        2 * (#pairs under (C) and (W)) + (#standalone gadgets under (R) or (U)).
    Control drops under (R) are counted separately as ``drops``.
    exact = total CCX - relphase_total - drops.
    """
    total = _count_ccx(qc)
    pairs = report["sites_applied"] + len(report.get("window_pairs_admitted", []))
    standalone = len(report["phase_aware_admitted"]) + len(report.get("rphase_admitted", []))
    drops = len(report.get("approx_admitted", []))
    relphase = 2 * pairs + standalone
    exact = total - relphase - drops
    return {
        "drops": drops,
        "ccx_total": total,
        "relphase": relphase,
        "relphase_pairs": 2 * pairs,
        "relphase_standalone": standalone,
        "exact": exact,
    }


def run_sweep(ratios, p1q_fixed=1e-3):
    """Sweep r = p2q/p1q with p1q fixed; p2q = r * p1q.

    Returns a list of row dicts (one per circuit x ratio).
    """
    rows = []
    for qc in _subset():
        for r in ratios:
            p2q = r * p1q_fixed
            em = HardwareErrorModel(p2q=p2q, p1q=p1q_fixed)
            sel = ErrorBudgetSelector(error_model=em, phase_aware=True)
            res = sel.select(qc, pinned_zero=__import__("_clean").clean_ancillas(qc))
            rep = res["report"]
            mix = _mix_from_report(qc, rep)
            inf_before = rep["infidelity_before"]
            inf_after = rep["infidelity_after"]
            rows.append({
                "mode": "ratio_sweep",
                "device": "",
                "circuit": qc.name,
                "relphase_safe": qc.relphase_safe,
                "p1q": p1q_fixed,
                "p2q": p2q,
                "ratio": r,
                "ccx_total": mix["ccx_total"],
                "exact_decomp": mix["exact"],
                "relphase_decomp": mix["relphase"],
                "relphase_pairs": mix["relphase_pairs"],
                "relphase_standalone": mix["relphase_standalone"],
                "two_qubit_before": rep["two_qubit_before"],
                "two_qubit_after": rep["two_qubit_after"],
                "infidelity_before": inf_before,
                "infidelity_after": inf_after,
                "infidelity_reduction_rel": (
                    (inf_before - inf_after) / inf_before if inf_before > 0 else 0.0
                ),
                "verified": rep["verified"],
            })
    return rows


def run_devices():
    """Evaluate the selector at each real device's (p2q, p1q)."""
    rows = []
    for dev in DEVICES:
        em = HardwareErrorModel(p2q=dev.p2q, p1q=dev.p1q)
        sel = ErrorBudgetSelector(error_model=em, phase_aware=True)
        for qc in _subset():
            res = sel.select(qc, pinned_zero=__import__("_clean").clean_ancillas(qc))
            rep = res["report"]
            mix = _mix_from_report(qc, rep)
            inf_before = rep["infidelity_before"]
            inf_after = rep["infidelity_after"]
            rows.append({
                "mode": "device",
                "device": dev.label,
                "circuit": qc.name,
                "relphase_safe": qc.relphase_safe,
                "p1q": dev.p1q,
                "p2q": dev.p2q,
                "ratio": dev.p2q / dev.p1q,
                "ccx_total": mix["ccx_total"],
                "exact_decomp": mix["exact"],
                "relphase_decomp": mix["relphase"],
                "relphase_pairs": mix["relphase_pairs"],
                "relphase_standalone": mix["relphase_standalone"],
                "two_qubit_before": rep["two_qubit_before"],
                "two_qubit_after": rep["two_qubit_after"],
                "infidelity_before": inf_before,
                "infidelity_after": inf_after,
                "infidelity_reduction_rel": (
                    (inf_before - inf_after) / inf_before if inf_before > 0 else 0.0
                ),
                "verified": rep["verified"],
            })
    return rows


def write_csv(rows, path):
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


# --------------------------------------------------------------------------- #
# Pretty printers                                                               #
# --------------------------------------------------------------------------- #
def print_sweep_table(rows, ratios):
    print("\n=== RATIO SWEEP (p1q = 1e-3 fixed; p2q = r*p1q) ===")
    print("Per circuit: does the SELECTED MIX (exact vs relphase) change with r?\n")
    by_circ = {}
    for row in rows:
        by_circ.setdefault(row["circuit"], []).append(row)
    hdr = f"{'circuit':24s} {'safe':>5} {'ccx':>4} | "
    hdr += "  ".join(f"r={r:g}".rjust(9) for r in ratios)
    print(hdr)
    print("-" * len(hdr))
    for circ, rws in by_circ.items():
        rws = sorted(rws, key=lambda x: x["ratio"])
        ccx = rws[0]["ccx_total"]
        safe = rws[0]["relphase_safe"]
        # show relphase-count per ratio (the SELECTION); should it vary?
        cells = "  ".join(f"{x['relphase_decomp']:>9d}" for x in rws)
        print(f"{circ:24s} {str(safe):>5} {ccx:>4} | {cells}")
    print("\n(values = number of Toffolis lowered with the 3-CX relative-phase "
          "gadget, per ratio r)")
    # Determine invariance
    changed = []
    for circ, rws in by_circ.items():
        vals = {x["relphase_decomp"] for x in rws}
        if len(vals) > 1:
            changed.append(circ)
    if changed:
        print(f"\nSELECTION CHANGES with r for: {changed}")
    else:
        print("\nSELECTION IS INVARIANT to r across the entire grid for ALL circuits.")
    # infidelity reduction does still vary numerically with r
    print("\n--- relative infidelity reduction (1 - infid_after/infid_before) by r ---")
    print(hdr)
    print("-" * len(hdr))
    for circ, rws in by_circ.items():
        rws = sorted(rws, key=lambda x: x["ratio"])
        cells = "  ".join(f"{x['infidelity_reduction_rel']*100:>8.2f}%" for x in rws)
        print(f"{circ:24s} {str(rws[0]['relphase_safe']):>5} {rws[0]['ccx_total']:>4} | {cells}")


def print_device_table(rows):
    print("\n\n=== PER-DEVICE OPERATING POINTS ===")
    by_dev = {}
    for row in rows:
        by_dev.setdefault(row["device"], []).append(row)
    for dev, rws in by_dev.items():
        r = rws[0]["ratio"]
        print(f"\n--- {dev}   (p2q={rws[0]['p2q']:.2e}, p1q={rws[0]['p1q']:.2e}, "
              f"r={r:.1f}) ---")
        print(f"{'circuit':24s} {'2q before':>10} {'2q after':>9} "
              f"{'infid before':>13} {'infid after':>12} {'reduction':>10}")
        tot_before = tot_after = 0.0
        for x in sorted(rws, key=lambda z: z["circuit"]):
            print(f"{x['circuit']:24s} {x['two_qubit_before']:>10d} "
                  f"{x['two_qubit_after']:>9d} {x['infidelity_before']:>13.4g} "
                  f"{x['infidelity_after']:>12.4g} "
                  f"{x['infidelity_reduction_rel']*100:>9.2f}%")
        # aggregate: treat the subset as a "workload" by combining fidelities
        # (product of per-circuit fidelities) -> an aggregate infidelity.
        f_before = f_after = 1.0
        for x in rws:
            f_before *= (1.0 - x["infidelity_before"])
            f_after *= (1.0 - x["infidelity_after"])
        agg_before = 1.0 - f_before
        agg_after = 1.0 - f_after
        red = (agg_before - agg_after) / agg_before if agg_before > 0 else 0.0
        print(f"{'  AGGREGATE (subset)':24s} {'':>10} {'':>9} "
              f"{agg_before:>13.4g} {agg_after:>12.4g} {red*100:>9.2f}%")


def main():
    ratios = [1, 3, 10, 30, 100, 300]
    sweep_rows = run_sweep(ratios)
    dev_rows = run_devices()

    print_sweep_table(sweep_rows, ratios)
    print_device_table(dev_rows)

    all_rows = sweep_rows + dev_rows
    csv_path = str(EXPERIMENTS_DIR / "fixtures" / "sensitivity_data.csv")
    write_csv(all_rows, csv_path)
    print(f"\n\nWrote {len(all_rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Certification times on the 12-24 qubit scale suite with the current ErrorBudgetSelector.

For every circuit of ``large_benchmarks.large_suite()`` (ancillas pinned to |0> as in scale_eval.py):
  * the stock ``ErrorBudgetSelector().select`` (deterministic certificate) and the opt-in ``certificate="monte_carlo"``: wall time,
    two-qubit gates before / after, verification method of the output;
  * the whole-circuit certificate of the selected circuit against the all-exact one, timed separately with every method that applies:
    blocks of difference (deterministic, blockcert.py), QCEC (decision diagrams), dense columns (C++, up to 14 qubits), Monte Carlo.
Crossover series: the ripple adder of growing width (the exact lowering against itself with every safe pair substituted), timed with
the exhaustive dense check (up to 12 qubits), QCEC, blocks, and Monte Carlo.

Results go to ``$TOPTOFFOLI_OUT/scale_cert_times.jsonl`` (one JSON row per circuit / width; rerun resumes, finished rows are skipped).
Run on an otherwise idle machine: the times are wall-clock times."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from _paths import EXPERIMENTS_DIR  # noqa: F401  (must be first)

import large_benchmarks as LB
from scale_eval import ScalableErrorBudgetSelector, certify as paper_certify
from _clean import clean_ancillas
from toffoli_optimizer.core.blockcert import block_certificate
from toffoli_optimizer.core.context_analysis import find_relative_phase_safe_sites
from toffoli_optimizer.core.cppcert import compare_up_to_phase, mc_compare
from toffoli_optimizer.core.decomposition_selector import ErrorBudgetSelector
from toffoli_optimizer.core.equivalence_verifier import ExactEquivalenceVerifier
from toffoli_optimizer.core.subspace_check import qcec_certify_on_subspace
from toffoli_optimizer.core.scalable_verification import verify_scalable

OUT = Path(os.environ.get("TOPTOFFOLI_OUT", str(EXPERIMENTS_DIR))) / "scale_cert_times.jsonl"
DENSE_MAX_QUBITS = 14


def _timed(fn):
    t = time.perf_counter()
    try:
        r = fn()
    except Exception as e:  # noqa: BLE001
        return None, time.perf_counter() - t, repr(e)[:120]
    return r, time.perf_counter() - t, None


def certify_all(exact, sel, pins, cert_inputs):
    """Time every whole-circuit certificate that applies to (exact, sel)."""
    n = exact.num_qubits
    out = {}
    r, t, err = _timed(lambda: block_certificate(exact, sel, max_qubits=12))
    out["blocks"] = {"t": t, "ok": bool(r and r[0]), "blocks": (r[1].get("blocks") if r else None), "err": err}
    r, t, err = _timed(lambda: qcec_certify_on_subspace(exact, sel, tuple(pins)))
    out["qcec"] = {"t": t, "ok": bool(r and r[0]), "err": err}
    if n <= DENSE_MAX_QUBITS:
        cols = sorted(set(cert_inputs))
        r, t, err = _timed(lambda: compare_up_to_phase(exact, sel, cols))
        out["dense"] = {"t": t, "ok": bool(r and r[0] == "equal"), "err": err}
    r, t, err = _timed(lambda: mc_compare(exact, sel, cert_inputs, k=24, seed=1))
    out["mc"] = {"t": t, "ok": bool(r and r[0] == "equal"), "failure_probability": (r[1].get("failure_probability") if r else None), "err": err}
    return out


def done_names():
    if not OUT.exists():
        return set()
    return {json.loads(l).get("key") for l in OUT.read_text().splitlines() if l.strip()}


def emit(row):
    with open(OUT, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run_suite(done):
    for qc in LB.large_suite():
        key = "suite:" + qc.name
        if key in done:
            continue
        pins = clean_ancillas(qc)
        row = {"key": key, "kind": "suite", "name": qc.name, "n": qc.num_qubits, "pinned": len(pins), "family": qc.metadata.get("family") if getattr(qc, "metadata", None) else None}
        for mode in ("exact", "monte_carlo"):
            sel_obj = ErrorBudgetSelector(certificate=mode)
            t = time.perf_counter()
            r = sel_obj.select(qc, pinned_zero=tuple(pins))
            dt = time.perf_counter() - t
            rep = r["report"]
            row[mode] = {"select_s": dt, "two_qubit_before": rep["two_qubit_before"], "two_qubit_after": rep["two_qubit_after"],
                         "fell_back": bool(rep.get("fell_back_to_exact")), "method": rep["verify_info"].get("method"),
                         "failure_probability": rep["verify_info"].get("failure_probability")}
            if mode == "exact":
                row["certificates"] = certify_all(r["exact"], r["circuit"], pins, r["cert_inputs"])
        emit(row)
        print(key, row["n"], {m: round(row[m]["select_s"], 2) for m in ("exact", "monte_carlo")}, row["exact"]["method"], flush=True)


def run_crossover(done):
    ver = ExactEquivalenceVerifier()
    sel_obj = ErrorBudgetSelector()
    for nbits in range(2, 11):
        key = f"cross2:{nbits}"
        if key in done:
            continue
        qc = LB.ripple_carry_adder(nbits)
        n = qc.num_qubits
        exact = sel_obj.decompose_exact_only(qc)
        idx = set()
        for s in find_relative_phase_safe_sites(qc):
            idx |= {s.compute_idx, s.uncompute_idx}
        cand = sel_obj._build(qc, idx)
        pins = clean_ancillas(qc)
        row = {"key": key, "kind": "cross", "nbits": nbits, "n": n, "gates_exact": len(exact.data), "gates_cand": len(cand.data)}
        r, t, err = _timed(lambda: qcec_certify_on_subspace(exact, cand, tuple(pins)))
        row["qcec"] = {"t": t, "ok": bool(r and r[0]), "err": err}
        r, t, err = _timed(lambda: verify_scalable(exact, cand, prefer="qcec"))
        row["qcec_unpinned"] = {"t": t, "ok": bool(r is not None and r.equivalent is True), "err": err}
        r, t, err = _timed(lambda: block_certificate(exact, cand, max_qubits=12))
        row["blocks"] = {"t": t, "ok": bool(r and r[0]), "err": err}
        full = range(1 << n)
        r, t, err = _timed(lambda: mc_compare(exact, cand, full, k=24, seed=1)) if n <= 22 else (None, None, "n > 22")
        row["mc"] = {"t": t, "ok": bool(r and r[0] == "equal"), "err": err}
        if n <= 12:
            r, t, err = _timed(lambda: ver.verify(exact, cand, allow_permutation=False))
            row["exhaustive"] = {"t": t, "ok": bool(r and r[0]), "err": err}
        emit(row)
        print(key, n, {k: (round(row[k]["t"], 3) if row[k]["t"] is not None else None) for k in ("qcec", "qcec_unpinned", "blocks", "mc", "exhaustive") if k in row}, flush=True)





def run_paper_outputs(done):
    """Certificates of the outputs reported in the paper (ScalableErrorBudgetSelector, QCEC-gated per pair), timed with the method of
    the paper (``certify``: dense up to 12 qubits, QCEC with the pinned qubits above) and with the current certificates."""
    for qc in LB.large_suite():
        key = "paper:" + qc.name
        if key in done:
            continue
        pins = clean_ancillas(qc)
        sres = ScalableErrorBudgetSelector().select(qc, pinned_zero=pins)
        exact, sel = sres["exact"], sres["circuit"]
        n = qc.num_qubits
        pm = sum(1 << q for q in pins)
        cert_inputs = range(1 << n) if not pins else [x for x in range(1 << n) if not x & pm]
        row = {"key": key, "kind": "paper", "name": qc.name, "n": n, "pinned": len(pins),
               "two_qubit_before": sres["two_qubit_before"], "two_qubit_after": sres["two_qubit_after"]}
        r, t, err = _timed(lambda: paper_certify(exact, sel, pinned_zero=pins if sres.get("r_gadgets") else ()))
        row["paper_certify"] = {"t": t, "method": getattr(r, "method", None), "ok": bool(r is not None and r.equivalent is True), "err": err}
        row["certificates"] = certify_all(exact, sel, pins, cert_inputs)
        emit(row)
        print(key, n, row["two_qubit_before"], row["two_qubit_after"], round(t, 3), row["paper_certify"]["method"], flush=True)


if __name__ == "__main__":
    done = done_names()
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "suite"):
        run_suite(done)
    if which in ("all", "paper"):
        run_paper_outputs(done)
    if which in ("all", "cross"):
        run_crossover(done)

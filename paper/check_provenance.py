#!/usr/bin/env python3
"""Check that the stored inputs of the paper in paper/ are byte-identical to the outputs of
the experiment drivers stored in experiments/ (the copies exist so that paper/ is
self-contained). Exits with status 1 on the first mismatch."""
import hashlib
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent
EXP = PAPER.parent / "experiments"

# paper copy  ->  experiment output (driver that writes it)
PAIRS = [
    ("tables/data/mirror_hw.json", "fixtures/mirror_test/wyniki_hw.json"),          # mirror_test_ibm.py odbierz
    ("tables/data/mirror_plan.json", "fixtures/mirror_test/plan_FakeMarrakesh.json"),  # mirror_test_ibm.py plan
    ("tables/data/mirror_aachen.json", "fixtures/mirror_test_aachen/wyniki_aachen.json"),  # mirror_test_ibm_aachen.py odbierz
    ("tables/data/idle_budget.json", "budget/idle_budget.json"),                   # budget/idle_budget.py
    ("tables/data/prediction.json", "budget/przewidywanie.json"),               # budget/budget_vs_hardware.py
    ("tables/data/orientation.jsonl", "budget/orientacja.jsonl"),                   # budget/orientation_eval.py
    ("tables/data/diag_ibm.json", "budget/diagnostic/analiza.json"),               # budget/diagnostic/analiza.py
    ("tables/data/sym_measured.json", "budget/diagnostic/sym_zmierzone.json"),    # budget/diagnostic/sym_measured.py
    ("tables/data/sym_measured_phaseCZ.json", "budget/diagnostic/sym_zmierzone_fazaCZ.json"),
    ("data/rerun/experiments/fixtures/sensitivity_data.csv", "fixtures/sensitivity_data.csv"),  # sensitivity_sweep.py
]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


bad = 0
for a, b in PAIRS:
    pa, pb = PAPER / a, EXP / b
    same = pa.exists() and pb.exists() and sha(pa) == sha(pb)
    print(("same " if same else "DIFF ") + f"paper/{a}  <->  experiments/{b}")
    bad += not same
sys.exit(1 if bad else 0)

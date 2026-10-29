# Reproduction guide

The complete map of the paper's tables, figures and generated numbers, with the scripts and
data behind each one, is [`paper/README.md`](../paper/README.md). This page gives the short
version.

## Environment

```bash
git clone https://github.com/barkol/qcpt
cd qcpt
python -m venv .venv && source .venv/bin/activate
pip install -e ".[experiments,dev]"
```

The versions used for the paper are listed in `paper/environment.txt` (Python 3.12,
qiskit 2.4.1, qiskit-aer 0.17.2, mqt.qcec 3.6.1, pytket 2.18.0). The figures need LaTeX.

Verify:

```bash
python -c "import toffoli_optimizer; print(toffoli_optimizer.__version__)"   # 1.2.1
pytest -q tests                                                             # 502 passed, 87 xfailed
```

## Rebuild the paper's outputs from the stored results (seconds)

```bash
paper/reproduce.sh
git status paper/        # empty: tables, figure data and numbers identical to the stored ones
```

## Rerun the experiments (hours)

In this order (details and runtimes in `paper/README.md`, section 3):

```bash
python experiments/safety_experiment.py
python experiments/ablation_b.py
python experiments/baselines.py
python experiments/sensitivity_sweep.py
python experiments/noisy_sim.py
python experiments/sync_benchmark.py
python experiments/sync_scale.py
python experiments/scale_eval_ckpt.py
python experiments/scale_program.py
python experiments/revlib/fetch_revlib.py
python experiments/revlib/revlib_eval.py all
python experiments/revlib/revlib_extra.py
python experiments/budget/idle_budget.py
python experiments/budget/budget_vs_hardware.py
python experiments/budget/orientation_eval.py
```

then point `paper/numbers/paper_numbers.py` and `paper/figures/figures_data.py` at the new run
directory (first argument) and rerun `paper/reproduce.sh`.

The 12-circuit drivers (`safety_experiment.py`, `ablation_b.py`, `baselines.py`,
`sensitivity_sweep.py`) reproduce the stored `*_results.md` files and
`fixtures/sensitivity_data.csv` byte for byte with the code of this repository. For the 12–24-qubit suite
and RevLib, `paper/compare_reruns.py` compares a rerun with the stored results (timings excluded).

The experiment drivers in `experiments/` find the repository root from their own location. Their run outputs go to `runs/` in the repository root; set `QCPT_RUNS` to use another directory.

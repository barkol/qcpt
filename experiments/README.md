# Experiment drivers

The drivers here produce the raw results behind the paper
**"Context-Verified, Error-Budget-Aware Decomposition Selection for Toffoli Networks"** (Bartkiewicz & Tulewicz,
arXiv:2606.31791). The tables, figures and numbers of the paper are
assembled from these results by the scripts in [`../paper/`](../paper/README.md), which also
gives the full map (paper artefact -> assembly script -> data -> driver) and the run order.

## Setup

From the repo root:

```bash
pip install -e ".[experiments]"     # qiskit-aer, pytket, mqt.qcec, scipy
```

or, without installing, run scripts directly; the `_paths.py` shim puts the repo root on
`sys.path` automatically.

## Map: driver -> output -> paper artefact

| Driver | Output | Used for | Runtime |
|---|---|---|---|
| `safety_experiment.py` | `safety_results.md` | Fig. `safety` | ~30 s |
| `ablation_b.py` | `ablation_b_results.md` | Fig. `safety` (no-gate bar) | ~10 s |
| `baselines.py` | `baseline_results.md`, stdout | Fig. `budget`, Table `semantics`, 12-circuit numbers | ~10 s |
| `sensitivity_sweep.py` | `fixtures/sensitivity_data.csv`, stdout | Fig. `device` | ~10 s |
| `noisy_sim.py` | stdout | density-matrix fidelities | ~10 s |
| `sync_benchmark.py`, `sync_scale.py` | stdout | Table `reset` | ~10 min |
| `scale_eval_ckpt.py` (uses `scale_eval.py`) | `scale_rows.jsonl`, `scale_results.md` | Fig. `scale`, 12–24-qubit numbers | ~5-15 min, resumable |
| `scale_program.py` (`scale_program_last.py` for a QCEC timeout) | `scale_program_rows.jsonl` | program semantics and count-greedy at 12–24 qubits | ~30 min, resumable |
| `mirror_test_ibm.py plan / wyslij / odbierz` | `fixtures/mirror_test/` | Table `mirror` | IBM job |
| `mirror_test_aachen_plan.py`, `mirror_test_ibm_aachen.py sprawdz / wyslij / odbierz` | `fixtures/mirror_test_aachen/` | Table `mirror2` | IBM job |
| `revlib/fetch_revlib.py` | `revlib/real/*.real` (checked against `revlib/real.sha256`) | Table `revlib` | ~1 min |
| `revlib/revlib_eval.py all`, `revlib/revlib_extra.py` | `revlib/wyniki.jsonl`, `revlib/wyniki_extra.jsonl` | Table `revlib` | ~30 min + timeouts, resumable |
| `budget/budget_vs_hardware.py` | `budget/przewidywanie.json` | budget vs hardware | seconds |
| `budget/idle_budget.py` | `budget/idle_budget.json` | idle errors and ZZ | ~4 min |
| `budget/orientation_eval.py` | `budget/orientacja.jsonl` | calibration-aware orientation | resumable |
| `budget/diagnostic/` (`diag.py`, `analiza.py`, `sym_measured.py`) | `analiza.json`, `sym_zmierzone*.json` | controlled-adder residual | IBM job + seconds |

The stored outputs used for the paper are kept in `fixtures/`, `revlib/`, `budget/` and
`../paper/data/rerun/`. `TOPTOFFOLI_OUT=<dir>` redirects the output of the checkpointed
drivers (`scale_eval_ckpt.py`, `scale_program.py`, `revlib/revlib_eval.py`,
`revlib/revlib_extra.py`) and `TOPTOFFOLI_QCEC_TIMEOUT=<s>` sets the limit of one QCEC call
(default 120 s).

`make_figures.py` is a pointer: the figures are built by `paper/figures/`.

## Files

- `_paths.py` — path-resolution shim (auto-imported by every script)
- `benchmarks.py` — 12-circuit primary suite (Table `bench`)
- `large_benchmarks.py` — 20-circuit scale suite (12–24 qubits)
- `naive_relphase.py` — unsafe count-greedy baseline
- `_clean.py` — detection of clean (constant-|0>) ancillas that the pass may pin
- `fixtures/sensitivity_data.csv` — device sweep used for Fig. `device`

## Optional dependencies

| Dependency | Needed by | Purpose |
|---|---|---|
| `qiskit-aer` | `noisy_sim.py`, `mirror_test_ibm.py plan`, `budget/` | noise simulation |
| `pytket`, `pytket-qiskit` | `baselines.py`, `revlib/` | tket baseline |
| `mqt.qcec` | `scale_eval.py`, `scale_program.py`, `sync_scale.py`, `revlib/` | decision-diagram certificates |
| `scipy` | `budget/` | statistics, fits |
| `qiskit-ibm-runtime` | `mirror_test_ibm.py`, `budget/diagnostic/diag.py` | IBM jobs, fake backend |

## Pickle files

The calibration snapshots `budget/kal_ibm_*.pkl`, `budget/marrakesh_props_jobtime.pkl` and `budget/diagnostic/props_wysylka.pkl` are Python pickles of Qiskit objects (backend properties and targets), read by the scripts in `budget/`. A pickle can execute code when loaded: check them against `budget/PICKLES.sha256` (`sha256sum -c`) before loading, and load only files from a checkout of this repository. We scanned them without loading: they contain only `qiskit` and builtin types.

# paper/ — reproducing the tables, figures and numbers of the paper

This directory rebuilds every generated table, figure and number of

> K. Bartkiewicz and P. Tulewicz, *Context-Verified, Error-Budget-Aware Decomposition Selection for
> Toffoli Networks* (2026).

It contains the assembly scripts used for the manuscript, the stored experiment outputs they
read, and the recertification and ablation checks. The experiment drivers that produce the raw
outputs live in `../experiments/`.

The pipeline runs in three layers:

```
experiments/  (drivers, hours)  ->  stored outputs (paper/data, paper/tables/data, experiments/*)
              ->  paper/tables/*.tex, paper/figures/data/make_figures.npz + fig_*.pdf, paper/numbers/paper_numbers.json
```

## 1. Quick check: rebuild everything from the stored data (seconds)

From the repository root, in an environment with the requirements of `pyproject.toml`
(`pip install -e ".[experiments]"`; the versions used for the paper are in `environment.txt`):

```bash
paper/reproduce.sh                   # PYTHON=/path/to/python paper/reproduce.sh to choose the interpreter
git status paper/                    # nothing listed = every output identical to the stored version
```

`reproduce.sh` stops at the first error. In order, it:

1. checks that the copies of experiment outputs kept in `paper/` are byte-identical to the files
   in `experiments/` (`check_provenance.py`);
2. rebuilds `tables/data/revlib.json` from the per-circuit RevLib results (`tables/revlib_data.py`);
3. writes `tables/tab_semantics.tex`, `tables/tab_mirror.tex`, `tables/tab_mirror_aachen.tex`,
   `tables/tab_revlib.tex` (generated, not stored in the repository);
4. writes `figures/data/make_figures.npz` (`figures/figures_data.py`) and draws
   `figures/fig_{safety,budget,device,scale}.pdf` (`figures/make_figures.py`; needs a LaTeX
   installation because the figures use `text.usetex`; `SKIP_FIGURES=1` skips the drawing);
5. writes `numbers/paper_numbers.json`, every number of the text that is not typed by hand
   (`numbers/paper_numbers.py`).

`paper/reproduce.sh compute` additionally recomputes the 12-circuit data of the semantics table,
the interference rows and the gain attribution with the code of this checkout (a few minutes).

The `.tex` tables, the npz file and `paper_numbers.json` come out byte-identical to the files used for
the manuscript. The figure PDFs come out pixel-identical; only the PDF `CreationDate` differs.
The PDFs stored in `figures/` are the ones in the manuscript.

The manuscript source has `@@KEY@@` markers where a number of `paper_numbers.json` goes;
`python numbers/fill_in.py main.template.tex main.tex` fills them and stops on a missing key.
The manuscript source itself is not part of this repository.

## 2. Map: paper artefact -> script -> data

### Tables

| Paper | Built by | Reads | Raw data produced by |
|---|---|---|---|
| Table `axes`, Table `taxonomy` | typed in the manuscript (no data) | — | — |
| Table `bench` (12-circuit suite) | typed in the manuscript | — | circuits: `experiments/benchmarks.py` |
| Table `semantics` | `tables/make_tab_semantics.py` | `tables/data/semantics.json`, `tables/data/interference.json`, `data/rerun/experiments/baseline_results.md` | `tables/semantics_data.py`, `tables/interference_data.py`, `experiments/baselines.py` |
| Table `mirror` (IBM mirror test) | `tables/make_tab_mirror.py` | `tables/data/mirror_hw.json` (= `experiments/fixtures/mirror_test/wyniki_hw.json`) | `experiments/mirror_test_ibm.py plan / wyslij / odbierz` (job id in `experiments/fixtures/mirror_test/job.json`) |
| Table `mirror2` (second mirror test, `ibm_aachen`) | `tables/make_tab_mirror_aachen.py` | `tables/data/mirror_aachen.json` (= `experiments/fixtures/mirror_test_aachen/wyniki_aachen.json`) | `experiments/mirror_test_ibm_aachen.py sprawdz / wyslij / odbierz` (plan: `experiments/mirror_test_aachen_plan.py`; job id in `fixtures/mirror_test_aachen/job.json`) |
| Table `revlib` | `tables/make_tab_revlib.py` | `tables/data/revlib.json` | `tables/revlib_data.py` <- `experiments/revlib/wyniki.jsonl` (`revlib_eval.py`), `wyniki_extra.jsonl` (`revlib_extra.py`), `nieuruchomione.json`; files from `fetch_revlib.py` |
| Table `reset` | numbers `RS*` in `numbers/paper_numbers.json` | `data/rerun/log_sync_scale.txt` | `experiments/sync_scale.py` |

### Figures

| Paper | Drawn by | Data (npz key) | Raw data produced by |
|---|---|---|---|
| Fig. `method`, Fig. `reset` | TikZ/quantikz in the manuscript (no data) | — | — |
| Fig. `safety` | `figures/make_figures.py` | `safety_bad`, `safety_n` | `experiments/safety_experiment.py` (`safety_results.md`), `experiments/ablation_b.py` (`ablation_b_results.md`) |
| Fig. `budget` | `figures/make_figures.py` | `budget_*` | `experiments/baselines.py` (`baseline_results.md`) |
| Fig. `device` | `figures/make_figures.py` | `dev_*` | `experiments/sensitivity_sweep.py` (`fixtures/sensitivity_data.csv`) |
| Fig. `scale` | `figures/make_figures.py` | `fam_*`, `qcec_*`, `cross_*`, `scale_*` | `experiments/scale_eval_ckpt.py` (`scale_rows.jsonl`, then `scale_results.md` with the crossover series of `experiments/scale_eval.py`) |

All four read `figures/data/make_figures.npz`, written by `figures/figures_data.py` from
`data/rerun/experiments/`. `figstyle.py` is the vendored figure-style module.

### Numbers of the text (`numbers/paper_numbers.json`)

| Keys | Meaning | Source read by `paper_numbers.py` | Produced by |
|---|---|---|---|
| `EX2Q`, `OUR2Q`, `EXINF`, `OURINF`, `DEP*`, `Q2RED*`, `VSQ*`, `VST*`, `Q2NCERT` | 12-circuit suite: totals and reductions vs exact, Qiskit, tket | `data/rerun/log_baselines.txt` | `experiments/baselines.py` |
| `AT*` | gain attribution to conditions (C), (W), (R) | `tables/data/attribution.json` | `tables/attribution_data.py` |
| `GR*`, `PR2Q`, `PRBAD*`, `NINTERF`, `GRINTP` | count-greedy and program semantics, interference circuits | `tables/data/semantics.json`, `interference.json` | `tables/semantics_data.py`, `tables/interference_data.py` |
| `F{CA,GO,NC}{EX,OUR}` | density-matrix fidelities | `data/rerun/log_noisy_sim.txt` | `experiments/noisy_sim.py` |
| `DEV*` | device operating points | `data/rerun/log_sensitivity_sweep.txt` | `experiments/sensitivity_sweep.py` |
| `RESET`, `RS*`, `RESMIN`, `RESMAX` | resetting-protocol circuits | `log_sync_benchmark.txt`, `log_sync_scale.txt` | `experiments/sync_benchmark.py`, `experiments/sync_scale.py` |
| `L*`, `F{GRO,MIX,CLA,RIP,MOD,MUL,LIV}*`, `LQ*`, `LND`, `LDT`, `X*` | 12–24-qubit suite, certificates, verification crossover | `data/rerun/experiments/scale_rows.jsonl`, `scale_results.md` | `experiments/scale_eval_ckpt.py` |
| `P*` (program mode), `G2Q`, `GNBAD`, `GNUND`, `GSAMEBAD`, `PMULTMIN` | 12–24-qubit suite in program semantics and count-greedy | `data/rerun/experiments/scale_program_rows.jsonl` | `experiments/scale_program.py`; the `array_mult_6b` row from `scale/qcec_mult6.py` (QCEC without the 300 s limit, outputs `data/rerun/array_mult_6b_*.json`) |
| `HW*` | mirror test on `ibm_marrakesh` | `tables/data/mirror_hw.json` | `experiments/mirror_test_ibm.py` |
| `AA*` | second mirror test on `ibm_aachen` | `tables/data/mirror_aachen.json` | `experiments/mirror_test_ibm_aachen.py` |
| `RL*` | RevLib suite | `tables/data/revlib.json` | see Table `revlib` |
| `PR*` (prediction), `PRLAM*`, `PRCZMED` | error budget vs hardware | `tables/data/prediction.json` (= `experiments/budget/przewidywanie.json`) | `experiments/budget/budget_vs_hardware.py` |
| `ID*`, `FULL*` | idle errors and ZZ in the budget | `tables/data/idle_budget.json` (= `experiments/budget/idle_budget.json`) | `experiments/budget/idle_budget.py` |
| `OR*` | calibration-aware gadget orientation | `tables/data/orientation.jsonl` (= `experiments/budget/orientacja.jsonl`) | `experiments/budget/orientation_eval.py` |
| `DG*` | controlled-adder residual (diagnostic IBM job) | `tables/data/diag_ibm.json` (= `experiments/budget/diagnostic/analiza.json`), `tables/data/sym_measured.json` | `experiments/budget/diagnostic/diag.py`, `analiza.py`, `sym_measured.py` (uses `symulacja.py`) |

`paper_numbers.py` stops on the first missing source or failed consistency assertion (for example,
the attribution totals must equal the baseline totals, and every RevLib output of the pass must
be certified).

## 3. Rerunning the experiments (hours)

The stored outputs in `data/rerun/` are those of the run used for the paper (the file names
of that run directory are replaced by `<run>` in the logs). To regenerate them, run from the
repository root, in this order (runtimes on a 28-core workstation):

```bash
python experiments/safety_experiment.py      # ~30 s
python experiments/ablation_b.py             # ~10 s
python experiments/baselines.py              # ~10 s
python experiments/sensitivity_sweep.py      # ~10 s
python experiments/noisy_sim.py              # ~10 s
python experiments/sync_benchmark.py         # ~10 min
python experiments/sync_scale.py             # ~1 min
python experiments/scale_eval_ckpt.py        # ~5-15 min, resumable (scale_rows.jsonl)
python experiments/scale_program.py          # ~30 min plus array_mult_6b, resumable (scale_program_rows.jsonl)
python paper/scale/qcec_mult6.py prog        # array_mult_6b in program mode: QCEC ~16 min
python paper/scale/qcec_mult6.py greedy
python experiments/revlib/fetch_revlib.py    # downloads RevLib files, checks real.sha256
python experiments/revlib/revlib_eval.py all # ~30 min plus 2 x 1 h timeouts (urf2, urf5), resumable
python experiments/revlib/revlib_extra.py    # tket with allow_swaps=False, ours -> Qiskit
python experiments/budget/idle_budget.py     # ~4 min
python experiments/budget/budget_vs_hardware.py
python experiments/budget/orientation_eval.py
python experiments/budget/diagnostic/analiza.py
python experiments/budget/diagnostic/sym_measured.py
```

Each driver writes next to itself (log lines go to stdout; redirect them to
`log_<driver>.txt` to feed `paper_numbers.py`). Two environment variables redirect the
checkpointed drivers (`scale_eval_ckpt.py`, `scale_program.py`, `revlib_eval.py`,
`revlib_extra.py`) without touching stored data: `TOPTOFFOLI_OUT=<dir>` sets the output directory
and `TOPTOFFOLI_QCEC_TIMEOUT=<s>` the limit of one QCEC call (default 120 s, the paper's value).
Point `paper_numbers.py` and `figures_data.py` at a new run directory with their first argument.
The IBM hardware stages (`mirror_test_ibm.py wyslij/odbierz`, `mirror_test_ibm_aachen.py wyslij/odbierz`,
`diag.py wyslij/odbierz`) need an
IBM Quantum account; the `plan` stages and everything downstream run without one from the stored
counts, ISA circuits (`experiments/budget/lustro_isa.qpy`) and calibrations (`*.pkl`).

## 4. Checks that accompany the paper

* `recert/` — recertification of every QCEC-certified result after the QCEC settings were
  corrected (`run_zx_checker=False`, `elide_permutations=False`, `trace_threshold=1e-12`), and an
  independent state-vector check on random superposed inputs with the pinned qubits in |0>
  (`sv_check.py`, `recert_large.py`). Results: `recert_large.jsonl` (12–24-qubit suite: the
  pass, program mode and count-greedy), `recert_reset_revlib.jsonl` (resetting circuits and RevLib
  above 12 lines), `recert_orient.jsonl` (oriented gadgets; stored run used code commit 34364bf,
  set `CODE=<checkout>` to choose the code), `prog_qcec.json` (`prog_qcec.py`: QCEC tolerance
  check). Every output of the pass is `equivalent` under QCEC and passes the state-vector test,
  and the published certificate verdicts of the program-mode and count-greedy outputs agree with
  the state-vector test on every circuit.
* `ablation/ablation_without_gate.py` — the pass with every whole-circuit and per-site
  verification disabled, so that only its own admission rules (C), (W), (R) decide. The
  outputs are compared with the exact decomposition as dense unitaries (no wire
  permutation): `corrupted: 0 / 12`.
* `compare_reruns.py` with `data/rerun_suites/` — rerun of the 12–24-qubit suite and of RevLib with the
  release code (see below).

## 5. Code version of the stored data

The stored outputs in `data/rerun/` were produced on 2026-10-01/02 with an earlier development
version. Its code differs from the released code in: the corrected QCEC options
above, fail closed also on an undecided (not only a failed) final certificate, an optional
gadget-orientation flag and optional idle/ZZ terms of the error model (both off by default).
To check that none of this changes a result of the paper, `data/rerun_suites/` holds a rerun with the code of this repository of the 12–24-qubit suite
(`scale_rows.jsonl`, `scale_program_rows.jsonl`, `scale_results.md`) and of RevLib
(`results.jsonl`), with a 600 s limit per QCEC call. `python paper/compare_reruns.py` lists every
difference in two-qubit counts, infidelities and certificate verdicts; timings are not
compared. See `data/rerun_suites/README.md` for the outcome.


## Hardware measurements of 29 October 2026 (probed regions)

The hardware results of the paper come from one series on `ibm_aachen` (29 October 2026), on regions chosen after a probe of every coupler:

| step | script (`../experiments/`) | stored output |
|---|---|---|
| probe of all qubits and couplers (4 circuits, 3 s of QPU time) | `aachen_probe.py send|fetch` (`PROBE_TAG`) | `fixtures/aachen_probe*/probe.json` |
| calibration snapshot and regions from the calibration | `aachen_calibration_next.py` (`EXCLUDE`) | `fixtures/aachen_next/` |
| regions from the probe (couplers above 1.2 % per CZ and qubits with readout flip above 0.2 dropped) | `select_from_probe.py` (`PROBE_TAG`, `SEL_TAG`) | `fixtures/aachen_repro/selection.json` |
| mirror circuits on the probed regions: nine circuits and scaling families (22 circuits, one job) | `aachen_repro_hw.py build|wyslij|odbierz` | `fixtures/aachen_repro_hw/hw_results.jsonl` |
| five circuits fixed in advance (`ripple_adder_4b` and four RevLib circuits) | `aachen_next_hw2.py` (and `aachen_next_hw.py`) | `fixtures/aachen_next_hw2/hw_results.jsonl` |
| check of the calibration (T1, echo T2, readout, randomized benchmarking) | `aachen_calcheck.py` | `fixtures/aachen_calcheck_2026-10-29/` |
| Draper adder on a probed region of 12 qubits | `qft_adder_probed.py`, `qft_ranking.py` | `fixtures/aachen_draper_2026-10-29/` |
| simulation of the 33 circuits, noise model of the calibration, corrected model, dynamometer fits | `sim_2026-10-29/step1_dump.py`, `step2_analyze.py`, `step5_build33.py`, `step5b_sim33.py` | `sim_2026-10-29/out/` |
| figures 7, 8 and 10, map of the failed couplers (figure 9) | `figures_resynthesis/figures/make_sim_hw.py`, `make_dyno_fam2.py`, `figures/probe_map_data.py`, `figures/make_probe_map.py` | `figures_resynthesis/data_in/new/manuscript_data.json`, `figures/data/probe_map.npz` |

Every driver refuses a second submission (`job.json` exists), a circuit outside the selected qubits, a circuit on a coupler marked dead and an estimate of the QPU time above the remaining budget. The intermediate pickle files (`items_hw.pkl`) are regenerated by `build` and are not part of the release.

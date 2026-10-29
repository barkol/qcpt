# paper/figures_resynthesis: figures of the resynthesis, hardware and scaling sections

Scripts that draw the figures of sections 5.4 to 5.7 and of the appendices that depend on the resynthesis and the ibm_aachen runs, with the data they read. The drawing layer only reads the stored data; recomputing the fits is done by the `*_data.py` scripts. LaTeX is needed (the figures use `text.usetex`). Run a script with `FIG_OUT=<directory>` to write the PDF there.

| figure of the paper | script | data |
|---|---|---|
| Fig. 1 `concept` | `figures/make_concept.py` | `figures/data/dyno_fam2.npz` (measured incrementer fit) |
| Fig. 7 `sim` and Fig. 8 `hw` | `figures/make_sim_hw.py` | `data_in/new/manuscript_data.json` (written by `experiments/make_manuscript_data.py`) |
| Fig. 9 `dyno` | `figures/make_dyno_fam2.py` | `figures/data/dyno_fam2.npz`, made by `figures/dyno_hw_data.py` from `data_in/new/scaling_hw_all.jsonl`, `pred_calib.json`, `pred_fit.json`; sensitivity of the ratios in `figures/dyno_sensitivity.py` (`figures/data/dyno_sensitivity.json`) |
| Fig. 10 `resynth` | `figures/make_results_figs.py` | `figures/paper_data.py` |
| Fig. 11 `compare` | `figures/make_compare_icons.py` | `figures/compare_data.py` (`figures/data/compare.npz`) |
| Fig. 14 `times` | `figures/make_times.py` | `figures/times_data.py` |
| Figs. 6, 13 (`cost`, `mirror`) | `figures/make_tables_as_figs.py` | `figures/tables_data.py` |
| Fig. 2 `flow` | `figures/make_flowchart.py` | none |

The corrected noise model of section 5.4 is fitted by `experiments/aachen_noise_fit.py` (fits and predictions in `experiments/fixtures/aachen_noise_fit/`; the two numbers of the out-of-sample check are in `eval_K1.0_R1.0.json` and `eval_K1.946_R0.124_T0.0264.json`, copied to `data_in/new/`). The simulation of the scaling families uses the calibration of 26 October 2026, 23:00 (`experiments/fixtures/aachen_calibration_2026-10-26_2300/`), which is numerically identical to the one of 23:30.

To rebuild from the stored data: run `paper/reproduce.sh` first (it writes `paper/tables/tab_*.tex`), then `figures/tables_data.py`, `paper_data.py`, `compare_data.py`, `times_data.py`, then the `make_*.py` scripts with `FIG_OUT=<directory>`. `make_compare_icons.py` needs `svgpath2mpl`. The cached results in `figures/data/*.npz` are stored, so the drawing scripts also run without the `*_data.py` step.

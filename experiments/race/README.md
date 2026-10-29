# Race, windows and scaling drivers

Python drivers of the C++ race and of the windowed resynthesis, as they were run for the paper (they live next to the repository in a working directory called qct_runs, and several of them add that directory to sys.path and read or write paths under it; set QCPT_BIN to the cpp/qcpt binary).

- race33n.py, race_cpp.py, portfolio.py: one race of the C++ core on one circuit (members, rounds, verification against the exact circuit).
- windows33.py, windows_scal.py, reach_run.py, reach_try.py: windowed resynthesis (windows of the exact tables, and windows with reachable context, toffoli_optimizer/core/window_reach.py).
- freeze_more.py, add_stage1.py, add_incr.py: freeze the cheapest certified circuit (race, windows or stage 1) into experiments/fixtures/nighthawk/frozen_scaling.json.
- race_energy.py: metered run (CPU time from /proc/stat).
Results used in the paper: experiments/fixtures/comparison_lm/ (race33_new, parity_new, scaling_race, scaling_reach, claude_to_best.json).

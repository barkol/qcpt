# Rerun with the release code (2026-10-02)

Code: commit `cee2e0d` (= v1.2 plus the `TOPTOFFOLI_OUT` / `TOPTOFFOLI_QCEC_TIMEOUT` options;
the core is byte-identical to v1.2 and to v1.2.1 up to docstrings). Environment:
`../environment.txt`. Limit of one QCEC call: 600 s (`TOPTOFFOLI_QCEC_TIMEOUT=600`); limit of
one RevLib circuit: 3600 s, as in the paper. The machine was shared with other jobs (load about
40 on 28 cores), so all wall-clock times are larger than in the paper run.

| Files | Driver |
|---|---|
| `scale_rows.jsonl`, `scale_results.md`, `log_scale_eval_ckpt.txt` | `experiments/scale_eval_ckpt.py` |
| `scale_program_rows.jsonl`, `log_scale_program.txt` | `experiments/scale_program.py` |
| `array_mult_6b_{prog,greedy}.json`, `log_qcec_mult6.txt` | `paper/scale/qcec_mult6.py` (QCEC without limit) |
| `wyniki.jsonl`, `log_revlib.txt` | `experiments/revlib/revlib_eval.py all` (stopped after the 58 circuits of the paper; urf1, urf3, urf4, urf6 not run, as in the paper) |
| `suite12/` | `safety_experiment.py`, `ablation_b.py`, `baselines.py`, `sensitivity_sweep.py`, `noisy_sim.py`, `sync_benchmark.py`, `sync_scale.py` |

Paths of the run directories are replaced by `<run>` and `<out>` in the logs.

## Outcome (`python paper/compare_reruns.py`)

* 12–24-qubit suite, subroutine semantics: 20/20 rows identical in two-qubit counts,
  infidelities, admitted pairs/windows/gadgets, certificate method and verdict.
* Program semantics and count-greedy: 20/20 rows identical. For `array_mult_6b` QCEC does not
  decide within 600 s (it ran 1833 s and 1713 s before returning undecided), exactly as the
  300 s limit of the paper run; the unbounded run gives `not_equivalent` for both variants
  (1334 s, 1230 s), the paper's verdict.
* RevLib: 58/58 rows identical (counts, infidelities, conditions, certificates of every method;
  urf2 and urf5 time out at 3600 s as in the paper). `tables/revlib_data.py` on these rows gives
  a `revlib.json` with the same aggregates, the same `tab_revlib.tex` and the same `RL*` numbers.
* 12-circuit suite and resetting circuits: logs and `*_results.md` byte-identical to
  `../data/rerun/`, except the verification wall-clock times in `log_sync_scale.txt`.
* `liczby.py` on this rerun gives the same 229 numbers as the paper except 16 that are
  wall-clock times or depend on them: `LDT`, `LQMAX`, `LQMED`, `LQFASTMAX`, `LQSLOW` (QCEC calls
  above 1 s: 2 instead of 1), `PMULTMIN`, `RS*T`, `XE10`, `XE12`, `XQ12`, `XQMAX`, `XRATIO`,
  `XRATIOR`. The figure data differ only in the timing arrays of Fig. `scale` (`qcec_t`,
  `cross_exh`, `cross_qcec`).

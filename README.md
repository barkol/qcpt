# qcpt

Companion code to

> K. Bartkiewicz and P. Tulewicz,
> **"Context-Verified, Error-Budget-Aware Decomposition Selection for Toffoli Networks"**,
> [arXiv:2606.31791](https://arxiv.org/abs/2606.31791) (2026).

A compiler pass that selects, per Toffoli gate, the decomposition that
minimizes a **hardware two-qubit-infidelity error budget** — not gate count —
and admits a context-dependent (relative-phase or control-drop) decomposition
**only under an operator condition on the reachable input subspace**, then certifies
the whole output against the exact decomposition. In the default subroutine semantics
the output equals the exact circuit on the input subspace up to one global phase, so
it can be used inside a larger algorithm. Pattern-matched relative-phase substitution
as deployed in current tooling is silently incorrect; the certified pass keeps most of
the savings without that risk.

The package ships the optimizer, the exact and scalable equivalence
verifiers, the reachable-subspace and phase-observability admissibility
checks, the experiment drivers under `experiments/`, and under `paper/` the scripts
and stored data that rebuild every table, figure and generated number of the paper.

## Method in three sentences

1. **Objective.** Minimize the two-qubit-infidelity budget of the emitted
   circuit under the device's `p_{2q}` — not the CCX or two-qubit gate count.
2. **Admission.** For each Toffoli, consider the exact 6-CX decomposition
   and the cheaper context-dependent alternatives (3-CX relative-phase gadgets,
   control drops). Admit one only under condition (C) (a compute/uncompute pair,
   checked exactly), (W) (a certified window between mirrored gates) or (R) (equality
   with CCX up to one phase on the span of the reachable basis states). Condition (U)
   (phase invisible to a terminal measurement) is available only in program semantics.
3. **Certification.** The whole output is certified against the exact decomposition on
   the input subspace: densely up to 12 qubits, by decision diagrams (MQT QCEC, pinned
   qubits as ancillas) above. If the certificate fails or is undecided, the pass
   returns the exact decomposition.

## Quickstart

```bash
git clone https://github.com/barkol/qcpt
cd qcpt
pip install -e ".[experiments,dev]"

pytest tests                                  # the test suite (needs the dev extra)
paper/reproduce.sh                            # rebuild tables, figures, numbers of the paper
git status paper/                             # empty: outputs identical to the stored ones
```

## Reproducing the paper

[`paper/README.md`](paper/README.md) maps every table, figure and generated number of the
paper to the script that builds it and the data it reads, and lists the order in which
to run the experiment drivers:

1. `experiments/` drivers (hours) write the raw results: 12-circuit suite
   (`safety_experiment.py`, `ablation_b.py`, `baselines.py`, `sensitivity_sweep.py`,
   `noisy_sim.py`), resetting circuits (`sync_benchmark.py`, `sync_scale.py`),
   12–24-qubit suite (`scale_eval_ckpt.py`, `scale_program.py`), RevLib
   (`revlib/fetch_revlib.py`, `revlib/revlib_eval.py`, `revlib/revlib_extra.py`),
   IBM mirror tests (`mirror_test_ibm.py`, `mirror_test_ibm_aachen.py`), error budget on hardware (`budget/`).
2. `paper/reproduce.sh` rebuilds from the stored results the tables
   (`paper/tables/make_tab_{semantics,mirror,mirror_aachen,revlib}.py`), the figure data and the
   figures (`paper/figures/figures_data.py`, `paper/figures/make_figures.py`) and the
   numbers of the text (`paper/numbers/paper_numbers.py` -> `paper_numbers.json`). The outputs are
   byte-identical to the files of the manuscript (figure PDFs pixel-identical).
3. `paper/recert/` holds the recertification of every QCEC result with the corrected
   QCEC settings and a state-vector check; `paper/ablation/` the ablation of all
   verification; `paper/data/rerun_suites/` a rerun of the 12–24-qubit suite and RevLib with
   the release code.

## Repository layout

```
qcpt/
├── toffoli_optimizer/    # the package (core algorithms, CLI, utilities)
├── experiments/          # experiment drivers and their stored raw results
├── paper/                # tables, figures and numbers of the paper (scripts + data)
├── tests/                # pytest suite
├── examples/             # standalone demos + example input circuits
├── benchmarks/           # micro-benchmarks (not the paper's suite)
└── docs/                 # methodology + reproduction guide
```

## Requirements

- Python ≥ 3.10 (the paper used 3.12)
- `qiskit` ≥ 2.4, `numpy`, `matplotlib` — always required
- Optional (for `experiments/` and `paper/`):
  - `qiskit-aer` ≥ 0.17 — density-matrix noise validation, mirror-test plan
  - `pytket` ≥ 2.18, `pytket-qiskit` ≥ 0.77 — the tket baseline in `baselines.py`
  - `mqt.qcec` ≥ 3.6.1 — decision-diagram certificates above 12 qubits. The certificates
    pass `run_zx_checker`, `elide_permutations` and `trace_threshold` and fail closed
    if the installed QCEC does not accept them.
  - `scipy` — statistics of the hardware analysis
  - `qiskit-ibm-runtime` — only for submitting the IBM jobs
  - a LaTeX installation — the figures are typeset with `text.usetex`
- The minimum versions are the versions used for the paper; the full list of that
  environment is in [`paper/environment.txt`](paper/environment.txt).

## Contents of the package

`toffoli_optimizer.core` contains the certified pass of the paper:

- stage 1 (selection): `decomposition_selector` (`ErrorBudgetSelector`), `context_analysis`, `reachable_subspace`, `reach_local`, `window_pairs`, `window_reach`, `phase_observability`, `orientation`, `error_model`;
- certificates: `subspace_check`, `equivalence_verifier`, `scalable_verification`;
- stage 2 (certified resynthesis): `resynth`, `template`, `cppsearch` (calls the C++ search in `cpp/`; build it with `make -C cpp`, check it with `cpp/tests/check_regression.sh`);
- `toffoli_count_reducer`, which reduces the number of atomic multi-controlled gates.

`ErrorBudgetSelector(elide_swaps=True)` removes the SWAP gates before the selection by relabelling the wires; the result then carries `output_permutation` (logical qubit `i` ends on wire `output_permutation[i]`), as a transpiled circuit carries its final layout. Off by default.

`ErrorBudgetSelector(simplify=...)` adds a final step after the certified selection: `"zx"` reduces the output with the ZX-calculus (`core/zx_simplify.py`, needs PyZX, extra `zx`), `"peephole"` applies commutation-aware CNOT cancellation and two-qubit block resynthesis (`core/peephole.py`) to the output and to the all-exact circuit, `"zx+peephole"` both. A candidate replaces the output only if its error budget is lower and it passes the same whole-circuit certificate; the cheapest certified candidate wins, otherwise the output of the selection is kept (`report["simplify"]` lists the candidates). Off by default.

`ErrorBudgetSelector(certificate=...)` selects the whole-circuit certificate. `"exact"` (default) is deterministic: a certificate by blocks of difference between the output and the exact circuit (`core/blockcert.py`; the blocks are compared on the qubits they touch, a common stretch that commutes with a block is slid away; undecided when the equality holds only on the input subspace), then dense columns (C++, up to 14 qubits) or QCEC. `"monte_carlo"` is opt-in: `mc_vectors` (default 24) random vectors on the input subspace are propagated through both circuits (`core/cppcert.py: mc_compare`; blocks of difference first for circuits above 16 qubits); a pair that is not equal up to one global phase (Frobenius deviation above 1e-8) is accepted with probability below (0.1 e^0.9)^(K-1), reported as `verify_info["failure_probability"]`; a counterexample is deterministic and an undecided test falls through to the deterministic certificate.

Limitations: a circuit above 12 qubits without pinned qubits and without (R) gadgets always returns the exact circuit, because the dense verifier refuses it and QCEC is used only with pinned qubits (the pass fails closed). `select` takes unitary circuits only and raises `ValueError` on measurements and resets; the count reducer returns circuits with classical bits unchanged. The error model charges atomic `mcx`, `cswap`, `rccx` and `c3sx` as one two-qubit gate (and does not charge the single-qubit gates of an atomic CCX); the numbers of the paper count two-qubit gates after decomposition. These conventions are pinned as strict `xfail` tests in `tests/unit/test_error_model.py`.

**Certificates:** all decision-diagram certificates call MQT QCEC with
`run_zx_checker=False`, `elide_permutations=False` and `trace_threshold=1e-12`.
With the QCEC defaults, QCEC could report `equivalent` for circuits that
differ on the input subspace when ancillas are declared, treated a SWAP as the
identity, and accepted differences of order 1e-4. Every QCEC-certified result of the
paper was re-certified with the corrected settings and, independently, by
state-vector simulation on random superposed inputs; all verdicts were unchanged.

## Tests

`pytest tests` runs regression tests for the soundness of the pass (including a counterexample to the basis-wise check) and property tests against dense-matrix ground truth on random circuits (these need `hypothesis`, part of the `dev` extra). Strict `xfail` tests document the known defects listed above.

## Citation

If you use this software or reproduce results from the paper, please cite:

```bibtex
@article{bartkiewicz2026qcpt,
  title   = {Context-Verified, Error-Budget-Aware Decomposition Selection for
             {Toffoli} Networks},
  author  = {Bartkiewicz, Karol and Tulewicz, Patrycja},
  journal = {arXiv preprint arXiv:2606.31791},
  year    = {2026},
  url     = {https://arxiv.org/abs/2606.31791}
}
```

Machine-readable metadata is provided in [`CITATION.cff`](CITATION.cff)
(rendered by GitHub as a "Cite this repository" widget in the sidebar).

## Use of AI in code development

The authors used a large language model (Claude, Anthropic) as an assistant in
writing and testing the code of this repository, including the optimizer and the
evaluation and figure scripts. The authors checked the code and take full
responsibility for it. The regression tests in `tests/` (run with `pytest tests`)
and the input-subspace certificate, which every output of the pass must pass,
are the safeguards we rely on rather than review alone.

## License

MIT — see [`LICENSE`](LICENSE).

## Acknowledgements

We acknowledge support from the EuroHPC JU under Horizon Europe Grant
No. 101194322 (**QEC4QEA**), co-funded by the Polish National Centre for
Research and Development (NCBiR) under Decision No.
DWM/EuroHPC/2023/429/2025.

The authors are with the Institute of Spintronics and Quantum Information,
Faculty of Physics and Astronomy, Adam Mickiewicz University, Poznań, Poland.

# Cost of a Toffoli on its reachable support (Proposition prop:cost, Appendix app:cost)

- `cost.py` — numerical search for the minimal number of CNOTs that implement CCX on span(L), L ⊆ {0,1}^3, up to one global phase, with arbitrary single-qubit gates (all CNOT placements without equal neighbours, random starts, PyTorch). `topup.py` fills in the smaller CNOT counts of the same parity where 4 or 5 CNOTs succeed. Results: `cost.jsonl` (one line per support and CNOT count). Runtime about 1.5 h on 20 cores.
- `class4_example.qpy/.json` — an explicit 4-CNOT circuit for class 4 (L = Q ∪ {000, 110}), verified with qiskit `Operator`.
- `prediction/` — static prediction of the two-qubit count from L(g), pairs and windows, compared with the pass on the primary suite, RevLib (≤ 12 lines) and the scale suite (`prediction.py`, `analysis.py`; results `results.jsonl`, `analysis.json`).
- `summary.py` — prints every number of the proposition paragraph and of Appendix app:cost from the stored data.

The lower bounds for the classes 1–4 are numerical evidence; the bound 6 for L = {0,1}^3 is the theorem of Shende and Markov.

# trasy_qct_topo.json: which route won the qct_topo points (ibm_aachen, 29 October 2026)

`trasy_qct_topo.json` records, for the 27 hardware points of the Table `probed` of the paper (22 circuits of the job
`dbhkvo94vo9s73d1kr10` and the 5 circuits marked with a dagger of the job `dbhkn1g8vbtc738e43bg`), which of the
candidates qct, qct_topo and qct_post was sent as the QCPT circuit, and, for the four points won by qct_topo, which route of its
portfolio won (QCPT's router, or SABRE on the circuit of stage 2 or of stage 1, best of 200 seeds), with the two-qubit gates of V
after routing for the three routes and the gates and the depth of the final hardware circuit.

| circuit | winning route | 2q gates of V: QCPT router / SABRE stage 2 / SABRE stage 1 | final circuit: CZ / 1q / depth |
|---|---|---|---|
| ripple_carry_adder_2b | QCPT router | 23 / 31 / 29 | 86 / 357 / 307 |
| grover_oracle_5c | SABRE on the circuit of stage 1 (best of 200 seeds) | 62 / 59 / 45 | 142 / 614 / 503 |
| mod_incr_5b | SABRE on the circuit of stage 2 (best of 200 seeds) | 32 / 26 / 37 | 128 / 516 / 386 |
| ripple_adder_4b | SABRE on the circuit of stage 1 (best of 200 seeds) | 74 / 93 / 57 | 200 / 815 / 637 |

Candidates over the 27 points: qct 15, qct_post 8, qct_topo 4.

Provenance. The values were read from the intermediate files of the hardware drivers
(`aachen_repro_hw/items_hw.pkl`, `aachen_next_hw2/items_hw.pkl`; fields `name`, `tag`, `layout`, `twoq_t`, `topo.chosen`,
`topo.n2q_*` and the transpiled circuit `t`) and the order of circuits of `job.json`. These pickles are not part of the release, so
this file is the record of the winning routes. Only the number of two-qubit gates is stored for the routes that did not win, so
single-qubit gates and depth are given only for the final circuits. The driver is `experiments/mirror_test_nighthawk.py`
(candidate qct_topo: `experiments/qcpt_topo.py`, `route_best`, 200 SABRE seeds, then one transpilation at optimization level 1,
seed 11).

sha256:

- `trasy_qct_topo.json` bc6a8fd7bf37dfbaf372b6a61071717e0972d561527943777afc5640f03cc4d9
- `aachen_repro_hw/items_hw.pkl` cae11b5f717996283dcb1e8729d378225d3d36b498774c128d52b58227d25f78
- `aachen_repro_hw/job.json` 2cf34b98b12a48e62e267091796fc3e6b3a99f0949aba55bbc607f3e749112ac
- `aachen_next_hw2/items_hw.pkl` 3037ef6bc3fc019fbf31732ec0a4c185b575879d456eee15d003d59700ce7cf1
- `aachen_next_hw2/job.json` 4ba3965a9bc2195b339d6365b967f6846595ee68705cd0f865e785eae24651ec

Date of the measurements: 29 October 2026 (jobs sent 14:35 and 14:53).

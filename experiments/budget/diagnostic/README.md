# Diagnostic of the controlled-adder residual (mirror test)

Record of the analysis behind the controlled-adder paragraph of the paper (2026-10-02).
The scripts are kept as run, except that every path is now relative to the script file
(they used to point at the working directory of that session). `diag.py` imports the
mirror-test driver `../../mirror_test_ibm.py`; `symulacja.py` and `hipotezy.py` read the
hardware results `../../fixtures/mirror_test/wyniki_hw.json` (identical to
`paper/tables/data/mirror_hw.json`). All inputs and outputs they used are stored here:

- `counts_all.json` — counts of the 15 mirror circuits (job davcn5o4oijs73e7b5m0, 2026-10-01);
- `counts.json`, `job.json`, `props_wysylka.pkl`, `analiza.json` — follow-up job davpkglj371s73dnb7hg
  (same-layout repeat, other-region run, T1 and Hahn-echo on q2, q16, q3, q6);
- `symulacja.py` — density-matrix simulation of the executed ISA circuits (ALAP schedule,
  T1/T2 relaxation, calibrated gate errors, coherent ZZ, optional CZ phase and excess error);
- `hipotezy*.json`, `sym_zmierzone*.json` — hypothesis scan and re-simulation with measured T1/T2;

The executed ISA circuits and the job-time calibration are in `../lustro_isa.qpy` and
`../marrakesh_props_jobtime.pkl`.

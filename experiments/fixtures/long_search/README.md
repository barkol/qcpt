# Long search for cheaper certified circuits

`experiments/long_search.py NAME FINAL` looks for a circuit with fewer than FINAL two-qubit gates for a classical circuit (bidirectional beam search,
22 configurations: library, beam width, heuristic weight, depth, exploration slots and noise; budget 1500 s, 25 min). `results/` holds the logs of
31 circuits (every configuration with its time and result); `circuits/` holds the six circuits it lowered, in OpenQASM 2 (`rccx` is the Qiskit
relative-phase Toffoli gate).

| circuit | before | after | found after |
|---|---|---|---|
| rl_rd53_138 | 33 | 23 | 998 s |
| controlled_adder_3b | 31 | 27 | 2 configurations |
| adder_live_carry | 14 | 13 | 41 s |
| rl_4mod5-v1_23 | 7 | 6 | 17 s |
| rl_rd32_271 | 9 | 8 | 30 s |
| rl_rd32_272 | 9 | 8 | 26 s |

Each circuit was verified with a dense `Operator` (Qiskit) against the exact decomposition on the valid inputs (one global phase, deviation below 1e-14)
and its CX count was taken after decomposition to cx and u. The other circuits did not improve; 13 of 31 did not finish all configurations in the budget,
so the search proves no optimality.

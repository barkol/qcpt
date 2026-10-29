"""Stage 2 of QCPT (certified beam resynthesis) with the schedule of the paper, timed; QCPT_GPU=0/1 selects the CPU pool or the GPU expansion.
Output: JSON line with cost found, wall time, time to the final circuit, energy (GPU: nvidia-smi power integral; CPU: RAPL package energy)."""
import os, sys, re, time, json, threading, subprocess
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); sys.path[:0] = [R, R + "/experiments", R + "/experiments/revlib"]
import numpy as np
from qiskit import transpile
import benchmarks as B, large_benchmarks as LB
from _clean import clean_ancillas
from revlib import parse as rl_parse
from toffoli_optimizer.core import resynth as RS
from toffoli_optimizer.core.decomposition_selector import ErrorBudgetSelector
from toffoli_optimizer.core.subspace_check import circuit_columns
name = sys.argv[1]; cur0 = int(sys.argv[2])                    # cur0 = best stage-1 candidate (CX count), as in the paper's driver
def n2q(c): return sum(1 for i in c.data if i.operation.num_qubits == 2 and i.operation.name != "barrier")
if name.startswith("rl_"): qc, pins, _ = rl_parse(f"{R}/experiments/revlib/real/{name[3:]}.real")
else:
    suite = {c.name: c for c in B.benchmark_suite()}
    if name in suite: qc = suite[name]
    else:
        m_ = re.fullmatch(r"(half_uncomputed|mod_incr|controlled_adder)_(\d+)(c|b)", name); qc = {"half_uncomputed": LB.half_uncomputed_oracle, "mod_incr": LB.modular_increment, "controlled_adder": B.controlled_adder}[m_.group(1)](int(m_.group(2)))
    pins = clean_ancillas(qc)
n = qc.num_qubits; inputs = [x for x in range(1 << n) if not any(x >> q & 1 for q in pins)]; E = ErrorBudgetSelector().select(qc, pinned_zero=pins)["exact"]
UE = circuit_columns(E, inputs)
def dev(c):
    C = circuit_columns(c, inputs); ph = np.vdot(UE.ravel(), C.ravel()); return float(np.abs(C - (ph / abs(ph)) * UE).max()) if abs(ph) > 1e-12 else 9.9

tgt = RS.classical_target(qc, inputs); m = len(inputs); sc = max(32.0 / m, 0.06); budget = float(os.environ.get("IMPROVE_BUDGET", "900")); t0 = time.time()
BASIC = {"x", "cx", "mg1", "mg2", "mg3", "mg4", "ccx"}; full = RS.library(n); fullx = RS.library(n, extra_signs=True); basic = [g for g in full if g[0] in BASIC]
libs = {"basic": basic, "full": full, "fullx": fullx}
confs = []                                            # (lib, beam, lam, depth_side, diversity, noise, seed): wider beams, other heuristic weights, exploration slots, noise
for beam in (30000, 100000):
    for lam in (0.4, 0.7, 1.0, 1.5):
        for lib in ("basic", "full"): confs.append((lib, beam, lam, 8, 0.0, 0.0, 0))
for seed in range(4): confs.append(("full", 30000, 0.7, 10, 0.25, 0.4, seed + 1)); confs.append(("fullx", 30000, 0.7, 10, 0.25, 0.4, seed + 11))
target = cur0 - 1; log = []; found = None
for lib, beam, lam, dsd, div, noise, seed in confs:
    if time.time() - t0 > budget: log.append(["BUDGET"]); break
    b = max(int(beam * sc), 500); t1 = time.time()
    r = RS.bidir_search(n, inputs, tgt, max_cost=target, beam=b, lam=lam, depth_side=dsd, lib=libs[lib], diversity=div, noise=noise, seed=seed)
    log.append([lib, b, lam, dsd, div, noise, seed, round(time.time() - t1, 1), None if not r else r[0]])
    if r:
        rc = RS.emit(n, r[1]); found = dict(cost=r[0], dev=dev(rc), conf=log[-1]); break
print("RESULT " + json.dumps(dict(circuit=name, n=n, final=cur0, target=target, nconf=len(log), done=len(log), wall_s=round(time.time() - t0, 1), improved=found, log=log)), flush=True)

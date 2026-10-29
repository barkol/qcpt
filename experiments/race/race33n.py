#!/usr/bin/env python3
"""Race on the C++ core for one circuit, metered and independently verified. Usage: race33.py CIRCUIT START BEST [--alpha A --gamma G --round 60]
Metering: machine-wide busy CPU seconds from /proc/stat, sampled every 0.1 s (the C++ core uses no GPU), so the energy up to the best circuit can be read off.
Verification: the winning circuit (template winners with their peeled prefix, database middles expanded) is emitted with Qiskit and compared with the exact circuit on the valid inputs."""
import os, sys, time, json, subprocess, argparse, threading
sys.path.insert(0, os.environ.get("QCPT_RUNS", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "runs")))
import numpy as np
import portfolio as PF
import race_cpp as RC
from toffoli_optimizer.core import resynth as RS, template as TP
from toffoli_optimizer.core.decomposition_selector import ErrorBudgetSelector
from toffoli_optimizer.core.subspace_check import circuit_columns

def busy():
    v = [int(x) for x in open("/proc/stat").readline().split()[1:]]
    return (v[0] + v[1] + v[2] + v[5] + v[6]) / os.sysconf("SC_CLK_TCK")

ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("start", type=int); ap.add_argument("best", type=int)
ap.add_argument("--alpha", type=float, default=0.0); ap.add_argument("--gamma", type=float, default=1.0); ap.add_argument("--round", type=float, default=60); ap.add_argument("--extra", type=int, default=0); ap.add_argument("--stop", action="store_true")
a = ap.parse_args()
for sd in range(2, 2 + a.extra): PF.BEAM[f"B4s{sd}"] = ("full", 30000, 0.7, 10, 0.25, 0.4, sd)        # extra noisy beams with other seeds: fills idle cores, more luck per round
probs = RC.problems(a.name, a.start - 1, 2)
if os.environ.get("QCPT_MEMBERS"): probs = {k: v for k, v in probs.items() if k in os.environ["QCPT_MEMBERS"].split(",")}
text = f"{a.start} {a.best if a.stop else 0} {a.round} {len(probs)} {a.alpha} {a.gamma}\n" + "".join(f"member {mid} {'beam' if argv[1] == 'beam' else 'template'} {off}\n{body}" for mid, (argv, body, off) in probs.items())
tl = []; stop = threading.Event(); b0 = busy(); t0 = time.time()
def sampler():
    while not stop.is_set(): tl.append((time.time() - t0, busy() - b0)); time.sleep(0.1)
th = threading.Thread(target=sampler, daemon=True); th.start()
p = subprocess.Popen([RC.QCPT, "race"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=dict(os.environ, QCPT_DB=RC.DB))
p.stdin.write(text); p.stdin.close()
steps = []; lines = []; arrival = None; final = None; winner = None; gates = []
for ln in p.stdout:
    t = ln.split()
    if not t: continue
    if t[0] == "STEP": steps.append(dict(member=t[1], cost=int(t[2]), round_s=float(t[3]), at_s=float(t[4]), wall=round(time.time() - t0, 3)))
    elif t[0] == "RESULT": final = int(t[1]); tb = float(t[2]); tot = float(t[3])
    elif t[0] == "WINNER": winner = t[1]
    elif t[0] != "END": gates.append(t)
p.wait(); wall = time.time() - t0; cpu = busy() - b0; stop.set(); th.join(1)
res = dict(circuit=a.name, start=a.start, best_known=a.best, final=final, reached=final is not None and final <= a.best, wall_s=round(wall, 2), cpu_s=round(cpu, 1), steps=steps, winner=winner)
if steps:
    res["t_best"] = steps[-1]["wall"]; res["t_best_core"] = steps[-1]["at_s"]
    res["cpu_s_at_best"] = round(next((c for t, c in tl if t >= steps[-1]["wall"]), cpu), 1)
# independent verification of the final circuit
if gates and winner:
    qc, pins, n, inputs = PF.load(a.name); G = []
    if winner in PF.TEMPLATE:
        peel, _ = PF.TEMPLATE[winner]
        ops = [(i.operation.name, tuple(qc.find_bit(x).index for x in i.qubits)) for i in qc.data if i.operation.name != "barrier"]; o2 = list(ops)
        while peel and o2 and o2[0][0] in ("x", "cx"): G.append(o2.pop(0))
    for t in gates:
        if t[0] == "G": G.append((t[1], tuple(int(v) for v in t[2:])))
        elif t[0] == "DBMID": G.append(("DBMID", tuple(int(v) for v in t[1:4]), tuple(int(v) for v in t[4:12])))
    DBP = f"{PF.R}/toffoli_optimizer/core/db3_costs.json"; G = TP.expand_db(G, DBP, TP.load_db3(DBP))
    E = ErrorBudgetSelector().select(qc, pinned_zero=pins)["exact"]; UE = circuit_columns(E, inputs); C = circuit_columns(RS.emit(n, G), inputs)
    ph = np.vdot(UE.ravel(), C.ravel()); res["dev"] = float(np.abs(C - (ph / abs(ph)) * UE).max()); res["cx"] = sum(RS.gate_cost(g[0]) for g in G)
    from qiskit import qasm2
    os.makedirs(os.path.join(os.environ.get("QCPT_RUNS", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "runs")), "race33_new/qasm"), exist_ok=True)
    open(os.path.join(os.environ.get("QCPT_RUNS", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "runs")), f"race33_new/qasm/{a.name}.qasm"), "w").write(qasm2.dumps(RS.emit(n, G)))
    json.dump([[g[0], list(g[1])] + ([list(g[2])] if len(g) > 2 else []) for g in G], open(os.path.join(os.environ.get("QCPT_RUNS", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "runs")), f"race33_new/qasm/{a.name}.gates.json"), "w"))
print("RESULT " + json.dumps(res))

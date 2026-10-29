#!/usr/bin/env python3
"""Race of all portfolio members on the C++ core (prototype of the single-process scheduler): every member is a `qcpt` process, the first success lowers the target for all
(a new round), the last round without success confirms. Usage: race_cpp.py CIRCUIT START [--round 120] [--threads 14]"""
import os, sys, time, json, signal, subprocess, argparse
sys.path.insert(0, os.environ.get("QCPT_RUNS", os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "runs")))
import portfolio as PF
from toffoli_optimizer.core import resynth as RS
QCPT = os.environ.get("QCPT_BIN") or os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "cpp/qcpt"); DB = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "cpp/data/db3.tab")
BASIC = ["x", "cx", "mg1", "mg2", "mg3", "mg4", "ccx"]
GAD = ("mg1", "mg2", "mg3", "mg4", "mgp21", "mgq21", "mgr21", "mgs21", "mgt21", "mgu21", "mgv21", "mgw21", "mgp12", "mgq12")

def run_cls(x, seq):
    for nm, q in seq:
        if nm == "x": x ^= 1 << q[0]
        elif nm == "cx": x ^= ((x >> q[0]) & 1) << q[1]
        else: x ^= (((x >> q[0]) & (x >> q[1])) & 1) << q[2]
    return x

def problems(name, target, nth):
    """member id -> (argv, stdin text, cost offset); templates only when n <= 7 and the circuit is classical x/cx/ccx."""
    qc, pins, n, inputs = PF.load(name); tgt = [int(v) for v in RS.classical_target(qc, inputs)]; m = len(inputs); sc = max(32.0 / m, 0.06) * float(os.environ.get("QCPT_BEAM_SCALE", "1")); out = {}
    for mid, (lib_, b, lam, dsd, div, noise, seed) in PF.BEAM.items():
        names = [] if lib_ == "full" else BASIC; beam = max(int(b * sc), 500)
        out[mid] = ([QCPT, "beam"], f"{n} {m}\n{' '.join(map(str, inputs))}\n{' '.join(map(str, tgt))}\n0\n{target} {beam} {lam} {dsd} {div} {noise} {seed} {nth} 0\nlib {len(names)} {' '.join(names)}\n", 0)
    ops = [(i.operation.name, tuple(qc.find_bit(x).index for x in i.qubits)) for i in qc.data if i.operation.name != "barrier"]
    if n <= int(os.environ.get("QCPT_TEMPLATE_MAXN", "7")) and all(o[0] in ("x", "cx", "ccx") for o in ops):
        for mid, (peel, cname) in PF.TEMPLATE.items():
            ga = ("mg1", "mg2", "mg3", "mg4") if cname == "A" else GAD; f1, f2 = (2, 2) if cname == "A" else (1, 1)
            pre = []; o2 = list(ops)
            while peel and o2 and o2[0][0] in ("x", "cx"): pre.append(o2.pop(0))
            if not peel and not any(o[0] in ("x", "cx") for o in ops[:1]): continue
            npre = sum(1 for g in pre if g[0] == "cx"); xin = [run_cls(x, pre) for x in inputs]
            out[mid] = ([QCPT, "template", DB], f"{n} {m}\n{' '.join(map(str, xin))}\n{' '.join(map(str, tgt))}\n{target - npre} {f1} 0 {f2} {nth} 0\nga {len(ga)} {' '.join(ga)}\ngb 0\nmid 0\n", npre)
    return out

def one_round(name, target, limit, nth):
    t0 = time.time(); procs = {}
    for mid, (argv, text, off) in problems(name, target, nth).items():
        p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, start_new_session=True); p.stdin.write(text); p.stdin.close(); procs[mid] = (p, off)
    win = None; alive = dict(procs)
    while alive and time.time() - t0 < limit and win is None:
        for mid, (p, off) in list(alive.items()):
            if p.poll() is None: continue
            del alive[mid]; first = (p.stdout.readline() or "").split()
            if first and first[0] == "COST": win = (mid, int(first[1]) + off); break
        time.sleep(0.005)
    el = time.time() - t0
    for p, _ in procs.values():
        try: os.killpg(p.pid, signal.SIGKILL)
        except Exception: pass
        p.wait()
    return win, el, sorted(set(procs) - set(alive))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("start", type=int); ap.add_argument("--round", type=float, default=120); ap.add_argument("--threads", type=int, default=14); ap.add_argument("--best", type=int, default=0)
    a = ap.parse_args(); T0 = time.time(); cur = a.start; steps = []
    while True:
        win, el, fin = one_round(a.name, cur - 1, a.round, a.threads)
        if win is None: break
        cur = win[1]; steps.append(dict(member=win[0], cost=cur, round_s=round(el, 2), at_s=round(time.time() - T0, 2)))
        if a.best and cur <= a.best: break
    print("RESULT " + json.dumps(dict(circuit=a.name, start=a.start, final=cur, t_best=steps[-1]["at_s"] if steps else None, total_s=round(time.time() - T0, 2), steps=steps)), flush=True)

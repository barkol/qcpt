#!/usr/bin/env python3
"""Portfolio schedule for stage 2 of QCPT with early stop.

Members (each searches for a certified circuit with at most `target` two-qubit gates): beam searches on the GPU with three settings, template searches
(four combinations, n <= 7) and a beam search with exploration slots and noise. The cheap members start together; the expensive ones start after
`--delay` seconds or as soon as the cheap ones are done. The first success stops all others (process groups are killed), the target drops to the cost found
minus one, and the round is repeated until a round finds nothing or the time limit of the round is over.
Usage: portfolio.py NAME START [--round 600] [--delay 90] [--budget 3600]"""
import os, sys, re, time, json, signal, argparse, multiprocessing as mp
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); sys.path[:0] = [R, R + "/experiments", R + "/experiments/revlib"]

BEAM = {  # id: (library, beam at 32 inputs, lambda, depth, diversity, noise, seed)
    "B1_full_l04": ("full", 60000, 0.4, 8, 0.0, 0.0, 0), "B2_basic_l04": ("basic", 100000, 0.4, 8, 0.0, 0.0, 0), "B3_basic_l06": ("basic", 30000, 0.6, 8, 0.0, 0.0, 0),
    "B4_div_noise": ("full", 30000, 0.7, 10, 0.25, 0.4, 1)}
TEMPLATE = {"T1_peel_C": (True, "C"), "T2_peel_A": (True, "A"), "T3_nopeel_C": (False, "C"), "T4_nopeel_A": (False, "A")}
WAVE1 = ["B1_full_l04", "B2_basic_l04", "B3_basic_l06", "T1_peel_C", "T2_peel_A", "T3_nopeel_C"]; WAVE2 = ["T4_nopeel_A", "B4_div_noise"]


def load(name):
    import numpy as np
    import benchmarks as B, large_benchmarks as LB
    from _clean import clean_ancillas
    from revlib import parse as rl_parse
    if name.startswith("rl_"): qc, pins, _ = rl_parse(f"{R}/experiments/revlib/real/{name[3:]}.real")
    else:
        suite = {c.name: c for c in B.benchmark_suite()}
        if name in suite: qc = suite[name]
        elif name == "half_uncomputed_min": qc = LB.half_uncomputed_minimal()
        else:
            m_ = re.fullmatch(r"(half_uncomputed|mod_incr|controlled_adder|ripple_adder|grover_oracle)_(\d+)(c|b)", name); qc = {"half_uncomputed": LB.half_uncomputed_oracle, "mod_incr": LB.modular_increment, "ripple_adder": LB.ripple_carry_adder, "grover_oracle": LB.grover_oracle, "controlled_adder": B.controlled_adder}[m_.group(1)](int(m_.group(2)))
        pins = clean_ancillas(qc)
    n = qc.num_qubits; inputs = [x for x in range(1 << n) if not any(x >> q & 1 for q in pins)]
    return qc, pins, n, inputs


def member(mid, name, target, q):
    """Runs in its own process group; puts (mid, cost or None, gates, seconds) on the queue."""
    os.setpgrp(); t0 = time.time()
    try:
        from toffoli_optimizer.core import resynth as RS
        qc, pins, n, inputs = load(name); tgt = RS.classical_target(qc, inputs); m = len(inputs); sc = max(32.0 / m, 0.06)
        if mid in BEAM:
            lib_, b, lam, dsd, div, noise, seed = BEAM[mid]; full = RS.library(n); BASIC = {"x", "cx", "mg1", "mg2", "mg3", "mg4", "ccx"}
            lib = full if lib_ == "full" else [g for g in full if g[0] in BASIC]
            os.environ["QCPT_GPU"] = "0" if noise > 0 else "1"; os.environ.setdefault("QCPT_NPROC", "8")
            r = RS.bidir_search(n, inputs, tgt, max_cost=target, beam=max(int(b * sc), 500), lam=lam, depth_side=dsd, lib=lib, diversity=div, noise=noise, seed=seed)
            q.put((mid, None if r is None else r[0], None if r is None else [(g[0], tuple(g[1])) for g in r[1]], time.time() - t0)); return
        from toffoli_optimizer.core import template as TP
        peel, cname = TEMPLATE[mid]; GAD = ("mg1", "mg2", "mg3", "mg4", "mgp21", "mgq21", "mgr21", "mgs21", "mgt21", "mgu21", "mgv21", "mgw21", "mgp12", "mgq12")
        cfg = dict(f1=2, fm=0, f2=2, ga=("mg1", "mg2", "mg3", "mg4")) if cname == "A" else dict(f1=1, fm=0, f2=1, ga=GAD)
        ops = [(i.operation.name, tuple(qc.find_bit(x).index for x in i.qubits)) for i in qc.data if i.operation.name != "barrier"]
        if n > 7 or any(o[0] not in ("x", "cx", "ccx") for o in ops): q.put((mid, None, None, time.time() - t0)); return
        def run(x, seq):
            for nm, qq in seq:
                if nm == "x": x ^= 1 << qq[0]
                elif nm == "cx": x ^= ((x >> qq[0]) & 1) << qq[1]
                else: x ^= (((x >> qq[0]) & (x >> qq[1])) & 1) << qq[2]
            return x
        pre = []; o2 = list(ops)
        while peel and o2 and o2[0][0] in ("x", "cx"): pre.append(o2.pop(0))
        if not peel and not any(o[0] in ("x", "cx") for o in ops[:1]): q.put((mid, None, None, time.time() - t0)); return
        npre = sum(1 for g in pre if g[0] == "cx"); DBP = f"{R}/toffoli_optimizer/core/db3_costs.json"; db = TP.load_db3(DBP); lib = RS.library(n, extra_signs=True)
        r = TP.template_search(n, [run(x, pre) for x in inputs], tgt, target - npre, lib=lib, nproc=int(os.environ.get("TP_NPROC", 8)), db=db, **cfg)
        if r is None: q.put((mid, None, None, time.time() - t0)); return
        gates = TP.expand_db(r[1], DBP, db); q.put((mid, r[0] + npre, [(nm, tuple(a)) for nm, a in pre] + [(g[0], tuple(g[1])) for g in gates], time.time() - t0))
    except BaseException as e:
        q.put((mid, None, "ERROR " + repr(e)[:200], time.time() - t0))


PIN = os.environ.get("PF_PIN", "0") == "1"
WEIGHT = {"B1_full_l04": 1, "B2_basic_l04": 1, "B3_basic_l06": 1, "B4_div_noise": 5, "T1_peel_C": 5, "T2_peel_A": 5, "T3_nopeel_C": 5, "T4_nopeel_A": 5}


def repin(procs, done):
    """Splits the cores among the live members in proportion to WEIGHT (contiguous blocks) and pins every process of each member to its block."""
    import psutil
    live = [m for m in procs if m not in done]
    if not live: return
    nc = os.cpu_count() or 1; tot = sum(WEIGHT[m] for m in live); lo = 0.0
    for m in live:
        hi = lo + nc * WEIGHT[m] / tot; cores = list(range(int(round(lo)), max(int(round(hi)), int(round(lo)) + 1))); lo = hi
        try:
            pr = psutil.Process(procs[m].pid)
            for c in [pr] + pr.children(recursive=True):
                try: c.cpu_affinity(cores)
                except Exception: pass
        except Exception: pass


def round_(name, target, delay, limit, log):
    ctx = mp.get_context("spawn"); q = ctx.Queue(); procs = {}; t0 = time.time(); done = {}
    def start(ids):
        for mid in ids:
            p = ctx.Process(target=member, args=(mid, name, target, q)); p.start(); procs[mid] = p
    def kill_all():
        for p in procs.values():
            try: os.killpg(p.pid, signal.SIGKILL)
            except Exception: pass
            try: p.kill()
            except Exception: pass
    start(WAVE1 + WAVE2 if PIN else WAVE1); wave2 = PIN; result = None; tpin = 0.0
    while True:
        try: mid, cost, gates, sec = q.get(timeout=0.3)
        except Exception: mid = None
        if mid is not None:
            done[mid] = (cost, round(sec, 1))
            if isinstance(gates, str): log.append(f"  {mid}: {gates}")
            if cost is not None and cost <= target: result = (mid, cost, gates, time.time() - t0); break
        el = time.time() - t0
        if PIN and el - tpin > 0.5: repin(procs, done); tpin = el
        if not wave2 and (el >= delay or all(m in done for m in WAVE1)): start(WAVE2); wave2 = True
        if wave2 and all(m in done for m in WAVE1 + WAVE2): break
        if el >= limit: break
    kill_all()
    for p in procs.values(): p.join(timeout=2)
    log.append(f"  round target<={target}: " + (f"SUCCESS {result[0]} cost {result[1]} at {result[3]:.1f}s" if result else f"no success in {time.time() - t0:.0f}s; finished members {sorted(done)}"))
    return result, done, time.time() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("start", type=int); ap.add_argument("--round", type=float, default=600); ap.add_argument("--delay", type=float, default=15); ap.add_argument("--budget", type=float, default=3600)
    a = ap.parse_args(); T0 = time.time(); cur = a.start; log = []; steps = []; best_gates = None
    while time.time() - T0 < a.budget:
        res, done, el = round_(a.name, cur - 1, a.delay, a.round, log)
        if res is None: break
        mid, cost, gates, t = res; cur = cost; best_gates = gates; steps.append(dict(member=mid, cost=cost, round_s=round(t, 1), at_s=round(time.time() - T0, 1)))
    # verification of the final circuit with the dense-free column check
    ver = None
    if best_gates:
        import numpy as np
        from toffoli_optimizer.core import resynth as RS
        from toffoli_optimizer.core.decomposition_selector import ErrorBudgetSelector
        from toffoli_optimizer.core.subspace_check import circuit_columns
        qc, pins, n, inputs = load(a.name); E = ErrorBudgetSelector().select(qc, pinned_zero=pins)["exact"]; UE = circuit_columns(E, inputs); C = circuit_columns(RS.emit(n, best_gates), inputs)
        ph = np.vdot(UE.ravel(), C.ravel()); ver = float(np.abs(C - (ph / abs(ph)) * UE).max())
    print("RESULT " + json.dumps(dict(circuit=a.name, start=a.start, final=cur, steps=steps, total_s=round(time.time() - T0, 1), dev=ver, last_round=log[-1].strip() if log else None)), flush=True)

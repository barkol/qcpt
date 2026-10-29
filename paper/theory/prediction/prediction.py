"""Prediction of the number of 2q gates of the ErrorBudgetSelector pass by a static per-Toffoli criterion.

Usage:
  prediction.py worker SUITE KEY      -> prints one JSON row to stdout
  prediction.py run [--suites a,b,c] [--timeout-revlib 300] [--timeout-scale 1800] [--jobs 3]
Checkpoint: results.jsonl (append, one row per circuit; a restart skips the finished keys).
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results.jsonl")
REVLIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "experiments", "revlib")


# ----------------------------------------------------------------- kryterium
def has(S, a=None, b=None, t=None):
    for x in S:
        if (a is None or (x & 1) == a) and (b is None or ((x >> 1) & 1) == b) \
                and (t is None or ((x >> 2) & 1) == t):
            return True
    return False


def c_full(S):
    """Full table: 0 / 1 / 3 (none of Q={100,101,010,011} missing) / 6."""
    if not has(S, a=1, b=1):
        return 0
    if not has(S, a=0, b=1) or not has(S, a=1, b=0):
        return 1
    Q = [1 + 0, 1 + 4, 2 + 0, 2 + 4]          # abt=100,101,010,011 -> x=a+2b+4t
    if not all(q in S for q in Q):
        return 3
    return 6


def c_pass(S):
    """Table restricted to the options that select() actually tries:
    relphase (phase on abt=100), relphase_m (phase on abt=101), without orientation swap."""
    if not has(S, a=1, b=1):
        return 0
    if not has(S, a=0, b=1) or not has(S, a=1, b=0):
        return 1
    if (1 not in S) or (5 not in S):
        return 3
    return 6


def act_cost(action):
    if action is None:
        return 6
    k = action[0]
    if k in ("relphase", "relphase_m"):
        return 3
    if k == "control_drop":
        return 0 if len(action[1]) == 0 else 1
    raise ValueError(action)


# ----------------------------------------------------------------- circuits
def load(suite, key):
    if suite == "primary":
        from benchmarks import benchmark_suite
        from _clean import clean_ancillas
        qc = [c for c in benchmark_suite() if c.name == key][0]
        return qc, clean_ancillas(qc)
    if suite == "scale":
        from large_benchmarks import large_suite
        from _clean import clean_ancillas
        qc = [c for c in large_suite() if c.name == key][0]
        return qc, clean_ancillas(qc)
    if suite == "revlib":
        sys.path.insert(0, REVLIB)
        from revlib import parse
        qc, pins, info = parse(os.path.join(REVLIB, "real", key))
        if qc is None:
            raise RuntimeError(f"parse failed: {info}")
        return qc, pins
    raise ValueError(suite)


def keys(suite):
    if suite == "primary":
        from benchmarks import benchmark_suite
        return [c.name for c in benchmark_suite()]
    if suite == "scale":
        from large_benchmarks import large_suite
        return [c.name for c in large_suite()]
    if suite == "revlib":
        d = json.load(open(os.path.join(REVLIB, "wybor.json")))
        return [x["file"] for x in d if x["lines"] <= 12]
    raise ValueError(suite)


# ----------------------------------------------------------------- worker
def worker(suite, key):
    from toffoli_optimizer.core.decomposition_selector import ErrorBudgetSelector, _CCX_NAMES
    from toffoli_optimizer.core.reach_local import local_reachable
    from toffoli_optimizer.core.reachable_subspace import reachable_overapprox
    from toffoli_optimizer.core.subspace_check import project_support

    qc, pins = load(suite, key)
    pins = tuple(pins)
    sel = ErrorBudgetSelector()
    t0 = time.time()
    res = sel.select(qc, pinned_zero=pins)
    t_sel = time.time() - t0
    rep = res["report"]
    actions = res["actions"]
    fell_back = bool(rep["fell_back_to_exact"])

    # S(g) dokladnie tak jak w select(): local_reachable, a jesli None -> nadaproksymacja
    space = "all_basis"
    if pins and qc.num_qubits <= 20:
        pm = sum(1 << q for q in pins)
        space = [x for x in range(1 << qc.num_qubits) if not x & pm]

    def S_of(idx, qbs):
        loc = local_reachable(qc, idx, qbs, pins)
        src = "exact"
        if loc is None:
            loc = project_support(reachable_overapprox(qc, idx, input_space=space), qbs)
            src = "overapprox"
        return sorted(int(v) for v in loc), src

    struct_pairs = set()
    from toffoli_optimizer.core.context_analysis import find_relative_phase_safe_sites
    all_sites = find_relative_phase_safe_sites(qc)
    applied_t = set(tuple(x) for x in rep["applied_sites"])
    for s in all_sites:
        if tuple(s) in applied_t:
            struct_pairs.update((s.compute_idx, s.uncompute_idx))
    rejected_site_idx = set()
    for s in all_sites:
        if tuple(s) not in applied_t:
            rejected_site_idx.update((s.compute_idx, s.uncompute_idx))
    win = set()
    for w in rep["window_pairs_admitted"]:
        win.update(w["pair"])
    if fell_back:
        # nothing is left after the fallback; we remember what was there before (from the window log)
        pass

    toffs = []
    nT = 0
    for idx, inst in enumerate(qc.data):
        name = inst.operation.name.lower()
        qb = [qc.find_bit(q).index for q in inst.qubits]
        if name not in _CCX_NAMES or len(qb) != 3:
            continue
        nT += 1
        S, src = S_of(idx, tuple(qb))
        cf, cp = c_full(S), c_pass(S)
        in_pair = idx in struct_pairs
        in_win = idx in win
        pred_full = 3 if (in_pair or in_win) else min(6, cf)
        pred_pass = 3 if (in_pair or in_win) else min(6, cp)
        a = actions.get(idx)
        toffs.append({"idx": idx, "qubits": qb, "S": S, "S_src": src,
                      "S_bits": ["%d%d%d" % (x & 1, (x >> 1) & 1, (x >> 2) & 1) for x in S],
                      "c_full": cf, "c_pass": cp, "pair_C": in_pair, "window_W": in_win,
                      "rejected_site": idx in rejected_site_idx,
                      "action": list(a) if a is not None else None,
                      "action_repr": repr(a), "cost_actual": act_cost(a),
                      "pred_full": pred_full, "pred_pass": pred_pass})

    twoq_before = rep["two_qubit_before"]
    twoq_after = rep["two_qubit_after"]
    nonT = twoq_before - 6 * nT
    row = {
        "suite": suite, "key": key, "n": qc.num_qubits, "pins": list(pins),
        "n_toffoli": nT, "t_select": t_sel, "fell_back": fell_back,
        "verify_info": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                        for k, v in dict(rep.get("verify_info") or {}).items()},
        "twoq_before": twoq_before, "twoq_actual": twoq_after, "twoq_nonT": nonT,
        "twoq_from_actions": nonT + sum(t["cost_actual"] for t in toffs),
        "twoq_pred_full": nonT + sum(t["pred_full"] for t in toffs),
        "twoq_pred_pass": nonT + sum(t["pred_pass"] for t in toffs),
        "sites_found": rep["sites_found"], "sites_applied": rep["sites_applied"],
        "n_window_pairs": len(rep["window_pairs_admitted"]),
        "toffolis": toffs,
    }
    print(json.dumps(row))


# ----------------------------------------------------------------- driver
def done_keys():
    out = set()
    if os.path.exists(OUT):
        for ln in open(OUT):
            ln = ln.strip()
            if ln:
                try:
                    r = json.loads(ln)
                    out.add((r["suite"], r["key"]))
                except Exception:
                    pass
    return out


def run(suites, timeouts, jobs):
    env = dict(os.environ)
    todo = []
    dk = done_keys()
    for s in suites:
        for k in keys(s):
            if (s, k) not in dk:
                todo.append((s, k))
    print(f"todo: {len(todo)}", flush=True)
    running = []
    while todo or running:
        while todo and len(running) < jobs:
            s, k = todo.pop(0)
            p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "worker", s, k],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
            running.append((p, s, k, time.time()))
            print(f"start {s} {k} pid={p.pid}", flush=True)
        time.sleep(0.5)
        still = []
        for p, s, k, t0 in running:
            if p.poll() is None:
                if time.time() - t0 > timeouts[s]:
                    p.kill()          # by PID (the Popen object), not by pattern
                    p.wait()
                    row = {"suite": s, "key": k, "skipped": True,
                           "reason": f"timeout > {timeouts[s]} s"}
                    with open(OUT, "a") as f:
                        f.write(json.dumps(row) + "\n")
                    print(f"TIMEOUT {s} {k}", flush=True)
                else:
                    still.append((p, s, k, t0))
                continue
            out, err = p.communicate()
            lines = [l for l in out.splitlines() if l.startswith("{")]
            if p.returncode == 0 and lines:
                row = json.loads(lines[-1])
            else:
                row = {"suite": s, "key": k, "skipped": True,
                       "reason": f"error rc={p.returncode}: {err[-600:]}"}
            with open(OUT, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"done {s} {k} {time.time()-t0:.1f}s "
                  f"act={row.get('twoq_actual')} pf={row.get('twoq_pred_full')} "
                  f"pp={row.get('twoq_pred_pass')}", flush=True)
        running = still


if __name__ == "__main__":
    if sys.argv[1] == "worker":
        worker(sys.argv[2], sys.argv[3])
    else:
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("--suites", default="primary,revlib,scale")
        ap.add_argument("--timeout-revlib", type=float, default=300)
        ap.add_argument("--timeout-scale", type=float, default=1800)
        ap.add_argument("--jobs", type=int, default=3)
        a = ap.parse_args()
        run(a.suites.split(","), {"primary": 600, "revlib": a.timeout_revlib,
                                  "scale": a.timeout_scale}, a.jobs)

"""Punkt 1: wybor orientacji gadzetow wg kalibracji. Dla kazdego urzadzenia i obwodu:
uklad kubitow L z transpilacji obwodu dokladnego (opt 3, ziarno 11); orientacja domyslna vs
wg kalibracji (orient); oba warianty transpilowane z tym samym L (opt 1, ziarna 1..5);
infidelity z kalibracji urzadzenia 1-prod(1-eps) po bramkach fizycznych (rz wolne, bez odczytu).
Wznawialne: orientacja.jsonl (klucz device|circuit). Kalibracje (kal_*.pkl) pobiera QiskitRuntimeService().backend(...).properties()."""
import json, os, pickle, sys, time
HERE0 = os.path.dirname(os.path.abspath(__file__)); C = os.path.dirname(os.path.dirname(HERE0)); sys.path.insert(0, C); sys.path.insert(0, C + "/experiments")
RL = os.path.join(C, "experiments", "revlib"); sys.path.insert(0, RL)
from _paths import EXPERIMENTS_DIR  # noqa
import benchmarks as B
from _clean import clean_ancillas
from revlib import parse
from qiskit import transpile
from toffoli_optimizer.core.decomposition_selector import ErrorBudgetSelector
from toffoli_optimizer.core.orientation import orient, pair_cost_from_calibration, edge_errors_from_properties
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "orientacja.jsonl")
SEEDS = [1, 2, 3, 4, 5]
DEV = []
for nm in ("ibm_marrakesh", "ibm_fez", "ibm_kingston"):
    d = pickle.load(open(os.path.join(HERE, f"kal_{nm}.pkl"), "rb")); DEV.append((nm, d["props"], d["target"]))
jt = pickle.load(open(os.path.join(HERE, "marrakesh_props_jobtime.pkl"), "rb"))
DEV.append(("ibm_marrakesh@2026-10-01", jt, DEV[0][2]))


def gerr_map(props):
    g = {}
    for x in props["gates"]:
        e = [p["value"] for p in x["parameters"] if p["name"] == "gate_error"]
        if e: g[(x["gate"], tuple(x["qubits"]))] = min(e[0], 1.0)
    return g


def calib_infid(c, gm):
    f = 1.0
    for ins in c.data:
        n = ins.operation.name
        if n in ("rz", "barrier", "delay", "measure"): continue
        qs = tuple(c.find_bit(q).index for q in ins.qubits)
        e = gm.get((n, qs)) if (n, qs) in gm else gm.get((n, qs[::-1]))
        if e is None: raise RuntimeError(f"brak kalibracji {n}{qs}")
        f *= 1 - e
    return 1 - f


def circuits():
    for qc in B.benchmark_suite():
        yield "Q2", qc.name, qc, clean_ancillas(qc)
    sel = sorted(json.load(open(os.path.join(RL, "wybor.json"))), key=lambda o: o["gates"])
    for o in sel:
        if o["gates"] > 1000: continue
        qc, pins, _ = parse(os.path.join(RL, "real", o["file"]))
        yield "RevLib", o["file"], qc, pins


done = {json.loads(l)["key"] for l in open(OUT)} if os.path.exists(OUT) else set()
sel_cache = {}
for suite, name, qc, pins in circuits():
    for dev, props, target in DEV:
        key = f"{dev}|{suite}|{name}"
        if key in done: continue
        if name not in sel_cache:
            s = ErrorBudgetSelector(); sel_cache[name] = (s, s.select(qc, pinned_zero=pins))
        s, res = sel_cache[name]
        n_g = sum(1 for a in res["actions"].values() if a[0] in ("relphase", "relphase_m"))
        row = {"key": key, "device": dev, "suite": suite, "circuit": name, "n": qc.num_qubits, "gadgets": n_g}
        if n_g == 0:
            row.update(oriented=0, skip="brak gadzetow")
        else:
            gm = gerr_map(props)
            L = transpile(res["exact"], target=target, optimization_level=3, seed_transpiler=11).layout.initial_index_layout()[: qc.num_qubits]
            cost = pair_cost_from_calibration(edge_errors_from_properties(props), L)
            t0 = time.time(); out, info = orient(s, qc, res, cost, pinned_zero=pins)
            row.update(oriented=info["oriented"], rejected_R=info["rejected_R"], certificate=info["certificate"],
                       kept_default=info["kept_default"], orient_s=round(time.time() - t0, 2), layout=list(L),
                       flags={str(k): (len(v) > 1) for k, v in info.get("actions", res["actions"]).items()})
            vals = {"default": [], "oriented": []}
            for sd in SEEDS:
                for tag, c in (("default", res["circuit"]), ("oriented", out)):
                    t = transpile(c, target=target, initial_layout=L, optimization_level=1, seed_transpiler=sd)
                    vals[tag].append(calib_infid(t, gm))
            row.update({f"infid_{k}": sum(v) / len(v) for k, v in vals.items()}, infid_seeds=vals)
        with open(OUT, "a") as fh: fh.write(json.dumps(row) + "\n")
        print(key, row.get("oriented"), round(row.get("infid_default", 0), 4), round(row.get("infid_oriented", 0), 4), flush=True)
print("KONIEC")

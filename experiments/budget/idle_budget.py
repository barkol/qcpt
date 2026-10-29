"""Budzet z bledami bezczynnosci, crosstalkiem ZZ (mediana zz z kalibracji) i nadmiarowym bledem CZ
(dopasowanym w budget_vs_hardware.py) na zestawie 12 obwodow i RevLib: T1, T2 = mediany z kalibracji
ibm_marrakesh z chwili testu lustrzanego, czasy bramek 68 ns (CZ) / 36 ns (SX). Wybor dekompozycji
jest ten sam (bezczynnosc nie zmienia decyzji passu); zmienia sie budzet i zysk. -> idle_budget.json"""
import json, os, pickle, statistics, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "experiments")); sys.path.insert(0, os.path.join(ROOT, "experiments", "revlib"))
from _paths import EXPERIMENTS_DIR  # noqa
import benchmarks as B
from _clean import clean_ancillas
from revlib import parse
from toffoli_optimizer.core.decomposition_selector import ErrorBudgetSelector
from toffoli_optimizer.core.error_model import HardwareErrorModel
P = pickle.load(open(os.path.join(HERE, "marrakesh_props_jobtime.pkl"), "rb"))
T1 = [p["value"] * 1e-6 for ps in P["qubits"] for p in ps if p["name"] == "T1"]
T2 = [p["value"] * 1e-6 for ps in P["qubits"] for p in ps if p["name"] == "T2"]
t1, t2 = statistics.median(T1), statistics.median(T2)
import pickle as _pk
_K = _pk.load(open(os.path.join(HERE, "kal_ibm_marrakesh.pkl"), "rb"))
_g = {x["name"]: x["value"] for x in P.get("general", [])}
ZZ = statistics.median(abs(v) * 1e9 for k, v in _g.items() if k.startswith("zz_"))
LAM = json.load(open(os.path.join(HERE, "przewidywanie.json")))["lam_all"]
EMS = {"gates": HardwareErrorModel(), "gates+idle": HardwareErrorModel(t1=t1, t2=t2),
       "full": HardwareErrorModel(t1=t1, t2=t2, zz_hz=ZZ, p_excess_2q=LAM)}
def suite():
    for qc in B.benchmark_suite(): yield "Q2", qc, clean_ancillas(qc)
    for o in json.load(open(os.path.join(ROOT, "experiments", "revlib", "wybor.json"))):
        if o["gates"] <= 1000:
            qc, pins, _ = parse(os.path.join(ROOT, "experiments", "revlib", "real", o["file"])); yield "RevLib", qc, pins
out = {"t1_median_s": t1, "t2_median_s": t2, "zz_median_hz": ZZ, "p_excess_2q": LAM, "suites": {}}
for name, qc, pins in suite():
    s = ErrorBudgetSelector(); res = s.select(qc, pinned_zero=pins)
    agg = out["suites"].setdefault(name, {k: {"exact": 0.0, "ours": 0.0} for k in EMS} | {"n": 0, "idle_us_exact": 0.0, "idle_us_ours": 0.0})
    agg["n"] += 1
    for k, em in EMS.items():
        agg[k]["exact"] += em.circuit_infidelity(res["exact"]); agg[k]["ours"] += em.circuit_infidelity(res["circuit"])
    em = EMS["gates+idle"]
    agg["idle_us_exact"] += sum(em.idle_times(res["exact"]).values()) * 1e6; agg["idle_us_ours"] += sum(em.idle_times(res["circuit"]).values()) * 1e6
for name, a in out["suites"].items():
    for k in EMS: a[k]["red_pct"] = 100 * (a[k]["exact"] - a[k]["ours"]) / a[k]["exact"]
    a["idle_red_pct"] = 100 * (a["idle_us_exact"] - a["idle_us_ours"]) / a["idle_us_exact"]
json.dump(out, open(os.path.join(HERE, "idle_budget.json"), "w"), indent=1); print(json.dumps(out, indent=1))

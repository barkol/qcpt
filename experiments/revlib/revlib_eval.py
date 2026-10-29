"""RevLib suite of the paper (Table 6). The .real files are fetched by fetch_revlib.py
from https://www.revlib.org/doc/real/ into real/.
Zestaw RevLib (NCT, <=24 linie) — te same metody i ten sam certyfikat co w pracy.
Metody: exact_only, ours (pass, semantyka podprocedury, stale-0 przypiete), qiskit_l3, tket, count-greedy.
Certyfikat kazdego wyjscia: scale_eval.certify (gesto na podprzestrzeni wejsc <=12 q, QCEC z ancillami wyzej).
Wznawialne: kazdy obwod -> jedna linia w wyniki.jsonl (osobny proces, twardy limit LIMIT s).
Uzycie: revlib_eval.py all | one <plik.real>"""
import json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); RUN = os.path.dirname(os.path.dirname(HERE))  # korzen repo
OUT = os.path.join(os.environ.get("TOPTOFFOLI_OUT", HERE), "wyniki.jsonl"); LIMIT = int(os.environ.get("LIMIT", "3600"))
PY = sys.executable


def one(fname):
    sys.path.insert(0, RUN + "/experiments"); sys.path.insert(0, RUN); sys.path.insert(0, HERE)
    from _paths import EXPERIMENTS_DIR  # noqa: F401
    import baselines as BL, scale_eval as SE
    from revlib import parse
    from toffoli_optimizer.core.decomposition_selector import ErrorBudgetSelector
    qc, pins, info = parse(os.path.join(HERE, "real", fname))
    EM = SE.EM; row = {"file": fname, **info, "pinned": pins}
    exact = ErrorBudgetSelector().decompose_exact_only(qc)
    t0 = time.time(); sel = ErrorBudgetSelector().select(qc, pinned_zero=pins); t_sel = time.time() - t0
    rep = sel["report"]
    row.update(sel_s=round(t_sel, 2), fell_back=rep.get("fell_back_to_exact"), pairs_C=rep["sites_applied"],
               pairs_W=len(rep.get("window_pairs_admitted", [])), gadgets_R=len(rep.get("rphase_admitted", [])),
               drops_R=len(rep.get("approx_admitted", [])), sel_cert=str(rep.get("verification", {}).get("certification", "")))
    outs = {"exact": exact, "ours": sel["circuit"], "qiskit_l3": BL.m_qiskit_l3(qc), "greedy": BL.m_count_greedy(qc),
            "ours_qiskit": BL.m_qiskit_l3(sel["circuit"])}
    try:
        outs["tket"] = BL.m_tket(qc)
    except Exception as e:
        row["tket_error"] = str(e)[:200]
    for k, c in outs.items():
        t0 = time.time(); r = SE.certify(exact, c, pinned_zero=pins)
        row[k] = {"twoq": EM.two_qubit_count(c), "infid": EM.circuit_infidelity(c), "cert": r.equivalent,
                  "cert_method": r.method, "cert_s": round(time.time() - t0, 2)}
    return row


if __name__ == "__main__":
    if sys.argv[1] == "one":
        print(json.dumps(one(sys.argv[2]), default=str)); sys.exit(0)
    sel = json.load(open(os.path.join(HERE, "wybor.json")))
    sel.sort(key=lambda o: o["gates"])
    done = set()
    if os.path.exists(OUT):
        done = {json.loads(l)["file"] for l in open(OUT)}
    for o in sel:
        f = o["file"]
        if f in done:
            continue
        t0 = time.time()
        try:
            r = subprocess.run([PY, os.path.abspath(__file__), "one", f], capture_output=True, text=True, timeout=LIMIT)
            line = [l for l in r.stdout.splitlines() if l.startswith("{")]
            row = json.loads(line[-1]) if line else {"file": f, "error": (r.stderr or "")[-400:]}
        except subprocess.TimeoutExpired:
            row = {"file": f, "error": f"timeout {LIMIT}s", **{k: o[k] for k in ("lines", "gates", "ccx")}}
        row["wall_s"] = round(time.time() - t0, 1)
        with open(OUT, "a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
        print(f, row.get("error", ""), "ours", row.get("ours", {}).get("twoq"), "exact", row.get("exact", {}).get("twoq"), row["wall_s"], "s", flush=True)
    print("KONIEC", flush=True)

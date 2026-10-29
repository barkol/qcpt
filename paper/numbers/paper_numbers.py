"""Collects the numbers of the text from the results of the recomputation (logs + json) -> paper_numbers.json.
Stops at the first missing source (no partial table). Usage: python paper_numbers.py [RUN]"""
import json, os, re, sys, csv
HERE = os.path.dirname(os.path.abspath(__file__))
RUN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "data", "rerun")  # logs + experiments/ of the paper run
TAB = os.path.join(HERE, "..", "tables", "data")
K = {}
def rd(p): return open(os.path.join(RUN, p)).read()
def pct(a, b): return f"{100*(a-b)/a:.1f}"

# --- suite of 12 circuits (baselines)
b = rd("log_baselines.txt")
def row(m):
    x = re.search(rf"^{m}\s+([\d.]+)\s+(\d+)\s+(\d+)", b, re.M); return float(x.group(1)), int(x.group(2)), int(x.group(3))
ex, q, t, ours = row("exact_only"), row("qiskit_l3"), row("tket"), row("ours")
assert re.search(r"^ours .* 12/12", b, re.M), "ours is not 12/12"
K.update(EX2Q=ex[1], EXINF=f"{ex[0]:.2f}", OUR2Q=ours[1], OURINF=f"{ours[0]:.2f}", DEPEX=ex[2], DEPOUR=ours[2],
         Q2RED2Q=pct(ex[1], ours[1]), Q2REDINF=pct(ex[0], ours[0]),
         VSQ2Q=pct(q[1], ours[1]), VSQINF=pct(q[0], ours[0]), VST2Q=pct(t[1], ours[1]), VSTINF=pct(t[0], ours[0]),
         Q2NCERT=12)
# --- attribution
A = json.load(open(os.path.join(TAB, "attribution.json")))
assert A["exact"]["twoq"] == ex[1] and A["CWR"]["twoq"] == ours[1], "attribution niespojna z baselines"
K.update(ATC=A["C"]["twoq"], ATCW=A["CW"]["twoq"], ATCWR=A["CWR"]["twoq"],
         ATNC=A["CWR"]["pairs_C"], ATNW=A["CWR"]["pairs_W"], ATNR=A["CWR"]["gadgets_R"], ATND=A["CWR"]["drops_R"],
         ATSC=ex[1]-A["C"]["twoq"], ATSW=A["C"]["twoq"]-A["CW"]["twoq"], ATSR=A["CW"]["twoq"]-A["CWR"]["twoq"])
K["ATPC"] = f"{100*K['ATSC']/(ex[1]-ours[1]):.0f}"; K["ATPW"] = f"{100*K['ATSW']/(ex[1]-ours[1]):.0f}"; K["ATPR"] = f"{100*K['ATSR']/(ex[1]-ours[1]):.0f}"
# --- semantics
S = json.load(open(os.path.join(TAB, "semantics.json"))); I = json.load(open(os.path.join(TAB, "interference.json")))
assert S["sub"]["twoq"] == ours[1]
K.update(GR2Q=S["greedy"]["twoq"], GRBADS=S["greedy"]["bad_sub"], GRBADP=S["greedy"]["bad_prog"],
         PR2Q=S["prog"]["twoq"], PRBADS=S["prog"]["bad_sub"], PRBADP=S["prog"]["bad_prog"],
         NINTERF=len(I), GRINTP=sum(not I[c]["greedy"]["prog"] for c in I))
# --- szum
n = rd("log_noisy_sim.txt")
for name, key in (("controlled_adder_2b", "CA"), ("grover_oracle_mcx4", "GO"), ("nested_compute_uncompute", "NC")):
    x = re.search(rf"^{name}\s+([\d.]+)\s+([\d.]+)", n, re.M); K[f"F{key}EX"] = f"{float(x.group(1)):.2f}"; K[f"F{key}OUR"] = f"{float(x.group(2)):.2f}"
# --- urzadzenia
s = rd("log_sensitivity_sweep.txt"); dev = s.split("=== PER-DEVICE OPERATING POINTS ===")[1]
reds = [float(x) for x in re.findall(r"AGGREGATE \(subset\)\s+[\d.]+\s+[\d.]+\s+([\d.]+)%", dev)]
assert len(reds) == 5, reds
blk = dev.split("---")[2]
pairs = re.findall(r"^\S+\s+(\d+)\s+(\d+)\s+[\d.]+", blk, re.M)
K.update(DEVMIN=f"{min(reds):.1f}", DEVMAX=f"{max(reds):.1f}", DEVB=sum(int(a) for a, _ in pairs), DEVA=sum(int(c) for _, c in pairs))
assert "SELECTION IS INVARIANT" in s
# --- resetujace (opcjonalnie, jesli gotowe)
if os.path.exists(os.path.join(RUN, "done_sync_benchmark")):
    r = rd("log_sync_benchmark.txt")
    R = re.findall(r"^(resetting_n\d+_L\d+)\s+q=\s*(\d+)\s+ccx=\s*(\d+)\s+2q\s+(\d+)->\s*(\d+)\s+\(\s*([\d.]+)%\).*verified=(\w+)", r, re.M)
    assert R and all(v == "True" for *_, v in R)
    K["RESET"] = [dict(name=a, n=int(b_), ccx=int(c), before=int(d), after=int(e), red=f) for a, b_, c, d, e, f, _ in R]
# --- large suite of 12-24 qubits
ck = os.path.join(RUN, "experiments", "scale_rows.jsonl")
if os.path.exists(os.path.join(RUN, "done_scale_eval_ckpt")):
    import statistics as st
    SR = [json.loads(l) for l in open(ck)]
    assert len(SR) == 20 and all(r["cert_equiv"] for r in SR), "large suite: certificate missing"
    b_ = sum(r["twoq_exact"] for r in SR); a_ = sum(r["twoq_ours"] for r in SR)
    ib = sum(r["infid_exact"] for r in SR); ia = sum(r["infid_ours"] for r in SR); qb = sum(r["qiskit_2q"] for r in SR)
    K.update(LN=len(SR), LEX2Q=b_, LOUR2Q=a_, L2Q=pct(b_, a_), LINF=pct(ib, ia), LVSQ2Q=pct(qb, a_),
             LNW=sum(r.get("window_pairs", 0) for r in SR), LNR=sum(r.get("r_gadgets", 0) for r in SR),
             LNC=sum(r["sites_applied"] for r in SR))
    def fam(f):
        v = [r["red_2q"] for r in SR if r["family"] == f]; return f"{min(v):.1f}", f"{max(v):.1f}", len(v)
    for f, k in (("grover/cu", "GRO"), ("grover/mixed", "MIX"), ("adder/cla", "CLA"), ("adder/cu", "RIP"),
                 ("modular/cu", "MOD"), ("multiplier/live", "MUL"), ("adder/live", "LIV")):
        K["F"+k+"MIN"], K["F"+k+"MAX"], K["F"+k+"N"] = fam(f)
    q = [r["cert_time"] for r in SR if r["cert_method"].startswith("qcec")]
    K.update(LQMAX=f"{max(q):.0f}", LQMED=f"{st.median(q):.2f}", LNQ=len(q),
             LQSLOW=sum(t > 1 for t in q), LQFASTMAX=f"{max(t for t in q if t <= 1):.2f}")
    d = [r for r in SR if r["cert_method"] == "dense_subspace"]; K["LND"] = len(d)
    K["LDT"] = f"{max(r['cert_time'] for r in d):.0f}" if d else "0"
    t = open(os.path.join(RUN, "experiments", "scale_results.md")).read()
    cr = re.findall(r"^\|\s*(\d+)\s*\|\s*([\d.]+|—)\s*\|\s*\S+\s*\|\s*([\d.]+)\s*\|", t, re.M)
    cr = {int(n_): (float(e) if e != "—" else None, float(qq)) for n_, e, qq in cr}
    K.update(XE10=f"{cr[10][0]:.1f}", XE12=f"{cr[12][0]:.0f}", XQ12=f"{cr[12][1]:.3f}",
             XRATIO=f"{cr[12][0]/cr[12][1]:.0f}", XRATIOR=("2\\times10^{3}" if 1500 <= cr[12][0]/cr[12][1] < 2500 else f"{cr[12][0]/cr[12][1]:.0f}"), XQMAX=f"{max(v[1] for v in cr.values()):.3f}", XNMAX=max(cr))
# --- large suite in program mode
if os.path.exists(os.path.join(RUN, "done_scale_program")):
    PR = [json.loads(l) for l in open(os.path.join(RUN, "experiments", "scale_program_rows.jsonl"))]
    assert len(PR) == 20, len(PR)
    eb = sum(r["twoq_exact"] for r in PR); ib = sum(r["infid_exact"] for r in PR)
    K.update(PNU=sum(r["U_admitted"] for r in PR),
             P2Q=pct(eb, sum(r["twoq_prog"] for r in PR)), PINF=pct(ib, sum(r["infid_prog"] for r in PR)),
             G2Q=pct(eb, sum(r["twoq_greedy"] for r in PR)),
             PNBAD=sum(r["sub_ok_prog"] is False for r in PR), GNBAD=sum(r["sub_ok_greedy"] is False for r in PR),
             PNUND=sum(r["sub_ok_prog"] is None for r in PR), GNUND=sum(r["sub_ok_greedy"] is None for r in PR),
             PN12=sum(r["n"] <= 12 for r in PR), PN12OK=sum(r["prog_ok_prog"] is True for r in PR),
             PNGAIN=sum(r["sub_ok_prog"] is True and r["twoq_prog"] < r["twoq_sub"] for r in PR),
             PMULTMIN=round(max(r["sub_t_prog"] or 0 for r in PR)/60), GSAMEBAD=sum(r["sub_ok_greedy"] is False and r["twoq_greedy"] == r["twoq_sub"] for r in PR))
    assert K["PN12OK"] == K["PN12"], "program: observational certificate did not pass"
# --- test lustrzany na IBM
LH = os.path.join(TAB, "mirror_hw.json")
if os.path.exists(LH):
    import math
    D = json.load(open(LH)); R = {(r["name"], r["tag"]): r for r in D["rows"]}
    def sig(a, b): return f"{abs(a['p0_hw']-b['p0_hw'])/math.hypot(a['sigma'], b['sigma']):.0f}"
    h = ("half_uncomputed",); K.update(HWBACK=D["job"]["backend"].replace("_", "\\_"), HWSHOTS=D["job"]["shots"], HWQPU=D["qpu_s"],
        HWHUS=f"{R[('half_uncomputed','sub')]['p0_hw']:.2f}", HWHUG=f"{R[('half_uncomputed','greedy')]['p0_hw']:.2f}",
        HWHUSIG=sig(R[('half_uncomputed','sub')], R[('half_uncomputed','greedy')]),
        HWMINSIG=min(int(sig(R[(n,'sub')], R[(n,'greedy')])) for n in ("half_uncomputed","live_and_chain","single_live_toffoli")),
        HWCAREF=f"{R[('controlled_adder_2b','ref')]['p0_hw']:.2f}", HWCAS=f"{R[('controlled_adder_2b','sub')]['p0_hw']:.2f}",
        HWCAG=f"{R[('controlled_adder_2b','greedy')]['p0_hw']:.2f}", HWCAGI=f"{R[('controlled_adder_2b','greedy')]['p0_ideal']:.2f}")
    # drugi test lustrzany, ibm_aachen (experiments/mirror_test_ibm_aachen.py)
    A = json.load(open(os.path.join(TAB, "mirror_aachen.json"))); RA = {(r["name"], r["tag"]): r for r in A["rows"]}
    def zz(a, b): return (a['p0_hw'] - b['p0_hw']) / math.hypot(a['sigma'], b['sigma'])
    GW = [n for n in dict.fromkeys(k[0] for k in RA) if RA[(n, 'greedy')]['p0_ideal'] < 0.999]
    SB = [n for n in GW if RA[(n, 'sub')]['twoq_V'] < RA[(n, 'ref')]['twoq_V']]
    K.update(AABACK=A["job"]["backend"].replace("_", "\\_"), AASHOTS=A["job"]["shots"], AAQPU=A["usage_after"], AANGW=len(GW), AANSB=len(SB),
        AAMINSIG=f"{min(zz(RA[(n,'sub')], RA[(n,'greedy')]) for n in GW):.0f}",
        AAREFGRMIN=f"{min(zz(RA[(n,'ref')], RA[(n,'greedy')]) for n in GW):.1f}",
        AANBETTER=sum(1 for n in SB if zz(RA[(n,'sub')], RA[(n,'ref')]) > 3),
        AABETMIN=f"{min(zz(RA[(n,'sub')], RA[(n,'ref')]) for n in SB if zz(RA[(n,'sub')], RA[(n,'ref')]) > 3):.1f}",
        AABETMAX=f"{max(zz(RA[(n,'sub')], RA[(n,'ref')]) for n in SB):.0f}",
        AA3CZ=f"{-zz(RA[('half_uncomputed_3c','sub')], RA[('half_uncomputed_3c','ref')]):.1f}",
        AANESTZ=f"{abs(zz(RA[('nested_compute_uncompute','sub')], RA[('nested_compute_uncompute','greedy')])):.1f}",
        AACAREF=f"{RA[('controlled_adder_2b','ref')]['p0_hw']:.2f}", AACAS=f"{RA[('controlled_adder_2b','sub')]['p0_hw']:.2f}", AACAG=f"{RA[('controlled_adder_2b','greedy')]['p0_hw']:.2f}")
# --- RevLib
RL = os.path.join(TAB, "revlib.json")
if os.path.exists(RL):
    D = json.load(open(RL)); A = D["agg"]; n = D["n"]
    assert A["ours"]["cert"] == n and A["ours_qiskit"]["cert"] == n, "RevLib: not all outputs of the pass are certified"
    skip = D["failed"] + D.get("not_run", [])
    K.update(RLN=n, RLNFAIL=len(D["failed"]), RLNNOT=len(D.get("not_run", [])), RLNSKIP=len(skip), RLSKIPMIN=min(x["gates"] for x in skip), RLSKIPMAX=max(x["gates"] for x in skip), RLMAXL=D["max_lines"], RLMAXG=D["max_gates"],
             RL2Q=pct(A["exact"]["twoq"], A["ours"]["twoq"]), RLINF=pct(A["exact"]["infid"], A["ours"]["infid"]),
             RLVSQ=pct(A["qiskit_l3"]["twoq"], A["ours"]["twoq"]), RLVST=pct(A["tket"]["twoq"], A["ours"]["twoq"]),
             RLOQVSQ=pct(A["qiskit_l3"]["twoq"], A["ours_qiskit"]["twoq"]), RLOQVST=pct(A["tket"]["twoq"], A["ours_qiskit"]["twoq"]),
             RLOQBETTER=D["n_oq_better_q"], RLOQWORSE=D["n_oq_worse_q"], RLGCERT=A["greedy"]["cert"],
             RLC=D["cond"]["C"], RLW=D["cond"]["W"], RLR=D["cond"]["R"], RLDROP=D["cond"]["drop"],
             RLFB=D["n_fell_back"], RLIMP=D["n_improved"], RLTKCERT=A["tket"]["cert"], RLQCERT=A["qiskit_l3"]["cert"])
# --- budget: hardware prediction (item 3) and calibration-aware orientation (item 1)
PZ = os.path.join(TAB, "prediction.json")
if os.path.exists(PZ):
    Z = json.load(open(PZ)); ca = next(r for r in Z["rows"] if r["name"] == "controlled_adder_2b" and r["tag"] == "ref")
    K.update(PRSPF=f"{Z['flat']['spearman']:.2f}", PRSPC=f"{Z['cal']['spearman']:.2f}", PRPF=f"{Z['flat']['pearson']:.2f}",
             PRPAIROK=Z["flat"]["pary_zgodne"], PRPAIRS=Z["flat"]["pary"], PRMAEF=f"{Z['flat']['mae']:.2f}",
             PRCAPRED=f"{ca['p0_pred_cal']:.2f}", PRCAHW=f"{ca['p0_hw']:.2f}", PRCAIDLE=f"{ca['p0_pred_cal_idle']:.2f}",
             PRSPI=f"{Z['cal_idle']['spearman']:.2f}", PRPI=f"{Z['cal_idle']['pearson']:.2f}", PRPC=f"{Z['cal']['pearson']:.2f}",
             PRMAEC=f"{Z['cal']['mae']:.2f}", PRMAEI=f"{Z['cal_idle']['mae']:.2f}", PRPAIROKI=Z["cal_idle"]["pary_zgodne"],
             PRPZ=f"{Z['cal_idle_zz']['pearson']:.2f}", PRMAEZ=f"{Z['cal_idle_zz']['mae']:.2f}",
             PRPL=f"{Z['full_loo']['pearson']:.2f}", PRSPL=f"{Z['full_loo']['spearman']:.2f}", PRMAEL=f"{Z['full_loo']['mae']:.2f}",
             PRPAIROKL=Z["full_loo"]["pary_zgodne"], PRLAM=f"{Z['lam_all']*1e3:.1f}", PRLAMLO=f"{Z['lam_loo_range'][0]*1e3:.1f}",
             PRLAMHI=f"{Z['lam_loo_range'][1]*1e3:.1f}", PRCAFULL=f"{ca['p0_pred_full_loo']:.2f}", PRCZMED=f"{Z['cz_median_used']*1e3:.1f}",
             PRSIGNOK=sum(1 for c in Z["ours_vs_ref"] if abs(c["pred_cal"]) > 1e-9 and c["pred_cal"] * c["hw"] > 0),
             PRSIGNN=sum(1 for c in Z["ours_vs_ref"] if abs(c["pred_cal"]) > 1e-9))
IB = os.path.join(TAB, "idle_budget.json")
if os.path.exists(IB):
    I = json.load(open(IB)); q2, rl = I["suites"]["Q2"], I["suites"]["RevLib"]
    K.update(IDT1=f"{I['t1_median_s']*1e6:.0f}", IDT2=f"{I['t2_median_s']*1e6:.0f}",
             IDQ2=f"{q2['gates+idle']['red_pct']:.1f}", IDRL=f"{rl['gates+idle']['red_pct']:.1f}",
             IDQ2T=f"{q2['idle_red_pct']:.0f}", IDRLT=f"{rl['idle_red_pct']:.0f}",
             IDZZ=f"{I['zz_median_hz']/1e3:.1f}", FULLQ2=f"{q2['full']['red_pct']:.1f}", FULLRL=f"{rl['full']['red_pct']:.1f}")
OR = os.path.join(TAB, "orientation.jsonl")
if os.path.exists(OR):
    import numpy as np
    from scipy.stats import wilcoxon
    O = [json.loads(l) for l in open(OR)]; A = [r for r in O if r.get("oriented", 0) > 0]
    d = np.array([r["infid_default"] for r in A]); o = np.array([r["infid_oriented"] for r in A])
    by = {}
    for r in A: by.setdefault(r["circuit"], {})[r["device"]] = r["flags"]
    tot = dif = 0
    for c, m in by.items():
        devs = [k for k in m if "@" not in k]
        for g in set().union(*[set(m[k]) for k in devs]) if devs else []:
            tot += 1; dif += len({m[k].get(g) for k in devs}) > 1
    t2 = d2 = 0
    for c, m in by.items():
        if "ibm_marrakesh" in m and "ibm_marrakesh@2026-10-01" in m:
            for g in set(m["ibm_marrakesh"]) | set(m["ibm_marrakesh@2026-10-01"]):
                t2 += 1; d2 += m["ibm_marrakesh"].get(g) != m["ibm_marrakesh@2026-10-01"].get(g)
    K.update(ORN=len(A), ORCIRC=len({r["circuit"] for r in A}), ORGAIN=f"{100*(d.sum()-o.sum())/d.sum():.2f}", ORWIN=int((o < d).sum()), ORLOSS=int((o > d).sum()),
             ORP=f"{wilcoxon(d, o).pvalue:.3f}", ORFLIP=dif, ORFLIPN=tot, ORFLIPPCT=f"{100*dif/tot:.0f}", ORDAY=d2, ORDAYN=t2, ORDAYPCT=f"{100*d2/t2:.0f}",
             ORCERTFAIL=sum(1 for r in O if r.get("certificate") is False), ORREJR=sum(r.get("rejected_R", 0) for r in O))
# --- diagnostyka sumatora (test IBM 02.10)
DG = os.path.join(TAB, "diag_ibm.json")
if os.path.exists(DG):
    G = json.load(open(DG)); q2 = G["C"]["2"]; q6 = G["C"]["6"]; A_ = {r["tag"]: r["p0"] for r in G["A"]}; B_ = {r["tag"]: r["p0"] for r in G["B"]}
    SZ = json.load(open(os.path.join(TAB, "sym_measured.json")))["zmierzone T1/T2 (q2,q16,q3,q6)"]
    K.update(DGQ2T1=f"{q2['T1_fit']:.0f}", DGQ2T2=f"{q2['T2echo_fit']:.0f}", DGQ2T1K=f"{q2['T1_kal']:.0f}", DGQ2T2K=f"{q2['T2_kal']:.0f}",
             DGQ6T1=f"{q6['T1_fit']:.0f}", DGQ6T1K=f"{q6['T1_kal']:.0f}",
             DGAREF=f"{A_['ref']:.3f}", DGASUB=f"{A_['sub']:.3f}", DGAGR=f"{A_['greedy']:.3f}",
             DGBREF=f"{B_['ref']:.3f}", DGBSUB=f"{B_['sub']:.3f}",
             DGSREF=f"{SZ['ref']['p0']:.3f}", DGSSUB=f"{SZ['sub']['p0']:.3f}", DGSGR=f"{SZ['greedy']['p0']:.3f}")
# --- reset table (sync_scale: backend and time)
if os.path.exists(os.path.join(RUN, "done_sync_scale")):
    r = rd("log_sync_scale.txt")
    R = re.findall(r"^resetting_n(\d+)_L(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+[\d.]+\s+(\w+)\s+(\w+)\s+([\d.]+)", r, re.M)
    assert len(R) == 4 and all(x[7] == "True" for x in R), R
    for st, le, n_, ccx, b_, a_, red, ok, meth, tt in R:
        k = f"RS{st}L{le}"; K.update({k+"N": n_, k+"B": b_, k+"A": a_, k+"R": red, k+"M": "exh." if meth == "exhaustive" else "QCEC", k+"T": f"{float(tt):.2f}"})
    reds = [float(x[6]) for x in R]; K["RESMIN"] = f"{min(reds):.1f}"; K["RESMAX"] = f"{max(reds):.1f}"
json.dump(K, open(os.path.join(HERE, "paper_numbers.json"), "w"), indent=1)
print(json.dumps(K, indent=0)[:3000])

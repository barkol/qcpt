"""Best connected regions from the probe of aachen (aachen_probe.py): couplers above PROBE_MAX per CZ and qubits with readout flip above QFLIP are dropped; the cost of a coupler is max(calibration, probe / INFL)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aachen_calibration_next as AC
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures"); OUT = os.path.join(HERE, os.environ.get("SEL_TAG", "aachen_next2")); os.makedirs(OUT, exist_ok=True)
PROBE_MAX, QFLIP, INFL = 0.012, 0.2, 3.5
snap = json.load(open(HERE + "/aachen_next/calibration.json")); P = json.load(open(HERE + "/" + os.environ.get("PROBE_TAG", "aachen_probe") + "/probe.json"))
Q = {int(i): d for i, d in snap["qubits"].items()}; cz = {tuple(map(int, k.split("-"))): v for k, v in snap["cz"].items()}
qf = {int(k): v for k, v in P["qubit"].items()}; cp = {tuple(map(int, k.split("-"))): v["err_per_cz"] for k, v in P["coupler"].items()}
bad = {i for i, d in Q.items() if d["t1"] <= 0 or d["t2"] <= 0 or d["ro"] >= 0.5} | {q for q, f in qf.items() if f > QFLIP} | {q for q, d in Q.items() if d["t1"] < 100 or d["t2"] < 100}
adj = {i: set() for i in Q if i not in bad}; ecs = {}
bad_pairs = set()
for (a, b), v in cz.items():
    pe = cp.get((a, b), cp.get((b, a)))
    if pe is None or pe > PROBE_MAX or v >= 0.5: bad_pairs.add((a, b)); continue
    if a in bad or b in bad: continue
    adj[a].add(b); adj[b].add(a); ecs[(a, b)] = ecs[(b, a)] = max(v, pe / INFL)
ecost = lambda a, b: ecs.get((a, b), 1.0)
reg = AC.regions(dict(q=Q), adj, (6, 8, 10, 12, 14), ecost)
# no region may contain both ends of a dropped coupler (the router could use it): drop one end and repeat
for _ in range(6):
    clash = [(a, b) for k, r in reg.items() for (a, b) in bad_pairs if a in r["qubits"] and b in r["qubits"] and (k in (6, 8, 10))]
    if not clash: break
    for a, b in clash:
        bad.add(max(a, b, key=lambda q: Q[q]["ro"]));
    adj = {i: {j for j in s if j not in bad} for i, s in adj.items() if i not in bad}
    reg = AC.regions(dict(q=Q), adj, (6, 8, 10, 12, 14), ecost)
for k, r in reg.items(): print(f"region {k:>2}: cost {r['cost']:.3f} | {r['qubits']}")
dead = sorted(bad_pairs)
json.dump(dict(calibration_taken=snap["taken"], span_us=20.0, regions={str(k): r for k, r in reg.items()}, bad_qubits=sorted(bad), independent=[], couplers=[], dead=[list(d) for d in dead], probe_max=PROBE_MAX, qflip=QFLIP), open(OUT + "/selection.json", "w"))
print("dropped couplers:", len(dead), "| dropped qubits:", sorted(bad)[:40])

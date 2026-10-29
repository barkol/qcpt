#!/usr/bin/env python3
"""Data layer of the figure of the failed couplers of ibm_aachen: heavy-hex coordinates reconstructed from the coupling graph, probe and calibration values
(inputs: data/rerun/experiments/aachen_probe/probe.json, calibration.json; output: figures/data/probe_map.npz)."""
import json, os
import numpy as np
H = os.path.dirname(os.path.abspath(__file__)); IN = os.path.join(H, "..", "data", "rerun", "experiments", "aachen_probe"); OUT = os.path.join(H, "data"); os.makedirs(OUT, exist_ok=True)
P = json.load(open(os.path.join(IN, "probe.json"))); C = json.load(open(os.path.join(IN, "calibration.json")))
n = len(C["qubits"])
cal = {tuple(map(int, k.split("-"))): v for k, v in C["cz"].items()}; probe = {tuple(map(int, k.split("-"))): v["err_per_cz"] for k, v in P["coupler"].items()}
edges = sorted(e for e in cal if cal[e] < 0.5 or e in probe)
# rows = maximal runs of consecutive indices joined by edges (a, a+1); the other qubits connect two rows
adj = {i: set() for i in range(n)}
for a, b in sorted(set(edges) | set(cal)): adj[a].add(b); adj[b].add(a)                          # dead couplers too: they only matter for the layout
row_of = {}; rows = []
for q in range(n):
    if q in row_of: continue
    run = [q]
    while run[-1] + 1 in adj[run[-1]]: run.append(run[-1] + 1)
    if len(run) >= 4:
        for x in run: row_of[x] = len(rows)
        rows.append(run)
conn = [q for q in range(n) if q not in row_of]
# x position of every row: the first row starts at 0, the next ones are shifted so that every connector is vertical
xs = {}; ys = {}
for x, q in enumerate(rows[0]): xs[q] = float(x); ys[q] = 0.0
changed = True
while changed:
    changed = False
    for c in conn:
        up = [v for v in adj[c] if v in row_of]
        if len(up) != 2: continue
        u, v = sorted(up, key=lambda t: row_of[t])
        ru, rv = row_of[u], row_of[v]
        if u in xs and v not in xs:
            shift = xs[u] - rows[rv].index(v)
            for k, q in enumerate(rows[rv]): xs[q] = shift + k; ys[q] = 2.0 * rv
            changed = True
        xs.setdefault(c, xs.get(u, xs.get(v, 0.0))); ys.setdefault(c, (ys.get(u, 2.0 * ru) + 2.0 * rv) / 2.0)
for q in range(n):
    if q not in xs: xs[q] = 0.0; ys[q] = 0.0
X = np.array([xs[q] for q in range(n)]); Y = -np.array([ys[q] for q in range(n)])
ea = np.array([e[0] for e in edges]); eb = np.array([e[1] for e in edges])
pe = np.array([probe.get(e, np.nan) for e in edges]); ce = np.array([cal.get(e, np.nan) for e in edges])
qflip = np.array([P["qubit"].get(str(q), np.nan) for q in range(n)])
regions = dict(used_6=[111, 112, 113, 119, 132, 133], used_8=[110, 111, 112, 113, 114, 119, 132, 133], used_10=[54, 55, 59, 73, 74, 75, 79, 93, 94, 95], used_12=[52, 53, 54, 55, 59, 73, 74, 75, 79, 93, 94, 95])   # regions chosen from the probe (selection.json of the measurements)
np.savez(os.path.join(OUT, "probe_map.npz"), x=X, y=Y, ea=ea, eb=eb, probe=pe, cal=ce, qflip=qflip, **{k: np.array(v) for k, v in regions.items()},
         median_probe=float(np.nanmedian(pe)), median_cal=float(np.nanmedian(ce)), n_rows=len(rows), taken=C["taken"])
print("saved", os.path.join(OUT, "probe_map.npz"), "| rows", len(rows), [len(r) for r in rows], "| connectors", len(conn), "| edges", len(edges))

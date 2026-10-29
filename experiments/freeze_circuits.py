"""Freeze the final QCPT circuits for a hardware run: for every circuit the cheapest certified gate list among the race result and the
windowed resynthesis (both checked against the exact circuit when they were made). Writes experiments/fixtures/nighthawk/frozen_circuits.json.
Usage: freeze_circuits.py RACE_DIR WINDOWS_DIR [OUT]   (RACE_DIR holds circuits.txt and qasm/<name>.gates.json; WINDOWS_DIR holds <name>.json)"""
import json, os, sys
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path[:0] = [R]
from toffoli_optimizer.core import resynth as RS

def cost(gates): return sum(RS.gate_cost(g[0]) for g in gates)

def main(race, windows, out):
    frozen = {}
    for ln in open(os.path.join(race, 'circuits.txt')):
        name = ln.split()[0]; cand = []
        gj = os.path.join(race, 'qasm', name + '.gates.json')
        if os.path.exists(gj): cand.append(('race', json.load(open(gj))))
        wj = os.path.join(windows, name + '.json')
        if os.path.exists(wj):
            w = json.load(open(wj))
            if 'gates' in w and w.get('dev', 1.0) < 1e-9: cand.append(('windows', w['gates']))
        if not cand: continue
        src, gates = min(cand, key=lambda c: cost(c[1]))
        frozen[name] = dict(source=src, cost=cost(gates), gates=gates)
    os.makedirs(os.path.dirname(out), exist_ok=True); json.dump(frozen, open(out, 'w'), indent=1)
    print(len(frozen), 'circuits, total cost', sum(v['cost'] for v in frozen.values()), '->', out)

if __name__ == '__main__':
    default = os.path.join(R, 'experiments', 'fixtures', 'nighthawk', 'frozen_circuits.json')
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else default)

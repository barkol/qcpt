"""Numbers of Proposition prop:cost, Sec. method-cost and Appendix app:cost from the stored data.
cost.jsonl: numerical search for the minimal CNOT count of CCX on span(L), L <- {0,1}^3 (cost.py, topup.py).
prediction/results.jsonl: per-Toffoli static prediction vs pass output (prediction/prediction.py)."""
import json, os, collections
H = os.path.dirname(os.path.abspath(__file__)); OK = 1e-8
lab = lambda x: f"{x & 1}{(x >> 1) & 1}{(x >> 2) & 1}"
def cstar(s):
    s = set(s); P = {x[:2] for x in s}; Q = {'100', '101', '010', '011'}
    if '11' not in P or P <= {'11'}: return 0
    if P == {'01', '10', '11'}: return 2
    if '01' not in P or '10' not in P: return 1
    if not Q <= s: return 3
    return 4 if len(s) < 8 else 6
by = collections.defaultdict(dict)
for l in open(os.path.join(H, 'cost.jsonl')):
    r = json.loads(l); by[tuple(r['S'])][r['k']] = r['best_loss']
def swap(x): return (x & 4) | ((x & 1) << 1) | ((x >> 1) & 1)
reps = {min(tuple(x for x in range(8) if m >> x & 1), tuple(sorted(swap(x) for x in range(8) if m >> x & 1))) for m in range(1, 256)}
agree = gap = 0; gaps = []
for S, d in by.items():
    c = cstar([lab(x) for x in S])
    if all(k >= c for k, L in d.items() if L < OK) and all(k < c for k, L in d.items() if L >= OK): agree += 1
    gaps += [L for k, L in d.items() if L >= OK]
print('supports up to exchange of controls:', len(reps))
print('supports searched numerically:', len(by), '| consistent with c*:', agree)
print('smallest loss below c* (gap):', round(min(gaps), 4))
suites = collections.Counter(); tof = 0; full6 = 0; extra = collections.Counter(); match = 0
for l in open(os.path.join(H, 'prediction', 'results.jsonl')):
    r = json.loads(l)
    if 'toffolis' not in r: continue
    suites[r['suite']] += 1; tof += len(r['toffolis'])
    for t in r['toffolis']:
        grp = t.get('pair_C') or t.get('window_W'); cs = cstar(t['S_bits'])
        if t['c_full'] == 6: full6 += 1
        extra[r['suite']] += (3 if grp else t['c_pass']) - (min(3, cs) if grp else cs)
print('circuits per suite:', dict(suites), 'total', sum(suites.values()), '| Toffolis:', tof)
print('Toffolis with L = {0,1}^3:', full6)
print('further two-qubit savings of the unused classes per suite:', dict(extra))
A = json.load(open(os.path.join(H, 'prediction', 'analysis.json')))
print('prediction vs reported outputs (analysis.json keys):', list(A)[:8])

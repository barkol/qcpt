"""Top-up: for S with tabela=6 for which k=5 or k=4 gives loss ~0, compute the next smaller k of the same parity
(k=3,1 or k=2,0) until the first failure. Result in cost.jsonl (same format). Run after cost.py has finished."""
import json, os, time, cost
cost.R = 8; cost.STEPS = 1200
rows = [json.loads(l) for l in open('cost.jsonl')]
have = {(tuple(r['S']), r['k']): r['best_loss'] for r in rows}
S6 = sorted({tuple(r['S']) for r in rows if r['tabela'] == 6})
PAS, NPAS = int(os.environ.get('PAS', '0')), int(os.environ.get('NPAS', '1'))
for n, S in enumerate(S6):
    if n % NPAS != PAS: continue
    for start in (5, 4):
        if have.get((S, start), 1) > 1e-8: continue
        k = start - 2
        while k >= 0:
            if (S, k) not in have:
                t0 = time.time(); L, arg = cost.best_loss(list(S), k)
                row = dict(S=list(S), labels=sorted(cost.lab(x) for x in S), tabela=6, k=k, best_loss=L, pattern=arg, s=round(time.time() - t0, 1), R=8, steps=1200)
                open('cost.jsonl', 'a').write(json.dumps(row) + '\n'); print(json.dumps(row), flush=True); have[(S, k)] = L
            if have[(S, k)] > 1e-8: break
            k -= 2
print('KONIEC')

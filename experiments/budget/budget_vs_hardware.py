"""Punkt 3 (+ bledy bezczynnosci, model (c): kalibracja + bezczynnosc z T1/T2 i szeregowania ALAP): czy budzet bledu przewiduje sprzet? Dla 15 obwodow testu lustrzanego (obwody ISA
faktycznie wykonane na ibm_marrakesh) liczymy przewidywana wiernosc F = prod(1-eps) dwoma modelami:
 (a) surogat z pracy (p2q=1e-2, p1q=1e-3, jednakowe), (b) kalibracja z chwili zadania (bledy bramek
 per krawedz/kubit + odczyt). Przewidywane P0 = P0_ideal*F + (1-F)/2^n (depolaryzacja do rozkladu
 jednostajnego). Wynik: korelacje z P0 zmierzonym i trafnosc kolejnosci w parach. -> przewidywanie.json"""
import json, pickle, math
import numpy as np
from qiskit import qpy
from scipy.stats import pearsonr, spearmanr
import os
W = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fixtures', 'mirror_test', 'wyniki_hw.json')))
_H = os.path.dirname(os.path.abspath(__file__)); os.chdir(_H)
circs = qpy.load(open('lustro_isa.qpy', 'rb'))
P = pickle.load(open('marrakesh_props_jobtime.pkl', 'rb'))
import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__)))
from bezczynnosc import durations_from_target, idle_times, idle_fidelity, t1t2_from_props, zz_map, zz_fidelity
_K = pickle.load(open('kal_ibm_marrakesh.pkl', 'rb')); DUR = durations_from_target(_K['target']); T1, T2 = t1t2_from_props(P)
ZZ = zz_map(P, _K['coupling'])
gerr, rerr = {}, {}
for g in P['gates']:
    e = [p['value'] for p in g['parameters'] if p['name'] == 'gate_error']
    if e: gerr[(g['gate'], tuple(g['qubits']))] = e[0]
for q, ps in enumerate(P['qubits']):
    for p in ps:
        if p['name'] == 'readout_error': rerr[q] = p['value']
def F_model(c, kind):
    f = 1.0; n_meas = 0
    for ins in c.data:
        name = ins.operation.name; qs = tuple(c.find_bit(q).index for q in ins.qubits)
        if name in ('barrier', 'delay', 'rz'): continue
        if name == 'measure':
            n_meas += 1
            if kind == 'cal': f *= 1 - rerr.get(qs[0], 0.0)
            continue
        if kind == 'flat':
            f *= 1 - (1e-2 if len(qs) == 2 else 1e-3)
        else:
            e = gerr.get((name, qs)) or gerr.get((name, qs[::-1]))
            if e is None: raise SystemExit(f'brak kalibracji {name} {qs}')
            f *= 1 - min(e, 1.0)
    return f, n_meas
rows = []
for k, r in enumerate(W['rows']):
    c = circs[k]; n = c.num_clbits
    out = {'name': r['name'], 'tag': r['tag'], 'p0_ideal': r['p0_ideal'], 'p0_hw': r['p0_hw']}
    out['n_cz'] = sum(1 for i in c.data if i.operation.name == 'cz')
    for kind in ('flat', 'cal', 'cal_idle', 'cal_idle_zz'):
        F, nm = F_model(c, 'flat' if kind == 'flat' else 'cal')
        if kind in ('cal_idle', 'cal_idle_zz'):
            tau = idle_times(c, DUR); F *= idle_fidelity(tau, T1, T2); out['tau_max_us'] = max(tau.values()) * 1e6
        if kind == 'cal_idle_zz':
            fz, npairs = zz_fidelity(c, DUR, ZZ); F *= fz; out['zz_pairs'] = npairs; out['f_zz'] = fz
        out[f'F_{kind}'] = F; out['n_meas'] = nm
        out[f'p0_pred_{kind}'] = r['p0_ideal'] * F + (1 - F) / 2 ** nm
    rows.append(out)
# leakage / nadmiarowy blad na CZ: jeden parametr lam, F *= (1-lam)^n_cz; dopasowanie LOO
import numpy as _np
LAMS = _np.linspace(0, 0.02, 2001)
def _pred(r, lam): F = r['F_cal_idle_zz'] * (1 - lam) ** r['n_cz']; return r['p0_ideal'] * F + (1 - F) / 2 ** r['n_meas']
def _fit(rr): return float(LAMS[_np.argmin([sum((_pred(r, l) - r['p0_hw']) ** 2 for r in rr) for l in LAMS])])
LAM_ALL = _fit(rows)
for i, r in enumerate(rows):
    lam_i = _fit(rows[:i] + rows[i + 1:]); r['lam_loo'] = lam_i
    r['p0_pred_full_loo'] = _pred(r, lam_i); r['p0_pred_full'] = _pred(r, LAM_ALL)
_used = {}
for c in circs:
    for ins in c.data:
        if ins.operation.name == 'cz':
            qs = tuple(c.find_bit(q).index for q in ins.qubits); _used[qs] = gerr.get(('cz', qs)) or gerr.get(('cz', qs[::-1]))
res = {'cz_median_used': float(_np.median(list(_used.values()))), 'rows': rows, 'lam_all': LAM_ALL, 'lam_loo_range': [min(r['lam_loo'] for r in rows), max(r['lam_loo'] for r in rows)]}
for kind in ('flat', 'cal', 'cal_idle', 'cal_idle_zz', 'full_loo'):
    x = [r[f'p0_pred_{kind}'] for r in rows]; y = [r['p0_hw'] for r in rows]
    res[kind] = {'pearson': pearsonr(x, y)[0], 'spearman': spearmanr(x, y)[0],
                 'mae': float(np.mean(np.abs(np.array(x) - np.array(y))))}
    # trafnosc kolejnosci: wszystkie pary obwodow, czy znak roznicy przewidywany = zmierzony
    ok = tot = 0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            d1 = x[i] - x[j]; d2 = y[i] - y[j]
            if abs(d2) > 2 * 0.008: tot += 1; ok += (d1 * d2 > 0)
    res[kind]['pary_zgodne'] = ok; res[kind]['pary'] = tot
# czy budzet poprawnie porzadkuje 'ours' vs 'ref' w kazdym obwodzie (te same idealne P0=1)
cmp = []
for nm in sorted({r['name'] for r in rows}):
    a = next(r for r in rows if r['name'] == nm and r['tag'] == 'ref'); b = next(r for r in rows if r['name'] == nm and r['tag'] == 'sub')
    cmp.append({'name': nm, 'pred_cal': b['p0_pred_cal'] - a['p0_pred_cal'], 'pred_cal_idle': b['p0_pred_cal_idle'] - a['p0_pred_cal_idle'], 'pred_full_loo': b['p0_pred_full_loo'] - a['p0_pred_full_loo'], 'hw': b['p0_hw'] - a['p0_hw']})
res['ours_vs_ref'] = cmp
json.dump(res, open('przewidywanie.json', 'w'), indent=1)
print('lam_all', LAM_ALL, 'loo', res['lam_loo_range'])
for kind in ('flat', 'cal', 'cal_idle', 'cal_idle_zz', 'full_loo'): print(kind, {k: round(v, 3) if isinstance(v, float) else v for k, v in res[kind].items()})
for r in rows: print(f"{r['name'][:22]:22s} {r['tag']:6s} hw={r['p0_hw']:.3f} cal={r['p0_pred_cal']:.3f} cal+idle={r['p0_pred_cal_idle']:.3f} +zz={r['p0_pred_cal_idle_zz']:.3f} +leak(LOO)={r['p0_pred_full_loo']:.3f} flat={r['p0_pred_flat']:.3f}")
for c in cmp: print(c)

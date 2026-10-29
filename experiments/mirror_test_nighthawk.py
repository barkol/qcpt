"""Mirror test of subroutine equivalence on IBM Nighthawk (square lattice), the counterpart of mirror_test_aachen.py.
M = H_d V^dag E H_d, return probability to |0...0>, E = exact decomposition (the reference), V = the circuit under test. Variants per circuit (default qiskit and qcpt, on all 33 circuits of the frozen set; NAMES and TAGS override):
qiskit (the original circuit with its Toffolis, decomposed and optimized by Qiskit itself at optimization level 3), qcpt (the frozen QCPT circuit,
optimization level 2) and qct_l3 (QCPT followed by Qiskit level 3); TAGS=ref,... adds the exact decomposition. One layout per circuit, taken from the
first variant. QCPT circuits come from fixtures/nighthawk/frozen_circuits.json (freeze_circuits.py), so the hardware measures exactly the circuits of the paper.
Stages: plan (offline: transpile to a Nighthawk-like 10x12 square lattice with cz/sx/rz/x, no account needed) | sprawdz (transpile to the real
backend, print counts and the QPU budget) | wyslij (one job; job id stored at once in job.json, refuses if it exists) | odbierz (fetch -> wyniki.json).
Account and backend via IBM_ACCOUNT / IBM_BACKEND (the backend name is set when access is granted); SHOTS, NAMES (comma list) override the defaults.
Assumption to confirm on the real backend: 120 qubits, 218 couplers (10x12 grid), native gates cz, sx, rz, x."""
import json, os, sys, time
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path[:0] = [R, R + '/experiments', R + '/experiments/revlib']
import numpy as np
import qcpt_topo as QT, qcpt_post as QP
import benchmarks as B, large_benchmarks as LB
from _clean import clean_ancillas
from revlib import parse as rl_parse
from mirror_test_ibm import mirror, p0_from_counts
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from toffoli_optimizer.core import resynth as RS
from toffoli_optimizer.core.decomposition_selector import ErrorBudgetSelector
ACC, BK, SHOTS = os.environ.get('IBM_ACCOUNT', ''), os.environ.get('IBM_BACKEND', 'ibm_nighthawk'), int(os.environ.get('SHOTS', '8000'))
NAMES = os.environ.get('NAMES', '').split(',') if os.environ.get('NAMES') else None            # default: all circuits of the frozen set (33)
TAG = os.environ.get('RUN_TAG', 'nighthawk')      # one fixtures directory per run (e.g. aachen_rerun)
HERE = os.path.join(R, 'experiments', 'fixtures', TAG); os.makedirs(HERE, exist_ok=True); J = os.path.join(HERE, 'job.json')
FROZEN = os.environ.get('FROZEN') or os.path.join(R, 'experiments', 'fixtures', 'nighthawk', 'frozen_circuits.json'); TAGS = tuple(os.environ.get('TAGS', 'qiskit,qct,qct_topo,qct_post').split(',')); SEC_PER_SHOT = float(os.environ.get('SEC_PER_SHOT', '293e-6'))

def load(name):
    if name.startswith('rl_'): qc, pins, _ = rl_parse(f'{R}/experiments/revlib/real/{name[3:]}.real'); return qc, pins
    suite = {c.name: c for c in B.benchmark_suite()}
    if name in suite: qc = suite[name]
    elif name == 'half_uncomputed_min': qc = LB.half_uncomputed_minimal()
    else:
        import re
        k, a, _ = re.fullmatch(r'(half_uncomputed|mod_incr|controlled_adder|ripple_adder|grover_oracle)_(\d+)(c|b)', name).groups()
        qc = {'half_uncomputed': LB.half_uncomputed_oracle, 'mod_incr': LB.modular_increment, 'ripple_adder': LB.ripple_carry_adder, 'grover_oracle': LB.grover_oracle, 'controlled_adder': B.controlled_adder}[k](int(a))
    return qc, clean_ancillas(qc)

TOPO_SKIPPED = {}; POST_NOTES = {}; POST_S = float(os.environ.get('POST_SECONDS', '40'))
SEND_CAND = os.environ.get('SEND_CANDIDATES', '0') == '1'      # also measure the QCPT candidates that are not chosen (default: only Qiskit and the final QCPT circuit)
SELECTION = os.environ.get('SELECTION'); TSEEDS = int(os.environ.get('TRANSPILE_SEEDS', '5')); TRIALS = int(os.environ.get('LAYOUT_TRIALS', '8'))

def mirror_b(E, V, data):
    """Mirror circuit M = H_d V^dagger E H_d as in mirror_test_ibm.mirror, with a barrier also between V^dagger and the last H layer: the segment of V^dagger is then delimited by two barriers (qcpt_post)."""
    n = E.num_qubits; m = QuantumCircuit(n)
    for q in data: m.h(q)
    m.compose(E, inplace=True); m.barrier(); m.compose(V.inverse(), inplace=True); m.barrier()
    for q in data: m.h(q)
    m.measure_all(); return m

def esp_of(t, tg):
    """Predicted success probability of a transpiled circuit from the calibration: product of (1 - error) over its gates and measured qubits (idle decoherence ignored)."""
    e = 1.0
    for ins in t.data:
        nm = ins.operation.name; q = tuple(t.find_bit(x).index for x in ins.qubits)
        if nm in ('cz', 'sx', 'x', 'measure') and nm in tg and q in tg[nm] and tg[nm][q] is not None and tg[nm][q].error is not None: e *= 1 - tg[nm][q].error
    return e

def check_isa(t, bk):
    """Violations that would make a circuit not runnable as sent: operations or qubit tuples outside the backend target, two-qubit gates on dead couplers."""
    tg = bk.target; bad = []
    for ins in t.data:
        nm = ins.operation.name; q = tuple(t.find_bit(x).index for x in ins.qubits)
        if nm == 'barrier': continue
        if not tg.instruction_supported(nm, q): bad.append((nm, q, 'not in target'))
        elif len(q) == 2 and tg[nm][q] is not None and tg[nm][q].error is not None and tg[nm][q].error >= 0.5: bad.append((nm, q, 'dead coupler'))
    return bad

def best_transpile(m, bk, lay, levels, seeds):
    """The cheapest physical circuit that Qiskit finds for m on a fixed layout: the highest predicted success probability (= the smallest sum of calibrated errors of the gates and readouts)
    over the given optimization levels and seeds. The same search is given to the Qiskit variant and to the QCPT variant routed by Qiskit."""
    best = None
    for lv in levels:
        for sd in range(seeds):
            t = transpile(m, bk, optimization_level=lv, seed_transpiler=11 + sd, **({} if lay is None else {'initial_layout': lay})); e = esp_of(t, bk.target)
            if best is None or e > best[0]: best = (e, t, lv, sd)
    return best[1], dict(level=best[2], seed=best[3])

def best_layout(m, bk, sel, n):
    """Placement of n logical qubits on a CONNECTED set of n qubits inside the smallest selected region that holds them (the router of qcpt_topo needs a connected set of couplers):
    TRIALS random connected sets and orderings, the placement with the highest predicted success probability wins."""
    k = min(int(x) for x in sel['regions'] if int(x) >= n); region = set(sel['regions'][str(k)]['qubits']); rng = np.random.default_rng(11); best = None
    alive = {(a, b) for (a, b), v in bk.target['cz'].items() if v is not None and v.error is not None and v.error < 0.5}
    adj = {q: sorted(x for x in region if (q, x) in alive or (x, q) in alive) for q in region}
    for _ in range(TRIALS):
        start = int(rng.choice(sorted(region))); chosen = [start]
        while len(chosen) < n:
            cand = sorted({x for q in chosen for x in adj[q]} - set(chosen))
            if not cand: break
            chosen.append(int(rng.choice(cand)))
        if len(chosen) < n: continue                                            # a component of the region smaller than n: try another start
        lay = [int(x) for x in rng.permutation(chosen)]; t = transpile(m, bk, optimization_level=2, seed_transpiler=11, initial_layout=lay); e = esp_of(t, bk.target)
        if best is None or e > best[0]: best = (e, lay)
    if best is None: raise RuntimeError(f"no connected set of {n} qubits in the region of {k}")
    return best[1]

def offline_backend():
    from qiskit.transpiler import CouplingMap
    from qiskit.providers.fake_provider import GenericBackendV2
    return GenericBackendV2(num_qubits=120, basis_gates=['cz', 'sx', 'rz', 'x'], coupling_map=CouplingMap.from_grid(10, 12), seed=1)

def build(bk):
    frozen = json.load(open(FROZEN)); out = []
    for nm in (NAMES or sorted(frozen)):
        qc, pins = load(nm); n = qc.num_qubits; data = [q for q in range(n) if q not in pins]
        E = ErrorBudgetSelector().decompose_exact_only(qc)
        g = frozen[nm]['gates']; V = RS.emit(n, [tuple([x[0], tuple(x[1])] + ([tuple(x[2])] if len(x) > 2 else [])) for x in g])
        if frozen[nm].get('qasm'):                                                     # stage-1 circuit cheaper than the resynthesized gate list (frozen_scaling.json)
            from qiskit import qasm2; V = qasm2.loads(frozen[nm]['qasm'], custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
        m_n = n; lay = None; sel = json.load(open(SELECTION)) if SELECTION and os.path.exists(SELECTION) else None
        variants = {'ref': (E, 2), 'qiskit': (qc, 3), 'qct': (V, 2), 'qct_l3': (V, 3)}
        for tag in TAGS:
            if tag == 'qct_post': continue                                              # built after this loop from the Qiskit variant
            if tag == 'qct_topo':                                                           # QCPT routes its own circuit on the couplers of the layout (qcpt_topo.py)
                assert lay is not None, 'qcpt_topo needs the layout of an earlier variant (put qiskit first in TAGS)'
                gl = [(x[0], tuple(x[1])) + ((tuple(x[2]),) if len(x) > 2 else ()) for x in g]
                s1 = ErrorBudgetSelector().select(qc, pinned_zero=tuple(pins))['circuit']          # stage 1 of the current release: a further input for the SABRE candidate of the portfolio (qcpt_topo.route_best)
                try: M, p0i, own, qis, info = QT.topo_variant(nm, m_n, gl, lay, bk, E, data, os.path.join(HERE, 'topo'), float(os.environ.get('TOPO_WIN_S', '20')), extra=[('stage1', s1)]); M.name = f'{nm}__{tag}'
                except (KeyError, RuntimeError) as e_:                          # a gadget needs three mutually coupled qubits (a triangle), which heavy-hex does not have, or the router found no placement
                    TOPO_SKIPPED[nm] = f'{type(e_).__name__}: {e_}'; continue
                t = transpile(M, bk, initial_layout=lay, optimization_level=1, seed_transpiler=11); ch = dict(level=1, seed=11)
                out.append(dict(name=nm, tag=tag, t=t, p0_ideal=p0i, choice=ch, layout=[int(x) for x in lay], esp_pred=esp_of(t, bk.target), phys=sorted({t.find_bit(x).index for ins in t.data if ins.operation.name != 'barrier' for x in ins.qubits}),
                                twoq_t=sum(1 for i in t.data if i.operation.num_qubits == 2 and i.operation.name != 'barrier'), twoq_V=own, twoq_V_qiskit_routing=qis, topo=info))
                continue
            circ, lvl = variants[tag]
            m = mirror_b(E, circ, data); m.name = f'{nm}__{tag}'
            p0i = float(abs(Statevector(m.remove_final_measurements(inplace=False)).data[0]) ** 2)
            if lay is None and sel is not None: lay = best_layout(m, bk, sel, m.num_qubits)            # placement in the best region of the current calibration, shared by all variants
            t, ch = best_transpile(m, bk, lay, (2, 3) if tag in ('qiskit', 'qct') else (lvl,), TSEEDS)
            if lay is None: lay = t.layout.initial_index_layout()[:m.num_qubits]
            out.append(dict(name=nm, tag=tag, t=t, p0_ideal=p0i, choice=ch, data=list(data), layout=[int(x) for x in lay], esp_pred=esp_of(t, bk.target), phys=sorted({t.find_bit(x).index for ins in t.data if ins.operation.name != 'barrier' for x in ins.qubits}), twoq_t=sum(1 for i in t.data if i.operation.num_qubits == 2 and i.operation.name != 'barrier'),
                            twoq_V=sum(1 for i in transpile(circ, basis_gates=['cx', 'rz', 'sx', 'x'], optimization_level=0).data if i.operation.num_qubits == 2)))
    by = {}
    for it in out: by.setdefault(it['name'], {})[it['tag']] = it
    if 'qct_post' in TAGS:                                                      # QCPT on the circuit Qiskit has already placed and routed: whole V^dagger segment, same qubits, same couplers
        for nm, v in list(by.items()):
            q = v.get('qiskit')
            if q is None: continue
            t2, st = QP.post_qcpt_segment(q['t'], bk, q['data'], seconds=POST_S); POST_NOTES[nm] = {k_: st[k_] for k_ in st if k_ in ('note', 'cz_segment', 'cz_new', 'cz_before', 'cz_after', 'verified', 'k')}
            if st.get('replaced'):
                item = dict(q); item.update(tag='qct_post', t=t2, esp_pred=esp_of(t2, bk.target), twoq_t=sum(1 for i in t2.data if i.operation.num_qubits == 2 and i.operation.name != 'barrier'), post=st); out.append(item); v['qct_post'] = item
    for nm, v in by.items():                                                    # the final QCPT circuit is never worse than Qiskit's: best of the candidates and the Qiskit circuit (fall-back)
        q = v.get('qiskit'); cands = [v[t] for t in ('qct', 'qct_topo', 'qct_post') if t in v]
        for it in v.values(): it['send'] = True if it['tag'] == 'qiskit' else (SEND_CAND or it['tag'] not in ('qct', 'qct_topo', 'qct_post'))
        if q is None or not cands: continue
        best = max(cands, key=lambda it: it['esp_pred']); fallback = best['esp_pred'] <= q['esp_pred']; src = 'qiskit_fallback' if fallback else best['tag']
        if not fallback and not SEND_CAND: best['send'] = True
        for it in v.values(): it['qct_final'] = src
    return out

def same_qubits(items):
    """For every circuit: do the Qiskit variant and the QCPT variant use exactly the same physical qubits (placement shared; routing may add qubits)?"""
    by = {}
    for it in items: by.setdefault(it['name'], {})[it['tag']] = set(it['phys'])
    both = {k: v for k, v in by.items() if 'qiskit' in v and len(v) > 1}                 # (all variants, sent or not)
    diff = {k: {t: (sorted(v['qiskit'] - w), sorted(w - v['qiskit'])) for t, w in v.items() if t != 'qiskit' and w != v['qiskit']} for k, v in both.items() if any(w != v['qiskit'] for t, w in v.items() if t != 'qiskit')}
    return len(both), diff

def cheapest(items):
    """QCPT as a producer of the cheapest physical circuit: per circuit the best of the two QCPT candidates (routed by Qiskit, routed by QCPT) and the Qiskit circuit, by predicted success
    probability (the smallest sum of calibrated errors). It is never worse than Qiskit by construction; where Qiskit wins, QCPT returns the Qiskit circuit (fall-back)."""
    by = {}
    for it in items: by.setdefault(it['name'], {})[it['tag']] = it
    rows = []
    for nm, v in by.items():
        c = [t for t in ('qct', 'qct_topo', 'qct_post') if t in v]
        if 'qiskit' in v and c:
            b = max(c, key=lambda t: v[t]['esp_pred']); q = v['qiskit']; src = 'qiskit_fallback' if v[b]['esp_pred'] <= q['esp_pred'] else b
            f = q if src == 'qiskit_fallback' else v[b]; rows.append((nm, src, -np.log(f['esp_pred']), -np.log(q['esp_pred']), f['twoq_t'], q['twoq_t']))
    return rows

def show(items, sent=None):
    for it in items: print(it['name'], it['tag'], 'twoq_V', it['twoq_V'], 'twoq_t', it['twoq_t'], 'p0_ideal', round(it['p0_ideal'], 3), *(['V alone with Qiskit routing:', it['twoq_V_qiskit_routing'], 'non-adjacent left:', it['topo']['non_adjacent_left']] if it['tag'] == 'qct_topo' else []))
    rows = cheapest(items)
    if POST_NOTES: print(f'qcpt_post (QCPT na obwodzie Qiskita): tanszy obwod dla {sum(1 for x in POST_NOTES.values() if x.get("cz_after") is not None and x["cz_after"] < x["cz_before"])} z {len(POST_NOTES)} obwodow; powody braku:', {k_: v_.get('note') for k_, v_ in POST_NOTES.items() if v_.get('note')})
    if TOPO_SKIPPED: print(f'qcpt_topo niemozliwy (gadzet wymaga trojkata sprzezen lub brak ulozenia) dla {len(TOPO_SKIPPED)} obwodow:', sorted(TOPO_SKIPPED))
    if rows:
        cnt = {s_: sum(1 for r in rows if r[1] == s_) for s_ in ('qct', 'qct_topo', 'qct_post', 'qiskit_fallback')}
        print(f'QCPT = najtanszy obwod fizyczny (najlepszy z qcpt, qcpt_topo i obwodu Qiskita wg przewidywanego ESP): wybrano {cnt}; suma -ln ESP: QCPT {sum(r[2] for r in rows):.2f}, Qiskit {sum(r[3] for r in rows):.2f}; bramki 2q: QCPT {sum(r[4] for r in rows)}, Qiskit {sum(r[5] for r in rows)}; QCPT gorszy od Qiskita na {sum(1 for r in rows if r[2] > r[3] + 1e-12)} obwodach')
    n, diff = same_qubits(items); print(f'te same kubity fizyczne dla wariantow qiskit i qcpt: {n - len(diff)} z {n} obwodow; rozne:', {k: v for k, v in list(diff.items())[:6]})
    sent = items if sent is None else sent; print('obwodow zbudowanych', len(items), '| do wyslania', len(sent), 'strzalow', SHOTS * len(sent), 'szac. QPU s', round(SHOTS * len(sent) * SEC_PER_SHOT, 1))

if __name__ == '__main__':
    stage = sys.argv[1]
    if stage == 'plan': show(build(offline_backend())); sys.exit()
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    svc = QiskitRuntimeService(name=ACC) if ACC else QiskitRuntimeService()
    if stage in ('sprawdz', 'wyslij'):
        bk = svc.backend(BK); items = build(bk); u = svc.usage(); show(items); est = SHOTS * len(items) * SEC_PER_SHOT
        print('pozostalo s', u.get('usage_remaining_seconds'))
        if stage == 'wyslij':
            if os.path.exists(J): sys.exit('job.json istnieje - nie wysylam ponownie (uzyj odbierz)')
            if u.get('usage_remaining_seconds', 0) < est * 1.5: sys.exit('za malo budzetu')
            job = SamplerV2(mode=bk).run([it['t'] for it in items], shots=SHOTS)
            json.dump(dict(job_id=job.job_id(), account=ACC, backend=BK, shots=SHOTS, order=[(it['name'], it['tag']) for it in items], p0_ideal=[it['p0_ideal'] for it in items],
                           twoq_t=[it['twoq_t'] for it in items], twoq_V=[it['twoq_V'] for it in items], sent=time.strftime('%Y-%m-%d %H:%M:%S')), open(J, 'w'), indent=1)
            print('wyslano', job.job_id(), flush=True)
    elif stage == 'odbierz':
        D = json.load(open(J)); job = svc.job(D['job_id']); print('status', job.status()); res = job.result(); rows = []
        for k, ((nm, tag), pid) in enumerate(zip(D['order'], D['p0_ideal'])):
            c = res[k].data.meas.get_counts(); n = len(next(iter(c))); p, s, tot = p0_from_counts(c, n)
            rows.append(dict(name=nm, tag=tag, p0_ideal=round(pid, 4), p0_hw=round(p, 4), sigma=round(s, 4), shots=tot, twoq_transpiled=D['twoq_t'][k], twoq_V=D['twoq_V'][k])); print(rows[-1])
        u = svc.usage(); json.dump(dict(job=D, rows=rows, metrics=str(job.metrics()), usage_after=u.get('usage_consumed_seconds')), open(os.path.join(HERE, 'wyniki.json'), 'w'), indent=1)
        print('zuzyto lacznie s', u.get('usage_consumed_seconds'))

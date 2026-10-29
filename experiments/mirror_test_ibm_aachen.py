"""Second mirror test of subroutine equivalence on ibm_aachen (Table tab:mirror2 of the paper).
Same construction as mirror_test_ibm.py: M = H_d V^dag E H_d, return probability to |0...0>.
Stages: sprawdz (build + transpile, print plan) | wyslij (submit once; job id stored at once in
fixtures/mirror_test_aachen/job.json, refuses if it exists) | odbierz (fetch -> wyniki_aachen.json).
Account name and backend via IBM_ACCOUNT / IBM_BACKEND (paper run: eu-de-aachen-156, ibm_aachen)."""
import json, os, sys, time
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path[:0] = [R, R + '/experiments']
import numpy as np
import benchmarks as B, large_benchmarks as LB, baselines
from _clean import clean_ancillas
from mirror_test_ibm import mirror, p0_from_counts
from qiskit import transpile
from qiskit.quantum_info import Statevector
from toffoli_optimizer.core.decomposition_selector import ErrorBudgetSelector
ACC, BK, SHOTS = os.environ.get('IBM_ACCOUNT', 'eu-de-aachen-156'), os.environ.get('IBM_BACKEND', 'ibm_aachen'), int(os.environ.get('SHOTS', '8000'))
NAMES = ['half_uncomputed', 'half_uncomputed_3c', 'half_uncomputed_4c', 'mod_incr_3b', 'controlled_adder_2b', 'nested_compute_uncompute']
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures', 'mirror_test_aachen'); os.makedirs(HERE, exist_ok=True); J = os.path.join(HERE, 'job.json')

def circuits():
    pool = {q.name: q for q in B.benchmark_suite()}
    pool.update({q.name: q for q in (LB.half_uncomputed_oracle(3), LB.half_uncomputed_oracle(4), LB.modular_increment(3))})
    return [pool[n] for n in NAMES]

def build(bk):
    out = []
    for qc in circuits():
        pins = clean_ancillas(qc); data = [q for q in range(qc.num_qubits) if q not in pins]
        E = ErrorBudgetSelector().decompose_exact_only(qc)
        V = {'ref': E, 'sub': ErrorBudgetSelector().select(qc, pinned_zero=pins)['circuit'], 'greedy': baselines.m_count_greedy(qc)}
        lay = None
        for tag in ('ref', 'sub', 'greedy'):
            m = mirror(E, V[tag], data); m.name = f'{qc.name}__{tag}'
            p0i = float(abs(Statevector(m.remove_final_measurements(inplace=False)).data[0]) ** 2)
            t = transpile(m, bk, optimization_level=2, seed_transpiler=11, **({} if lay is None else {'initial_layout': lay}))
            if lay is None: lay = t.layout.initial_index_layout()[:m.num_qubits]
            out.append(dict(name=qc.name, tag=tag, t=t, p0_ideal=p0i, twoq_t=sum(1 for i in t.data if i.operation.num_qubits == 2 and i.operation.name != 'barrier'),
                            twoq_V=sum(1 for i in V[tag].data if i.operation.num_qubits == 2)))
    return out

if __name__ == '__main__':
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    stage = sys.argv[1]; svc = QiskitRuntimeService(name=ACC)
    if stage in ('sprawdz', 'wyslij'):
        bk = svc.backend(BK); items = build(bk); u = svc.usage()
        PF = os.path.join(HERE, 'plan_aachen.jsonl'); plan = {json.loads(l)['name']: json.loads(l) for l in open(PF)} if os.path.exists(PF) else {}
        for it in items: print(it['name'], it['tag'], 'twoq_t', it['twoq_t'], 'plan', plan.get(it['name'], {}).get(it['tag'], {}).get('twoq_t'), 'p0_ideal', round(it['p0_ideal'], 3))
        est = SHOTS * len(items) * 293e-6
        print('obwodow', len(items), 'strzalow', SHOTS * len(items), 'szac. QPU s', round(est, 1), 'pozostalo s', u.get('usage_remaining_seconds'))
        if stage == 'wyslij':
            if os.path.exists(J): sys.exit('job.json istnieje - nie wysylam ponownie (uzyj odbierz)')
            if u.get('usage_remaining_seconds', 0) < est * 1.5: sys.exit('za malo budzetu')
            job = SamplerV2(mode=bk).run([it['t'] for it in items], shots=SHOTS)
            json.dump(dict(job_id=job.job_id(), account=ACC, backend=BK, shots=SHOTS, order=[(it['name'], it['tag']) for it in items],
                           p0_ideal=[it['p0_ideal'] for it in items], twoq_t=[it['twoq_t'] for it in items], twoq_V=[it['twoq_V'] for it in items],
                           sent=time.strftime('%Y-%m-%d %H:%M:%S')), open(J, 'w'), indent=1)
            print('wyslano', job.job_id(), flush=True)
    elif stage == 'odbierz':
        D = json.load(open(J)); job = svc.job(D['job_id']); print('status', job.status())
        res = job.result(); rows = []
        for k, ((nm, tag), pid) in enumerate(zip(D['order'], D['p0_ideal'])):
            c = res[k].data.meas.get_counts(); n = len(next(iter(c))); p, s, tot = p0_from_counts(c, n)
            rows.append(dict(name=nm, tag=tag, p0_ideal=round(pid, 4), p0_hw=round(p, 4), sigma=round(s, 4), shots=tot, twoq_transpiled=D['twoq_t'][k], twoq_V=D['twoq_V'][k]))
            print(rows[-1])
        u = svc.usage()
        json.dump(dict(job=D, rows=rows, metrics=str(job.metrics()), usage_after=u.get('usage_consumed_seconds')), open(os.path.join(HERE, 'wyniki_aachen.json'), 'w'), indent=1)
        print('zuzyto lacznie s', u.get('usage_consumed_seconds'))

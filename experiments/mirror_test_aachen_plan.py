"""Planning run for mirror_test_ibm_aachen.py (noisy simulation with the current ibm_aachen calibration).
Plan drugiego testu lustrzanego (pkt 5): kandydaci, w ktorych przebieg SAM podstawia gadzety.
Dla kazdego kandydata: ref (dokladny), sub (przebieg), greedy (count-greedy); p0 idealne, 2q po transpilacji
na ibm_marrakesh, p0 z symulacji szumu (AerSimulator.from_backend, kalibracja bieżąca), z-score sub-vs-greedy i
sub-vs-ref przy SHOTS strzalach. Wynik dopisywany per kandydat do plan.jsonl (wznawialne). Nic nie wysyla na QPU."""
import json, os, sys, time
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path[:0] = [R, R + '/experiments']
import numpy as np
import benchmarks as B, large_benchmarks as LB, baselines
from _clean import clean_ancillas
from mirror_test_ibm import mirror
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from toffoli_optimizer.core.decomposition_selector import ErrorBudgetSelector
WYBOR={'half_uncomputed','half_uncomputed_3c','half_uncomputed_4c','mod_incr_3b','nested_compute_uncompute','controlled_adder_2b'}
SHOTS = 8000; OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures', 'mirror_test_aachen', 'plan_aachen.jsonl')
done = {json.loads(l)['name'] for l in open(OUT)} if os.path.exists(OUT) else set()
from qiskit_ibm_runtime import QiskitRuntimeService
be = QiskitRuntimeService(name='eu-de-aachen-156').backend('ibm_aachen'); sim = AerSimulator.from_backend(be)
cands = [qc for qc in B.benchmark_suite()]
for f, args in ((LB.half_uncomputed_oracle, (3, 4)), (LB.grover_oracle, (3, 4)), (LB.ripple_carry_adder, (2,)), (LB.modular_increment, (3,)), (LB.carry_lookahead_adder, (2,))):
    for a in args:
        try: cands.append(f(a))
        except Exception as e: print('pomijam', f.__name__, a, e)
for qc in cands:
    if qc.name in done or qc.name not in WYBOR: continue
    pins = clean_ancillas(qc); data = [q for q in range(qc.num_qubits) if q not in pins]
    E = ErrorBudgetSelector().decompose_exact_only(qc)
    V = {'ref': E, 'sub': ErrorBudgetSelector().select(qc, pinned_zero=pins)['circuit'], 'greedy': baselines.m_count_greedy(qc)}
    row = {'name': qc.name, 'n': qc.num_qubits}
    lay = None
    for tag in ('ref', 'sub', 'greedy'):
        m = mirror(E, V[tag], data)
        p0i = float(abs(Statevector(m.remove_final_measurements(inplace=False)).data[0]) ** 2)
        t = transpile(m, be, optimization_level=2, seed_transpiler=11, **({} if lay is None else {'initial_layout': lay}))
        if lay is None: lay = t.layout.initial_index_layout()[:m.num_qubits]
        c = sim.run(t, shots=SHOTS, seed_simulator=7).result().get_counts()
        p0 = c.get('0' * m.num_qubits, 0) / SHOTS
        row[tag] = {'twoq_V': sum(1 for i in V[tag].data if i.operation.num_qubits == 2),
                    'twoq_t': sum(1 for i in t.data if i.operation.num_qubits == 2 and i.operation.name != 'barrier'),
                    'p0_ideal': round(p0i, 4), 'p0_sim': round(p0, 4)}
    s = lambda p: np.sqrt(max(p * (1 - p), 1e-6) / SHOTS)
    a, b, r = row['sub']['p0_sim'], row['greedy']['p0_sim'], row['ref']['p0_sim']
    row['z_sub_greedy'] = round(abs(a - b) / np.hypot(s(a), s(b)), 1); row['z_sub_ref'] = round(abs(a - r) / np.hypot(s(a), s(r)), 1)
    row['pass_substitutes'] = row['sub']['twoq_V'] < row['ref']['twoq_V']
    row['greedy_wrong'] = row['greedy']['p0_ideal'] < 0.999
    open(OUT, 'a').write(json.dumps(row) + '\n'); print(json.dumps(row), flush=True)
print('KONIEC')

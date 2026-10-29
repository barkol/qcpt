"""QCPT on the circuit that Qiskit has already placed and routed ("qct_post"): same qubits, same couplers, nothing is moved.
In the transpiled mirror circuit only the V^dagger segment is touched (between the first barrier and the last one), never the reference E. Blocks of at most four qubits are collected
(Qiskit CollectMultiQBlocks); a block whose unitary is a monomial matrix with Z8 phases (a permutation with phases, up to a global phase) is searched again with the library restricted to the couplers
that exist between its physical qubits (resynth.library(k, coupling=...)); the replacement is accepted only if it has fewer CZ gates AND a smaller sum of calibrated errors, and it is checked against the block
(operator equal up to a phase). Blocks that are not monomial (Hadamard-like gates inside) are left alone, so the context of a Toffoli (reachable states) is not used here: this pass finds exact savings only."""
import time
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import Operator
from qiskit.transpiler import PropertySet
from qiskit.transpiler.passes import CollectMultiQBlocks
from toffoli_optimizer.core import resynth as RS

BASIS = ['cz', 'sx', 'rz', 'x']; _SEARCH = {}; _LIB = {}

def _err(tg, a, b):
    v = tg['cz'].get((a, b)) or tg['cz'].get((b, a)); return -np.log(1 - v.error) if v is not None and v.error is not None else 0.0

def _same(U, V):
    a = np.vdot(U.ravel(), V.ravel()); return abs(a) > 0 and float(np.abs(V - (a / abs(a)) * U).max()) < 1e-8

def _lib(k, cp):
    if (k, cp) not in _LIB: _LIB[(k, cp)] = [g for g in RS.library(k, coupling=cp) if not g[0].startswith('mined')]
    return _LIB[(k, cp)]

def post_qcpt(t, bk, seconds=30.0, passes=3):
    """Returns (new transpiled circuit, stats). The input circuit is not modified."""
    tg = bk.target; alive = {(a, b) for (a, b), v in tg['cz'].items() if v is not None and v.error is not None and v.error < 0.5}
    bars = [i for i, ins in enumerate(t.data) if ins.operation.name == 'barrier']
    cz0 = sum(1 for ins in t.data if ins.operation.name == 'cz')
    if len(bars) < 2: return t, dict(replaced=0, cz_before=cz0, cz_after=cz0, searched=0, note='no segment')
    b0, b1 = bars[-3], bars[-2]                                                # E | barrier | V^dagger | barrier | last H layer | barrier | measurement (E may contain barriers of its own)
    head, seg_ops, tail = t.data[:b0 + 1], t.data[b0 + 1:b1], t.data[b1:]
    seg = t.copy_empty_like()
    for ins in seg_ops: seg._append(ins)
    t0 = time.time(); replaced = 0; searched = 0
    for _ in range(passes):
        dag = circuit_to_dag(seg); p = CollectMultiQBlocks(max_block_size=4); p.property_set = PropertySet(); p.run(dag); blocks = p.property_set['block_list'] or []; changed = False
        for block in blocks:
            if time.time() - t0 > seconds: break
            if any(n.op.name not in BASIS for n in block): continue
            qs = sorted({dag.find_bit(q).index for n in block for q in n.qargs}); k = len(qs)
            if k < 2 or k > 4: continue
            c = sum(1 for n in block if n.op.name == 'cz')
            if c < 2: continue
            pos = {q: i for i, q in enumerate(qs)}; sub = QuantumCircuit(k)
            for n in block: sub.append(n.op, [pos[dag.find_bit(q).index] for q in n.qargs])
            try: perm, ph = RS._table_norm(sub)                                   # raises AssertionError if the block is not a monomial matrix with Z8 phases
            except (AssertionError, ValueError): continue
            cp = frozenset((x, y) for x in range(k) for y in range(x + 1, k) if (qs[x], qs[y]) in alive or (qs[y], qs[x]) in alive)
            if not cp: continue
            key = (k, perm.tobytes(), ph.tobytes(), cp, c); searched += 1
            if key not in _SEARCH:
                _SEARCH[key] = RS.bidir_search(k, list(range(1 << k)), perm, max_cost=c - 1, beam=3000, lam=0.6, depth_side=8, lib=_lib(k, cp), target_phase=ph)
            r = _SEARCH[key]
            if not r: continue
            new = transpile(RS.emit(k, [(g[0], tuple(g[1])) for g in r[1]]), basis_gates=BASIS, optimization_level=1)
            if sum(1 for i in new.data if i.operation.name == 'cz') >= c or not _same(Operator(sub).data, Operator(new).data): continue
            old_e = sum(_err(tg, qs[n_.qargs and pos[dag.find_bit(n_.qargs[0]).index]], qs[pos[dag.find_bit(n_.qargs[1]).index]]) for n_ in block if n_.op.name == 'cz')
            new_e = sum(_err(tg, qs[new.find_bit(i.qubits[0]).index], qs[new.find_bit(i.qubits[1]).index]) for i in new.data if i.operation.name == 'cz')
            if new_e >= old_e: continue
            dag.replace_block_with_op(block, new.to_gate(), {dag.qubits[q]: pos[q] for q in qs}, cycle_check=True); replaced += 1; changed = True
        seg = transpile(dag_to_circuit(dag), basis_gates=BASIS, optimization_level=0)
        if not changed or time.time() - t0 > seconds: break
    out = t.copy_empty_like()
    for ins in head: out._append(ins)
    for ins in seg.data: out._append(ins)
    for ins in tail: out._append(ins)
    cz1 = sum(1 for ins in out.data if ins.operation.name == 'cz')
    return out, dict(replaced=replaced, cz_before=cz0, cz_after=cz1, searched=searched, seconds=round(time.time() - t0, 1))


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Whole-segment version: the entire V^dagger segment (all qubits it touches, up to 10) is one monomial block, so it is resynthesized as a whole with the coupler filter of the C++ core,
# on the subspace that the mirror test uses: after V^dagger every qubit that does not carry data (ancillae pinned to 0, qubits used only for routing) is back in |0>.
from toffoli_optimizer.core import cppsearch as CS


BEAMS = ((60000, 0.6, 10, 0.0, 0.0, 0), (120000, 0.4, 10, 0.0, 0.0, 0), (60000, 0.7, 12, 0.25, 0.4, 1), (60000, 0.7, 12, 0.25, 0.4, 2), (200000, 0.6, 12, 0.0, 0.0, 0))      # beam, lambda, depth, diversity, noise, seed (like the members of the race)


def post_qcpt_segment(t, bk, data, seconds=120.0, max_qubits=10, beams=BEAMS):
    """Resynthesis of the V^dagger segment of the transpiled mirror circuit t on its valid subspace under the couplers of the backend. data = logical data qubits (the others are pinned ancillae).
    Returns (new circuit or t, stats); the new circuit is returned only if it has fewer CZ gates and a smaller sum of calibrated errors, and has been verified on the valid inputs."""
    tg = bk.target; alive = {(a, b) for (a, b), v in tg['cz'].items() if v is not None and v.error is not None and v.error < 0.5}
    bars = [i for i, ins in enumerate(t.data) if ins.operation.name == 'barrier']
    cz0 = sum(1 for ins in t.data if ins.operation.name == 'cz'); st = dict(replaced=0, cz_before=cz0, cz_after=cz0)
    if len(bars) < 3: return t, dict(st, note='no segment')
    b0, b1 = (bars[-3], bars[-2]) if len(bars) >= 3 else (bars[0], bars[1]) # E | barrier | V^dagger | barrier | last H layer | barrier | measurement (E may contain barriers of its own)
    head, seg_ops, tail = t.data[:b0 + 1], t.data[b0 + 1:b1], t.data[b1:]
    qs = sorted({t.find_bit(q).index for ins in seg_ops for q in ins.qubits}); k = len(qs)
    if k < 2 or k > max_qubits: return t, dict(st, note=f'{k} active qubits')
    pos = {q: i for i, q in enumerate(qs)}; S = QuantumCircuit(k)
    for ins in seg_ops: S.append(ins.operation, [pos[t.find_bit(q).index] for q in ins.qubits])
    c = sum(1 for ins in S.data if ins.operation.name == 'cz')
    try: perm, ph = RS._table_norm(S)                                              # the whole segment is a permutation with Z8 phases
    except (AssertionError, ValueError): return t, dict(st, note='segment not monomial')
    fl = t.layout.final_index_layout(); dphys = {fl[q] for q in data}
    free_local = [pos[p] for p in dphys if p in pos]                              # data qubits touched by the segment; every other active qubit is |0> after V^dagger
    zero_mask = sum(1 << i for i in range(k) if i not in free_local)
    inputs = [r for r in range(1 << k) if (int(perm[r]) & zero_mask) == 0]        # states that V^dagger maps into the valid output subspace
    if not inputs or len(inputs) > 4096: return t, dict(st, note=f'{len(inputs)} valid inputs')
    target = [int(perm[r]) for r in inputs]; phase = [int((ph[r] - ph[inputs[0]]) % 8) for r in inputs]
    cp = frozenset((x, y) for x in range(k) for y in range(x + 1, k) if (qs[x], qs[y]) in alive or (qs[y], qs[x]) in alive)
    if not cp: return t, dict(st, note='no couplers')
    allow = CS.allow_indices(k, cp); best = None; t0 = time.time(); mc = c - 1; calls = 0
    while mc >= 0 and time.time() - t0 < seconds:
        found = None
        for beam, lam, ds, dv, nz, sd in beams:
            if time.time() - t0 > seconds: break
            calls += 1
            try: found = CS.beam_search(k, inputs, target, phase, mc, beam=beam, lam=lam, depth_side=ds, diversity=dv, noise=nz, seed=sd, nthreads=8, allow=allow, timeout=max(30.0, seconds))
            except Exception: found = None                                          # time limit of one search

            if found: break
        if not found: break
        best = found; mc = found[0] - 1
    st.update(k=k, valid_inputs=len(inputs), cz_segment=c, searches=calls, seconds=round(time.time() - t0, 1))
    if best is None: return t, dict(st, note='no cheaper circuit found')
    new = transpile(RS.emit(k, [(g[0], tuple(g[1])) for g in best[1]]), basis_gates=BASIS, optimization_level=1)
    U = Operator(new).data; Uo = Operator(S).data; r0 = inputs[0]; g0 = U[target[0], r0] / Uo[target[0], r0] if abs(Uo[target[0], r0]) > 0 else 0
    ok = abs(abs(g0) - 1) < 1e-8 and all(np.abs(U[:, r] - g0 * Uo[:, r]).max() < 1e-8 for r in inputs)      # equal on the valid inputs, one global phase
    n_cz = sum(1 for i in new.data if i.operation.name == 'cz')
    old_e = sum(_err(tg, qs[S.find_bit(i.qubits[0]).index], qs[S.find_bit(i.qubits[1]).index]) for i in S.data if i.operation.name == 'cz')
    new_e = sum(_err(tg, qs[new.find_bit(i.qubits[0]).index], qs[new.find_bit(i.qubits[1]).index]) for i in new.data if i.operation.name == 'cz')
    st.update(cz_new=n_cz, err_old=round(old_e, 4), err_new=round(new_e, 4), verified=bool(ok))
    if not ok or n_cz >= c or new_e >= old_e: return t, dict(st, note='rejected')
    out = t.copy_empty_like()
    for ins in head: out._append(ins)
    for ins in new.data: out.append(ins.operation, [t.qubits[qs[new.find_bit(q).index]] for q in ins.qubits])
    for ins in tail: out._append(ins)
    return out, dict(st, replaced=1, cz_after=sum(1 for ins in out.data if ins.operation.name == 'cz'))

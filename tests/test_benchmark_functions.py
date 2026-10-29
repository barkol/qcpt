"""The ripple adders and Grover oracles of the scale suite compute their named function on
every basis input with clean ancillas (the paper states this). The modular incrementer
increments x correctly but leaves car[0] dirty (see its docstring); the carry-lookahead,
live-carry and array-multiplier generators are structural models."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
import large_benchmarks as LB


def _run(qc, s):
    for ins in qc.data:
        qb = [qc.find_bit(q).index for q in ins.qubits]; n = ins.operation.name
        if n == "barrier": continue
        if n == "x": s ^= 1 << qb[0]
        elif n in ("cx", "ccx", "mcx", "mcx_gray"):
            if all((s >> c) & 1 for c in qb[:-1]): s ^= 1 << qb[-1]
        else: raise AssertionError("non-classical gate " + n)
    return s


def _reg(qc, name):
    return [qc.find_bit(q).index for q in next(r for r in qc.qregs if r.name == name)]


def _val(s, idx): return sum(((s >> q) & 1) << k for k, q in enumerate(idx))
def _set(idx, v): return sum(((v >> k) & 1) << q for k, q in enumerate(idx))


def test_ripple_adder_adds():
    for nb in (3, 4):
        qc = LB.ripple_carry_adder(nb); a, b, z = _reg(qc, "a"), _reg(qc, "b"), _reg(qc, "z")
        for av in range(2 ** nb):
            for bv in range(2 ** nb):
                s = _run(qc, _set(a, av) | _set(b, bv))
                assert _val(s, b) + (_val(s, z) << nb) == av + bv and _val(s, a) == av


def test_modular_increment():
    for nb in (3, 4):
        qc = LB.modular_increment(nb); c, x, car = _reg(qc, "ctrl"), _reg(qc, "x"), _reg(qc, "car")
        for cv in (0, 1):
            for xv in range(2 ** nb):
                s = _run(qc, _set(c, cv) | _set(x, xv))
                assert _val(s, x) == (xv + cv) % 2 ** nb   # increment correct; car[0] not restored


def test_grover_oracle_marks_all_ones():
    for nc in (3, 4, 5):
        qc = LB.grover_oracle(nc); c, anc, t = _reg(qc, "c"), _reg(qc, "anc"), _reg(qc, "t")
        for cv in range(2 ** nc):
            s = _run(qc, _set(c, cv))
            assert _val(s, t) == int(cv == 2 ** nc - 1) and _val(s, anc) == 0 and _val(s, c) == cv

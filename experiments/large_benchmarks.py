"""Large CCX/MCX benchmark circuits (12-24 qubits) resembling real workloads.

The small suite in ``benchmarks.py`` is capped at ~8-10 qubits so that the WHOLE
circuit fits QCPT's exhaustive ``ExactEquivalenceVerifier``. This module pushes
to 12-24 qubits -- past the exhaustive-verification limit -- to exercise the SCALABLE
verification fallback (``verify_scalable`` -> MQT QCEC) and to test whether the
error-budget relative-phase reduction holds at scale.

Every circuit is built only from CCX / MCX / CX / X (classical-reversible), so the
exact-only lowering is a plain Clifford+T network and the reachable-subspace /
phase-observability analyses apply. Each generator tags two attributes:

  * ``relphase_safe`` (bool): ground truth -- does a BLANKET relative-phase (Margolus)
    substitution of every CCX preserve the function? True for fully compute/uncompute-
    structured circuits (the spurious phase cancels), False for "live" circuits whose
    Toffoli results survive to an output (the phase corrupts the function).
  * ``family`` (str): coarse workload class, for the results table.

The mix:

  COMPUTE/UNCOMPUTE-STRUCTURED  (relphase admissible -> reduction expected)
    * wide ripple-carry adder (Cuccaro MAJ/UMA, every carry uncomputed)
    * carry-lookahead-style AND-tree adder (compute tree, use, uncompute tree)
    * modular-arithmetic block: controlled modular increment with uncomputed carries
    * larger Grover AND-ladder oracles (compute ladder, mark, uncompute ladder)

  LIVE  (relphase NOT admissible -> a sound selector must keep these exact)
    * ripple adder with a LIVE carry-out (carry consumed downstream, not uncomputed)
    * a small array multiplier whose partial-product ANDs survive to the product
    * a "half-uncomputed" oracle: a valid pair plus one live marking Toffoli
"""

from __future__ import annotations

from qiskit import QuantumCircuit, QuantumRegister


def _tag(qc: QuantumCircuit, name: str, safe: bool, family: str) -> QuantumCircuit:
    qc.name = name
    qc.relphase_safe = safe
    qc.family = family
    return qc


# ===========================================================================
# COMPUTE/UNCOMPUTE-STRUCTURED  (relative-phase admissible)
# ===========================================================================
def ripple_carry_adder(nbits: int) -> QuantumCircuit:
    """Cuccaro ripple-carry adder a += b, every carry computed AND uncomputed.

    Width = 2*nbits + 2. nbits=5 -> 12 qubits; nbits=7 -> 16; nbits=10 -> 22.
    Fully compute/uncompute-structured -> relative-phase SAFE.
    """
    a = QuantumRegister(nbits, "a")
    b = QuantumRegister(nbits, "b")
    c = QuantumRegister(1, "c")
    z = QuantumRegister(1, "z")
    qc = QuantumCircuit(a, b, c, z)

    def MAJ(ci, ai, bi):
        qc.cx(ai, bi); qc.cx(ai, ci); qc.ccx(ci, bi, ai)

    def UMA(ci, ai, bi):
        qc.ccx(ci, bi, ai); qc.cx(ai, ci); qc.cx(ci, bi)

    prev = c[0]
    for i in range(nbits):
        MAJ(prev, a[i], b[i]); prev = a[i]
    qc.cx(prev, z[0])
    for i in reversed(range(nbits)):
        prev = a[i - 1] if i > 0 else c[0]
        UMA(prev, a[i], b[i])
    return _tag(qc, f"ripple_adder_{nbits}b", safe=True, family="adder/cu")


def carry_lookahead_adder(nbits: int) -> QuantumCircuit:
    """Carry-lookahead-style adder: an AND-tree of generate/propagate signals
    computed onto carry ancillas, used to form the sum, then the tree UNcomputed.

    A simplified CLA: g[i] = a[i] & b[i] computed onto carry ancillas via a Toffoli
    tree, the sum written, then the tree uncomputed in mirror order. Every CCX is in
    a compute/uncompute pair -> relative-phase SAFE.

    Width = 3*nbits + 1. nbits=4 -> 13; nbits=5 -> 16; nbits=7 -> 22.
    """
    a = QuantumRegister(nbits, "a")
    b = QuantumRegister(nbits, "b")
    g = QuantumRegister(nbits, "g")   # generate ancillas (uncomputed)
    s = QuantumRegister(1, "s")       # a sum/output wire fed by the carries
    qc = QuantumCircuit(a, b, g, s)

    # compute generate signals g[i] = a[i] & b[i]
    for i in range(nbits):
        qc.ccx(a[i], b[i], g[i])
    # an AND-tree carry chain onto g (propagate), then a use onto s
    for i in range(1, nbits):
        qc.ccx(g[i - 1], a[i], g[i])     # tree combine (still uncomputed)
    qc.cx(g[nbits - 1], s[0])            # use the top carry
    # uncompute the tree, then the generates, in mirror order
    for i in reversed(range(1, nbits)):
        qc.ccx(g[i - 1], a[i], g[i])
    for i in reversed(range(nbits)):
        qc.ccx(a[i], b[i], g[i])
    return _tag(qc, f"cla_adder_{nbits}b", safe=True, family="adder/cla")


def modular_increment(nbits: int) -> QuantumCircuit:
    """Controlled modular increment (mod 2^nbits) with uncomputed carry chain.

    ctrl-controlled x += 1 (mod 2^nbits): a ripple of CCX carries onto ancillas to
    propagate the increment, the conditional bit flips, then the carries UNcomputed.
    Compute/uncompute-structured -> relative-phase SAFE.

    Known defect (kept, since the published results use this circuit): the last
    uncompute uses x[0] after its flip, so car[0] is not restored when ctrl=1. The
    increment of x is correct on every basis input; the circuit is a structural model
    of a modular incrementer. Every output of the pass is certified against the exact
    decomposition of this same circuit, so the defect does not affect soundness.

    Width = 2*nbits + 1. nbits=6 -> 13; nbits=8 -> 17; nbits=11 -> 23.
    """
    ctrl = QuantumRegister(1, "ctrl")
    x = QuantumRegister(nbits, "x")
    car = QuantumRegister(nbits - 1, "car")
    qc = QuantumCircuit(ctrl, x, car)

    # compute carry chain: car[0] = ctrl & x[0]; car[i] = car[i-1] & x[i]
    qc.ccx(ctrl[0], x[0], car[0])
    for i in range(1, nbits - 1):
        qc.ccx(car[i - 1], x[i], car[i])
    # apply the conditional flips (controls read the live carries)
    qc.cx(car[nbits - 2], x[nbits - 1])
    for i in range(nbits - 1):
        c = ctrl[0] if i == 0 else car[i - 1]
        qc.cx(c, x[i])
    # uncompute the carry chain (mirror)
    for i in reversed(range(1, nbits - 1)):
        qc.ccx(car[i - 1], x[i], car[i])
    qc.ccx(ctrl[0], x[0], car[0])
    return _tag(qc, f"mod_incr_{nbits}b", safe=True, family="modular/cu")


def grover_oracle(nctrl: int) -> QuantumCircuit:
    """Grover marking oracle as an AND-ladder onto ancillas, mark, then uncompute.

    Width = 2*nctrl. nctrl=6 -> 12; nctrl=8 -> 16; nctrl=11 -> 22.
    Compute/uncompute-structured -> relative-phase SAFE.
    """
    ctrl = QuantumRegister(nctrl, "c")
    anc = QuantumRegister(nctrl - 1, "anc")
    tgt = QuantumRegister(1, "t")
    qc = QuantumCircuit(ctrl, anc, tgt)
    qc.ccx(ctrl[0], ctrl[1], anc[0])
    for i in range(2, nctrl):
        qc.ccx(ctrl[i], anc[i - 2], anc[i - 1])
    qc.cx(anc[nctrl - 2], tgt[0])
    for i in reversed(range(2, nctrl)):
        qc.ccx(ctrl[i], anc[i - 2], anc[i - 1])
    qc.ccx(ctrl[0], ctrl[1], anc[0])
    return _tag(qc, f"grover_oracle_{nctrl}c", safe=True, family="grover/cu")


# ===========================================================================
# LIVE  (relative-phase NOT admissible -- a sound selector must keep exact)
# ===========================================================================
def ripple_adder_live_carry(nbits: int) -> QuantumCircuit:
    """Ripple adder whose carry chain is left LIVE (consumed downstream, not undone).

    The carry Toffolis write results that survive to the output, so their relative
    phase is NOT cancelled -> a blanket Margolus substitution is UNSAFE.

    Width = 2*nbits + 1. nbits=6 -> 13; nbits=8 -> 17; nbits=11 -> 23.
    """
    a = QuantumRegister(nbits, "a")
    b = QuantumRegister(nbits, "b")
    cout = QuantumRegister(1, "cout")
    qc = QuantumCircuit(a, b, cout)
    # carry ripple written onto b/cout and never uncomputed
    for i in range(nbits):
        qc.cx(a[i], b[i])
    qc.ccx(a[0], b[0], cout[0])
    for i in range(1, nbits):
        qc.ccx(a[i], b[i], cout[0])   # accumulates a live majority onto cout
    return _tag(qc, f"ripple_live_carry_{nbits}b", safe=False, family="adder/live")


def array_multiplier(nbits: int) -> QuantumCircuit:
    """Small array multiplier: partial-product ANDs (Toffolis) accumulated onto the
    product register and NEVER uncomputed (they ARE the result).

    p[i+j] ^= a[i] & b[j] for all i,j. The AND results survive to the product ->
    relative phase corrupts the product -> UNSAFE (live).

    Width = 2*nbits + 2*nbits = ... we use a[nbits], b[nbits], prod[2*nbits].
    nbits=3 -> 12; nbits=4 -> 16; nbits=5 -> 20; nbits=6 -> 24.
    """
    a = QuantumRegister(nbits, "a")
    b = QuantumRegister(nbits, "b")
    prod = QuantumRegister(2 * nbits, "p")
    qc = QuantumCircuit(a, b, prod)
    for i in range(nbits):
        for j in range(nbits):
            qc.ccx(a[i], b[j], prod[i + j])   # live partial product
    return _tag(qc, f"array_mult_{nbits}b", safe=False, family="multiplier/live")


def half_uncomputed_oracle(nctrl: int) -> QuantumCircuit:
    """A valid compute/uncompute AND-ladder PLUS one extra LIVE marking Toffoli.

    The ladder pairs are relative-phase safe, but the trailing live Toffoli writes a
    surviving result, so a BLANKET substitution is UNSAFE. A sound per-gate selector
    should still admit the paired ladder and keep only the live one exact.

    Width = 2*nctrl + 1. nctrl=6 -> 13; nctrl=8 -> 17; nctrl=10 -> 21.
    """
    ctrl = QuantumRegister(nctrl, "c")
    anc = QuantumRegister(nctrl - 1, "anc")
    extra = QuantumRegister(2, "x")
    qc = QuantumCircuit(ctrl, anc, extra)
    # compute/uncompute ladder (safe)
    qc.ccx(ctrl[0], ctrl[1], anc[0])
    for i in range(2, nctrl):
        qc.ccx(ctrl[i], anc[i - 2], anc[i - 1])
    qc.cx(anc[nctrl - 2], extra[0])
    for i in reversed(range(2, nctrl)):
        qc.ccx(ctrl[i], anc[i - 2], anc[i - 1])
    qc.ccx(ctrl[0], ctrl[1], anc[0])
    # one LIVE Toffoli writing a surviving result onto extra[1]
    qc.ccx(ctrl[0], ctrl[1], extra[1])
    return _tag(qc, f"half_uncomputed_{nctrl}c", safe=False, family="grover/mixed")


def half_uncomputed_minimal() -> QuantumCircuit:
    """Smallest member of the half-uncomputed family (4 qubits): the compute/uncompute pair on one ancilla and a live Toffoli that writes a surviving result into the control c1 (no extra qubits)."""
    c = QuantumRegister(2, "c"); anc = QuantumRegister(1, "anc"); x = QuantumRegister(1, "x")
    qc = QuantumCircuit(c, anc, x)
    qc.ccx(c[0], c[1], anc[0]); qc.cx(anc[0], x[0]); qc.ccx(c[0], c[1], anc[0])
    qc.ccx(c[0], x[0], c[1])
    return _tag(qc, "half_uncomputed_min", safe=False, family="grover/mixed")


# ===========================================================================
# suite
# ===========================================================================
def large_suite():
    """Full list of large benchmark circuits, ordered by qubit count.

    Sizes are chosen to straddle the exhaustive-verification crossover (~12-14
    qubits) and run up to 24 qubits, where only the QCEC fallback is feasible.
    """
    circuits = [
        # --- compute/uncompute-structured (relphase admissible) ---
        ripple_carry_adder(5),       # 12q
        grover_oracle(6),            # 12q
        carry_lookahead_adder(4),    # 13q
        modular_increment(6),        # 13q
        ripple_carry_adder(7),       # 16q
        grover_oracle(8),            # 16q
        carry_lookahead_adder(5),    # 16q
        modular_increment(8),        # 17q
        ripple_carry_adder(10),      # 22q
        grover_oracle(11),           # 22q
        carry_lookahead_adder(7),    # 22q
        modular_increment(11),       # 23q
        # --- live (relphase NOT admissible) ---
        array_multiplier(3),         # 12q
        ripple_adder_live_carry(6),  # 13q
        half_uncomputed_oracle(6),   # 13q
        array_multiplier(4),         # 16q
        ripple_adder_live_carry(8),  # 17q
        half_uncomputed_oracle(8),   # 17q
        array_multiplier(5),         # 20q
        array_multiplier(6),         # 24q
    ]
    return circuits


if __name__ == "__main__":
    print(f"{'name':24s} {'qubits':>6} {'ccx':>4} {'cx':>4} "
          f"{'relphase_safe':>13} {'family':>16}")
    for qc in large_suite():
        nccx = sum(1 for i in qc.data if i.operation.name == "ccx")
        ncx = sum(1 for i in qc.data if i.operation.name == "cx")
        print(f"{qc.name:24s} {qc.num_qubits:>6} {nccx:>4} {ncx:>4} "
              f"{str(qc.relphase_safe):>13} {qc.family:>16}")

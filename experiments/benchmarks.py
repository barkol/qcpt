"""Benchmark suite of CCX/MCX circuits resembling real reversible workloads.

Every circuit is small enough (<= ~8-10 qubits) for EXACT verification by
QCPT's ExactEquivalenceVerifier.

The suite deliberately mixes two regimes for the relative-phase (Margolus / RCCX)
Toffoli substitution:

  * COMPUTE/UNCOMPUTE-structured circuits, where each CCX is later undone by an
    identical CCX (V ... V^dagger). Here a relative-phase Toffoli IS valid: the
    spurious relative phase introduced on the compute is cancelled on the
    uncompute, so the overall function is preserved.

  * "Live" circuits where Toffolis are NOT uncomputed -- their target feeds
    further logic / outputs. Here the relative-phase Toffoli is INVALID: the
    extra phase survives and silently corrupts the computed function.

Each generator returns a qiskit QuantumCircuit with a `.name` and an attribute
`relphase_safe` (bool) recording whether a blanket relative-phase substitution is
expected to preserve the function (ground truth for the safety experiment).
"""

from __future__ import annotations

from qiskit import QuantumCircuit, QuantumRegister


def _tag(qc: QuantumCircuit, name: str, safe: bool) -> QuantumCircuit:
    qc.name = name
    # ground-truth: does a BLANKET relphase substitution preserve the function?
    qc.relphase_safe = safe
    return qc


# --------------------------------------------------------------------------- #
# (a) reversible-arithmetic style: ripple-carry adders / controlled adders     #
# --------------------------------------------------------------------------- #
def ripple_carry_adder(nbits: int = 2) -> QuantumCircuit:
    """Cuccaro-style ripple-carry adder a += b built from CX/CCX.

    Carries are computed AND uncomputed (the classic MAJ / UnMAJ structure), so
    every CCX appears in a compute/uncompute pair -> relative-phase is SAFE.

    Layout: [c0, a0, b0, a1, b1, ..., c_out] ... we use a compact register.
    """
    # registers: a[n], b[n], carry c (1 ancilla), high carry z (1)
    a = QuantumRegister(nbits, "a")
    b = QuantumRegister(nbits, "b")
    c = QuantumRegister(1, "c")   # carry-in ancilla
    z = QuantumRegister(1, "z")   # carry-out
    qc = QuantumCircuit(a, b, c, z)

    def MAJ(ci, ai, bi):
        qc.cx(ai, bi)
        qc.cx(ai, ci)
        qc.ccx(ci, bi, ai)

    def UMA(ci, ai, bi):
        qc.ccx(ci, bi, ai)
        qc.cx(ai, ci)
        qc.cx(ci, bi)

    carries = [c[0]] + [a[i] for i in range(nbits)]  # reuse a[i] as running carry slot
    # Standard Cuccaro: forward MAJ chain, the high carry into z, then UMA chain.
    prev = c[0]
    for i in range(nbits):
        MAJ(prev, a[i], b[i])
        prev = a[i]
    qc.cx(prev, z[0])
    for i in reversed(range(nbits)):
        prev = a[i - 1] if i > 0 else c[0]
        UMA(prev, a[i], b[i])
    return _tag(qc, f"ripple_carry_adder_{nbits}b", safe=True)


def controlled_adder(nbits: int = 2) -> QuantumCircuit:
    """Controlled-add: ctrl-controlled b += a, CCX-heavy. Carries uncomputed.

    Built as a compute/uncompute network -> relative-phase SAFE.
    """
    ctrl = QuantumRegister(1, "ctrl")
    a = QuantumRegister(nbits, "a")
    b = QuantumRegister(nbits, "b")
    car = QuantumRegister(nbits, "car")  # carry ancillas, all uncomputed
    qc = QuantumCircuit(ctrl, a, b, car)

    # compute carries (ripple), conditionally add, then uncompute carries.
    # carry[i+1] = MAJ(a[i], b[i], carry[i]); we model a clean compute/uncompute.
    for i in range(nbits):
        qc.ccx(a[i], b[i], car[i])           # generate
        if i + 1 < nbits:
            qc.ccx(car[i], a[i + 1], car[i + 1])  # propagate-ish (still uncomputed)
    # controlled sum onto b
    for i in range(nbits):
        qc.ccx(ctrl[0], car[i], b[i])
    # uncompute carries (mirror)
    for i in reversed(range(nbits)):
        if i + 1 < nbits:
            qc.ccx(car[i], a[i + 1], car[i + 1])
        qc.ccx(a[i], b[i], car[i])
    # NOTE: although carries are uncomputed, the controlled-sum Toffolis
    # (ccx(ctrl, car[i], b[i])) read the carry while it is live and write a
    # surviving result onto b -- their relative phase is NOT cancelled, so a
    # BLANKET substitution is in fact unsafe here. This is exactly the subtle
    # case a count-greedy pass gets wrong; we therefore tag it UNSAFE.
    return _tag(qc, f"controlled_adder_{nbits}b", safe=False)


# --------------------------------------------------------------------------- #
# (b) Grover-oracle style multi-controlled blocks                              #
# --------------------------------------------------------------------------- #
def grover_oracle_mcx(nctrl: int = 4) -> QuantumCircuit:
    """Grover-style marking oracle: a multi-controlled-X flip realised by a
    Toffoli ladder onto ancillas (compute), the marking CX, then UNcompute.

    Standard MCX-via-ancilla: compute/uncompute -> relative-phase SAFE.
    """
    ctrl = QuantumRegister(nctrl, "c")
    anc = QuantumRegister(nctrl - 1, "anc")
    tgt = QuantumRegister(1, "t")
    qc = QuantumCircuit(ctrl, anc, tgt)

    # compute AND-ladder
    qc.ccx(ctrl[0], ctrl[1], anc[0])
    for i in range(2, nctrl):
        qc.ccx(ctrl[i], anc[i - 2], anc[i - 1])
    # mark
    qc.cx(anc[nctrl - 2], tgt[0])
    # uncompute AND-ladder (mirror)
    for i in reversed(range(2, nctrl)):
        qc.ccx(ctrl[i], anc[i - 2], anc[i - 1])
    qc.ccx(ctrl[0], ctrl[1], anc[0])
    return _tag(qc, f"grover_oracle_mcx{nctrl}", safe=True)


def grover_oracle_native_mcx(nctrl: int = 4) -> QuantumCircuit:
    """Grover oracle using a native MCXGate (no ancilla ladder).

    There are no CCX gates to substitute (the substitution pass targets ccx),
    so this circuit is trivially unchanged -> SAFE, and serves as a control.
    """
    ctrl = QuantumRegister(nctrl, "c")
    tgt = QuantumRegister(1, "t")
    qc = QuantumCircuit(ctrl, tgt)
    qc.h(tgt[0])
    qc.mcx(list(ctrl), tgt[0])
    qc.h(tgt[0])
    return _tag(qc, f"grover_native_mcx{nctrl}", safe=True)


# --------------------------------------------------------------------------- #
# (c) compute/uncompute-structured circuits (relative-phase VALID)             #
# --------------------------------------------------------------------------- #
def compute_uncompute_pair() -> QuantumCircuit:
    """Minimal V . CX . V^dagger where V is a single CCX. The phase cancels."""
    qc = QuantumCircuit(4)
    qc.ccx(0, 1, 3)
    qc.cx(3, 2)        # use the AND result
    qc.ccx(0, 1, 3)    # uncompute
    return _tag(qc, "compute_uncompute_pair", safe=True)


def nested_compute_uncompute() -> QuantumCircuit:
    """Two nested compute/uncompute scopes; all CCX paired -> SAFE."""
    a = QuantumRegister(2, "a")
    b = QuantumRegister(2, "b")
    anc = QuantumRegister(2, "anc")
    qc = QuantumCircuit(a, b, anc)
    qc.ccx(a[0], a[1], anc[0])         # compute 1
    qc.ccx(anc[0], b[0], anc[1])       # compute 2 (nested)
    qc.cx(anc[1], b[1])                # use
    qc.ccx(anc[0], b[0], anc[1])       # uncompute 2
    qc.ccx(a[0], a[1], anc[0])         # uncompute 1
    return _tag(qc, "nested_compute_uncompute", safe=True)


# --------------------------------------------------------------------------- #
# (d) circuits where Toffolis are NOT uncomputed (relative-phase INVALID)      #
# --------------------------------------------------------------------------- #
def live_and_chain() -> QuantumCircuit:
    """Chain of ANDs whose results survive to the output -- no uncompute.

    The relative phase on each |11..>-conditioned subspace survives, so a blanket
    relative-phase substitution changes the function -> UNSAFE.
    """
    qc = QuantumCircuit(5)
    qc.ccx(0, 1, 4)
    qc.ccx(2, 3, 4)   # result accumulates on wire 4; never uncomputed
    return _tag(qc, "live_and_chain", safe=False)


def phase_sensitive_oracle() -> QuantumCircuit:
    """Toffoli whose controls are in superposition and whose target is a phase
    kickback wire (|->). A relative-phase Toffoli flips the marked phase ->
    silently wrong oracle. Classic place where Margolus is INVALID without
    surrounding uncompute. UNSAFE."""
    ctrl = QuantumRegister(2, "c")
    tgt = QuantumRegister(1, "t")
    qc = QuantumCircuit(ctrl, tgt)
    qc.h(ctrl[0])
    qc.h(ctrl[1])
    qc.x(tgt[0]); qc.h(tgt[0])   # |->
    qc.ccx(ctrl[0], ctrl[1], tgt[0])  # phase-kickback marking, NOT uncomputed
    return _tag(qc, "phase_sensitive_oracle", safe=False)


def single_live_toffoli() -> QuantumCircuit:
    """A lone Toffoli used as a logical AND output. UNSAFE."""
    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    return _tag(qc, "single_live_toffoli", safe=False)


def half_uncomputed() -> QuantumCircuit:
    """Three Toffolis: one compute/uncompute pair (safe) PLUS one live Toffoli
    (unsafe). The presence of a single live Toffoli makes a BLANKET substitution
    unsafe, but a per-gate gate keeps only the safe pair. UNSAFE for naive."""
    a = QuantumRegister(2, "a")
    b = QuantumRegister(1, "b")
    anc = QuantumRegister(2, "anc")
    qc = QuantumCircuit(a, b, anc)
    # safe pair
    qc.ccx(a[0], a[1], anc[0])
    qc.cx(anc[0], b[0])
    qc.ccx(a[0], a[1], anc[0])
    # live toffoli onto anc[1] (never uncomputed) -> survives to output
    qc.ccx(a[0], b[0], anc[1])
    return _tag(qc, "half_uncomputed", safe=False)


def adder_then_live_carry() -> QuantumCircuit:
    """A 1-bit full-adder structure where the carry-out Toffoli is left live
    (not uncomputed) -- realistic when the carry is consumed downstream. UNSAFE."""
    a = QuantumRegister(1, "a")
    b = QuantumRegister(1, "b")
    cin = QuantumRegister(1, "cin")
    cout = QuantumRegister(1, "cout")
    sumr = QuantumRegister(1, "s")
    qc = QuantumCircuit(a, b, cin, cout, sumr)
    # sum = a xor b xor cin
    qc.cx(a[0], sumr[0]); qc.cx(b[0], sumr[0]); qc.cx(cin[0], sumr[0])
    # cout = majority(a,b,cin) via live Toffolis (NOT uncomputed)
    qc.ccx(a[0], b[0], cout[0])
    qc.ccx(a[0], cin[0], cout[0])
    qc.ccx(b[0], cin[0], cout[0])
    return _tag(qc, "adder_live_carry", safe=False)


# --------------------------------------------------------------------------- #
# suite                                                                        #
# --------------------------------------------------------------------------- #
def benchmark_suite():
    """Return the full list of (name -> QuantumCircuit) benchmark circuits."""
    circuits = [
        # (a) reversible arithmetic
        ripple_carry_adder(2),
        controlled_adder(2),
        # (b) Grover oracles
        grover_oracle_mcx(3),
        grover_oracle_mcx(4),
        grover_oracle_native_mcx(4),
        # (c) compute/uncompute (relphase valid)
        compute_uncompute_pair(),
        nested_compute_uncompute(),
        # (d) NOT uncomputed (relphase invalid)
        live_and_chain(),
        phase_sensitive_oracle(),
        single_live_toffoli(),
        half_uncomputed(),
        adder_then_live_carry(),
    ]
    return circuits


if __name__ == "__main__":
    for qc in benchmark_suite():
        nccx = sum(1 for inst in qc.data if inst.operation.name == "ccx")
        print(f"{qc.name:28s} qubits={qc.num_qubits:2d} ccx={nccx:2d} "
              f"relphase_safe={qc.relphase_safe}")

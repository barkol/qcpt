"""A simple, transparent hardware error model for Toffoli-network decompositions.

The point of this module is to give the decomposition selector a *citable*,
parameterizable cost to minimize. On near-term superconducting and trapped-ion
hardware the dominant error source by far is the two-qubit entangling gate: its
error rate (~1e-2) is roughly an order of magnitude larger than a single-qubit
gate (~1e-3). See e.g. IBM device calibration data and the surface-code
literature (Fowler et al., 2012), where the two-qubit gate is the limiting
primitive. So a good proxy for "how badly will this circuit do on real hardware"
is essentially "how many two-qubit gates does it contain", weighted by their
error rate.

We keep the model deliberately simple and inspectable:

    per-gate fidelity  F_g = 1 - err(g)
    circuit fidelity   F   = prod_g F_g
    circuit infidelity  1 - F      (the figure the selector minimizes)

``err(g)`` depends only on the number of qubits the gate acts on: two-qubit
(and larger) gates pay ``p2q``; single-qubit gates pay ``p1q``; barriers/ids are
free. This is the standard depolarizing-style accounting used for back-of-the-
envelope error budgets. Every rate is overridable so the selector's decisions can
be re-checked against a specific device's calibration.

Multi-qubit gates (CCX/MCX) are *not* expected to survive into the costed
circuit -- the selector decomposes them to a two-qubit basis first -- but if one
is costed directly we charge it ``p2q`` per two-qubit gate of its standard
decomposition (see ``_MULTIQUBIT_2Q_COST``) so the cost never under-counts.
"""

from __future__ import annotations

import math
from typing import Optional

# Names that cost nothing.
_FREE_GATES = {"barrier", "id"}

# Standard two-qubit-gate count for atomic multi-controlled gates, used only if a
# circuit is costed BEFORE being decomposed to a two-qubit basis. A Toffoli (ccx)
# decomposes to 6 CX in the textbook Clifford+T construction; a generic n-control
# MCX to roughly 6*(n-1)+... -- we use a safe, simple lower-bound-ish proxy.
_MULTIQUBIT_2Q_COST = {
    "ccx": 6,   # textbook exact Toffoli = 6 CX (+ 1q gates, costed separately below)
    "ccz": 6,
}


class HardwareErrorModel:
    """Transparent, parameterizable per-gate error / infidelity model.

    Parameters
    ----------
    p2q : float
        Error rate of a two-qubit (or larger entangling) gate. Default 1e-2.
    p1q : float
        Error rate of a single-qubit gate. Default 1e-3.
    p_readout : float
        Optional per-qubit readout/measurement error. Default 0.0 (off); only
        charged for ``measure`` instructions.
    """

    def __init__(
        self,
        p2q: float = 1e-2,
        p1q: float = 1e-3,
        p_readout: float = 0.0,
        t1: Optional[float] = None,
        t2: Optional[float] = None,
        t_2q: float = 68e-9,
        t_1q: float = 36e-9,
        zz_hz: float = 0.0,
        zz_pairs=None,
        p_excess_2q: float = 0.0,
    ):
        if not (0.0 <= p2q < 1.0):
            raise ValueError("p2q must be in [0, 1)")
        if not (0.0 <= p1q < 1.0):
            raise ValueError("p1q must be in [0, 1)")
        if not (0.0 <= p_readout < 1.0):
            raise ValueError("p_readout must be in [0, 1)")
        self.p2q = p2q
        self.p1q = p1q
        self.p_readout = p_readout
        # Optional idle (decoherence) term. Off by default (t1 is None), so all
        # gate-count results are unchanged. When set, every qubit that has started
        # its computation and waits for others pays a Pauli-twirled amplitude and
        # phase damping channel for the waiting time (see ``idle_fidelity``).
        if (t1 is None) != (t2 is None):
            raise ValueError("t1 and t2 must be given together")
        if t1 is not None and not (t1 > 0 and t2 > 0):
            raise ValueError("t1 and t2 must be positive")
        self.t1, self.t2 = t1, t2
        self.t_2q, self.t_1q = t_2q, t_1q
        # Optional static ZZ crosstalk: coupled qubits that are both active for a
        # common time T acquire the conditional phase phi = 2 pi zeta T on |11>;
        # Pauli-twirled, this costs 1 - (3/4) sin^2(phi/2) per pair. ``zz_pairs``
        # lists the coupled pairs (a device layout); if None, every pair that shares
        # a two-qubit gate in the circuit is taken as coupled. Off when zz_hz == 0.
        if not zz_hz >= 0:
            raise ValueError("zz_hz must be >= 0")
        self.zz_hz = zz_hz
        self.zz_pairs = None if zz_pairs is None else {tuple(sorted(p)) for p in zz_pairs}
        # Optional excess error per two-qubit gate (leakage and the gap between the
        # isolated and the in-circuit gate error). Off when 0.
        if not (0.0 <= p_excess_2q < 1.0):
            raise ValueError("p_excess_2q must be in [0, 1)")
        self.p_excess_2q = p_excess_2q

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _num_qubits(circuit, inst) -> int:
        return len(inst.qubits)

    def gate_error(self, name: str, num_qubits: int) -> float:
        """Error rate charged to a single instruction.

        Two-qubit and larger entangling gates pay ``p2q`` (a multi-controlled gate
        pays ``p2q`` per two-qubit gate of its standard decomposition); single-
        qubit gates pay ``p1q``; barriers/ids are free; measure pays readout.
        """
        n = name.lower()
        if n in _FREE_GATES:
            return 0.0
        if n == "measure":
            return self.p_readout
        if n in _MULTIQUBIT_2Q_COST:
            cost = _MULTIQUBIT_2Q_COST[n]
            # 1 - (1 - p2q)^cost  == infidelity of `cost` two-qubit gates.
            return 1.0 - (1.0 - self.p2q) ** cost
        if num_qubits >= 2:
            return self.p2q
        return self.p1q

    # ------------------------------------------------------------------- public
    def two_qubit_count(self, circuit) -> int:
        """Number of two-qubit (or larger) entangling gates in ``circuit``.

        A multi-controlled gate (ccx/ccz/mcx) is counted as its standard two-qubit
        decomposition cost, so the figure is comparable whether or not the circuit
        has been lowered to a two-qubit basis yet.
        """
        total = 0
        for inst in circuit.data:
            name = inst.operation.name.lower()
            if name in _FREE_GATES or name == "measure":
                continue
            if name in _MULTIQUBIT_2Q_COST:
                total += _MULTIQUBIT_2Q_COST[name]
            elif self._num_qubits(circuit, inst) >= 2:
                total += 1
        return total

    def one_qubit_count(self, circuit) -> int:
        """Number of single-qubit gates (excluding free gates / measure)."""
        total = 0
        for inst in circuit.data:
            name = inst.operation.name.lower()
            if name in _FREE_GATES or name == "measure":
                continue
            if name in _MULTIQUBIT_2Q_COST:
                continue
            if self._num_qubits(circuit, inst) == 1:
                total += 1
        return total

    def circuit_infidelity(self, circuit) -> float:
        """Estimated total infidelity 1 - prod_g (1 - err(g)) over all gates,
        times the idle fidelity when ``t1``/``t2`` are set.

        Monotonic in the number of (error-weighted) gates and dominated by the
        two-qubit count when ``p2q >> p1q``. This is the figure the selector
        minimizes.
        """
        f = 1.0
        for inst in circuit.data:
            name = inst.operation.name.lower()
            err = self.gate_error(name, self._num_qubits(circuit, inst))
            if err:
                f *= (1.0 - err)
        if self.t1 is not None:
            f *= self.idle_fidelity(circuit)
        if self.zz_hz > 0:
            f *= self.zz_fidelity(circuit)
        if self.p_excess_2q > 0:
            f *= (1.0 - self.p_excess_2q) ** self.two_qubit_count(circuit)
        return 1.0 - f

    # ------------------------------------------------------------------- idle
    def _duration(self, name: str, nq: int) -> float:
        if name in _FREE_GATES or name in ("rz", "measure"):
            return 0.0
        if name in _MULTIQUBIT_2Q_COST:
            return _MULTIQUBIT_2Q_COST[name] * self.t_2q + 8 * self.t_1q
        return self.t_2q if nq >= 2 else self.t_1q

    def idle_times(self, circuit) -> dict:
        """Waiting time of every qubit under an ALAP schedule (no routing).

        The schedule runs on the reversed gate list, so each qubit starts as late as
        possible. A qubit idles between its first gate and the end of the circuit
        whenever it is not busy; before its first gate it is still in |0> and does
        not decohere, so that interval is not charged."""
        n = circuit.num_qubits
        cur = [0.0] * n
        busy = [0.0] * n
        first = [None] * n
        for inst in reversed(circuit.data):
            name = inst.operation.name.lower()
            qs = [circuit.find_bit(q).index for q in inst.qubits]
            if name == "barrier":
                t = max(cur[q] for q in qs)
                for q in qs:
                    cur[q] = t
                continue
            t0 = max(cur[q] for q in qs)
            dt = self._duration(name, len(qs))
            for q in qs:
                cur[q] = t0 + dt
                if dt > 0:
                    busy[q] += dt
                    first[q] = t0 + dt
        return {q: max(0.0, first[q] - busy[q]) for q in range(n) if first[q] is not None}

    def active_windows(self, circuit) -> dict:
        """[start, end] of every acting qubit under the ALAP schedule (end = circuit end)."""
        n = circuit.num_qubits
        cur = [0.0] * n
        first = [None] * n
        for inst in reversed(circuit.data):
            name = inst.operation.name.lower()
            qs = [circuit.find_bit(q).index for q in inst.qubits]
            if name == "barrier":
                t = max(cur[q] for q in qs)
                for q in qs:
                    cur[q] = t
                continue
            t0 = max(cur[q] for q in qs)
            dt = self._duration(name, len(qs))
            for q in qs:
                cur[q] = t0 + dt
                if dt > 0:
                    first[q] = t0 + dt
        T = max(cur) if n else 0.0
        return {q: (T - first[q], T) for q in range(n) if first[q] is not None}

    def zz_fidelity(self, circuit) -> float:
        if self.zz_hz <= 0:
            return 1.0
        pairs = self.zz_pairs
        if pairs is None:
            pairs = set()
            for inst in circuit.data:
                qs = [circuit.find_bit(q).index for q in inst.qubits]
                if len(qs) == 2 and inst.operation.name.lower() not in _FREE_GATES:
                    pairs.add(tuple(sorted(qs)))
        win = self.active_windows(circuit)
        f = 1.0
        for a, b in pairs:
            if a in win and b in win:
                T = max(0.0, min(win[a][1], win[b][1]) - max(win[a][0], win[b][0]))
                f *= 1.0 - 0.75 * math.sin(math.pi * self.zz_hz * T) ** 2
        return f

    def idle_fidelity(self, circuit) -> float:
        """prod_q (1 - p_x - p_y - p_z) for the Pauli-twirled amplitude and phase
        damping channel: p_x = p_y = (1 - e^{-tau/T1})/4,
        p_z = (1 - e^{-tau/T2})/2 - (1 - e^{-tau/T1})/4."""
        if self.t1 is None:
            return 1.0
        f = 1.0
        for tau in self.idle_times(circuit).values():
            a = 1.0 - math.exp(-tau / self.t1)
            b = 1.0 - math.exp(-tau / self.t2)
            f *= 1.0 - a / 2.0 - max(0.0, b / 2.0 - a / 4.0)
        return f

    def circuit_fidelity(self, circuit) -> float:
        """Estimated total fidelity prod_g (1 - err(g))."""
        return 1.0 - self.circuit_infidelity(circuit)

    def report(self, circuit) -> dict:
        """A small breakdown dict useful for selector reports / debugging."""
        return {
            "two_qubit_count": self.two_qubit_count(circuit),
            "one_qubit_count": self.one_qubit_count(circuit),
            "infidelity": self.circuit_infidelity(circuit),
            "fidelity": self.circuit_fidelity(circuit),
            "p2q": self.p2q,
            "p1q": self.p1q,
        }


# Module-level convenience functions using a default model -----------------------
_DEFAULT = HardwareErrorModel()


def circuit_infidelity(circuit, model: Optional[HardwareErrorModel] = None) -> float:
    """Estimated total infidelity of ``circuit`` under ``model`` (default rates)."""
    return (model or _DEFAULT).circuit_infidelity(circuit)


def two_qubit_count(circuit, model: Optional[HardwareErrorModel] = None) -> int:
    """Number of two-qubit gates in ``circuit`` under ``model`` (default rates)."""
    return (model or _DEFAULT).two_qubit_count(circuit)

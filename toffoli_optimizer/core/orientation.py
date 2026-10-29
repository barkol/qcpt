"""Calibration-aware orientation of the admitted three-CNOT gadgets.

Both relative-phase gadgets (Margolus and its mirror image) apply CX(b,t), CX(a,t),
CX(b,t): one control-target pair carries two CNOTs and the other carries one. Which
control plays the role of ``b`` is a free choice as far as the gate count goes, but
not as far as the error budget goes once the two pairs sit on hardware edges of
different quality. This module picks, for every admitted gadget, the orientation
that puts the doubled CNOT on the cheaper pair, using per-pair costs derived from a
device calibration and a fixed qubit layout.

Soundness is never assumed. Swapping the roles changes the diagonal phase of a
gadget (from the branch abt=100 to abt=010 for the Margolus gadget, from abt=101 to
abt=011 for the mirror), so

* a standalone gadget admitted under condition (R) keeps the new orientation only if
  the swapped gadget again equals CCX on the local reachable subspace;
* the two gadgets of a structural pair (C) or of a certified window (W) are oriented
  together, because their phases cancel only as a pair;
* the whole oriented circuit must pass the same input-subspace certificate as the
  default selection (dense up to 12 qubits, QCEC with the pinned qubits as ancillas
  above). If it does not, the default orientation is returned.
"""

from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .subspace_check import check_on_subspace, gadget_unitary

Pair = Tuple[int, int]


# --------------------------------------------------------------------- pair costs
class PairCost:
    """Additive cost w(i, j) = -log(1 - eps) of one CX between LOGICAL qubits i, j."""

    def __init__(self, w: Dict[Pair, float], default: float):
        self._w = {tuple(sorted(k)): v for k, v in w.items()}
        self.default = default

    def __call__(self, i: int, j: int) -> float:
        return self._w.get(tuple(sorted((i, j))), self.default)


def _w(eps: float) -> float:
    return -math.log(max(1e-12, 1.0 - min(eps, 0.999999)))


def pair_cost_from_calibration(edge_error: Dict[Pair, float], layout: Sequence[int]) -> PairCost:
    """Per-logical-pair CX cost on a device.

    ``edge_error`` maps physical coupling pairs to two-qubit gate errors, ``layout``
    maps logical qubit k to physical qubit ``layout[k]``. A CX between physical
    qubits that are not coupled is costed as a SWAP chain along the cheapest path
    (three CX per hop except the last), which is what a router has to do."""
    adj: Dict[int, List[Tuple[int, float]]] = {}
    for (p, q), e in edge_error.items():
        w = _w(e)
        adj.setdefault(p, []).append((q, w))
        adj.setdefault(q, []).append((p, w))

    def path_cost(src: int, dst: int) -> float:
        # Dijkstra on per-edge weight; cost of the routed CX = 3*sum(w) - 2*w_last,
        # i.e. 3 CX per SWAP hop plus one CX on the final edge.
        dist = {src: 0.0}
        prev: Dict[int, Tuple[int, float]] = {}
        pq = [(0.0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == dst:
                break
            if d > dist.get(u, math.inf):
                continue
            for v, w in adj.get(u, []):
                nd = d + w
                if nd < dist.get(v, math.inf):
                    dist[v] = nd
                    prev[v] = (u, w)
                    heapq.heappush(pq, (nd, v))
        if dst not in dist:
            return math.inf
        w_last = prev[dst][1] if dst in prev else 0.0
        return 3.0 * dist[dst] - 2.0 * w_last

    n = len(layout)
    w = {}
    for i in range(n):
        for j in range(i + 1, n):
            w[(i, j)] = path_cost(layout[i], layout[j])
    finite = [v for v in w.values() if math.isfinite(v)]
    return PairCost(w, default=max(finite) if finite else 1.0)


def edge_errors_from_properties(props: dict, gate_names=("cz", "ecr", "cx")) -> Dict[Pair, float]:
    """Two-qubit gate errors per coupling pair from ``BackendProperties.to_dict()``."""
    out: Dict[Pair, float] = {}
    for g in props.get("gates", []):
        if g.get("gate") in gate_names and len(g.get("qubits", [])) == 2:
            e = [p["value"] for p in g["parameters"] if p["name"] == "gate_error"]
            if e and e[0] < 1.0:
                key = tuple(sorted(g["qubits"]))
                out[key] = min(out.get(key, 1.0), e[0])
    return out


# --------------------------------------------------------------------- orientation
def _gadget_cost(a: int, b: int, t: int, swap: bool, cost: PairCost) -> float:
    a_, b_ = (b, a) if swap else (a, b)
    return 2.0 * cost(b_, t) + cost(a_, t)


def _swapped_local_unitary(kind: str) -> np.ndarray:
    """Unitary, in the local order (a, b, t), of the gadget called with (b, a, t)."""
    from .decomposition_selector import append_relative_phase_ccx, append_relative_phase_ccx_mirror
    fn = append_relative_phase_ccx if kind == "relphase" else append_relative_phase_ccx_mirror
    return gadget_unitary(lambda qc, a, b, t: fn(qc, b, a, t))


def orient(selector, circuit, result: dict, cost: PairCost, pinned_zero=(), input_space="all_basis"):
    """Return (circuit, info): the calibration-oriented version of ``result`` from
    ``selector.select(circuit, ...)`` if it passes the certificate, else the input."""
    from .decomposition_selector import _U_CCX
    from .reach_local import local_reachable
    from .reachable_subspace import reachable_overapprox
    from .subspace_check import project_support

    actions = dict(result.get("actions", {}))
    rep = result["report"]
    info = {"oriented": 0, "groups": 0, "rejected_R": 0, "certificate": None, "kept_default": False}
    if not actions:
        return result["circuit"], info

    # Groups whose gadgets must share one orientation.
    groups: List[List[int]] = []
    in_group = set()
    for s_ in rep.get("applied_sites", []):
        g = [s_[0], s_[1]] if isinstance(s_, (tuple, list)) else [s_.compute_idx, s_.uncompute_idx]
        groups.append(g); in_group.update(g)
    for w in rep.get("window_pairs_admitted", []):
        g = list(w["pair"]); groups.append(g); in_group.update(g)
    singles = [i for i, a in actions.items() if a[0] in ("relphase", "relphase_m") and i not in in_group]

    qubits = {i: [circuit.find_bit(q).index for q in circuit.data[i].qubits] for i in actions}
    new = dict(actions)

    def choose(idx_list):
        c0 = sum(_gadget_cost(*qubits[i], False, cost) for i in idx_list)
        c1 = sum(_gadget_cost(*qubits[i], True, cost) for i in idx_list)
        return c1 < c0 - 1e-12

    for g in groups:
        if not all(actions.get(i, ("",))[0] in ("relphase", "relphase_m") for i in g):
            continue
        info["groups"] += 1
        if choose(g):
            for i in g:
                new[i] = (actions[i][0], "swap")
            info["oriented"] += len(g)

    pinned_zero = tuple(pinned_zero)
    for i in singles:
        info["groups"] += 1
        if not choose([i]):
            continue
        a, b, t = qubits[i]
        loc = local_reachable(circuit, i, (a, b, t), pinned_zero) if input_space == "all_basis" else None
        if loc is None:
            space = input_space
            if pinned_zero and input_space == "all_basis" and circuit.num_qubits <= 20:
                pm = sum(1 << q for q in pinned_zero)
                space = [x for x in range(1 << circuit.num_qubits) if not x & pm]
            from .reachable_subspace import local_overapprox
            loc = local_overapprox(circuit, i, (a, b, t), input_space=space)
        ok, _, _ = check_on_subspace(_swapped_local_unitary(actions[i][0]), _U_CCX, loc, 0.0)
        if ok:
            new[i] = (actions[i][0], "swap"); info["oriented"] += 1
        else:
            info["rejected_R"] += 1

    if info["oriented"] == 0:
        return result["circuit"], info

    cand = selector._build(circuit, new)
    ok = _certify(result["exact"], cand, result.get("cert_inputs"), pinned_zero)
    info["certificate"] = ok
    if not ok:
        info["kept_default"] = True
        return result["circuit"], info
    info["actions"] = new
    return cand, info


def _certify(exact, cand, cert_inputs, pinned_zero) -> bool:
    """Same whole-circuit certificate as the selector: dense on the input subspace up
    to 12 qubits, QCEC with pinned qubits as ancillas above. Fail closed."""
    from .subspace_check import certify_on_input_subspace, qcec_certify_on_subspace
    if cand.num_qubits <= 12 and cert_inputs is not None:
        from qiskit.quantum_info import Operator
        ok, _, _ = certify_on_input_subspace(Operator(exact).data, Operator(cand).data, cert_inputs, "subroutine")
        return bool(ok)
    ok, _ = qcec_certify_on_subspace(exact, cand, pinned_zero)
    return bool(ok)

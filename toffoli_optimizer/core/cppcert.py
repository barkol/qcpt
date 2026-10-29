"""Whole-circuit certificate on the input subspace computed by the C++ core (``qcpt certify``).

The columns of both circuits for the given basis inputs are propagated one by one in cache-resident state vectors on all
cores; the result is the Frobenius norm of A - e^{i th} B over the columns with th from the first column. If that norm is below
``atol`` the circuits are equal up to ONE global phase on the input subspace (an upper bound of the operator-norm deviation).
Anything else (binary missing, unsupported gate, a norm above ``atol``) returns None: the caller then uses the numpy certificate."""
from __future__ import annotations

import os
import subprocess
from typing import Optional, Sequence

_NUMPY_MAX_QUBITS = int(os.environ.get("QCPT_NUMPY_CERT_QUBITS", "9"))      # up to this size the numpy path is faster than starting the C++ process
_NUMPY_MAX_WORK = float(os.environ.get("QCPT_NUMPY_CERT_WORK", "1e8"))     # above this much work the C++ core (threads) is faster
_BIN = os.environ.get("QCPT_BIN") or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "cpp", "qcpt")


def available() -> bool:
    return os.environ.get("QCPT_CERT_CPP", "1") != "0" and os.access(_BIN, os.X_OK)


def _serialize(circuit, ids, mats):
    from .subspace_check import _gate_matrix
    lines = []
    for inst in circuit.data:
        op = inst.operation
        if op.name == "barrier":
            continue
        G = _gate_matrix(op)
        key = id(G)
        if key not in ids:
            if G.shape[0] > 64:
                raise ValueError("gate too wide")
            ids[key] = len(mats)
            k = G.shape[0].bit_length() - 1
            mats.append(f"{k} " + " ".join(f"{float(z.real)!r} {float(z.imag)!r}" for z in G.ravel()))
        qs = [circuit.find_bit(q).index for q in inst.qubits]
        lines.append(f"{ids[key]} " + " ".join(map(str, qs)))
    return f"{len(lines)} {float(getattr(circuit, 'global_phase', 0.0))!r}\n" + "\n".join(lines)


def _numpy_compare(a, b, cols, atol, reject):
    """Same quantity as the C++ core, for small circuits in numpy (no process, no serialisation)."""
    import numpy as np
    from .subspace_check import _columns_serial
    cols = list(cols)
    A = _columns_serial(a, cols)
    B = _columns_serial(b, cols)
    ov = np.vdot(A[:, 0], B[:, 0])
    ph = ov / abs(ov) if abs(ov) > 1e-15 else 1.0
    fro = float(np.linalg.norm(B - ph * A))
    if fro != fro:
        return None, None
    return ("equal" if fro <= atol else "different" if fro > reject else None), fro


def compare_up_to_phase(a, b, cols: Sequence[int], atol: float = 1e-8, reject: float = 1e-5):
    """(status, fro): 'equal' if the Frobenius deviation is <= atol, 'different' if it is > reject, otherwise (and when the C++
    core is not usable) status None, i.e. undecided -> use the numpy certificate."""
    if a.num_qubits != b.num_qubits or a.num_qubits > 24 or not cols:
        return None, None
    if os.environ.get("QCPT_CERT_CPP", "1") == "0":              # switch off every fast certificate (the numpy certificate decides)
        return None, None
    work = (len(a.data) + len(b.data)) * (1 << a.num_qubits) * len(cols)            # gate applications on amplitudes
    if a.num_qubits <= _NUMPY_MAX_QUBITS and (work <= _NUMPY_MAX_WORK or not available()):
        try:
            return _numpy_compare(a, b, cols, atol, reject)
        except Exception:
            return None, None
    if not available():
        return None, None
    try:
        ids, mats = {}, []
        ca, cb = _serialize(a, ids, mats), _serialize(b, ids, mats)
        n = a.num_qubits
        text = f"{n} {len(cols)}\n{' '.join(map(str, cols))}\n{len(mats)}\n" + "\n".join(mats) + "\n" + ca + "\n" + cb + "\n"
        out = subprocess.run([_BIN, "certify"], input=text, capture_output=True, text=True, timeout=3600).stdout.split()
        if len(out) < 8 or out[0] != "FRO":
            return None, None
        fro = float(out[1])
    except Exception:
        return None, None
    if fro != fro:
        return None, None
    return ("equal" if fro <= atol else "different" if fro > reject else None), fro


def certify_equal_up_to_phase(a, b, cols: Sequence[int], atol: float = 1e-8) -> Optional[float]:
    """Frobenius deviation if it is <= atol (certified), else None."""
    st, fro = compare_up_to_phase(a, b, cols, atol)
    return fro if st == "equal" else None


# ---------------------------------------------------------------------------------------------------------------------------
# Monte Carlo certificate (opt-in: ``ErrorBudgetSelector(certificate="monte_carlo")``)
#
# K random vectors g_i (independent complex Gaussian components, unit variance, supported on the input subspace of dimension d) are
# propagated through both circuits. With the phase ph from g_0 let T_i = ||(B - ph A) g_i||^2 and M = (B - ph A) P (P = projector onto the
# input subspace). E T_i = ||M||_F^2 and T_i = sum_j lam_j e_j with e_j ~ Exp(1) independent, sum_j lam_j = ||M||_F^2, hence the
# Chernoff bound  P(T_i <= s ||M||_F^2) <= s e^{1-s}  (inf over theta of e^{theta s mu} / prod(1 + theta lam_j), and
# prod(1 + theta lam_j) >= 1 + theta mu). Accepting when T_i <= s eps^2 for all i = 1..K-1 therefore certifies
# min_theta ||A - e^{i theta} B||_F <= eps on the input subspace except with probability delta <= (s e^{1-s})^{K-1}.
# A single vector with ||(B - ph A) g||^2 > r_diff ||g||^2 is a deterministic counterexample (operator norm above sqrt(r_diff)).
# Anything in between is undecided (None): the caller then uses the deterministic certificate.
# ---------------------------------------------------------------------------------------------------------------------------
import math

MC_DEFAULT_K = 24
MC_DEFAULT_S = 0.1
_MC_REJECT = 1e-5


def mc_failure_probability(k: int, s: float = MC_DEFAULT_S) -> float:
    """Upper bound on the probability of accepting a pair whose Frobenius deviation on the input subspace exceeds eps."""
    return float((s * math.exp(1.0 - s)) ** (k - 1))


def mc_compare(a, b, cols, k: int = MC_DEFAULT_K, seed: Optional[int] = None, eps: float = 1e-8, s: float = MC_DEFAULT_S):
    """(status, info) with status 'equal' (up to one global phase on span{|x>: x in cols}, Frobenius deviation <= eps, failure
    probability info['failure_probability']), 'different' (deterministic counterexample) or None (undecided)."""
    if a.num_qubits != b.num_qubits or not (len(cols) if not isinstance(cols, range) else True) or k < 2:
        return None, {"reason": "unsupported input"}
    n = a.num_qubits
    if seed is None:
        import secrets
        seed = secrets.randbits(48)
    full = isinstance(cols, range) and len(cols) == (1 << n)
    support = [] if full else sorted(set(int(x) for x in cols))
    d = (1 << n) if full else len(support)
    T = nr = None
    if available() and 10 <= n <= 26:                                   # smaller vectors: numpy, no process start
        try:
            ids, mats = {}, []
            ca, cb = _serialize(a, ids, mats), _serialize(b, ids, mats)
            text = (f"{n} {k} {int(seed)} {len(support)}\n{' '.join(map(str, support))}\n{len(mats)}\n" + "\n".join(mats) + "\n" + ca + "\n" + cb + "\n")
            out = subprocess.run([_BIN, "certify_mc"], input=text, capture_output=True, text=True, timeout=3600).stdout.split("\n")
            if out and out[0].startswith("MC"):
                rows = [list(map(float, r.split())) for r in out[1:1 + k]]
                T = [r[0] for r in rows]
                nr = [r[1] for r in rows]
        except Exception:
            T = nr = None
    if T is None and n <= 24:                                           # numpy fallback (no C++ binary)
        try:
            import numpy as np
            from .subspace_check import _columns_serial
            rng = np.random.default_rng(int(seed))
            G = np.zeros((1 << n, k), dtype=complex)
            idx = np.arange(1 << n) if full else np.array(support, dtype=np.int64)
            G[idx, :] = (rng.standard_normal((len(idx), k)) + 1j * rng.standard_normal((len(idx), k))) / math.sqrt(2.0)
            try:                                                        # tiny tensordots: one BLAS thread is much faster than a thread pool
                from threadpoolctl import threadpool_limits
                _lim = threadpool_limits(limits=1)
            except Exception:
                _lim = None
            Av, Bv = _columns_serial(a, [], init=G), _columns_serial(b, [], init=G)
            if _lim is not None:
                _lim.restore_original_limits()
            ov = np.vdot(Av[:, 0], Bv[:, 0])
            ph = ov / abs(ov) if abs(ov) > 1e-15 else 1.0
            R = Bv - ph * Av
            T = [float(np.vdot(R[:, i], R[:, i]).real) for i in range(k)]
            nr = [float(np.vdot(G[:, i], G[:, i]).real) for i in range(k)]
        except Exception:
            T = nr = None
    if T is None:
        return None, {"reason": "not available"}
    info = {"method": "monte_carlo", "vectors": k, "seed": int(seed), "subspace_dim": d, "s": s,
            "failure_probability": mc_failure_probability(k, s), "max_T": max(T[1:]), "eps": eps}
    if any(t != t for t in T):
        return None, info
    if any(T[i] > _MC_REJECT ** 2 * nr[i] for i in range(k)):
        return "different", info                                        # deterministic counterexample
    if max(T[1:]) <= s * eps * eps:
        return "equal", info
    return None, info

"""Window search on the C++ core: one `qcpt beam` process per window (n <= 4 qubits, 2^n inputs, exact Z8 phases up to a global phase).
Same problem and result format as resynth.bidir_search(k, range(2**k), perm, max_cost, beam, lam, depth_side, lib=library(k), target_phase=ph)."""
import os, subprocess

_BIN = os.environ.get("QCPT_BIN") or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "cpp", "qcpt")


def available():
    return os.environ.get("QCPT_WIN_CPP", "1") != "0" and os.access(_BIN, os.X_OK)


def window_search(k, perm, phase, max_cost, beam=3000, lam=0.6, depth_side=8):
    """Returns (cost, [(name, qubits), ...]) or None. The C++ side certifies the circuit against (perm, phase) before it reports it."""
    m = 1 << k
    text = (f"{k} {m}\n{' '.join(map(str, range(m)))}\n{' '.join(str(int(v)) for v in perm)}\n1\n{' '.join(str(int(v)) for v in phase)}\n"
            f"{max_cost} {beam} {lam} {depth_side} 0 0 0 1 0\nlib 0\n")
    out = subprocess.run([_BIN, "beam"], input=text, capture_output=True, text=True, check=True).stdout.split("\n")
    if not out or out[0].strip() == "NONE": return None
    cost = int(out[0].split()[1]); gates = [(t[1], tuple(int(v) for v in t[2:])) for t in (ln.split() for ln in out[1:]) if t and t[0] == "G"]
    if "CERT ok" not in out: raise RuntimeError("C++ certificate failed for a window")
    return cost, gates


_DUMP = {}


def _dump(n):
    """Gates of the C++ library of n qubits in its order: [(name, args)]."""
    if n not in _DUMP:
        out = subprocess.run([_BIN, "--dump-lib", str(n)], capture_output=True, text=True, check=True).stdout.splitlines()
        _DUMP[n] = [(ln.split("|")[0].split()[0], tuple(int(x) for x in ln.split("|")[0].split()[1:])) for ln in out if ln.strip()]
    return _DUMP[n]


def allow_indices(n, coupling):
    """Indices (C++ library order) of the gates whose required couplers all exist: the same gates as resynth.library(n, coupling=coupling). coupling = None: all gates."""
    from . import resynth as RS
    keys = {(g[0], tuple(g[1])) for g in RS.library(n, coupling=coupling) if not g[0].startswith("mined")}
    return [i for i, k in enumerate(_dump(n)) if k in keys]


def beam_search(n, inputs, target, phase, max_cost, beam=6000, lam=0.6, depth_side=8, diversity=0.0, noise=0.0, seed=0, nthreads=1, allow=None, timeout=None):
    """One `qcpt beam` run on the valid inputs only (target: image of each input, phase: Z8 phase of each, global phase free), with an optional coupler filter. Returns (cost, [(name, qubits)]) or None."""
    m = len(inputs)
    text = (f"{n} {m}\n{' '.join(map(str, inputs))}\n{' '.join(str(int(v)) for v in target)}\n1\n{' '.join(str(int(v)) for v in phase)}\n"
            f"{max_cost} {beam} {lam} {depth_side} {diversity} {noise} {seed} {nthreads} 0\nlib 0\n" + (f"allow {len(allow)} {' '.join(map(str, allow))}\n" if allow is not None else ""))
    out = subprocess.run([_BIN, "beam"], input=text, capture_output=True, text=True, check=True, timeout=timeout).stdout.split("\n")
    if not out or out[0].strip() == "NONE": return None
    gates = [(t[1], tuple(int(v) for v in t[2:])) for t in (ln.split() for ln in out[1:]) if t and t[0] == "G"]
    if "CERT ok" not in out: raise RuntimeError("C++ certificate failed")
    return int(out[0].split()[1]), gates

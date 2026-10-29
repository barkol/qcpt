"""The tolerance-0 fast path of certify_on_input_subspace agrees with the full
phase minimisation on equal, phase-shifted, relative-phase and perturbed cases."""
import numpy as np
from toffoli_optimizer.core import subspace_check as S


def _slow(U_ex, U_sel, inputs):
    cols = sorted(set(inputs))
    dev, _ = S._min_scalar_phase_opnorm(U_sel[:, cols], U_ex[:, cols])
    return dev <= 1e-8


def _rand_unitary(n, rng):
    q, r = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    return q * (np.diag(r) / np.abs(np.diag(r)))


def test_fast_path_matches_full_minimisation():
    rng = np.random.default_rng(7)
    for trial in range(40):
        n = 16
        U = _rand_unitary(n, rng)
        inputs = sorted(rng.choice(n, size=rng.integers(1, n + 1), replace=False).tolist())
        kind = trial % 4
        if kind == 0:
            V = U.copy()
        elif kind == 1:
            V = np.exp(1j * rng.uniform(0, 2 * np.pi)) * U
        elif kind == 2:
            D = np.diag(np.exp(1j * rng.uniform(0, 2 * np.pi, size=n)))
            V = U @ D
        else:
            V = U @ _rand_unitary(n, rng) if rng.random() < 0.5 else U + 1e-6 * rng.normal(size=(n, n))
        fast = S.certify_on_input_subspace(U, V, inputs, "subroutine")[0]
        assert fast == _slow(U, V, inputs), (trial, kind)


if __name__ == "__main__":
    test_fast_path_matches_full_minimisation(); print("ok")

"""Noisy density-matrix validation of the error-budget reduction.

The selector ranks decompositions by a first-order additive two-qubit-infidelity
surrogate. This experiment checks that the surrogate's ranking translates into a
real fidelity gain on hardware-like noise: for representative circuits we build
the all-exact decomposition and our gated (verified) output, run both under a
depolarizing noise model in Qiskit Aer (density matrix), and compare the state
fidelity to the ideal output. Because our substitutions are verified sound, both
circuits share the SAME ideal target, so a higher fidelity is a genuine gain.

Run:
    pip install -e ".[experiments]"
    python experiments/noisy_sim.py
"""
from __future__ import annotations

from _paths import EXPERIMENTS_DIR  # noqa: F401  (must be first)

from qiskit import transpile
from qiskit.quantum_info import Statevector, state_fidelity, DensityMatrix
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

import baselines as B
import benchmarks

P2Q, P1Q = 1e-2, 1e-3   # matches the paper's representative error model


def noise_model():
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(P2Q, 2), ["cx"])
    nm.add_all_qubit_quantum_error(
        depolarizing_error(P1Q, 1),
        ["u", "u3", "rz", "sx", "x", "ry", "h"])
    return nm


def main():
    sim = AerSimulator(method="density_matrix", noise_model=noise_model())
    suite = {c.name: c for c in benchmarks.benchmark_suite()}
    targets = ["controlled_adder_2b", "grover_oracle_mcx4",
               "nested_compute_uncompute"]
    print(f"depolarizing noise p2q={P2Q}, p1q={P1Q}")
    print(f"{'circuit':24s} {'F_exact':>8} {'F_ours':>8} {'gain':>8}")
    for name in targets:
        c = suite[name]
        ideal = Statevector.from_instruction(c)
        fids = []
        for fn in (B.m_exact_only, B.m_ours):
            qc = transpile(fn(c), sim, optimization_level=0)
            qc.save_density_matrix()
            dm = DensityMatrix(sim.run(qc).result().data(0)["density_matrix"])
            fids.append(state_fidelity(ideal, dm))
        print(f"{name:24s} {fids[0]:8.4f} {fids[1]:8.4f} {fids[1]-fids[0]:+8.4f}")


if __name__ == "__main__":
    main()

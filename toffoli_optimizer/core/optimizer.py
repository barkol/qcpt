"""Facade that re-exports the public classes of the certified pass."""

# Import the exact equivalence verifier (proves a rewrite preserves the function)
from .equivalence_verifier import ExactEquivalenceVerifier

# Import the Toffoli-COUNT reducer (reduces the NUMBER of atomic CCX/MCX gates,
# keeping each multi-controlled gate atomic rather than decomposing it).
from .toffoli_count_reducer import ToffoliCountReducer

# Error-budget-aware, context-verified Toffoli decomposition selection.
# - HardwareErrorModel: transparent per-gate infidelity / two-qubit-count model.
# - find_relative_phase_safe_sites: detect compute/uncompute pairs where the cheap
#   relative-phase (Margolus) Toffoli is admissible.
# - ErrorBudgetSelector: choose per-Toffoli decompositions minimizing the error
#   budget, applying relative-phase gadgets ONLY at exact-verified sites.
from .error_model import HardwareErrorModel
from .context_analysis import find_relative_phase_safe_sites, RelativePhaseSite
from .decomposition_selector import ErrorBudgetSelector

# Reachable-subspace + phase-observability admissibility (the sound, more permissive
# criterion that admits standalone relative-phase substitutions whose phase is
# provably unobservable on the states that actually reach the gate).
from .reachable_subspace import reachable_basis_states, reachable_overapprox
from .phase_observability import is_phase_unobservable, default_affected_qubits

__all__ = ['ExactEquivalenceVerifier',
           'ToffoliCountReducer',
           'HardwareErrorModel', 'find_relative_phase_safe_sites', 'RelativePhaseSite',
           'ErrorBudgetSelector',
           'reachable_basis_states', 'reachable_overapprox',
           'is_phase_unobservable', 'default_affected_qubits']

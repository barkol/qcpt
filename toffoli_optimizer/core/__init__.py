"""Core components of QCPT: admissibility analysis, selection, verification and resynthesis."""

from .decomposition_selector import ErrorBudgetSelector  # noqa: F401
from .error_model import HardwareErrorModel  # noqa: F401

__version__ = "2.0.0"
__author__ = "Karol Bartkiewicz and Patrycja Tulewicz"

__all__ = ["ErrorBudgetSelector", "HardwareErrorModel"]

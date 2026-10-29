"""QCPT: certified, context-dependent Toffoli decompositions with error accounting."""

__version__ = "2.0.0"
__author__ = "Karol Bartkiewicz and Patrycja Tulewicz"

from toffoli_optimizer.core.decomposition_selector import ErrorBudgetSelector  # noqa: F401
from toffoli_optimizer.core.error_model import HardwareErrorModel  # noqa: F401

# Environment variables of the compiler are named QCPT_* (QCPT_GPU, QCPT_NPROC, QCPT_BIN, ...). The old names QCT_*
# (before the rename of QCT to QCPT) are deprecated aliases: once at import, every QCT_X that is set is copied to QCPT_X.
import os as _os
for _k in [k for k in _os.environ if k.startswith("QCT_")]:
    _os.environ.setdefault("QCPT_" + _k[4:], _os.environ[_k])
del _os

__all__ = ["ErrorBudgetSelector", "HardwareErrorModel"]

"""Path setup for reproducibility scripts.

After ``pip install -e .`` in the repo root, ``import toffoli_optimizer`` works
directly with no path hacks. This shim is a fallback for users who prefer to
run scripts from a clean clone without installing.

Every experiment script imports this module first::

    from _paths import EXPERIMENTS_DIR  # noqa: F401  (must be first)

which (a) makes ``toffoli_optimizer`` importable, (b) puts the ``experiments/``
directory on ``sys.path`` so sibling modules (``benchmarks``, ``naive_relphase``)
resolve, and (c) exposes ``EXPERIMENTS_DIR`` for writing result artefacts next
to the script that produced them.
"""

from __future__ import annotations

import sys
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENTS_DIR.parent

try:
    import toffoli_optimizer  # noqa: F401
except ImportError:
    sys.path.insert(0, str(REPO_ROOT))
    import toffoli_optimizer  # noqa: F401

if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

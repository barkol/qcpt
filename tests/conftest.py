"""Shared pytest configuration.

Ensures the repo root (which contains the ``toffoli_optimizer`` package) is
on ``sys.path`` even when tests are invoked without ``pip install -e .``.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

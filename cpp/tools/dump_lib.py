#!/usr/bin/env python3
"""Prints the Python reference library, one gate per line: name args | cost | perm | phase.
The C++ core (qcpt --dump-lib N [extra]) must reproduce it line by line; see tests/check_library.sh."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
from toffoli_optimizer.core import resynth as RS
n = int(sys.argv[1]); extra = len(sys.argv) > 2 and sys.argv[2] == "extra"
for g in RS.library(n, extra_signs=extra):
    print(g[0], " ".join(map(str, g[1])), "|", g[2], "|", " ".join(map(str, g[3])), "|", " ".join(str(int(v)) for v in g[4]))

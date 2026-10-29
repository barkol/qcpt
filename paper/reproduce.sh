#!/bin/bash
# Rebuild the tables, the figure data, the figures and the numbers of the paper from the
# stored experiment outputs. Stops at the first error (no partially updated outputs).
#
#   paper/reproduce.sh            assemble from stored data (seconds; figures need LaTeX)
#   paper/reproduce.sh compute    also recompute the 12-circuit semantics, attribution and
#                                 interference data with the code of this checkout (minutes)
#
# The experiment drivers themselves (hours) are listed in paper/README.md, section 3.
set -euo pipefail
P="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"
export PYTHONPATH="$P/..${PYTHONPATH:+:$PYTHONPATH}"
cd "$P"

echo "== 0. stored copies vs experiment outputs"
"$PY" check_provenance.py

if [ "${1:-}" = compute ]; then
  echo "== 1. recompute 12-circuit data (semantics, interference, attribution)"
  "$PY" tables/semantics_data.py
  "$PY" tables/interference_data.py
  "$PY" tables/attribution_data.py
fi

echo "== 2. RevLib aggregate from per-circuit results (experiments/revlib/wyniki*.jsonl)"
"$PY" tables/revlib_data.py > /dev/null

echo "== 3. tables"
for t in semantics mirror mirror_aachen revlib; do "$PY" tables/make_tab_$t.py > /dev/null; echo "tables/tab_$t.tex"; done

echo "== 4. figure data and figures"
"$PY" figures/figures_data.py
if [ "${SKIP_FIGURES:-0}" = 1 ]; then echo "SKIP_FIGURES=1: PDFs not rebuilt"; else "$PY" figures/make_figures.py; fi

echo "== 4b. map of the failed couplers (probe of 29 October 2026)"
"$PY" figures/probe_map_data.py
if [ "${SKIP_FIGURES:-0}" != 1 ]; then "$PY" figures/make_probe_map.py; fi

echo "== 5. numbers of the text"
"$PY" numbers/paper_numbers.py > /dev/null
echo "numbers/paper_numbers.json"
echo "done; 'git status paper/' shows any output that differs from the stored version"

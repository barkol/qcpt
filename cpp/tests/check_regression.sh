#!/bin/bash
# Regression test of the search core. Usage: tests/check_regression.sh [path/to/qcpt]   (default ./qcpt, run from cpp/)
#  1. beam searches reproduce the golden outputs byte for byte (same costs, same circuits, same certificate line);
#  2. a race of a beam and a template (needs data/db3.tab) stops soon after the beam wins: the template must react to cancellation (limit RACE_MAX_S, default 8 s;
#     before cancellation reached the inner loops of the template search it took about 15 s).
QCPT=${1:-./qcpt}; HERE=$(cd "$(dirname "$0")" && pwd); G=$HERE/golden; DB=$HERE/../data/db3.tab; RACE_MAX_S=${RACE_MAX_S:-8}; fail=0
[ -x "$QCPT" ] || { echo "no executable $QCPT"; exit 2; }
for f in "$G"/beam_*.in; do
  b=$(basename "$f" .in); if "$QCPT" beam < "$f" | cmp -s - "$G/$b.out"; then echo "ok    $b"; else echo "FAIL  $b: output differs from the golden file"; fail=1; fi
done
t0=$(date +%s%N); out=$(QCPT_DB=$DB "$QCPT" race < "$G/race_cancel_mcx3.in"); t1=$(date +%s%N); ms=$(( (t1 - t0) / 1000000 ))
if echo "$out" | grep -q "^RESULT 12 " && [ "$ms" -lt $((RACE_MAX_S * 1000)) ]; then echo "ok    race_cancel_mcx3 (${ms} ms, limit $((RACE_MAX_S * 1000)) ms)"; else echo "FAIL  race_cancel_mcx3: ${ms} ms (limit $((RACE_MAX_S * 1000)) ms), result: $(echo "$out" | grep RESULT)"; fail=1; fi
[ $fail -eq 0 ] && echo "all regression checks passed" || echo "regression checks FAILED"; exit $fail

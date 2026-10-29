# Bounded-approximate-on-reachable: measured numbers

Hardware error model: p2q=1e-2, p1q=1e-3 (defaults).

Benchmark `guard_chain(k)`: 2k Toffolis on 2k+2 qubits. Half are guard-controlled (control 0 is pinned to |0> by the input domain, so the Toffoli is identity on every reachable input); half are free-controlled (genuinely active, must not be dropped). Input domain restricts qubit 0 to |0>.


## guard_chain(k=1) -- 4 qubits, 2 Toffolis

| variant | 2q gates | infidelity | certified budget | drops fired | eps spent | verified |
|---|---|---|---|---|---|---|
| exact-only | 12 | 0.12944 | 0.12944 | 0 | 0 | True |
| relphase-only | 6 | 0.066025 | 0.066025 | 0 | 0 | True |
| +approx eps=0 | 3 | 0.033576 | 0.033576 | 1 | 0 | True |
| +approx eps=0.1 | 3 | 0.033576 | 0.033576 | 1 | 0 | True |
| +approx eps=1.41 | 3 | 0.033576 | 0.033576 | 1 | 0 | True |

## guard_chain(k=2) -- 6 qubits, 4 Toffolis

| variant | 2q gates | infidelity | certified budget | drops fired | eps spent | verified |
|---|---|---|---|---|---|---|
| exact-only | 24 | 0.24212 | 0.24212 | 0 | 0 | True |
| relphase-only | 12 | 0.12769 | 0.12769 | 0 | 0 | True |
| +approx eps=0 | 6 | 0.066025 | 0.066025 | 2 | 0 | True |
| +approx eps=0.1 | 6 | 0.066025 | 0.066025 | 2 | 0 | True |
| +approx eps=1.41 | 6 | 0.066025 | 0.066025 | 2 | 0 | True |

## guard_chain(k=3) -- 8 qubits, 6 Toffolis

| variant | 2q gates | infidelity | certified budget | drops fired | eps spent | verified |
|---|---|---|---|---|---|---|
| exact-only | 36 | 0.34021 | 0.34021 | 0 | 0 | True |
| relphase-only | 18 | 0.18529 | 0.18529 | 0 | 0 | True |
| +approx eps=0 | 9 | 0.097385 | 0.097385 | 3 | 0 | True |
| +approx eps=0.1 | 9 | 0.097385 | 0.097385 | 3 | 0 | True |
| +approx eps=1.41 | 9 | 0.097385 | 0.097385 | 3 | 0 | True |

## rarely_set -- 3 qubits, 1 Toffoli, |11> branch fires on 1 of 4 reachable inputs

| variant | 2q gates | infidelity | certified budget | drops fired | eps spent | verified |
|---|---|---|---|---|---|---|
| exact-only | 6 | 0.066959 | 0.066959 | 0 | 0 | True |
| relphase-only | 3 | 0.033576 | 0.033576 | 0 | 0 | True |
| +approx eps=0 | 3 | 0.033576 | 0.033576 | 0 | 0 | True |
| +approx eps=0.1 | 3 | 0.033576 | 0.033576 | 0 | 0 | True |
| +approx eps=1.41 | 3 | 0.033576 | 0.033576 | 0 | 0 | True |

Note: the control-drop NEVER fires here at eps in {0, 0.1, ~sqrt(2)-} -- a single firing basis input costs the full sqrt(2) deviation, which no eps below sqrt(2) tolerates. 'Rare' does not buy a small eps under basis-state reachability; see the soundness caveat below.


## Honest read of the numbers

* The bounded-approximate-on-reachable path DOES deliver a real budget gain beyond
  relative-phase, BUT the entire gain observed here comes from the eps == 0
  (exact-on-reachable) control-drop, not from any eps > 0 tolerance. On
  `guard_chain`, every guard-controlled Toffoli has a control provably |0> on the
  restricted reachable subspace, so dropping it to identity is EXACT (deviation 0).
  That halves the two-qubit count again on top of relative-phase (e.g. k=3: exact 36
  -> relphase 18 -> +approx 9), and it certifies.

* eps > 0 added NOTHING on these benchmarks: the rows for eps = 0, 0.1, and
  ~sqrt(2) are identical, and `eps spent` is 0.0 everywhere. This is not a tuning
  accident -- it is structural. The deviation is measured per REACHABLE BASIS STATE,
  and for a control-drop the per-basis-state phase-insensitive deviation is exactly
  0 (the |11> branch does not fire on that basis state) or exactly sqrt(2) (it does).
  There is no intermediate value to tolerate: a control is either provably constant
  on the reachable basis set (eps = 0 admits the drop) or it is genuinely set on at
  least one reachable basis state (deviation sqrt(2), admitted only by the trivial
  eps >= sqrt(2)). The `rarely_set` row makes this explicit: one firing input in five
  already forces the full sqrt(2), so no eps below sqrt(2) admits the drop.

* WHERE eps > 0 WOULD matter: amplitude-weighted reachability. If a control fires on
  a reachable input that carries only a TINY amplitude in the actual run-time
  superposition, the true state-deviation is small even though the basis state is
  "reachable". Capturing that requires propagating amplitudes (or amplitude bounds),
  not just the basis-state SET. The verifier method implemented here
  (`verify_on_reachable_basis_approx`) already computes the correct continuous
  phase-insensitive deviation and is exercised with a smooth, tunable knob in
  `test_reachable_approx.py::test_tunable_deviation_admitted_then_rejected`; the gap
  is purely in the reachability front-end, which is set-based and so quantises the
  per-input deviation to {0, sqrt(2)} for control-drops.

## Soundness caveats a referee should know

1. The reachable set used is `reachable_overapprox`, a SOUND over-approximation. The
   reported `max_deviation` is therefore an UPPER BOUND on the true worst-case
   reachable deviation; the check can only reject a valid drop, never admit an
   invalid one.

2. The restricted `input_space` is an ASSERTION by the caller that the whole circuit
   is never run outside that domain. Every soundness claim is conditional on it. With
   the default `all_basis`, a reversible prefix makes every (control, control) pair
   reachable, so NO control-drop can fire -- the gain requires a genuinely restricted
   domain (e.g. ancilla pinned to |0>).

3. The deviation is the worst-case L2 state-distance over reachable basis inputs,
   minimised over a global phase per input. Admitting an eps-drop adds at most eps to
   the per-input output error; the report's `certified_error_budget` charges the eps
   spent on top of the hardware infidelity, so the selector only admits a drop when
   the 2q-infidelity SAVED exceeds the eps SPENT. Here eps spent is always 0, so the
   budget gain is unambiguous.

4. The basis-state metric does NOT account for run-time amplitudes: a "rare" firing
   input is penalised the same sqrt(2) as a certain one. A referee should read the
   eps > 0 path as SOUND-BUT-INERT on set-based reachability, and genuinely active
   only with an amplitude-aware reachability front-end (future work).


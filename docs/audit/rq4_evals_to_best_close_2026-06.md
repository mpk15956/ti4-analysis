---
title: "RQ4 close: NSGA-II evals_to_best targeted fill and the bit-identical splice"
date: 2026-06-18
type: audit
tags: [audit, rq4, evals-to-best, nsga2, friedman, construct-validity, provenance]
---

## The gap

RQ4 (the Friedman omnibus on `evals_to_best`, the anytime-composite metric) could
not be answered from banked data: NSGA-II's `evals_to_best` was never instrumented
in the multi-algorithm run and sat at the `-1` "not recorded" sentinel for all
2,400 of its rows. It is not backfillable from the saved archives (the NSGA-II
trajectory is hypervolume-keyed, not composite-keyed), so the only honest route
was to instrument NSGA-II and re-run it to record the value. The instrumentation
(a pure, return-ignored trajectory callback computing the per-generation rank-0
minimum composite at the run's canonical 1:1:1 weights, proven a byte-identical
sink by `tests/test_nsga2_optimizer.py::test_trajectory_callback_is_pure_sink`)
landed in commit `fb4ddde`.

## The decision: targeted fill, not a full re-run (Option 1)

A full six-algorithm re-run was unnecessary. The five scalar searches already
carry a real `evals_to_best`; only NSGA-II was missing it. So the fill re-ran
**NSGA-II alone**, at the banked NSGA-II hyperparameters
(`blob=0.74948, mut=0.032985, warm=0.0`, weights 1:1:1), across budgets
1000–200000 (100 seeds × 7 budgets × 3 chains = 2,100 rows), in the rebuilt
provenance-stamped `.sif`. Job `28684` ran 72.5 h on teach and completed 700/700
seed-evaluations clean.

Two scope choices follow from "spend rigor where a skeptic could move the result":

- **Budget ceiling 200k, not 500k.** The fill stops at 200k, the largest budget
  where all six algorithms then carry a real `evals_to_best`. RQ4's six-way
  omnibus is therefore reported at **200k canonical**; at 500k it is five-way
  descriptive (NSGA-II was not re-run there). This is decoupled from the 500k
  RQ1/RQ2 canonical in both the generator (`--rq4-budget`, default 200000) and the
  pins (`RQ4_BUDGET`), so RQ1/RQ2 stay 500k-canonical.
- **Construct caveat, not a narrowed test.** A scalar `evals_to_best` is a genuine
  internal-convergence measure for the five composite optimizers; for NSGA-II it
  is the eval index at which the best-so-far *composite* first appeared among a
  population it never selects on that scalarization. The omnibus is kept over all
  six on the operationally identical, black-box definition (anytime delivery of a
  good composite map); the mechanism weight rests on RQ2 (hypervolume, fair by
  construction), with RQ4 the corroborating operational companion. See
  `feedback_construct_decoupling_for_metric_mismatches`.

## The load-bearing check: does the splice import an environmental confound?

Splicing the fresh NSGA-II `evals_to_best` onto the banked six-algorithm dataset
is only legitimate if the freshly rebuilt `.sif` reproduces the banked NSGA-II
**result**, not just a plausible one. If the rebuild perturbed numerics, the
spliced rows would live in a slightly different metric space than the five banked
scalars, and the omnibus would silently compare across that seam.

`scripts/rq4_verify_cross_container.py` answers it directly. Joining fresh vs
banked NSGA-II on `[seed, budget, chain_id]` for all 2,100 rows at budget ≤200k:

- `final_tile_layout` exact-match: **2100/2100 (100.00%)** — the integer tile
  assignment the search converged to is identical every time.
- every derived metric (`balance_gap, morans_i, jains_index, jfi_resources,
  jfi_influence, lisa_penalty, composite_score, front_size`): **max|Δ| = 0**.
- precondition confirmed: banked `evals_to_best` all `-1`, fresh all ≥ 0.

The seam is closed **bit-identically**. The rebuilt `.sif` is objectively
equivalent to the banked environment for NSGA-II, so the only thing the splice
changes is the column that was never instrumented. This is the strongest possible
answer to the confound question, and it is *why* the splice is trustworthy, not
the fact that the job "completed."

The reason one exact-match column dominates tolerance-checking the downstream
floats is worth keeping: the converged `final_tile_layout` is the endpoint of
every float-comparison branch the search took (every accept/reject, every
dominance test). A byte-identical layout therefore certifies that no comparison
forked differently across the two environments, and every derived metric is then a
deterministic function of that identical discrete path, identical by construction
rather than identical within tolerance. Verifying the discrete solution is
stronger and cheaper than diffing the continuous metrics it produces; the rule
transfers to any multi-environment experiment whose output is the endpoint of a
branchy computation.

## The splice

`scripts/rq4_build_merged_results.py` makes the minimal possible edit: for each
banked NSGA-II row at budget ≤200k, overwrite **only** `evals_to_best`
(`-1` → real), matched on `[seed, budget, chain_id]`, fail-loud on any unmatched
row or shape change. 2,100 cells change; every other cell, every other algorithm,
and NSGA-II at 500k (left at the five-way-descriptive sentinel) are untouched.
Because RQ2/RQ3/Track B read the archives and the unchanged columns — never
`evals_to_best` — they are provably identical before and after the splice. The
merged CSV is a regenerable artifact (banked + fresh via the tracked recipe), not
committed.

## Result

Six-way Friedman on `evals_to_best`, all six algorithms (`rs, hc, sa, sga, nsga2,
ts`), df = 5, n = 100, significant at **every** budget ≤200k:

| budget | chi2 | p |
|-------:|-----:|---|
| 1000 | 389.05 | 6.8e-82 |
| 5000 | 424.28 | 1.7e-89 |
| 10000 | 436.41 | 4.2e-92 |
| 25000 | 446.73 | 2.5e-94 |
| 50000 | 439.91 | 7.4e-93 |
| 100000 | 450.30 | 4.2e-95 |
| **200000 (canonical)** | **456.80** | **1.7e-96** |

The algorithms differ in anytime-composite delivery at every budget, and the
separation strengthens gently with budget. NSGA-II's median `evals_to_best` scales
832 → 134,100 across budgets; at 200k it spends ~67% of the budget reaching its
best composite — the operational signature of the depth-vs-breadth story §3.10
argues on RQ2's own terms (the breadth-first method pays to extract a single
composite optimum it never selects for).

## Provenance and verification

- Recipe (tracked, arg-driven, reproducible): `scripts/rq4_verify_cross_container.py`,
  `scripts/rq4_build_merged_results.py`.
- Canonical RQ4 read path: `generate_manuscript_values.py --rq4-budget 200000`
  reads `stats_b200000/stats/rq4_friedman_evals_to_best.csv`; the value flows into
  `manuscript_values_phase6.json` (`rq4_evals_to_best`, with `provenance.rq4_budget`).
- Pins: `tests/test_phase6_canonical_values.py` — the three RQ4 pins
  (available / all-six-df-5 / real-statistic) pass, and the RQ2/RQ3/Track B pins
  reproduce the banked values from the same generator (single-source proof).
  10/10 green on elis; the RQ4 Friedman is a rank test (environment-insensitive)
  and RQ2/RQ3/Track B are the banked in-SIF CSVs reused verbatim, so the assembled
  JSON is what the SIF emits.

## Folding into §3.10 (done at close-out)

The §3.10 prose is now drafted from the user/advisor-provided text: the RQ2 and
RQ4 **Result** lines carry the exact crossover and omnibus numbers, and a
**Mechanism: depth versus breadth** block carries the depth-vs-breadth argument,
the construct caveat (a scalar `evals_to_best` is a real convergence measure for
the five composite optimizers but a projection for NSGA-II), and the portfolio
framing (population breadth is the deliverable, not a tax, when a set of maps is
wanted). Every cited number is single-sourced and pinned twice over:
`tests/test_phase6_canonical_values.py` on the json<-csv edge (now 11/11), and
seven new §3.10 entries in `tests/manuscript_values.yaml` on the prose<->json
edge.

**The breadth-tax figure was made canonical, not hand-copied.** The "134,100
(~67% of budget)" anchor is the raw median of NSGA-II's `evals_to_best` across all
300 runs (100 seeds x 3 chains) at b = 200,000, computed by a new generator
function (`generate_manuscript_values.py::rq4_breadth_tax`, fail-loud on any
surviving `-1` sentinel) and emitted into `manuscript_values_phase6.json`. One
honesty note recorded so it is not mis-stated in revision: the *fractional* tax is
largest at small budgets (about 83% at b = 1,000) and *eases* to ~67% at
b = 200,000 even as the absolute median rises (832 -> 134,100). So the prose claims
only the canonical 67% and never that the fraction grows with budget; the easing
fraction is itself consistent with the RQ2 crossover (NSGA-II amortizes its breadth
cost as budget grows).

The depth-vs-breadth argument remains the user's to review and refine; this pass
places their provided prose and guarantees its numbers.

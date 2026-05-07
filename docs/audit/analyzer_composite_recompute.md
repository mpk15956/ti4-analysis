# Analyzer composite-recompute drift

**Category:** 3 (computation drift) — two systems that should compute the
same quantity diverged at the *formula level*, not just parameter level.
Plus category-1 echoes (`n_spatial=37` hand-copied across three call sites).

**Commit:** `1c517d0`.

## Question

When `scripts/analyze_benchmark.py` recomputes composite scores under
alternative weight configurations (the $\tau$ weight-invariance result that
lives in `sensitivity_rank_stability.txt`), does it use the same scoring
function the optimizer actually optimized?

## What the audit found

The analyzer's `recompute_composite` (and two structurally identical
functions, `rank_stability_and_kendall`'s inline body and `multi_jain_ablation`)
were computing a *different* fitness function than the one
`MultiObjectiveScore.composite_score` produces:

| Component | Optimizer (per Methodology §3.1) | Analyzer (pre-fix) |
|-----------|----------------------------------|---------------------|
| Spatial penalty | $\max(0,\ I - E[I])$ — one-sided hinge above the null | $|I|$ — symmetric absolute value |
| $E[I]$ basis | $-1/(\|G\|-1)$ with $\|G\| = 31$ | Implicit $-1/(n-1)$ with $n = 37$ hardcoded |
| LSAP divisor | $\|G\|(\|G\|-1) = 31 \cdot 30 = 930$ | $37 \cdot 36 = 1332$ |
| Weights | Live from `MultiObjectiveScore.weights` | Independent `WEIGHT_CONFIGS` dict |

For optimized maps with $I < -1/(\|G\|-1) \approx -0.033$ (the regime all
SA-optimized maps live in), the two scoring functions diverge qualitatively:

- $|I|$ symmetrically penalizes both clustering AND dispersion. For
  $I = -0.6$ (typical optimized map), $|I| \approx 0.6$.
- $\max(0,\ I - E[I])$ saturates at zero once $I < E[I]$. For the same
  $I = -0.6$ on the n=31 graph, the hinge is $\max(0, -0.6 + 0.033) = 0$.

So the analyzer was rewarding maps close to $I = 0$ (random spatial
pattern), while the methodology explicitly says (§3.1) that the optimizer
penalizes only positive autocorrelation: "negative autocorrelation
(spatial dispersion) incurs zero penalty, reflecting the design goal of
preventing resource clustering rather than enforcing spatial randomness."

## Stakes

Two layers:

1. **Magnitude divergence.** The analyzer's median composite scores in the
   pre-fix `summary_table.csv` were $\approx 0.23$ across algorithms. The
   optimizer's actual median composite scores at budget=500k are
   $\approx 0.0003$ (cheat sheet's "HC wins on composite score
   (median 0.000333)"). 700× scale gap, hidden by both being internally
   self-consistent.

2. **Ranking question.** The composite-weight $\tau = 1.000$ result was
   computed under the buggy $|I|$ formula. Under that formula, the spatial
   term dominated the composite (because $|I| \approx 0.6$ for optimized
   maps overwhelms the JFI gap and LSAP normalization which are at $10^{-3}$).
   Did the rankings under the buggy formula match what they would have been
   under the corrected hinge formula? *That* was the substantive question
   the re-run answered.

## Result of the re-run

Job 28078 ran the refactored analyzer on the production CSV at budget=500k.
Both the original and corrected formulas produce **$\tau = 1.000$ for all
ten pairwise weight-config comparisons** (legacy_5_5_3, equal_1_1_1,
jfi_dominant, spatial_dom, lisa_dominant — choose 2). The composite-weight
invariance claim survives the formula correction.

But the substantive interpretation is sharper than "result robust to bug":

- Under the buggy $|I|$ formula, the analyzer was computing a quantity
  that wasn't the optimizer's objective. *Of course* rankings were
  stable across weight perturbations applied to a non-target quantity —
  the result was vacuous.
- Under the corrected hinge formula, $\tau = 1.000$ means rankings are
  stable under perturbations of the actual scoring function the optimizer
  was trained on. That's a substantively different claim, and it's the
  claim the manuscript can defend.

The rank-stability sub-result *did* change qualitatively:

- Pre-fix `sensitivity_rank_stability.txt`: every pairwise comparison
  showed "0/5" significant wins across the five weight configs. (The
  $|I|$-dominated composite obscured the JFI/LSAP differences that
  actually drive rankings.)
- Post-fix: clear "5/5" winners on most pairs (e.g., hc vs nsga2 →
  nsga2 5/5; rs vs hc → hc 5/5).

So the corrected analyzer is *more* discriminating than the buggy one,
not less. Algorithm rankings are not just preserved across weight
perturbations — they are *significantly* preserved at $p < 0.05$ across
all five weight configurations on the corrected scoring function.

## Resolution

`1c517d0` introduced a single-source helper architecture:

1. **`_compute_normalized_terms(df, n_spatial)`** — vectorized version
   of `MultiObjectiveScore.raw_objective_terms()` operating on a
   DataFrame. One source for the term arithmetic; `recompute_composite`,
   `rank_stability_and_kendall`, and `multi_jain_ablation` all call it
   instead of duplicating the inline formula.

2. **`--n-spatial` CLI arg** with default 31. Replaces three hand-copy
   sites of `n_spatial = 37`. The benchmark CSV does not currently
   carry $n_{\text{spatial}}$ per row, so it is supplied externally;
   if a future benchmark writes it per row, prefer that source.

3. **`morans_i_null` import** from `map_topology` (the canonical helper
   added in `b2c65f4`). The analyzer cannot drift from the optimizer's
   definition of $E[I]$ because both source from the same module-level
   helper.

4. **`SPATIAL_DEGENERATE_THRESHOLD` import** from `spatial_optimizer`.
   Same regime-aware degenerate-graph guard as the optimizer; the
   analyzer's spatial terms degrade to 0.0 on the same boundary the
   optimizer does.

5. **Rename `current_5_5_3` → `legacy_5_5_3`** in `WEIGHT_CONFIGS`,
   updated the rank-stability ref/other defaults to put the
   manuscript's nominal weighting (`equal_1_1_1`) in the reference
   position. "current" was a lie — live nominal is 1:1:1; the 5:5:3
   configuration is retained as a sensitivity-grid probe demonstrating
   rankings hold under the historical weighting too.

6. **`multi_jain_ablation` weights replaced** from hardcoded 5:5:3 to
   `WEIGHT_CONFIGS["equal_1_1_1"]`. Spatial terms now match the
   optimizer's hinge formulation.

## The structural fix the audit recommends but doesn't implement

The deeper architectural lesson, articulated in
[README.md](README.md)'s "Backfill recipe" section: **logging is a
contract; recomputation is a duplicate implementation.**

The analyzer was forced to recompute the composite from raw inputs
because the benchmark CSV schema doesn't log the intermediate normalized
terms. The structural fix is to extend the CSV schema to v2:

```
seed, algorithm, budget, ..., morans_i, morans_hinge, jains_index,
jfi_gap, lisa_penalty, lisa_norm, composite_score, ...
```

Then `recompute_composite` becomes a one-line linear combination of v2
columns; `--n-spatial` and the `morans_i_null` import disappear from the
analyzer entirely. Computation drift surface: zero.

The schema upgrade is **backfill-cheap** for historical artifacts: a
deterministic pass through existing rows applying the canonical helpers,
plus a bridge test that re-derives `composite_score` from the new
columns and asserts agreement to float tolerance. **Not implemented in
this audit pass** because the immediate workaround (helpers + CLI arg)
gets to a clean manuscript without rerunning 12,000 benchmark rows; but
it's the canonical category-3 fix recipe and is the right thing to do
post-submission.

The general rule for what to log preemptively: **backfill is available
iff the quantity is a function of the optimizer's outputs alone, not of
its acceptance dynamics.** Linear reweighting is backfill-cheap because
changing the weights post-hoc doesn't change the trajectory the
optimizer took; hard-vs-softplus hinge is backfill-impossible because
the SA acceptance probabilities depend on which one was used. The
schema-design rule reduces to "log everything that's a deterministic
function of the run's inputs+outputs," which is finite and finishable.

## Where it landed in code

- `scripts/analyze_benchmark.py` — `_compute_normalized_terms` helper,
  `recompute_composite(df, weights, n_spatial)`, `--n-spatial` CLI arg
  threaded through `weight_sensitivity_analysis`,
  `rank_stability_and_kendall`, `multi_jain_ablation`. Imports
  `morans_i_null` and `SPATIAL_DEGENERATE_THRESHOLD` from the optimizer
  module.

## What this leaves available for the manuscript

When the manuscript reaches the composite-weight invariance section
(probably late §3 or early §4), the citation surface is the post-fix
`sensitivity_rank_stability.txt` with $\tau = 1.000$ across 10 pairwise
weight-config comparisons. The claim is supported on the corrected
scoring function, not the buggy one — a substantively stronger result
than what the pre-fix output supported.

## Related

The general "logging is a contract; recomputation is a duplicate
implementation" principle is articulated more fully in
[README.md](README.md). This is the canonical category-3 case in this
codebase. The JFI parallel-implementation question
([jfi_audit_TODO.md](jfi_audit_TODO.md)) is the next category-3
candidate to investigate.

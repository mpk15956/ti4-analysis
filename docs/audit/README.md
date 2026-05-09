# Codebase audit — May 2026

This directory documents an audit pass over the optimizer/analyzer/diagnostic
code that surfaced a cluster of latent bugs and drift surfaces between the
manuscript's pixi+apptainer migration (commit `916df74`) and the start of
manuscript drafting. The audit was structural rather than test-driven: it
went through the code looking for inconsistencies between independent
manifestations of the same logical concept, on the principle that any logical
quantity encoded in multiple places is free to drift between them, and tests
that exercise only the canonical run path will not catch the drift.

The audit produced four substantive `fix:` commits and reframed the
manuscript's §3.3 graph declaration. **No reported result is affected** —
the canonical 6p layout used in all production runs (Phase 1, Phase 2b,
Phase 4 LISA validation, distance-weight sensitivity, variance equalization)
never reached any of the affected code paths. The bugs were latent: they
would have surfaced the moment anyone applied the codebase to a different
layout (4-player, anomaly-only experimental boards, unit tests with synthetic
small graphs, alternative weight vectors). Fixing them before submission
turns "no reported result is affected" from a reasoned claim into a
mechanically verifiable invariant for the next paper that builds on this
codebase.

## Drift taxonomy

The audit caught examples of three structurally distinct drift categories.
The names below are deliberately teachable so future researchers (and AAAG
reviewers) can index whatever they're looking at against the framework.

| # | Category | Pattern | Fix recipe |
|---|----------|---------|------------|
| 1 | **Documentation drift** | One canonical value, multiple authoritative-looking copies in docs/docstrings/cheat-sheets, all become stale when the value changes. | Derived values get computed once, exposed as a property, referenced everywhere else. Docstrings cite the formula and the attribute, not the literal value. |
| 2 | **Configuration drift** | One logical concept (e.g., the composite weighting), multiple runtime sources free to diverge — typically because each consumer encodes the concept independently. | One source flowing to all consumers; consumers are arguments that read from the source, not separate declarations. |
| 3 | **Computation drift** | Two systems that should compute the same quantity have diverged at the *formula level*, not just the parameter level — different scoring functions, different actual outputs. | Shared helper that both consumers call. Verify outputs agree on canonical inputs *before* declaring the unification done. |

The three categories are not mutually exclusive (the analyzer audit caught
category 1 and category 3 in the same code, manifested as parallel hand-copies
of two diverging formulas). They're useful as orthogonal lenses for finding
drift, not as bin labels for individual bugs.

## Findings

| Question | Stakes | Resolution | Where it landed |
|----------|--------|------------|-----------------|
| What is the spatial graph $G$ that Moran's I and LSAP operate on, and how does its size $|G|$ relate to the manuscript's E[I] citation? | §3.3 cites E[I] as a manuscript-defining null expectation. Four documents/docstrings cited inconsistent N values (37, 31, 30, 14). Wrong N → wrong E[I] → wrong margin in the C0 vs null comparison. | Single-source: `MapTopology.n_spatial` and `morans_i_null(n)` properties; §3.3 declares G explicitly; limitations.md and docstrings reconciled. | [n_reconciliation.md](n_reconciliation.md) — commits `d307d79`, `b2c65f4`. Category 1 with category 3 echoes (the `lisa_penalty_thresholded` docstring conflated geometric vs spatial-graph N). |
| Does the variance equalization diagnostic compute its share verdict against the same composite the manuscript declares as nominal? | §3.8 prose cites the diagnostic's "92% Moran-share at Gen-0" verdict. The diagnostic was hardcoded to 5:5:3 weights for the share calculation while running its embedded SA under 1:1:1 (the optimizer's defaults). Two parallel weight sources, internally inconsistent within a single script. | `--weights` CLI argument with default `1,1,1`, single source flowing to both SA and reporting; `--sensitivity-probe` guard required for non-default weights; per-row weight columns in CSV so the artifact is self-describing. | [parallel_weight_sources.md](parallel_weight_sources.md) — commit pending step 3. Category 2. |
| Does the analyzer's "what would composite be under different weights?" recompute the same scoring function the optimizer optimized? | The composite-weight Kendall τ result lives in `sensitivity_rank_stability.txt` and would be cited as evidence of weight-invariance. The analyzer's `recompute_composite` was using `|I|` (not the optimizer's one-sided hinge) and `n_spatial=37` (not 31). Different scoring function, not just different weights. | `_compute_normalized_terms` shared helper using the canonical `morans_i_null`; `--n-spatial` CLI arg; rename `current_5_5_3` → `legacy_5_5_3` (because "current" was a lie — live nominal is 1:1:1). Re-run produced τ = 1.000 across all 10 pairwise weight-config comparisons under the corrected formula. | [analyzer_composite_recompute.md](analyzer_composite_recompute.md) — commit `1c517d0`. Category 3 with category 1 echoes (the `n_spatial=37` was hand-copied across three call sites). |
| Latent: does the composite_score gracefully degrade on degenerate spatial graphs (n_spatial < 3)? | Discovered while refactoring the four E[I] hand-copy sites in `spatial_optimizer.py`. The `max(1, n - 1)` syntactic clamp prevented division-by-zero but produced a spurious `1.0` hinge at n_spatial = 0 — *maximum* spatial penalty on a graph with no spatial nodes. | Regime-aware guard at the call site: `if n < SPATIAL_DEGENERATE_THRESHOLD: spatial_term = 0.0`. The library function (`morans_i_null`) keeps its strict mathematical contract; the application code (composite_score, etc.) explicitly handles the degenerate regime. | Folded into `b2c65f4`. Category 1 *plus* a regime-aware-vs-syntactic safety-bound lesson. See [safety_bounds.md](safety_bounds.md). |
| TODO: do the two `_jfi` implementations (`fast_map_state.py` for the optimizer, `spatial_metrics.py` for standalone analysis) agree on the all-zeros convention, and is that convention documented? | JFI returns 1.0 by convention on zero-allocation inputs ("perfect fairness on a vacuously-fair allocation"). Two parallel implementations exist; they may or may not both encode this convention. Asymmetric direction with the spatial penalty (1.0 = best for JFI vs 0.0 = best for spatial penalty) is undocumented. | Not yet addressed. Follow-up audit pass; flagged so the framework looks complete on day one rather than partial. | [jfi_audit_TODO.md](jfi_audit_TODO.md). Category 2 (parallel implementations) plus possibly category 3 (if the conventions differ). |

## Backfill recipe for category 3 (recomputation drift)

The most durable fix for category 3 is **logging-as-contract**: the optimizer
writes every intermediate quantity any downstream consumer might want, and
downstream consumers compose those quantities by linear combination rather
than recomputing from raw inputs. The benchmark CSV currently logs raw
inputs + final composite; a future schema-v2 should add the intermediate
normalized terms (`morans_hinge`, `jfi_gap`, `lisa_norm`).

The schema upgrade is **backfill-cheap** for historical artifacts: a
deterministic pass through existing rows applying the canonical helpers,
plus a bridge test that re-derives `composite_score` from the new columns
and asserts agreement with the original column to float tolerance. If the
bridge passes, the v2 artifact is *certifiably* a faithful re-derivation
and the audit's "no reported result is affected" assertion is upgraded
from informal-reasoning to mechanically-verified.

The general rule for what to log preemptively in any new schema:

> **Backfill is available iff the quantity is a function of the optimizer's
> outputs alone, not of its acceptance dynamics.** Linear reweighting is
> backfill-cheap because changing the weights post-hoc doesn't change the
> trajectory the optimizer took to those terminal states — you're scoring
> the same trajectory differently. Hard-vs-softplus hinge is
> backfill-impossible because the SA acceptance probabilities depend on
> which one was used, so different trajectories, different terminal states.

The schema-design rule reduces to: **log everything that's a deterministic
function of the run's inputs+outputs.** That's a finite, enumerable set,
which means the schema is finishable, not perpetually under-specified.

The `results.csv` v1→v2 upgrade is documented as a planned post-submission
task in [analyzer_composite_recompute.md](analyzer_composite_recompute.md);
not implemented in this audit pass because the immediate workaround
(helpers + CLI args) gets to a clean manuscript without the schema migration
work.

## Audit-driven vs test-driven development

A note on methodology, since the audit is itself a methodological artifact
relevant to AAAG reviewers:

None of the four bugs caught here would have triggered any failing test
on the canonical run path. They were all latent — invisible to test-driven
development because the tests run on the same canonical inputs production
does. They're catchable only by going through the code looking for
inconsistencies between independent manifestations of the same logical
concept. Call this *audit-driven development* to distinguish from
test-driven: the test suite is a sample, the audit pass is a structural
review. A publishable methods paper benefits from at least one
audit pass that goes beyond "do the tests pass" to "does every consumer
of every derived quantity agree with every other consumer."

This audit was that pass. The findings here — three drift categories,
five bugs across three commits — are what reviewers would otherwise
be asked to surface during peer review. Surfacing them in advance is
both faster (cheaper than a review round) and more credible (the fixes
were made by the author, not under reviewer pressure).

## Index of detail files

- [n_reconciliation.md](n_reconciliation.md) — the four-N drift, §3.3 declaration, postcondition assertion, regression tests.
- [parallel_weight_sources.md](parallel_weight_sources.md) — variance equalization diagnostic's two-weight-source bug, `--weights` CLI, per-row provenance.
- [analyzer_composite_recompute.md](analyzer_composite_recompute.md) — `|I|` vs hinge formula divergence, `n_spatial = 37` vs `31`, the v2 schema backfill recipe.
- [safety_bounds.md](safety_bounds.md) — regime-aware guards vs syntactic clamps; the composite_score zero-tile-hinge bug.
- [jfi_audit_TODO.md](jfi_audit_TODO.md) — flagged but not addressed in this pass; documents what a future audit pass would investigate.

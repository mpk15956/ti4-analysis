# JFI audit — TODO

**Status:** Flagged but not addressed in the May 2026 audit pass. Documents
what a future audit pass would investigate so the framework looks complete
on day one rather than partial.

**Category:** 2 (parallel implementations) plus possibly category 3 (if the
two implementations' edge-case conventions differ).

## Question

Do the two `_jfi` implementations in this codebase agree on the
all-zeros-input convention, and is that convention documented for reviewers
who would expect to see it justified?

## What is known from the May 2026 pass

A quick read during the spatial-penalty audit surfaced two parallel
implementations of Jain's Fairness Index in the codebase:

1. **`src/ti4_analysis/algorithms/fast_map_state.py:351-361`** — the
   `_jfi` static method on `FastMapState`. Used by the optimizer's
   composite scoring. Has two explicit guards:
   ```python
   if n == 0:           return 1.0
   if sum_x2 == 0.0:    return 1.0
   ```
   Encodes the convention: 1.0 (perfect fairness) on vacuously-fair
   allocations.

2. **`src/ti4_analysis/spatial_stats/spatial_metrics.py:276-310`** —
   the standalone `jains_fairness_index` function. Used by the
   `comprehensive_spatial_analysis` helper. Has a `sum_x_sq == 0` guard,
   but the audit pass did not verify what value it returns at that
   guard, nor what it returns at `n == 0`.

## What the audit pass did not check

- Whether `spatial_metrics.jains_fairness_index` returns 1.0 on the
  same edge cases as `fast_map_state._jfi`, or whether it diverges.
- Whether the convention "1.0 on vacuously-fair allocations" is
  documented anywhere — manuscript, docstring, or otherwise. The audit
  could not find a doc that justifies the convention with the
  limit-as-$\mathbf{x} \to \mathbf{0}$ argument that makes it
  defensible.
- Which downstream consumers of `comprehensive_spatial_analysis`
  depend on its JFI value, and whether any of them encounter the
  zero-allocation edge case.

## Stakes

Smaller than the spatial-penalty audit because the failure surface is
narrower:

- The optimizer's composite uses `fast_map_state._jfi`. Its convention
  is correct (1.0 on zeros), and that's the path all production
  results were generated on.
- The standalone implementation might have a different convention.
  If a future researcher uses `comprehensive_spatial_analysis` for
  per-map analysis and encounters a degenerate input, they may get a
  different answer than the optimizer would have produced — silent
  drift between two parallel implementations of the same logical
  operation.
- The convention itself, even if consistent across implementations,
  is currently undocumented. Reviewers familiar with JFI will check
  the boundary behavior and ask "why 1.0?" — a methodology section
  that doesn't pre-empt this question is a soft spot.

## Asymmetric directionality (a separate sub-audit)

A related observation worth documenting:

- Spatial penalty terms (Moran's hinge, LSAP) are in $[0, 1]$ with
  $0.0$ = best. Degenerate-graph convention: 0.0 (no penalty applies).
- JFI is in $[1/n, 1]$ with $1.0$ = best. Degenerate-allocation
  convention: 1.0 (perfect fairness applies).

The two terms have opposite "best" directions. That's correct —
$1 - \text{JFI}$ is the JFI gap that enters the composite as a penalty,
matching the directionality of the spatial penalties. But the
asymmetric edge-case conventions (0.0 for one, 1.0 for the other)
look superficially inconsistent if a reader doesn't notice the
sign-flip. Worth a sentence in the methodology that makes both
conventions explicit and cites the limit-argument justification for
each.

## Recommended fix when this audit is run

1. **Verify both `_jfi` implementations** return 1.0 at `n == 0` and
   at `sum_x_sq == 0`. If they don't agree, the divergent one needs
   to be aligned with the convention.

2. **Collapse to a single source.** Both implementations encode the
   same logical operation; one of them should call the other (or both
   should call a shared helper in `spatial_metrics`). Same fix recipe
   as `morans_i_null` — single-source the formula, callers reference
   it.

3. **Document the 1.0-on-zeros convention** in the methodology,
   ideally with the limit-as-$\mathbf{x} \to \mathbf{0}$ argument and
   a brief note on why the convention is operationally appropriate
   (e.g., "no allocation to be unfair about implies no fairness
   deficit").

4. **Document the asymmetric directionality** in §3.1: spatial
   penalty conventions (0.0 best, 0.0 on degenerate input) and JFI
   conventions (1.0 best, 1.0 on degenerate input) are dual under
   the $1 - \text{JFI}$ transform that enters the composite, so the
   composite term direction is consistent even though the underlying
   conventions look opposite.

## Triggering condition

This audit should be run before any of:

- Adding a new layout or game configuration that produces edge cases
  the canonical 6p layout doesn't (e.g., experimental small-n maps,
  4-player layouts where home tiles dominate the board, anomaly-only
  test boards).
- Citing JFI behavior in the manuscript at the boundary (e.g., a
  reviewer-defense response that needs to justify edge-case handling).
- Making any change to `comprehensive_spatial_analysis` that affects
  its callers' interpretation of JFI.

If none of those triggers, the audit can wait — the parallel
implementations have not measurably diverged in production, and
deferring is responsible scope discipline.

## Related

This is a category-2 (parallel implementations) case if the two
implementations agree, and category-3 (formula/convention divergence)
if they don't. The fix recipe is the same in either case: single
source, exposed, referenced. See the [README.md](README.md) drift
taxonomy and the related [parallel_weight_sources.md](parallel_weight_sources.md)
audit which addressed the same pattern in the variance equalization
diagnostic.

# N reconciliation — the four spatial-graph N values

**Category:** 1 (documentation drift) with category-3 echoes (one of the
hand-copies conflated geometric vs spatial-graph N).

**Commits:** `d307d79` (declaration + doc reconciliation), `b2c65f4`
(properties + postcondition + spatial_optimizer refactor + tests).

## Question

What is the spatial graph $G$ that Moran's I, LSAP, and the post-hoc
conditional-permutation LISA all operate on, and how does $|G|$ relate to
the manuscript's $E[I] = -1/(|G|-1)$ citation in §3.3?

## What the audit found

Four documents and docstrings cited inconsistent values for the spatial-graph
N, none of which agreed on the actual runtime value:

| Source | Cited N | Status |
|--------|---------|--------|
| `docs/07_POST_RUN_CHEAT_SHEET.md` (deleted in `6d12a55`) | 37 | Wrong — used as basis for $E[I] \approx -0.028$ |
| `docs/limitations/limitations.md` (small-N section) | 37 throughout | Wrong — claimed N=37 as the spatial-graph N |
| `docs/limitations/limitations.md` (boundary-violations passage) | "swappable subgraph (n=14) ... full graph (n=30)" | Off-by-one (30 vs 31), and 14 referred to an auxiliary metric not used in primary results |
| `src/ti4_analysis/algorithms/fast_map_state.py:243` `lisa_penalty_thresholded` docstring | "n = 37 swappable tiles, $E[I_i] \approx -0.028$" | Wrong on three counts (n≠37, not "swappable tiles", $E[I]$ wrong) |
| Actual runtime (verified in `validation_results.csv` `n_positions` column across all 120 maps) | **31** | Ground truth |

The runtime ground truth: `MapTopology.from_ti4_map` builds `spatial_indices`
from system-tiles-with-non-None-systems (which excludes the 6 home tiles by
space type), purges zero-degree nodes after impassable-edge excision (typically
removes a Supernova if seated isolated), and reports `spatial_W.shape[0]` as
the final $|G|$. For the canonical 6p layout, that's 31 across all production
runs.

## Stakes

- $E[I] = -1/(N-1)$. With wrong N=37: $E[I] \approx -0.028$. Correct
  N=31: $E[I] = -1/30 \approx -0.033$.
- The C0 vs null margin in the manuscript's central finding moves from
  $-0.086 - (-0.028) = -0.058$ (under wrong N) to $-0.086 - (-0.033) = -0.053$
  (correct N). Substantively identical claim — "C0 sits essentially at the
  null" — but the arithmetic was off by one decimal in the second place.
- Reviewers familiar with spatial autocorrelation will check the $E[I]$
  arithmetic against the stated N. It takes them ten seconds. Shipping
  the inconsistency would be embarrassing under peer review.

## Resolution

Two commits, deliberately separated:

**`d307d79` — methods declaration and doc reconciliation:**
- New §3.3 "The spatial graph $G$" subsection with explicit construction
  (system-tile-type filter, impassable-edge excision, zero-degree purge),
  $|G| = 31$ for canonical 6p verified against `validation_results.csv`,
  and a callout noting the optimizer–validator alignment (both endpoints
  source `z` from `FastMapState.spatial_values()` and $W$ from
  `MapTopology.spatial_W` — same graph by construction).
- Corrected $E[I] = -1/(|G|-1) \approx -0.033$.
- Corrected composite-bound formula to $\approx [0, 1.033]$ (was $[0, 1.028]$).
- Added Caldas de Castro & Singer (2006) citation for per-map BH-at-q=0.05
  FDR correction across the $|G| = 31$ locations.
- Replaced $|I|$-vs-37 references in `limitations.md` small-N section with
  $|G|=31$ throughout; "Note on N values" callout documents the historical
  drift so future readers don't re-introduce it.
- New "Boundary violations on the primary n=31 graph" subsection in
  `limitations.md` with correct attribution to the primary
  `FastMapState.morans_i()` metric (not the auxiliary `morans_i_swappable`)
  and the asymmetric-W relaxation citation (de Jong, Sprenger & van Veen
  1984; Anselin & Rey 2014).
- Fixed the `lisa_penalty_thresholded` docstring inline pointer to §3.3.

**`b2c65f4` — single-source properties + postcondition + spatial_optimizer
refactor:**
- `MapTopology.n_spatial` property (returns `spatial_W.shape[0]`; always
  well-defined).
- `MapTopology.morans_i_null_expectation` property delegating to a
  module-level `morans_i_null(n)` helper. Strict mathematical contract:
  raises `ValueError` for $n < 2$ with an informative message pointing
  callers at the regime-aware-guard pattern.
- Postcondition assertion in `from_ti4_map`: `assert n_spatial == 0 or
  n_spatial >= 2`. Converts the "n=1 is unreachable" property from
  "true because of how the code happens to flow" to "true because the
  constructor refuses to violate it." Future refactors that accidentally
  bypass the zero-degree purge fire this assertion at construction time
  on the production path, not at test time on a synthetic path.
- Four hand-copy E[I] sites in `spatial_optimizer.py`
  (`composite_score`, `raw_objective_terms`, `objective_values_for_pareto`,
  `dominates`) refactored to use `morans_i_null` inside a regime-aware
  `if n < SPATIAL_DEGENERATE_THRESHOLD: spatial_term = 0.0` guard.
- Three-category test file `tests/test_morans_i_null_invariants.py` with
  failure-response taxonomy comment block, plus a synthetic-helper bridge
  test pinning that the synthetic test fixture agrees with real
  `from_ti4_map` output at the canonical $|G| = 31$.

## Where it landed in code

- `MapTopology.n_spatial` — `src/ti4_analysis/algorithms/map_topology.py`
- `morans_i_null(n)` helper + `MapTopology.morans_i_null_expectation` property
  — same file.
- Postcondition assertion — same file, in `from_ti4_map` after the
  zero-degree purge.
- Spatial-optimizer guards — `src/ti4_analysis/algorithms/spatial_optimizer.py`
  (`SPATIAL_DEGENERATE_THRESHOLD` constant + four refactored call sites
  + new `_morans_hinge_term` helper consumed by `composite_score` and
  `dominates` so the guard logic doesn't drift between scoring and
  Pareto comparison).
- Tests — `tests/test_morans_i_null_invariants.py` (12 tests, all passing
  under apptainer in job 28076).
- Manuscript — `docs/methodology/Methodology_Section.md` §3.3 (graph
  declaration), `docs/limitations/limitations.md` (small-N section + Note
  on N values + boundary violations subsection), `docs/limitations/anomalies.md`
  (SA seed=37 anomaly added separately in `6d12a55`).

## What this leaves available for §3.3 prose

Direct citation surface for the manuscript:

> Spatial autocorrelation metrics operate on the spatial graph $G$ defined
> in §3.3, with $|G| = 31$ for the canonical 6-player layout used throughout
> this study. Under no spatial association, $E[I] = -1/(|G|-1) \approx -0.033$.

Both `|G|` and the formula are sourced from
`MapTopology.morans_i_null_expectation` at runtime; the manuscript number
cannot drift from the runtime value without a regression test in
`tests/test_morans_i_null_invariants.py` failing.

## Related

The structural prevention principle articulated by this audit — *derived
values get computed once, exposed as an attribute, referenced everywhere
else* — is the documentation-drift specialization of the more general
"single source, exposed, referenced" rule. See [README.md](README.md)
category 1 and the memory entry
`feedback_no_handcopied_derived_values.md`.

# Safety bounds — regime-aware vs syntactic clamps

**Category:** 1 + a methodological lesson on defensive coding.

**Commit:** `b2c65f4` (folded into the spatial_optimizer refactor).

## Question

When `MultiObjectiveScore.composite_score` was applied to a degenerate
spatial graph (n_spatial < 3), did it produce a meaningful answer?

## What the audit found

The pre-fix code had two compounding defensive clamps:

1. **`MultiObjectiveScore.__init__:76`**: `self.n_spatial = max(1, n_spatial)
   # guard against zero-division`. Silently clamped n_spatial=0 to 1.

2. **`composite_score:124`**: `hinge_raw = self.morans_i + 1.0 / max(1, n - 1)`.
   Floored the divisor at 1 to prevent division-by-zero.

Together, on a 0-system-tile experimental board:
- `morans_i()` returns 0.0 (the existing `n < 3` guard at `fast_map_state.py:184`).
- `n_spatial = max(1, 0) = 1` (constructor clamp).
- `n - 1 = 0`, but `max(1, 0) = 1` (composite_score clamp).
- `hinge_raw = 0.0 + 1.0/1 = 1.0`.
- `morans_hinge = max(0.0, 1.0) = 1.0`.

A 0-tile map produced a *maximum* spatial penalty of 1.0 — the worst
possible direction for the bug to point. Both `max(1, ...)` clamps
prevented the runtime crash they were defending against; neither
preserved the graceful-degradation semantic the existing
`fast_map_state.py:184` guard was trying to install.

## Stakes

- No reported result is affected. The canonical 6p layout produces
  `n_spatial = 31` throughout all production runs (Phase 1, Phase 2b,
  Phase 4 LISA validation, distance-weight sensitivity, variance
  equalization), and the affected `n < 3` path is structurally
  unreachable from `from_ti4_map` for that layout. The bug was latent.
- The bug would have surfaced the moment anyone ran the optimizer on
  a different layout — a 4-player map, an anomaly-only experimental
  board, a unit test with a synthetic small graph. Researchers
  legitimately submitting such inputs and trusting the output would
  have inferred extreme spatial-penalty regions of the search space
  that were artifacts of the guard arithmetic.

## The lesson — regime-aware vs syntactic safety bounds

This is the methodological abstraction the audit produced, beyond the
specific bug fix.

**Defensive bounds have two failure modes, and only one is obvious:**

1. **Obvious failure mode** — the runtime crash. `ZeroDivisionError`,
   `NaN` propagation, type errors. Visible at runtime, surfaced through
   stack traces, easy to attribute when debugging. Defensive coding
   catches these by clamping inputs into safe ranges.

2. **Subtle failure mode** — the clamp prevents the crash but the
   clamped value isn't meaningful in the regime where the clamp fires.
   The output is type-correct, magnitude-plausible, and silently wrong.
   No exception to catch; no log line to grep for.

The reframe: **distinguish syntactic safety bounds from regime-aware
safety bounds.**

- A `max(1, n-1)` is a *syntactic* safety bound — it answers "how do I
  prevent the divisor from being zero?"
- An `if n < 3: return 0.0` is a *regime-aware* safety bound — it
  answers "in what input regime is this formula meaningful, and what
  should the function do outside that regime?"

The structural rule that falls out: **prefer regime-aware guards at the
call site over syntactic clamps inside the arithmetic.** The library
function's contract becomes strict on its meaningful regime; the
application code explicitly identifies the regime boundary and handles
the edge case there. That's the math.sqrt-vs-caller pattern: the strict
mathematical contract pushes responsibility to the caller, who must
explicitly handle the edge regime.

## Resolution

`b2c65f4` applied the layered design throughout `spatial_optimizer.py`:

1. **`morans_i_null(n)` keeps a strict mathematical contract** — raises
   `ValueError` for $n < 2$ with an informative message pointing callers
   at the regime-aware-guard pattern. The library function does not
   silently return a sentinel from an undefined regime.

2. **`SPATIAL_DEGENERATE_THRESHOLD = 3` constant** at module level. Matches
   the existing `n < 3` guards in `FastMapState.morans_i` and `lisa_penalty`.
   One named threshold, three consumers.

3. **Four call sites guarded** at the call site, returning `0.0` for
   spatial penalty terms before `morans_i_null` is consulted:

   ```python
   if n < SPATIAL_DEGENERATE_THRESHOLD:
       morans_hinge = 0.0
       lisa_norm = 0.0
   else:
       lisa_norm = self.lisa_penalty / (n * (n - 1))
       hinge_raw = self.morans_i - morans_i_null(n)
       morans_hinge = max(0.0, hinge_raw)
   ```

4. **Removed `max(1, n_spatial)` clamp at construction.** `n_spatial` is
   now stored as-passed; the regime-boundary detection happens at the
   four call sites that actually consume it, not silently at construction.

5. **Regression test (`test_composite_score_zero_on_zero_tile_topology`)**
   asserts `composite_score == 0.0` and individual term values are 0.0
   on a zero-tile topology. Documents the pre-fix bug inline and dates
   the test to the same commit as the fix, so a future reader running
   `git blame` on the test gets routed to the bug-fix commit and this
   audit doc.

## Code-review questions for future reference

When reviewing code that has a clamp inside an arithmetic expression,
ask:

1. **What input regime triggers this clamp?** If you can't articulate
   it cleanly, the clamp is probably hiding undefined behavior rather
   than handling an edge case with meaning.

2. **Is the clamped output semantically interpretable to consumers, or
   is it an artifact?** A `1.0` from `max(1, n-1)` at $n=0$ is an
   artifact — no consumer can correctly interpret "the spatial penalty
   is 1.0 on a graph with zero spatial nodes."

3. **Could the clamp fire in production, and if so, would the test
   suite catch it?** The `max(1, n-1)` clamp couldn't fire in
   production for the canonical 6p layout, which is why the bug stayed
   latent — but it would fire silently the moment anyone ran the
   optimizer on a different layout.

If the clamped output is an artifact, the right fix is to gate the
regime explicitly upstream, not to clamp arithmetic. Move the safety
check to the regime boundary, not into the formula itself.

## Generalization

This pattern shows up wherever defensive coding meets mathematical
formulas:

- `max(eps, sigma)` for variance stabilization
- `min(1.0, prob)` for probability clamping
- `softplus` substitutions for `max` in optimization
- `1 + epsilon` denominators in normalization

All of these are worth re-examining for "is the clamped output a
meaningful value in the regime where the clamp fires, or an artifact
that downstream consumers can't interpret correctly?" In research
code, where outputs flow into manuscripts, the second case is academic
debt accumulating silently.

## Where it landed in code

- `SPATIAL_DEGENERATE_THRESHOLD` constant —
  `src/ti4_analysis/algorithms/spatial_optimizer.py`.
- Four refactored call sites in the same file (`composite_score`,
  `raw_objective_terms`, `objective_values_for_pareto`, `dominates`,
  with a new `_morans_hinge_term` helper consolidating the guard logic
  shared between `composite_score` and `dominates`).
- `morans_i_null(n)` — `src/ti4_analysis/algorithms/map_topology.py`
  (single-source helper, raises `ValueError` for n < 2).
- Regression test — `tests/test_morans_i_null_invariants.py`
  (`test_composite_score_zero_on_zero_tile_topology`, category 2).

## Related

The general "single source, exposed, referenced" principle (categories
1 and 2 in [README.md](README.md)) interacts with this category in a
specific way: the strict library contract is what makes the layered
design work. If `morans_i_null` returned a sentinel on $n < 2$, the
call-site guards would be redundant and easy to remove "for clarity";
the strict contract forces the call sites to be regime-aware. The
strictness of the library function and the responsibility of the
caller are dual aspects of the same architectural decision.

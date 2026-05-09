# Parallel weight sources in the variance equalization diagnostic

**Category:** 2 (configuration drift) — one logical concept (the composite
weight vector), multiple runtime sources free to diverge.

**Commit:** `6ae5061`.

## Question

Does `scripts/variance_equalization_diagnostic.py` compute its share verdict
against the same composite-objective formulation that the embedded SA optimizes,
and against the manuscript's nominal weighting (1:1:1, per Methodology §3.1)?

## What the audit found

Two parallel weight sources within a single script, internally inconsistent:

1. **`NOMINAL_WEIGHTS`** — module-level constant hardcoded to 5:5:3 (a
   historical formulation). Used by `_dominance_verdict` and `_print_table`
   to compute the weighted-share percentages and the WARNING/OK verdict.

2. **The embedded SA call** — `improve_balance_spatial(ti4_map, evaluator,
   iterations=args.sa_budget)` passed *no* `weights=` argument. The optimizer
   defaults to 1:1:1 (per `spatial_optimizer.py:82-88`), so the SA composite
   was actually optimized under 1:1:1.

The diagnostic was therefore claiming "92% Moran-share at Gen-0 under the
nominal 5:5:3 weights" while the convergence sample was an SA trajectory
optimized against an entirely different (1:1:1) composite. Two parallel
weight sources, and they encoded different actual weights — invisible from
any single read because the σ values look like data and the share calculation
looks like reporting.

## Stakes

- The §3.8 manuscript prose was about to cite the diagnostic's "92%
  Moran-share" verdict. That number is by definition a *weighted* share;
  citing a 5:5:3-weighted share in a 1:1:1 paper is exactly the unit
  mismatch a careful reviewer catches — and an uncareful reviewer doesn't,
  which is worse, because it survives review and embarrasses you at
  conference Q&A.
- The σ values themselves are weight-independent at Gen-0 (uniform
  sampling), but at convergence they depend on which weights the SA used.
  Since SA was actually using 1:1:1, the convergence σ values reflected
  1:1:1 trajectories — but the diagnostic was reporting them under a
  5:5:3 frame, attributing properties of one regime to a different regime.

## The qualitative shift the correction surfaces

The σ values reproduced bit-identically across the buggy run and the fixed
run (same SA-under-1:1:1 trajectories), so the "did the optimizer behave
correctly?" answer is yes. But the share verdict moves:

| | 5:5:3 (buggy) | 1:1:1 (correct) |
|---|---:|---:|
| Gen-0 hinge_morans_i share | 92.0% | **90.0%** |
| Gen-0 jfi_gap share | 4.7% | 4.6% |
| Gen-0 lisa_norm share | 3.3% | 5.4% |
| Convergence largest-share term | jfi_gap @ 56.6% | **lisa_norm @ 56.1%** |
| Convergence verdict | OK | OK |

Two qualitative shifts:

1. **Gen-0 hinge dominance moves 92% → 90%.** Two-regime story unchanged:
   one term dominates Markov-chain acceptance dynamics early.
2. **Convergence largest-share term flips from `jfi_gap` to `lisa_norm`.**
   The §3.8 prose pointing at "no single term dominates, largest share is
   X" must now use `lisa_norm`, not `jfi_gap`. This is the substantive
   sentence-level change in the manuscript that the audit produces.

The two-regime principle for §3.8 framing (cite Gen-0 dominance AND
convergence collapse together) survives intact — the σ-shift table
(0.138→0.000, 0.007→0.002, 0.008→0.003) is the same robust story under
either weighting.

## Resolution

Three structural changes in the script (`6ae5061`):

1. **`--weights` CLI argument** with default `1,1,1`. The parsed weight
   vector flows to BOTH `improve_balance_spatial(weights=...)` AND the
   share-verdict computation. One source, two consumers — agreement
   guaranteed by construction.

2. **`--sensitivity-probe` flag** required when `--weights` is not the
   manuscript default. Without this flag, non-default weights are
   rejected with an informative `SystemExit`. Prevents accidental
   authoritative-looking output from a sensitivity probe that doesn't
   match the live nominal weighting.

3. **Per-row weight columns in the CSV** (`weight_morans_i`,
   `weight_jains_index`, `weight_lisa_penalty`, `weight_label`). Every
   row carries its own provenance — future concatenation across
   sensitivity sweeps survives the merge without the weight vector
   having to be inferred from the filename or the run's metadata.

4. **`_extract_terms` delegated to `MultiObjectiveScore.raw_objective_terms`**
   (added in `b2c65f4`) instead of recomputing the term arithmetic
   inline. The diagnostic does not duplicate the formula; it reads from
   the optimizer's helper.

## Artifacts in git

Both the buggy "before" CSV and the corrected "after" CSV are force-added
past the `output/` gitignore so the audit-archive demonstration of the
parallel-source bug has its evidence in git history:

- `output/verify_pixi_apptainer_20260506_205045/variance_equalization_20260506_210008.csv`
  — pre-fix artifact (5:5:3 verdict reported on 1:1:1-SA trajectories).
- `output/verify_pixi_apptainer_20260506_205045/variance_equalization_20260506_224438.csv`
  — post-fix artifact (1:1:1-everywhere, single source).

Both have `weight_label` columns; the after-CSV's column says `1:1:1`,
the before-CSV doesn't have the column at all (the schema upgrade is part
of the fix).

## What this leaves available for §3.8 prose

The corrected after-CSV is the canonical citation target. Suggested
manuscript wording:

> At Gen-0 the Moran's-I hinge term accounts for 90% of weighted variance
> in random configurations under nominal 1:1:1 weights, dominating
> Markov-chain acceptance dynamics. SA collapses this variance to
> $\sigma \approx 0$ by convergence, after which no single term dominates
> ($\sigma$-share table omitted; largest residual share is the LSAP
> normalization term at 56%). The two-regime structure — Moran-dominance
> at Gen-0, variance collapse at convergence — supports the use of the
> nominal weights as operationally consistent across the optimization
> trajectory.

## Related

This is the configuration-drift specialization of the same single-source
principle that fixed the documentation-drift case. See
[n_reconciliation.md](n_reconciliation.md) and category 2 in
[README.md](README.md). The general rule: any logical parameter that
flows through multiple consumers should have one source, one runtime
path; consumers are arguments that read from the source, not separate
declarations of it.

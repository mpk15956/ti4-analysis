---
title: "Phase 6/7 latent defects surfaced at full scale (500k tail recovery)"
date: 2026-06-06
type: audit
tags: [audit, track-b, igd-plus, hypervolume, chains, methodology]
---

# Phase 6/7 latent defects surfaced at full scale

## How they were found

Job 28399 (the Paper-1 multi-algorithm canonical run, Stage 2) was killed by the
120 h walltime partway through budget = 500k. Recovery finished the missing
seeds (16–99) as array job 28494, merged them into the banked run (budget 500k →
100/100 seeds), and re-ran Phase 6/7 on the **complete** dataset for the first
time. Phase 6/7 had **never executed at full scale before** — the original
Stage-2 job died in Phase 1 — so this was the first time these scripts ran on
100 seeds × 3 chains × 8 budgets (800 NSGA-II Pareto archives; 14,400 unified
archives). Four latent defects surfaced, each fixed against documented intent
(README / Methodology §3.8–3.9), not merely made to run.

## Defect 1 — IGD+ reference set was the raw union, not the non-dominated front

`quality_indicators.py` (Track B) and `cross_method_igd.py` built the IGD+
reference as `np.vstack(...)` of **all** archive points with no non-dominated
filtering. At 100 seeds this pooled **204,599** points (Track B, additionally
pooled across *all* budgets). Ishibuchi et al. (2015) require the IGD+ reference
to approximate the true Pareto front — the non-dominated subset of the union —
so including dominated points biases IGD+ upward (distance to a cloud that is
~99.9 % dominated). Methodology §3.8 specifies "merging all observed Pareto
points **across seeds**" — i.e. per-budget, non-dominated.

**Fix.** Reference is now built **per budget** and **non-dominated-filtered**
(`nondominated_filter`). The non-dominated subset is 8–76 points per budget
(was 2,983–70,585 raw). Post-fix Track B IGD+ mean = 0.0011 (fronts sit on the
reference), which is the expected magnitude once the reference is correct.

## Defect 2 — `igd_plus` was a pure-Python double loop

`for r in reference_front: for p in front: ...` — O(|ref|·|front|) interpreted,
per archive, ×800. Against the 204k reference this is tens of billions of
iterations and was the proximate cause of the Phase 6a timeout.

**Fix.** New single-source module `src/ti4_analysis/algorithms/moo_indicators.py`
holds a vectorized `igd_plus` + `nondominated_filter`, imported by both scripts
(removing the two divergent copies — see `feedback_canonical_objective_single_source`).
`tests/test_moo_indicators.py` pins the vectorized forms to the original loops
to 1e-12 (both-sides fixtures) and the filter to the naive O(n²) predicate.

## Defect 3 — Phase 6c mishandled multi-chain archives (silent data loss)

`unified_hv_analysis.py::parse_meta_from_stem` expected exactly
`unified_archive_algo{a}_seed{s}_b{b}` and raised `ValueError` on the
`_chain{id}` suffix `benchmark_engine.py` emits when `chains>1`. Worse, had the
parse merely stripped `_chain`, `run_stats`' `data[algo][seed] = hv` would have
**silently kept only the last of 3 chains** (iteration-order-dependent,
non-reproducible).

**Fix.** Parse the optional `_chain{id}`; group by the documented unit
(algo, seed, budget) and aggregate the 3 chains by **mean of per-chain HV** —
matching Methodology §3.9 ("3 chains aggregated to one observation per seed via
mean") and the chain handling already in `analyze_benchmark.py`,
`analyze_rq2_unified_hv.py`, and `analyze_rq3_spearman.py`. (Mean of per-chain
HV, *not* HV of the unioned points, to stay consistent with that convention.)

## Defect 4 — Phase 6c reloaded all 14,400 archives 8× (once per budget)

`unified_hv.csv` already contains every budget in a single call; `--budget` only
selects which descriptive stats file is written. The submit script nonetheless
invoked `unified_hv_analysis.py` once per budget — ~8× redundant load + HV.

**Fix.** A single invocation writes `unified_hv.csv` (all budgets) plus one
stats file per budget present; `analyze_rq2_unified_hv.py` derives budgets from
the CSV, so one directory suffices.

## Impact on the manuscript

Track B (HV/IGD+/Spacing) and RQ2 (NSGA-II HV vs scalars) had not been exercised
under the canonical run before (see `claim_evidence_audit.md`: "Track B not
currently exercised under canonical"). These are therefore the **first**
canonical Track B / RQ2 numbers, not a correction to published ones. Headline
RQ2: NSGA-II's HV significantly exceeds RS/HC/SGA/TS at nearly all budgets
(VDA 0.7–0.98, Holm p ≪ 0.05) and is statistically tied with SA (VDA ≈ 0.44–0.55)
— the substantive Track-B finding. Final outputs:
`output/paper1_500k_finalize_phase67_20260606_101032/`.

## Lesson

Post-processing/analysis code that has only ever run on toy or partial data
hides defects that surface the first time it runs at full declared scale. Run
the real scale early, and verify each fix against the *documented* intent
(README/Methodology), not just "it executes."

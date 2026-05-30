# Project status

> **Snapshot of long-term state.** This file mirrors observable repo state
> (SLURM queue, pre-submission gate, manifest, tests) at the moment of last
> update. It is not a hand-maintained TODO list — it should be regenerable
> from the commands in the *Verification* section below. The fine-grained
> in-session todo list lives in Claude Code's `TodoWrite` (ephemeral, this
> conversation only); this file is the thing you read to pick up cold.

**Last updated:** 2026-05-09 (during session that landed §3.4 rewrite, float64 migration, manifest validator, and submitted the canonical re-run)

---

## Where the canonical re-run is

| Job | Purpose | Status | Notes |
|-----|---------|--------|-------|
| **28171** | First paper1 attempt | **FAILED** in 3 sec | Stale container missing `optuna` + missing `--corrected-landscape` flag — caught by `optimize_hyperparameters.py` argparse error |
| **28172** | Container rebuild (`build_sif.slurm`) | **COMPLETED** in 11:53 | New 4.7 GB `ti4-analysis.sif` (May 9 13:39); fix landed: `optuna ≥ 4.0` promoted to base `[tool.pixi.pypi-dependencies]` in `pyproject.toml` |
| **28173** | Paper1 canonical re-run (current) | **RUNNING** on rb1-3 | Phase 0 (SA hyperparameter tuning, 50 trials × 100 seeds) in progress; output at `output/paper1_canonical_20260509_134024/` |

Phase order this job will execute (sequential, ~24-36 hr total):

1. **Phase 0** — SA tuning. Output: `optuna_*/best_params.json` (cv_mean, cv_std, held_out_mean, held_out_std + tuned `initial_acceptance_rate`, `min_temp`). ~6 hr.
2. **Phase 1** — Five-condition ablation (`jfi_only`, `moran_only`, `lsap_only`, `jfi_moran`, `full_composite`). Output: `benchmark_*/results.csv`. **The primary empirical demonstration of the paper.** ~12-24 hr.
3. **Phase 4** — LISA proxy validation, 9,999-permutation tests, FDR-corrected at q<0.05. Output: `lisa_validation_*/proxy_validation_summary.json` + `lisa_proxy_per_map_diagnostic.json`. ~3 hr.
4. **Phase 5** — Distance-weight sensitivity (Kendall τ across 6 weight tables). Output: `dist_sensitivity_*/sensitivity_report.txt`. ~6 hr.

All four run under the canonical `--corrected-landscape` formulation defined in Methodology §3.4. Every artifact gets a `run_config.json` sidecar via `src/ti4_analysis/utils/run_config.py` (records `git_hash`, `git_dirty`, resolved weights, per-file SHA-256 hashes of the four metric-defining files).

---

## Pre-submission gate

`scripts/pre_submission_check.sh` runs three checks. Current state:

| Check | State | Detail |
|-------|-------|--------|
| 1. Placeholder tokens (manuscript-facing dirs) | **FAIL** | One marker remaining: `PENDING_CANONICAL_PHASE_0` in `docs/methodology/Methodology_Section.md` §3.7 (B). Will resolve when Phase 0 produces `best_params.json` and the four cv/held-out values get filled in. |
| 2. Methodology §X.Y cross-references | **PASS** | Every §-ref resolves to a defined section in `Methodology_Section.md`; pinned by `tests/test_methodology_cross_refs.py`. |
| 3. Manifest values vs source artifacts | **PASS** | Six entries in `tests/manuscript_values.yaml`, all verified end-to-end against on-disk artifacts (legacy validation_results.csv, post-fix variance_equalization CSV, sensitivity_report.txt, proxy_validation_summary.json, lisa_proxy_per_map_diagnostic.json, lsap_threshold_summary.json). |

---

## Manuscript fill-ins outstanding

All gated on canonical artifacts physically existing on disk:

| Section | Outstanding | Source artifact (will exist when Phase X finishes) |
|---------|-------------|-----------------------------------------------------|
| §3.7 (B) Resiliency of the mean | 4 PENDING values: `cv_mean`, `cv_std`, `held_out_mean`, `held_out_std` | Phase 0 → `optuna_*/best_params.json` |
| §3.9 Ablation Study | per-condition medians + IQRs + Wilcoxon p-values for C0→C4 across 100 seeds × 8 budgets | Phase 1 → `benchmark_*/results.csv` |
| §3.8 Analysis Tracks | unified HV per-algorithm if Track B comparison is included; presently §3.8's prose stops at the variance-equalization cross-reference | Phase 1 → `unified_archives/` (or revisit whether it's needed for the geography paper) |
| §3.3 Metrics references to LISA validation | post-canonical `n_positions` distribution (might still be 29/30/31; might shift) | Phase 4 → `lisa_validation_*/validation_results.csv` |
| Manifest entries | ~20 more entries beyond the current 6 | Phase 0/1/4/5 outputs as they land |

When Phase 0 lands, the immediate workflow:
1. Read `output/paper1_canonical_20260509_134024/optuna_*/best_params.json`
2. Replace 4 PENDING_CANONICAL_PHASE_0 markers in §3.7 (B) with real numbers
3. Add 4 manifest entries with verifier functions reading `best_params.json`
4. Run `pre_submission_check.sh` — Check 1 should now pass

---

## Tests

| Suite | State |
|-------|-------|
| `pytest tests/` | **156/156 passing** |
| `tests/test_metric_parity.py` | 10 tests; `FastMapState` ↔ `spatial_metrics` parity at bit-equality post-float64-migration |
| `tests/test_methodology_cross_refs.py` | 3 tests; every §X.Y resolves; §3.4 subsections 3.4.1–3.4.4 pinned |
| `tests/test_manuscript_values.py` | 3 tests; manifest validator + section-index hierarchy + custom-boundary regex |
| `tests/test_morans_i_null_invariants.py` | 12 tests; `MapTopology.n_spatial`, `morans_i_null`, regime-aware degenerate-graph guards |
| Float64 witness | `tests/_pre_float64_witness.json` carries pre + post blocks; idempotent capture script at `scripts/capture_float64_witness.py` |

---

## Uncommitted state

Current working tree carries the May 2026 session's work uncommitted:

- 5 new infrastructure assets (run_config helper, capture_float64_witness, renumber_methodology_sections, manuscript_values.yaml + validator, test_metric_parity, test_methodology_cross_refs, test_manuscript_values, submit_paper1.sh, _pre_float64_witness.json)
- 19 modified files spanning manuscript text, Phase 0/5 corrected-landscape refactor, float64 migration in `src/`, scripts plumbing the run_config helper
- 2 deletions (root-level stale docs `ACADEMIC_APPROACH.md` + `ADJACENCY_LOGIC_VERIFICATION.md`)

Suggested commit boundaries (in order, when you're ready):
1. `refactor: float32→float64 migration of optimizer hot path; pre/post witness committed`
2. `feat: tests/test_metric_parity.py — golden values + cross-impl parity + dtype assertions`
3. `feat: ti4_analysis.utils.run_config — single-source phase metadata helper`
4. `fix: Phase 0/5 scripts accept --corrected-landscape; pyproject promotes optuna to base`
5. `feat: submit_paper1.sh — geography-paper-scope orchestrator (apptainer-based)`
6. `docs: §3.4 canonical fitness landscape; renumber §3.5–§3.11 cascade; pin via cross-ref regression test`
7. `feat: manuscript_values.yaml + validator + pre_submission_check.sh upgrade`
8. `docs(limitations): Goodhart three-test reframe; manifest +=2 (rho_fdr, lsap_threshold_tau)`
9. `docs: §3.3 / §3.7 / §3.8 canonical-number integration with PENDING markers for Phase-0 deferred values`
10. `docs: Future Work reframe (Problem A swap), drop human-validation framing across docs/`
11. `chore: clean up stale root-level docs; add docs/README.md index; .gitignore += slurm-out`

---

## Verification (how to regenerate this snapshot)

```bash
# Job state
squeue -u $USER
sacct -j 28173 --format=JobID,JobName,State,Elapsed,Submit

# Gate state
bash scripts/pre_submission_check.sh

# Test state
pixi run pytest tests/ -q

# Manuscript outstanding placeholders (just check 1 of the gate)
grep -rE 'PENDING_CANONICAL_[A-Z0-9_]+|\[INSERT\]|\bTBD\b' docs/methodology/ docs/limitations/

# Canonical artifacts on disk
ls -d output/paper1_canonical_*/optuna_*/best_params.json 2>/dev/null
ls -d output/paper1_canonical_*/benchmark_*/results.csv 2>/dev/null
ls -d output/paper1_canonical_*/lisa_validation_*/proxy_validation_summary.json 2>/dev/null
ls -d output/paper1_canonical_*/dist_sensitivity_*/sensitivity_report.txt 2>/dev/null
```

If any of those commands disagree with what this file says, the file is stale and should be updated. Update is a manual edit; refresh-on-meaningful-change is the discipline, not a periodic schedule.

---

## What this file is *not*

- Not a TODO list. The fine-grained in-session work tracking lives in `TodoWrite` (Claude Code's tool-internal task list), not here.
- Not a roadmap. Long-term direction (the next paper on Problem A / anomaly swap) lives in `README.md` *Future Work*.
- Not a reproducibility manifest. That's `tests/manuscript_values.yaml` (artifact-derived numbers) + `*/run_config.json` sidecars (per-run provenance).

This is the answer to "what state is the project in right now" — nothing more, nothing less.

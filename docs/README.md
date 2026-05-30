# `docs/` — index

This directory contains the manuscript material for the geography methodology
paper (Multi-Jain bottleneck JFI + Moran's I + LSAP, with the canonical
formulation defined in Methodology §3.4), plus the audit and limitations
support that backs it. Quick map:

## Manuscript core (the paper itself)

| File | Role |
|------|------|
| [`methodology/Methodology_Section.md`](methodology/Methodology_Section.md) | The §3 methodology section as it will appear in the paper. Section structure: §3.1 Problem Formulation → §3.2 Distance-Weighting → §3.3 Metrics → §3.4 The canonical fitness landscape → §3.5 Algorithms → §3.6 Experimental Protocol → §3.7 Hyperparameter Validation → §3.8 Analysis Tracks → §3.9 Ablation Study → §3.10 Null Hypotheses → §3.11 Statistical Methods. |
| [`methodology/Design_Rationale.md`](methodology/Design_Rationale.md) | Single source of truth for paper scope and experimental design. Pins: contribution is geographic/methodological (not algorithmic, not behavioral); scoping decisions for reviewer feedback in CS or human-subject lanes. |
| [`methodology/FFI_Specification.md`](methodology/FFI_Specification.md) | Forward-looking: the Rust/WASM FFI bridge contract for porting the metric core. Bit-exact reproducibility hooks land here (per-file hashes recorded by `run_config.json`). |

## Limitations (manuscript-active)

Each file documents one specific limitation with the formal reframe / defence
that the manuscript cites:

| File | Topic |
|------|-------|
| [`limitations/limitations.md`](limitations/limitations.md) | Master limitations doc — small-N (\|G\| ≤ 31), permutation-based inference, FDR multi-testing, Goodhart three-test defence, Moran's I boundary violations on row-standardized W. |
| [`limitations/lsap-proxy-goodhart.md`](limitations/lsap-proxy-goodhart.md) | Goodhart's Law full reframe (the three-test defence the limitations.md summary lifts from). |
| [`limitations/anomalies.md`](limitations/anomalies.md) | Asteroid impassability + nebula transparency: Round-0 baseline justification for the strict-block / 0-cost choices; the SA seed=37 temperature-collapse anomaly. |
| [`limitations/held_out_validation_variance.md`](limitations/held_out_validation_variance.md) | Why held-out variance is a property of starting-state difficulty, not hyperparameter overfitting. |
| [`limitations/row-standardization-edge-effects.md`](limitations/row-standardization-edge-effects.md) | Bounded-tessellation edge effects + the √k local-variance correction (now §3.4.3 in the canonical formulation). |
| [`limitations/drf-terminology-framing.md`](limitations/drf-terminology-framing.md) | Multi-Jain bottleneck framing relative to Dominant Resource Fairness (Ghodsi et al. 2011) — DRF-inspired, not DRF-claiming. |

## Audit (May 2026 codebase audit pass)

The May 2026 audit caught a cluster of latent drift bugs (documentation,
configuration, computation) between the pixi+apptainer migration and the
manuscript draft. All findings have a fix commit; this archive documents the
drift taxonomy and the structural lessons.

| File | Topic |
|------|-------|
| [`audit/README.md`](audit/README.md) | Audit overview, three drift categories, findings table, audit-driven vs test-driven dev methodology note. |
| [`audit/n_reconciliation.md`](audit/n_reconciliation.md) | The four spatial-graph N values (37 / 31 / 30 / 14), reconciled to per-map \|G\| via `MapTopology.n_spatial`. |
| [`audit/parallel_weight_sources.md`](audit/parallel_weight_sources.md) | Variance-equalization diagnostic's two-weight-source bug (5:5:3 in reporting vs 1:1:1 in SA); single-source `--weights` resolution. |
| [`audit/analyzer_composite_recompute.md`](audit/analyzer_composite_recompute.md) | Analyzer's composite-recompute used \|I\| + n=37 instead of the optimizer's hinge + n=31; shared-helper resolution. |
| [`audit/safety_bounds.md`](audit/safety_bounds.md) | Regime-aware vs syntactic safety bounds (the n_spatial=0 zero-tile-hinge latent bug). |
| [`audit/jfi_audit_TODO.md`](audit/jfi_audit_TODO.md) | **RESOLVED** — JFI parallel-implementation parity now pinned by `tests/test_metric_parity.py`; retained as historical context. |

## Methods justification (single-issue files)

| File | Topic |
|------|-------|
| [`tabu_search_justification.md`](tabu_search_justification.md) | Why TS is included as a methodological control alongside SA / SGA / NSGA-II. |
| [`bayesian_optimization_exclusion.md`](bayesian_optimization_exclusion.md) | Why BO is *excluded* from the algorithm panel (cost-model mismatch with cheap fitness evaluations). |

## Reviewer-facing

| File | Role |
|------|------|
| [`Response_to_Reviewers_Template.md`](Response_to_Reviewers_Template.md) | Template responses to anticipated reviewer concerns (LSAP-as-heuristic, weighting, small-N inference, etc.); fill in after first peer review. |

## Literature review

| Directory | Content |
|-----------|---------|
| [`lit_review/`](lit_review/) | Research notes from external sources (curated `.md` consensus reviews + raw `.txt` AI-conversation captures). Internal research material; not part of the manuscript itself. |

## Companion machinery (outside `docs/`)

The manuscript's verification infrastructure lives in `tests/` and is invoked
by `scripts/pre_submission_check.sh`:

- [`../tests/test_methodology_cross_refs.py`](../tests/test_methodology_cross_refs.py) — every §X.Y reference in the manuscript resolves to a defined section.
- [`../tests/test_manuscript_values.py`](../tests/test_manuscript_values.py) — every artifact-derived value cited in the manuscript matches its source artifact and appears in the prose; manifest at [`../tests/manuscript_values.yaml`](../tests/manuscript_values.yaml).
- [`../tests/test_metric_parity.py`](../tests/test_metric_parity.py) — `FastMapState` ↔ `spatial_metrics` parity at bit-equality + golden-value pins.
- [`../tests/_pre_float64_witness.json`](../tests/_pre_float64_witness.json) — frozen pre/post float64 migration witness; the migration's evidence artifact.

## What's *not* here, deliberately

- A future-work / next-steps doc — the only future direction in scope (Problem A swap, anomaly placement as the swap variable) is documented inline in `README.md` *Future Work* and cross-referenced from `methodology/Design_Rationale.md` §6 and `methodology/Methodology_Section.md` §3.8.
- Deleted root-level docs (May 2026 cleanup): `ACADEMIC_APPROACH.md` (subsumed by README + Methodology), `ADJACENCY_LOGIC_VERIFICATION.md` (March 9 verification, retired G1/G3-D vocabulary), `07_POST_RUN_CHEAT_SHEET.md` (deleted in `6d12a55`, content absorbed into the audit/limitations files).

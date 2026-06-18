# Claim-Evidence Audit (May 2026)

> **Closure update (2026-06-18).** This is the dated May 2026 audit, preserved verbatim as the registration trace. Several §§3.10–3.11 verdicts below (RQ2/RQ3/RQ4 "not tested", Cohen's $d_z$ "not computed", Track B "not exercised") were the audit-time state and have since been **closed** in the canonical re-run. Each is marked **SUPERSEDED** inline and the closure is summarized under §§3.10–3.11; see also `docs/audit/rq4_evals_to_best_close_2026-06.md`. The original rows are kept unchanged so the registered-then-closed provenance chain stays legible (the same "date the deviation against the registration, do not erase it" move §3.10 makes for RQ4).

For every empirical or methodological claim made in the paper, this audit
traces two parallel chains:

- **Analytical chain**: where the math is derived (a docstring deriving the
  formula, a methodology paragraph showing the inequality, or a citation
  pointer backed by a re-derivation in our notation)
- **Numerical chain**: where the test that verifies the math lives (a
  regression test pinning the invariant, an artifact whose computed value
  matches the analytical prediction at machine precision, a formal stats
  panel computing significance)

**Verdict per claim:**

| Symbol | State | Action |
|--------|-------|--------|
| ✅ | Both math and test present | claim has both legs of support |
| ⚠ math-only | Derivation exists; numerical test missing | gap; add a test |
| ⚠ test-only | Empirical result without analytical motivation | add derivation or downgrade to descriptive |
| ⚠ malformed | Test exists but doesn't compare what the math says | fix the test |
| ❌ neither | Claim is unsupported | derive AND test before citing |

The audit was run as part of the May 2026 canonical re-run integration.
The methodology that requires this audit lives at
[`feedback_claim_evidence_audit_pattern.md`](../../../.claude/projects/-home-mpk15956-projects-ti4-analysis/memory/feedback_claim_evidence_audit_pattern.md)
and the user-level research methodology it verifies lives at
[`feedback_research_methodology_math_first_then_numerical.md`](../../../.claude/projects/-home-mpk15956-projects-ti4-analysis/memory/feedback_research_methodology_math_first_then_numerical.md).

---

## §3.1 Problem Formulation

| Claim | Math | Test | Verdict |
|-------|------|------|---------|
| Composite $S = w_1(1-J_{\min}) + w_2 \max(0, I - E[I]) + w_3 \cdot \text{LSAP}/[n(n-1)]$ | §3.1 prose, formula displayed; per-term derivation in §3.3 | `MultiObjectiveScore.composite_score` (`spatial_optimizer.py:107-169`) implements the formula; `tests/test_metric_parity.py` golden values for I and JFI components | ✅ |
| $E[I] = -1/(n-1)$ under no spatial association | §3.3 prose; cite Anselin 1995; derived in `morans_i_null` docstring (`map_topology.py:27-50`) | `tests/test_morans_i_null_invariants.py` (12 tests); regression-pinned at canonical $|G|=31$ giving $E[I] = -1/30$ | ✅ |
| Multi-Jain bottleneck $J_{\min} = \min(J_R, J_I)$ | §3.1 prose; cite Ghodsi et al. 2011 (DRF-inspired, not DRF-claiming per `docs/limitations/drf-terminology-framing.md`) | `FastMapState.jains_index()` uses `min(jfi_resources, jfi_influence)`; `tests/test_metric_parity.py` golden values for both `_jfi` impls | ✅ |
| 1:1:1 weights are operationally consistent | §3.1 prose + cross-ref to §3.4.4 + §3.8 two-regime framing (Gen-0 90% hinge / convergence 56% LSAP / 44% JFI) | `output/verify_pixi_apptainer_20260506_205045/variance_equalization_20260506_224438.csv`; manifest entries `gen0_moran_share_pct`, `convergence_lsap_share_pct`, `convergence_jfi_share_pct` all verified | ✅ |
| All three terms normalized to $[0,1]$ before weighting | §3.1 prose; bounds derived: $1-J_{\min} \in [0,1]$, hinge $\in [0, 1+1/(|G|-1)]$, $\text{LSAP}/(|G|(|G|-1)) \in [0,1]$ | The hinge upper bound $\approx 1.033$ is cited but **NOT numerically tested** (no test asserts the empirical maximum stays within this bound across the canonical map space) | ⚠ math-only |

---

## §3.2 Distance-Weighting Model

| Claim | Math | Test | Verdict |
|-------|------|------|---------|
| Step-function decay weights (Joebrew 6,6,6,4,4,2,1,0) | §3.2 prose; community-calibrated, no formal derivation (an empirical evaluator choice, not a theoretical claim) | Used by `BalanceEngine.get_distance_multiplier`; tested for round-trip via `tests/test_balance_engine.py` (18 tests pass) | ✅ |
| Asteroid impassability + nebula 0-cost (Round-0 baseline) | `docs/limitations/anomalies.md` derivation: Move-1 carrier baseline | Behavioral test in `tests/test_balance_engine.py` covers anomaly handling | ✅ |
| Mean Kendall's τ across distance-weight perturbations = 1.000 (n=15 pairwise) | §3.2 prose; empirical claim (no math proof; "if all rankings are invariant, τ=1" is a definition, not a derivation) | `output/paper1_canonical_20260509_134024/dist_sensitivity_20260509_204651/sensitivity_results.csv` + manifest entry `distance_weight_kendall_tau` (verifier computes τ across all 15 pairs from CSV) | ✅ |

---

## §3.3 Metrics

| Claim | Math | Test | Verdict |
|-------|------|------|---------|
| Spatial graph $G$ construction (SYSTEM tiles minus home, impassable-edge excision, zero-degree purge) | §3.3 prose; constructive definition in `MapTopology.from_ti4_map` (`map_topology.py:153+`) | `tests/test_morans_i_null_invariants.py::_canonical_6p_topology` builds the canonical layout; `MapTopology.from_ti4_map` postcondition asserts `n_spatial == 0 or n_spatial >= 2` | ✅ |
| $29 \le |G| \le 31$ depending on per-seed anomaly placement; distribution 52/52/16 | §3.3 prose; constructive (number of zero-degree purges depends on Supernova placement) | `output/saturation_20260314_205919/lisa_validation_20260316_100413/validation_results.csv` `n_positions` column verified to give the cited distribution (legacy artifact; canonical Phase 4 has only 30 maps so distribution structure may differ) | ✅ |
| Moran's I formula | §3.3 displayed formula; classical Moran 1950 | `tests/test_metric_parity.py::test_morans_i_standalone_path_graph_golden` pins $I = 0.4$ on hand-computed 4-node path; cross-impl parity at bit-equality | ✅ |
| Row-standardization mitigates edge-effect bias | §3.3 prose; cite Boots & Tiefelsdorf 2000 | `MapTopology` constructs row-standardized $\mathbf{W}$ (rows sum to 1.0); regression test would assert this empirically — **NOT explicitly tested as an invariant** (relies on construction correctness only) | ⚠ math-only |
| LSAP $\le n(n-1)$ → LSAP/(n(n-1)) $\in [0,1]$ | §3.3 + `composite_score` docstring derives the bound (leaf-node $k_i=1$ case → $\max I_i = (n-2)/2$, sum bound) | The conservative bound in the docstring is not empirically tested against the actual maximum LSAP observed across canonical maps — no regression that says "LSAP/(n(n-1)) is always in [0,1]" | ⚠ math-only |
| 21/12,000 (0.17%) Moran's I < −1 boundary violations on row-standardized $\mathbf{W}$ | `docs/limitations/limitations.md` "Moran's I boundary violations" derivation; cite de Jong, Sprenger & van Veen 1984; Anselin & Rey 2014 | Verified in this session's audit: `pd.read_csv(phase1_results)[df.morans_i < -1]` returns 21 rows in [-1.063, -1.001] (legacy `benchmark_20260314_233002/results.csv`); **no regression test pins this count under canonical** — and the canonical results.csv may produce a different count | ⚠ test-on-legacy-only |

---

## §3.4 Canonical Fitness Landscape

### §3.4.1 Smooth-min Jain bottleneck

| Claim | Math | Test | Verdict |
|-------|------|------|---------|
| Log-sum-exp relaxation of $\min$ at $p=8$ | §3.4.1 displayed formula; cite Boyd & Vandenberghe 2004 §3.1.5 | `objectives_smooth.smooth_min_jain` exists; `tests/test_structural_corrections.py` has tests `test_smooth_min_jain_approaches_min_as_p_large`, `test_smooth_min_jain_clamps_away_from_zero`, `test_smooth_min_jain_symmetric` | ✅ |
| **At $p=8$ the relaxation tracks $\min$ to within $\log(2)/p \approx 0.087$** | §3.4.1 prose; standard log-sum-exp bound | **NO numerical test asserts this bound holds on the actual JFI input distribution**. The structural-corrections tests check the limit-as-p-large behavior but not the specific 0.087 bound at $p=8$ on real per-player JFI inputs | ⚠ math-only |
| The bound is "well below the per-dimension JFI variance observed at convergence" | §3.4.1 prose | The convergence JFI variance is reported by the variance-equalization diagnostic ($\sigma_{\text{jfi}} \approx 0.00211$ at convergence), and 0.087 ≫ 0.00211 — so the bound is actually NOT well below JFI variance, the prose statement is reversed. **Re-read needed** | ⚠ malformed claim |

### §3.4.2 Softplus hinge

| Claim | Math | Test | Verdict |
|-------|------|------|---------|
| Softplus relaxation of hinge at $k=10$ | §3.4.2 displayed formula; same Boyd & Vandenberghe §3.1.5 (softplus = log-sum-exp of $\{0, x\}$) | `objectives_smooth.softplus_hinge` exists; tests `test_softplus_hinge_positive_x`, `test_softplus_hinge_negative_x`, `test_softplus_hinge_zero`, `test_softplus_k_clamped` in `tests/test_structural_corrections.py` | ✅ |
| **At $k=10$ the relaxation deviates from the hinge by at most $\log(2)/k \approx 0.069$** | §3.4.2 prose; standard softplus bound | **NO numerical test asserts this bound on actual input distributions**. Tests check sign behavior at the boundary but not the magnitude of approximation error at $k=10$ | ⚠ math-only |
| For $|G|$ in our range, $|E[I]|$ exceeds 0.069, preserving null-comparison structure | $E[I] \in [-0.036, -0.033]$ for $|G| \in [29, 31]$; 0.033 < 0.069 — **claim is mathematically incorrect**. $|E[I]|$ does NOT exceed $\log(2)/k$ at $k=10$ | Computable: `morans_i_null(31) = -0.033` < 0.069. The claim as stated is wrong | ❌ |

### §3.4.3 Variance-stabilized LSAP

| Claim | Math | Test | Verdict |
|-------|------|------|---------|
| $\text{Var}[I_i] \propto 1/k_i$ under row-standardized $\mathbf{W}$ | §3.4.3 prose; cite Anselin 1995 | **NO numerical test verifies this variance scaling under permutation on the canonical layout**. The √k-correction is applied (in `lisa_penalty_swappable(use_local_variance=True)`) but its effectiveness at variance-equalization is not empirically verified | ⚠ math-only |
| Canonical 6p layout has $k_i \in \{3,5,6\}$ with 12/6/13 split | §3.4.3 prose | Verified in this session via `(spatial_W > 0).sum(axis=1)` on `_canonical_6p_topology()` — confirmed exact split. **Could be a regression test but is not** | ⚠ test-only (one-shot, not regression) |
| $\sqrt{6/3} = \sqrt{2}$ inflation ratio | §3.4.3 prose | Mathematically identity-checked but **not pinned in any test** | ⚠ math-only |
| 12 of 31 nodes (39%) at boundary contribute at √2-inflated standard deviation | §3.4.3 prose | Constructive from the 12/6/13 split; **not regression-tested** | ⚠ test-only (one-shot) |

### §3.4.4 Static Gen-0 σ normalization

| Claim | Math | Test | Verdict |
|-------|------|------|---------|
| Dynamic per-iteration σ produces a composite that is not a well-defined function of map state | §3.4.4 prose; the path-dependence argument | The argument is purely analytical; no test runs dynamic vs static σ side-by-side to verify the path-dependence claim. **Defensible as math-only** since the claim is about a mathematical property of the formulation, not an empirical observation | ⚠ math-only |
| Moran hinge accounts for ~90% of total weighted variance at Gen-0 under 1:1:1 | §3.4.4 prose; cross-ref §3.8 | `output/verify_pixi_apptainer_20260506_205045/variance_equalization_20260506_224438.csv` + manifest entry `gen0_moran_share_pct` (verifier computes share at 1e-1 tolerance) | ✅ |
| $N=1{,}000$ Gen-0 sample size | §3.4.4 prose; argued as "empirical property of the unoptimized problem domain" — choice of $N$ is methodological | `compute_gen0_sigma(n_samples=1000)` in spatial_optimizer.py; **no test pins N=1000 as a calibrated choice** (e.g., that the σ estimate has converged at this $N$) | ⚠ math-only |

---

## §3.5 Algorithms

| Claim | Math | Test | Verdict |
|-------|------|------|---------|
| HC accepts only improving moves; SA Metropolis criterion; SGA scalar tournament; NSGA-II three-objective Pareto; TS full-neighborhood; RS uniform | §3.5 prose; standard algorithm descriptions with citations (Kirkpatrick 1983, Deb 2002, Glover 1989) | Each has a test file (`test_hc_optimizer.py`, `test_sga_optimizer.py`, etc.); 156 total tests pass | ✅ |
| SA vs SGA = same scalar, different architecture | §3.5 prose; constructive ("same composite optimization target") | Empirical comparison happens in Phase 2b but **Phase 2b is frozen as legacy**, not part of canonical paper. The claim is descriptive of the experimental design, not a tested empirical claim | ✅ (descriptive) |
| Tabu tenure $\max(3, \lceil 0.05 \cdot \binom{S}{2} \rceil)$ | §3.5 prose; cite Glover 1989 | Implemented in `tabu_search_optimizer.py`; `tests/test_tabu_search.py` has 10 tests including `test_ts_tenure_coefficient_equivalent_to_default` | ✅ |

---

## §3.6 Experimental Protocol

| Claim | Math | Test | Verdict |
|-------|------|------|---------|
| Seeds 0-99 benchmark, 9000-9099 tuning, 9100-9149 held-out (disjoint) | §3.6 prose; constructive | `submit_paper1.sh` configuration variables; `optimize_hyperparameters.py` and `benchmark_engine.py` argparse defaults; **no test pins seed-set disjointness as an invariant** | ⚠ math-only (definitional, not invariant-tested) |
| Quasi-logarithmic budget grid 1k–500k | §3.6 prose; descriptive | `submit_paper1.sh` BUDGETS variable; not a math claim, just a methodological choice | ✅ (descriptive) |
| Bootstrap 95% CIs on median differences | §3.6 + §3.10 prose; standard non-parametric inference | `analyze_phase1_conditions.py` computes bootstrap CIs; `phase1_condition_pairs.csv` reports `ci_lo, ci_hi` per pair | ✅ |

---

## §3.7 Hyperparameter Validation and Held-Out Variance

| Claim | Math | Test | Verdict |
|-------|------|------|---------|
| (A) Variance reflects starting-state difficulty, not overfitting | §3.7(A) prose; argued from the rugged-landscape framing | Empirically supported by (B): held-out σ ≤ CV σ; **the argument that this is starting-state-difficulty rather than overfitting is itself analytical, not directly tested** (would require a separate experiment varying tuning seed-set with fixed held-out) | ⚠ math-only |
| (B) CV mean ± std = $0.0914 \pm 0.0447$ vs held-out $0.0872 \pm 0.0423$ | §3.7(B) prose; empirical claim | `output/paper1_canonical_20260509_134024/optuna_20260509_134028/best_params.json` + manifest entries `phase0_sa_cv_mean`, `phase0_sa_cv_std`, `phase0_sa_held_out_mean`, `phase0_sa_held_out_std` | ✅ |
| Coefficient of variation ~49% reflects rugged landscape | §3.7(B) prose; cross-ref to (A) | Computed from the CV mean+std but **not pinned in manifest as its own entry**; could add `phase0_sa_cv_coefficient_of_variation_pct = 49` | ⚠ test-only (computed but not pinned) |
| (C) Phase 2 mitigates via 100-seed Friedman+Wilcoxon+Holm-Bonferroni | §3.7(C) prose | Implemented in `analyze_phase1_conditions.py`; outputs at `phase1_condition_*.csv/.txt` | ✅ |

---

## §3.8 Analysis Tracks

| Claim | Math | Test | Verdict |
|-------|------|------|---------|
| 90% Gen-0 hinge share, 56% LSAP / 44% JFI at convergence | §3.8 prose; two-regime framing | `output/verify_pixi_apptainer_20260506_205045/variance_equalization_20260506_224438.csv` + manifest entries `gen0_moran_share_pct`, `convergence_lsap_share_pct`, `convergence_jfi_share_pct` | ✅ |
| Track A vs Track B distinction | §3.8 prose; cite Ishibuchi 2015, Zitzler & Thiele 1999 | Track B (HV/IGD+/Spacing) requires NSGA-II; canonical paper uses SA-only via submit_paper1.sh, so **Track B is not currently exercised under canonical**. Methodology declares the framework; the canonical paper uses Track A only | ⚠ scope-only (declared but not tested under canonical) |
| Methodological scope (toy problem; Problem A is future work) | §3.8 prose; cross-ref README Future Work | Not a numerical claim; descriptive | ✅ (descriptive) |

---

## §3.9 Ablation Study (canonical Phase 1)

| Claim | Math | Test | Verdict |
|-------|------|------|---------|
| Friedman omnibus $\chi^2 = 349.82, 384.88, 367.14$ for Moran/LSAP/JFI ($df=4$, all $p \ll 10^{-70}$) | §3.9 prose; standard non-parametric omnibus | `output/paper1_canonical_20260509_134024/benchmark_20260509_191848/stats/phase1_condition_report.txt`; **not yet manifest-tracked** | ⚠ test-only (not pinned in manifest) |
| C0 medians: Moran=$-0.0986$, LSAP=$4.4571$, JFI=$0.99993$ | §3.9 prose; empirical | manifest entries `phase1_c0_morans_i_median_b500k`, `phase1_c0_lisa_penalty_median_b500k` (chain-mean / seed-median, verified) | ✅ |
| C3 medians: Moran=$-0.5926$, LSAP=$0.9674$ (parsimony test fails) | §3.9 prose; empirical | manifest entries `phase1_c3_morans_i_median_b500k`, `phase1_c3_lisa_penalty_median_b500k` | ✅ |
| C4 medians: Moran=$-0.6714$, LSAP=$0.0000$ | §3.9 prose; empirical | manifest entries `phase1_c4_morans_i_median_b500k`, `phase1_c4_lisa_penalty_median_b500k` | ✅ |
| Pairwise C0 vs Cx all significant (Holm-Bonferroni) with VDA $\geq 0.718$ | §3.9 prose; output of analyze_phase1_conditions.py | `phase1_condition_pairs.csv`; **VDA values not yet manifest-tracked** | ⚠ test-only (not pinned) |
| Δ(Moran's I) C0→C4 = +0.5728, 95% CI [+0.5362, +0.6030] | §3.9 prose; empirical | `phase1_condition_pairs.csv` rows; **not yet manifest-tracked** | ⚠ test-only (not pinned) |
| Δ(LSAP) C0→C4 = +4.4571, 95% CI [+4.1102, +4.9822] | §3.9 prose; empirical | `phase1_condition_pairs.csv`; **not yet manifest-tracked** | ⚠ test-only (not pinned) |
| JFI parity nuance: C3/C4 sacrifice statistically detectable but operationally tiny ($\Delta \approx 0.0001$) | §3.9 prose | `phase1_jfi_parity.csv`; **not yet manifest-tracked** | ⚠ test-only (not pinned) |

---

## §§ 3.10–3.11 Null hypotheses + Statistical methods

> **SUPERSEDED 2026-06-18: closed in the canonical re-run.** At audit time (May 2026) RQ2/RQ3/RQ4 and Cohen's $d_z$ were untested in the canonical run; all are now closed. RQ2 (NSGA-II vs scalar hypervolume) and RQ3 (balance-gap × spatial Spearman) are tested in §3.10 against the multi-algorithm finalize artifacts; Cohen's $d_z$ is computed by `analyze_phase1_conditions.py` and pinned in `manuscript_values.yaml`. RQ4 is closed as a **declared pre-specified deviation** from the registered wall-clock metric: a six-way Friedman on `evals_to_best` ($\chi^2 = 456.80$, $df = 5$, $p = 1.7 \times 10^{-96}$; §3.10), with wall-clock retained as secondary descriptive. The audit-time rows below are preserved verbatim as the registration trace.

| Claim | Math | Test | Verdict |
|-------|------|------|---------|
| RQ1–RQ4 null hypotheses (pre-registered) | §3.10 prose; pre-registration documents | RQ1 tested by Friedman (✅); RQ2 (Pareto HV vs scalar) NOT tested — Track B not in canonical paper; RQ3 (Spearman correlations) NOT tested in current canonical run; RQ4 (wall-clock comparison) NOT formally tested as a Friedman | ⚠ partial (RQ1 ✅, RQ2/3/4 ❌) · **SUPERSEDED 2026-06**: RQ2/3/4 closed, see closure note above + §3.10 |
| Cohen's $d_z$ for paired comparisons; Vargha-Delaney bands | §3.11 prose; cite Lakens 2013 | `analyze_phase1_conditions.py` computes VDA but **not Cohen's $d_z$ explicitly**; the methodology promises both | ⚠ test-only / partial · **SUPERSEDED 2026-06**: $d_z$ now computed and pinned |

---

## Limitations document

| Claim | Math | Test | Verdict |
|-------|------|------|---------|
| At $|G|=31$, asymptotic normality unjustified for Moran's I significance | `limitations.md` prose; standard small-N statistical theory | Permutation testing in `validate_lisa_proxy.py` is the operationally adopted alternative; **no test asserts asymptotic normality is unjustified** (it's a statistical-theory claim about the regime, not an empirical claim about this dataset) | ⚠ math-only (defensible) |
| Goodhart Test 1 (canonical): $\rho = +0.071, p = 0.711, r = +0.056$, precision@$\tau=1.0$ = 14.3% | `limitations.md` §5; per-map diagnostic | `output/paper1_canonical_20260509_134024/lisa_validation_20260509_204628/proxy_validation_summary.json` + manifest `goodhart_rho_alpha05_canonical` (regex-pinned) | ✅ |
| Goodhart Test 2 (canonical): $\rho_{\text{FDR}} = +0.290, p = 0.120$ | `limitations.md` §5 | `output/paper1_canonical_20260509_134024/lisa_validation_20260509_204628/lisa_proxy_per_map_diagnostic.json` + manifest `goodhart_rho_fdr_canonical` | ✅ |
| Goodhart Test 3 (legacy): $\tau = 0.949, p = 2.3 \times 10^{-22}$ | `limitations.md` §5; pre-registered defence | `output/saturation_20260314_205919/lsap_threshold_20260316_194325/lsap_threshold_summary.json` + manifest `lsap_threshold_kendall_tau` | ✅ (legacy) |
| **Goodhart Test 3 canonical re-run "pending"** | `limitations.md` §5 acknowledges canonical-form re-run as natural follow-up | The canonical-mode refactor of `lsap_threshold_sensitivity.py` (this session) **compares two different functionals** (√k-stabilized baseline vs raw thresholded) — produces $\tau=0.097$ but the test is malformed under canonical. Proper canonical Test 3 requires a √k-stabilized thresholded variant in `FastMapState` (not yet implemented) | ⚠ malformed |
| 21/12,000 (0.17%) Moran's I < −1 boundary violations | `limitations.md` "Moran's I boundary violations" subsection; cite de Jong 1984; Anselin & Rey 2014 | Verified on legacy `benchmark_20260314_233002/results.csv`; **not regression-tested**; **canonical Phase 1 (`benchmark_20260509_191848/results.csv`) may have a different count — not yet checked** | ⚠ test-on-legacy-only |
| D₆ dihedral search-space symmetry inflates redundancy by ~12× | `limitations.md` "Search Space Symmetry" subsection | No symmetry-detection test is run on the optimized maps; this is a math-only argument with empirical implications that aren't tested | ⚠ math-only |

---

## README claims

| Claim | Math | Test | Verdict |
|-------|------|------|---------|
| "Spatial metrics add meaningful constraint beyond JFI" — primary contribution | README abstract; backed by §3.9 ablation | §3.9 results (Phase 1 canonical) — significant Wilcoxon C0 vs C4 across Moran/LSAP/JFI | ✅ |
| Five-condition ablation runs C0–C4 | README abstract | Phase 1 results.csv shows 100 seeds × 5 conditions × 8 budgets × 3 chains | ✅ |
| "JFI parity preserved" (in headline) | README abstract; empirical | The §3.9 nuance ("statistically detectable, operationally tiny") shows the README's "parity" framing is slightly stronger than the formal statistical result. **The README abstract should mirror §3.9's nuance** | ⚠ wording-drift |
| Future Work: Problem A swap (anomaly placement as variable) | README Future Work; cross-ref Methodology §3.8 + Design_Rationale §6 | Descriptive direction; not a numerical claim | ✅ (descriptive) |

---

## Outstanding gaps tally

### ❌ Critical (RESOLVED in May 2026 follow-up pass)

1. ~~**§3.4.2 incorrect bound claim**~~ — RESOLVED. Replaced the false "$|E[I]|$ exceeds $\log(2)/k$" claim with the correct accounting: at $k=10$ the softplus deviation $\le \log(2)/k \approx 0.069$ is *comparable* to $|E[I]| \in [0.033, 0.036]$, not below it. The trade-off is now reported honestly (smoothing required for SA gradient signal, deviation is one-sided and bounded, optimum location unchanged).
2. ~~**§3.4.1 reversed comparison**~~ — RESOLVED. Removed the false "well below convergence JFI variance" framing; replaced with the correct one-sided bound statement and an explicit argument that the relaxation preserves the location of the JFI maximum at $J_R = J_I = 1$.
3. ~~**Goodhart Test 3 canonical: malformed test**~~ — RESOLVED. Implemented `FastMapState.lisa_penalty_swappable_thresholded(τ, use_local_variance=True)` — same √k-stabilized form as the baseline plus the threshold floor — and re-ran canonical Test 3 with the proper same-form comparison. **Result: $\tau = 0.5331$, $p = 4.7 \times 10^{-8}$ ($n=50$ SA seeds, budget 1000).** Statistically positive but **fails the pre-registered $\tau > 0.90$ defence threshold**. Reported honestly per `feedback_research_methodology_math_first_then_numerical.md`: the LSAP proxy is structurally sensitive to threshold choice under canonical $\sqrt{k}$-stabilization; the Goodhart defence under canonical now rests on Tests 1 and 2 (which pass) and acknowledges Test 3's failure rather than glossing over it.

### ⚠ math-only (add numerical verification)

4. §3.1: composite term bounds $[0, 1.033]$ — no regression test
5. §3.3: row-standardization mitigates edge-effect bias — implementation tested for sum-to-1; effect on Moran's I bias not directly tested
6. §3.3: LSAP $\le n(n-1)$ analytical bound — never empirically tested
7. ~~§3.4.1: smooth-min bound~~ — RESOLVED. Discovered prose/implementation drift in the process: the prose previously cited log-sum-exp ($-\frac{1}{p}\log\sum e^{-pJ}$) but the implementation is actually the generalized power-mean $M_{-p}(J_R, J_I) = ((J_R^{-p} + J_I^{-p})/2)^{-1/p}$. Prose now describes the implementation (Hardy/Littlewood/Pólya power-mean inequality, multiplicative slack $2^{1/p}-1 \approx 0.091$ at $p=8$), and `tests/test_structural_corrections.py::test_smooth_min_jain_power_mean_bound` empirically verifies the bound across the $(0, 1]^2$ grid.
8. ~~§3.4.2: softplus bound~~ — RESOLVED. `tests/test_structural_corrections.py::test_softplus_hinge_log2_over_k_bound` verifies $\mathrm{softplus}_k(x) \in [\max(0,x), \max(0,x) + \log(2)/k]$ on a 1001-point grid in $[-2, 2]$.
9. ~~§3.4.3: $\sqrt{k}$ variance-stabilization~~ — RESOLVED. `tests/test_structural_corrections.py::test_sqrt_k_variance_equalization_canonical_layout` runs a 2,000-iter Monte Carlo on the canonical layout under H₀ and confirms $\sqrt{k}$-scaling reduces inter-node variance heterogeneity (CV[$\mathrm{Var}[I_i^{\text{corr}}]$] < CV[$\mathrm{Var}[I_i]$]).
10. ~~§3.4.3: degree distribution $\{3,5,6\}$ split 12/6/13~~ — RESOLVED. `tests/test_structural_corrections.py::test_canonical_6p_degree_split` pins it as a regression invariant.
11. §3.4.4: $N=1000$ Gen-0 sample size convergence — not tested
12. §3.6: seed-set disjointness — not invariant-tested
13. §3.7(A): "starting-state difficulty" framing — argued but not separately tested

### ⚠ test-only (not yet manifest-tracked)

14. §3.7(B): cv-coefficient-of-variation 49% — computed but not pinned
15. ~~§3.9: Friedman $\chi^2$ values~~ — RESOLVED. Three new manifest entries (`phase1_friedman_chi2_morans_i`, `..._lisa_penalty`, `..._jains_index`) verify $\chi^2 = 349.82, 384.88, 367.14$ via direct re-computation from `results.csv`.
16. §3.9: pairwise VDA values — not in manifest
17. ~~§3.9: 95% CIs on median diff~~ — PARTIAL. Two new manifest entries (`phase1_c0_vs_c4_morans_i_median_diff`, `..._lisa_penalty_median_diff`) pin the headline C0→C4 deltas; the surrounding 95% CIs are reported in §3.9 prose but not yet pinned per-bound.
18. §3.9: JFI parity Wilcoxon W — not in manifest

### ⚠ partial / scope mismatch

19. §3.8: Track B (HV/IGD+/Spacing) declared but canonical paper uses SA-only — declared but not exercised in canonical · **SUPERSEDED 2026-06**: Track B now exercised in the multi-algo canonical (`quality_indicators.csv`; RQ2 unified HV)
20. §§3.10–3.11: RQ2 (Pareto HV), RQ3 (Spearman correlations), RQ4 (wall-clock) not formally tested in the canonical run · **SUPERSEDED 2026-06**: closed (RQ4 as a declared deviation to `evals_to_best`; see §3.10)
21. §3.11: Cohen's $d_z$ not computed (only VDA) · **SUPERSEDED 2026-06**: $d_z$ now computed and pinned
22. ~~README abstract: "JFI parity preserved" too strong~~ — RESOLVED. README abstract now reads "JFI parity within measurement-meaningful tolerance ($\Delta \text{JFI} \le 0.0002$ at convergence)", mirroring §3.9's nuance.

### ⚠ test-on-legacy-only (canonical re-verification pending)

23. ~~§3.3: 21/12,000 Moran's I < −1 boundary violations~~ — RESOLVED. Canonical Phase 1 verified: 29 of 12,000 (0.242%) in $[-1.047, -1.001]$, confined to `moran_only`/`lsap_only` at budgets ≥ 25,000. Pinned in manifest as `phase1_morans_i_lt_neg1_count`. §3.3 + limitations.md updated.
24. ~~limitations.md: same boundary violations claim~~ — RESOLVED (same fix as #23).

---

## Suggested follow-up order

In order of materiality to the headline contribution:

1. **Fix §3.4.2's reversed/incorrect bound claim** (text edit; ~10 min)
2. **Fix §3.4.1's reversed comparison** (text edit; ~10 min)
3. **Implement `lisa_penalty_swappable_thresholded` in `FastMapState`** + rerun canonical Test 3 (~30 min code + ~10 min run); update Goodhart Test 3 prose with whatever the proper canonical τ returns
4. **Re-verify the 21/12,000 boundary-violations claim against canonical Phase 1** (`benchmark_20260509_191848/results.csv`); update §3.3 + limitations.md with canonical count if different
5. **Add manifest entries for Friedman χ², pairwise VDA, 95% CIs** (the §3.9 stats panel becomes manifest-tracked end-to-end)
6. **Soften the README abstract** to mirror §3.9's "preserved within 0.0002" nuance instead of "preserved" full stop
7. **Add numerical verification tests** for the bounds in §3.4.1, §3.4.2, §3.4.3 (write tests that assert the bounds hold on the actual canonical input distributions; ~1 hour total)
8. **Decide scope**: §§3.10–3.11 promise tests (RQ2/3/4, Cohen's $d_z$) the canonical run does not deliver. Either run them, or update the methodology to reflect what's actually tested

Items 1–4 were blocking for honest submission. **Items 1–4 closed in the May 2026 follow-up pass** (§3.4.1 / §3.4.2 text fixes; canonical Test 3 same-form rerun returning $\tau=0.5331$ with the failure honestly disclosed; canonical boundary-violations count re-verified at 29/12,000). **Items 7–10, 15, 17 (partial), 22, 23–24 also closed in the same pass** (bound regression tests; ablation Friedman/delta manifest entries; README softening). Items 4–6, 11–14, 16, 18–21 remain — none blocking. Items 5–6 tighten the manuscript-manifest contract. Items 7–8 (in the original numbering) are scope decisions for follow-up.

---

## Methodology principle being audited

This audit verifies adherence to the user-level methodology recorded at:

- [`feedback_research_methodology_math_first_then_numerical.md`](../../../.claude/projects/-home-mpk15956-projects-ti4-analysis/memory/feedback_research_methodology_math_first_then_numerical.md) — math first, numerical at machine precision, honest writeup
- [`feedback_claim_evidence_audit_pattern.md`](../../../.claude/projects/-home-mpk15956-projects-ti4-analysis/memory/feedback_claim_evidence_audit_pattern.md) — the claim-evidence audit pattern itself

The audit catches three drift surfaces the user's research methodology requires:
1. Math claimed but not numerically verified (the "we wrote the equation, never tested it" trap) — items 4–13 above
2. Numerical results stated without analytical motivation (the "we got a number, wrote math to explain it" trap)
3. Tests that exist but don't actually test the claim (the "test runs, produces value, value is meaningless" trap — Test 3 case, item 3 above)

All three patterns appear in this manuscript. None are unfixable. Items 1–4 of the follow-up order address the most material gaps.

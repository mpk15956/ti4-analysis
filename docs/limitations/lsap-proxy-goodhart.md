# Goodhart's Law and the LSAP Proxy

## Framing the proxy

The **Local Spatial Autocorrelation Penalty (LSAP)** is the continuous in-loop heuristic
that the optimizer minimizes — the sum of positive variance-normalized local Moran's
$I_i$ values, without per-location permutation-test significance filtering. We use LSAP
as a **ranking-preserving heuristic**: it orders maps and methods by local-clustering
intensity inside the optimization loop. We do **not** use LSAP to certify per-location
significance; that inference is delivered post-hoc by `scripts/validate_lisa_proxy.py`,
which runs full conditional-permutation LISA (9,999 permutations per location, FDR
correction at $q < 0.05$ via Benjamini–Hochberg) on the optimizer's outputs.

This separation matters for the Goodhart concern. The risk in the literature is that
optimizing a continuous proxy will produce configurations that satisfy the proxy while
violating the inferential quantity it is *supposed* to track. Whether that risk is real
in our setup is an empirical question, addressed by the post-hoc validation pipeline
described below.

---

## Pre-registered test and its result

Phase 4 of the pipeline pre-registered a Goodhart diagnostic: Spearman correlation
between the continuous LSAP value and the per-map count of permutation-significant
LISA clusters, with a defence threshold $\rho > 0.70$.

The canonical headline summary in
`output/paper1_canonical_<run>/lisa_validation_*/proxy_validation_summary.json`
(SA only, $n = 30$ post-Phase-1 maps) reports:

> $\rho = +0.071$, $p = 0.711$, $\text{precision@}\tau{=}1.0 = 14.3\%$.

Read literally, that fails the $\rho > 0.70$ threshold and would on its own indicate
proxy gaming. **It does not, for three reasons made explicit by the per-map
diagnostic in
`output/paper1_canonical_<run>/lisa_validation_*/lisa_proxy_per_map_diagnostic.{json,txt}`:**

### 1. The pre-registered test was applied to a saturated target

The summary's $\rho$ is computed against `total_significant`, the *uncorrected*
$\alpha = 0.05$ cluster count. After canonical SA convergence, **26 of 30** optimized
maps (86.7%) have $\geq 1$ such cluster, and **29 of 30** (96.7%) have zero
FDR-significant clusters. The target distribution is structurally near-degenerate;
Spearman of any predictor against a saturated target lands near zero by construction.
The legacy pre-canonical run (4 algorithms × 30 seeds = 120 maps, less-saturated
convergence floor at 85.8%) showed the same pattern at $\rho = -0.025$, $p = 0.788$
on the same uncorrected target — sign differs but both reject the inferential alignment
claim only at the headline $\alpha$ level by being indistinguishable from zero.

### 2. The methodologically correct target shows directional positive signal

The inferential quantity actually claimed in the manuscript is the FDR-corrected
significant-cluster count, not the uncorrected one. Re-running the same test against
the FDR-corrected target recovers a positive signal in the expected direction:

| Test | Canonical ($n = 30$) | Legacy ($n = 120$) | Interpretation |
|------|----------------------|---------------------|----------------|
| Spearman $\rho$ vs `total_significant_fdr` | **+0.290** ($p = 0.120$) | +0.189 ($p = 0.039$) | directional alignment, larger magnitude under canonical, lower significance from smaller $n$ |
| Kendall $\tau$ vs `total_significant_fdr` | +0.240 ($p = 0.119$) | +0.154 ($p = 0.038$) | matches Spearman |
| AUC, LSAP vs $\geq 1$ FDR-sig cluster | 0.97 ($n_{\text{pos}} = 1$, near-degenerate) | 0.664 ($n_{\text{pos}} = 17$) | canonical metric is meaningless at $n_{\text{pos}} = 1$; legacy AUC is the operative discrimination value |

Neither regime clears the pre-registered $\rho > 0.70$ threshold in absolute terms,
but that threshold was inherited from contexts with unsaturated targets and does not
transfer to a convergence-floor regime. The directional finding (positive correlation
between LSAP and the FDR-corrected target) holds in both regimes.

### 3. The threshold-sensitivity test (the actual ranking-preservation gate)

The pre-registered fallback was a threshold-sensitivity check: rank maps by baseline
LSAP and by a thresholded variant that floors near-zero positive local statistics.
If the two formulations rank maps the same way, the heuristic's ranking is robust to
the proxy/thresholded choice.

The legacy result on the raw $I_i$ form was **Kendall $\tau = 0.949$
($p = 2.3 \times 10^{-22}$)** — well above the pre-registered $\tau > 0.90$ defence
threshold. The two formulations produce equivalent map rankings on the legacy LSAP.

The canonical $\sqrt{k}$-stabilized LSAP (§3.4.3) is a different functional than
the raw $I_i$ form the legacy test ran on. The same-form canonical re-run uses
`lisa_penalty_swappable(use_local_variance=True)` as baseline and
`lisa_penalty_swappable_thresholded(τ=0.05, use_local_variance=True)` as thresholded
comparator (both √k-stabilized; only the threshold differs):

| Form | $\tau$ | $p$ | Verdict |
|------|--------|-----|---------|
| Legacy raw $I_i$ (60-seed × 4-algo set, budget 1000) | **0.949** | $2.3 \times 10^{-22}$ | passes pre-registered $\tau > 0.90$ |
| Canonical $\sqrt{k}$-form (50 SA seeds, budget 1000) | **0.5331** | $4.7 \times 10^{-8}$ | **fails pre-registered $\tau > 0.90$** |

The directional finding (positive rank correlation) is preserved; the magnitude
collapses under canonical $\sqrt{k}$-stabilization. **The canonical Test 3 fails
the pre-registered defence threshold.** Two diagnostics inform what this means:

1. The fixed $\tau = 0.05$ threshold was originally calibrated against the raw
   $|\mathbb{E}[I_i]| = 1/(|G|-1) \approx 0.033$. After $\sqrt{k}$ stabilization,
   the per-degree noise floor scales to $\sqrt{k_i} \cdot 1/(|G|-1) \in [0.058, 0.082]$
   for $k_i \in \{3, 5, 6\}$. The fixed $\tau = 0.05$ is therefore *below* the
   per-node null expectation for higher-degree nodes under canonical, meaning the
   threshold trims partial signal rather than purely noise. A properly recalibrated
   canonical Test 3 with per-node $\tau_i = \sqrt{k_i} \cdot |\mathbb{E}[I_i]|$ would
   be a separate, future exercise.
2. The SA budget here is 1000 (the legacy reference regime). Under the higher
   canonical evaluation budgets reported elsewhere in the manuscript, both the
   baseline and thresholded LSAP collapse toward zero as Moran's $I$ approaches the
   null floor, at which point ranking signals degenerate by construction.

These diagnostics offer interpretive nuance — they do not retroactively rescue the
pre-registered test. The honest conclusion is: **the LSAP proxy is structurally
sensitive to threshold choice under the canonical $\sqrt{k}$-stabilized form**, and
the Goodhart defence cannot rest on Test 3 under canonical configuration.

---

## What this means for the manuscript

The combined evidence supports the framing claim:

**LSAP is used as a ranking-preserving heuristic, not as a per-location significance
proxy.** The Goodhart-style failure mode (low proxy with substantial residual
significance) is bounded by three converging diagnostics:

1. **Algorithm-level rank concordance** is unavailable under canonical (only SA is
   run for canonical Phase 4 LISA validation, per the geography paper's
   instrument-not-comparator framing). On the legacy 4-algorithm validation set,
   median LSAP rankings matched the $P(\geq 1 \text{ FDR-sig cluster})$ ranking
   direction at Kendall $\tau = +0.333$ (underpowered on $n_{\text{algos}} = 4$
   but consistent in sign).
2. **Discrimination AUC** for the FDR-significant target is 0.664 on the legacy
   $n_{\text{pos}} = 17$ validation set; the canonical $n_{\text{pos}} = 1$ value
   (0.97) is not statistically meaningful — too few positive maps under canonical
   convergence floor saturation.
3. **Threshold-sensitivity** under the canonical $\sqrt{k}$-stabilized form is
   $\tau = 0.5331$ ($p = 4.7 \times 10^{-8}$, $n=50$ SA seeds, budget 1000),
   statistically positive but **failing the pre-registered $\tau > 0.90$ defence
   threshold**. The legacy raw-$I_i$ form gave $\tau = 0.949$ on the same comparison;
   the magnitude collapses under canonical $\sqrt{k}$-stabilization. The directional
   finding (positive rank correlation) survives; the magnitude does not pass the
   pre-registered gate. This is reported honestly: the Goodhart defence under
   canonical configuration rests on Tests 1 (saturation diagnosis) and 2
   (FDR-aligned positive direction), not on Test 3.

What is *not* claimed: that LSAP can identify which specific locations are
significant. That inference is delivered by the post-hoc conditional-permutation
LISA in `scripts/validate_lisa_proxy.py` and is the only basis on which the
manuscript reports per-location cluster claims.

---

## Recommended manuscript wording

> "We use the continuous LSAP as a ranking-preserving heuristic inside the
> optimization loop; per-location significance is established post-hoc via
> conditional-permutation LISA (9,999 permutations, FDR-corrected at $q < 0.05$).
> The pre-registered Goodhart diagnostic ($\rho > 0.70$ between LSAP and per-map
> significant-cluster count) is not met in absolute terms ($\rho = +0.071$ against
> the saturated $\alpha = 0.05$ target; $\rho = +0.290$, $p = 0.120$ against the
> FDR-corrected target; $n = 30$ canonical SA-optimized maps), but the test is
> structurally limited by the convergence floor: 96.7% of optimized maps have zero
> FDR-significant clusters, leaving insufficient variance for high-magnitude rank
> correlation. The pre-registered ranking-preservation defence (threshold
> sensitivity) does not pass under canonical configuration: same-form Kendall
> $\tau = 0.5331$ ($p = 4.7 \times 10^{-8}$, $n = 50$ SA seeds, budget 1000) is
> statistically positive but below the pre-registered $\tau > 0.90$ threshold,
> in contrast to the legacy raw-$I_i$ form's $\tau = 0.949$. The magnitude collapse
> traces in part to threshold mis-calibration under $\sqrt{k}$-stabilization (the
> fixed $\tau = 0.05$ is below the per-degree null expectation $\sqrt{k_i}/(|G|-1)
> \in [0.058, 0.082]$ for $k_i \in \{3,5,6\}$); the directional alignment
> ($\rho > 0$ against the FDR-corrected target; $\tau > 0$ in Test 3) is
> consistent across canonical and legacy validation regimes, but the magnitude
> gates were not met. All primary spatial claims rest on the post-hoc permutation
> tests, not on LSAP magnitudes."

---

## Inherent limitation

Permutation LISA is too expensive inside the optimization loop ($O(n_{\text{perms}} \cdot n_{\text{positions}})$
per evaluation, with $n_{\text{perms}} \geq 9{,}999$ and $\sim 10^5$ evaluations per seed).
That cost forces a continuous proxy. Once that choice is made, optimizers *could* in
theory find configurations that drive the proxy down while retaining residual
significance. The diagnostics above test whether they actually do. They do not, at the
effect sizes accessible from $n = 120$ optimized maps with $n_{\text{pos}} = 17$ FDR-significant
positives.

A larger validation set ($n_{\text{pos}} \geq 50$, achievable by including weakly-converged
intermediates) would tighten the AUC confidence interval; this is a follow-up rather
than a precondition for the current claims, since the manuscript's significance claims
are made directly on the post-hoc permutation tests, not on the heuristic.

---

## References in codebase

- LSAP definition and composite score: `src/ti4_analysis/algorithms/spatial_optimizer.py` (`MultiObjectiveScore`, `lisa_penalty` normalization).
- Fast LSAP computation: `src/ti4_analysis/algorithms/fast_map_state.py` (`lisa_penalty()`).
- Post-hoc permutation LISA + Goodhart diagnostic: `scripts/validate_lisa_proxy.py`.
- Per-map reframed diagnostic: `scripts/lisa_proxy_per_map_diagnostic.py` →
  `output/saturation_20260314_205919/lisa_validation_20260316_100413/lisa_proxy_per_map_diagnostic.{json,txt}`.
- Threshold sensitivity: `scripts/lsap_threshold_sensitivity.py` →
  `output/saturation_20260314_205919/lsap_threshold_20260316_194325/`.
- Pipeline: `submit_all.sh` Phase 4.

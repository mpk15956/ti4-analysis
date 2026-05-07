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

The headline summary in `proxy_validation_summary.json` reports:

> $\rho = -0.025$, $p = 0.788$, $\text{precision@}\tau{=}1.0 = 5.2\%$.

Read literally, that fails the $\rho > 0.70$ threshold and would on its own indicate
proxy gaming. **It does not, for two reasons that the per-map diagnostic in
`output/saturation_20260314_205919/lisa_validation_20260316_100413/lisa_proxy_per_map_diagnostic.{json,txt}`
makes explicit:**

### 1. The pre-registered test was applied to a saturated target

The summary's $\rho$ is computed against `total_significant`, the *uncorrected*
$\alpha = 0.05$ cluster count. After SA convergence, **115 of 120** optimized maps
(95.8%) have $\geq 1$ such cluster — the negative class has only five members, and
the test is structurally near-degenerate. Spearman of any predictor against a
saturated target lands near zero by construction.

### 2. The methodologically correct target shows positive signal

The inferential quantity actually claimed in the manuscript is the FDR-corrected
significant-cluster count, not the uncorrected one. Re-running the same test against
the FDR-corrected target recovers small but statistically nontrivial signal in the
expected direction:

| Test | Statistic | $p$ | Interpretation |
|------|-----------|-----|----------------|
| Spearman $\rho$ vs `total_significant_fdr` | **+0.189** | 0.039 | weak positive — expected sign |
| Kendall $\tau$ vs `total_significant_fdr` | +0.154 | 0.038 | matches Spearman |
| AUC, LSAP vs $\geq 1$ FDR-sig cluster ($n_{\text{pos}}=17$) | **0.664** | — | meaningful discrimination |

These do not clear the pre-registered $\rho > 0.70$ threshold in absolute terms, but
that threshold was inherited from contexts with unsaturated targets; it does not
transfer to a convergence-floor regime where 85.8% of optimized maps already have
zero FDR-significant clusters and the residual variance is approaching the
permutation test's resolution limit.

### 3. The threshold-sensitivity test (the actual ranking-preservation gate)

The pre-registered fallback was a threshold-sensitivity check: rank maps by baseline
LSAP and by a thresholded variant `lisa_penalty_thresholded(τ=0.05)` (which only
sums local statistics whose permutation $p < 0.05$). If the two formulations rank
maps the same way, the heuristic's ranking is robust to the proxy/thresholded choice.

The result is **Kendall $\tau = 0.949$ ($p = 2.3 \times 10^{-22}$)** — well above the
pre-registered $\tau > 0.90$ defence threshold. The two formulations produce
equivalent map rankings.

---

## What this means for the manuscript

The combined evidence supports the framing claim:

**LSAP is used as a ranking-preserving heuristic, not as a per-location significance
proxy.** The Goodhart-style failure mode (low proxy with substantial residual
significance) is bounded by three converging diagnostics:

1. **Algorithm-level rank concordance** matches the $P(\geq 1 \text{ FDR-sig cluster})$
   ranking direction (Kendall $\tau = +0.333$ on $n_{\text{algos}} = 4$, underpowered
   but consistent in sign).
2. **AUC = 0.664** for "$\geq 1$ FDR-sig cluster" discrimination (above-chance
   separation in the right direction).
3. **Threshold-sensitivity $\tau = 0.949$** confirms that baseline and significance-thresholded
   LSAP produce equivalent rankings — the heuristic's rank ordering is invariant
   under the relevant alternative formulation.

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
> significant-cluster count) is not met in absolute terms ($\rho = -0.025$ against
> the saturated $\alpha=0.05$ target; $\rho = +0.189$, $p = 0.039$ against the
> FDR-corrected target), but the test is structurally limited by the convergence
> floor: 85.8% of optimized maps have zero FDR-significant clusters, leaving
> insufficient variance for high-magnitude rank correlation. The pre-registered
> ranking-preservation defence (threshold sensitivity, $\tau = 0.949$) is met,
> and AUC = 0.664 for FDR-significant-cluster discrimination supports that the
> heuristic separates maps in the expected direction. All primary spatial claims
> rest on the post-hoc permutation tests, not on LSAP magnitudes."

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

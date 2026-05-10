# Limitations: Small-N Problem in Spatial Statistics

## The flaw

Standard spatial statistics (e.g. Moran's I) rely on **asymptotic normality** for variance calculations and z-scores (e.g. the 1.96 threshold). The TI4 spatial graph $G$ used by both the optimizer's in-loop metrics and the post-hoc permutation pipeline has **|G| = 31** for the canonical 6-player layout (the 37 board hex positions minus the 6 frozen home tiles, after zero-degree purge of nodes isolated by impassable-edge excision; see Methodology §3.3 for the formal definition). At **N = 31**, asymptotic approximations are mathematically shaky. Top reviewers will reject papers that claim statistical significance purely from analytical variance on graphs this small. We do not use Getis-Ord $G_i^*$; it has been removed from the codebase and methodology because at N ≈ 31 its asymptotic z-score is unreliable and because LISA/LSAP are better suited to spatial-outlier detection (see README and Methodology §3.3).

> **Note on N values.** Earlier internal documents cited "N = 37 swappable system tiles" as the spatial-graph size. That conflated the geometric hex count of the 6-player board (37) with the spatial-graph N the metric operates on (31, after the home-tile and zero-degree exclusions). All small-N arguments in this section are made against |G| = 31. The 37 figure remains relevant only as a board-geometry input to the search-space symmetry argument at the end of this document.

## The fix (what we do)

- **Exact conditional permutation testing** is required for significance.
- The codebase leans on **`scripts/validate_lisa_proxy.py`** for evaluation: it re-runs optimization for a subset of seeds, then applies **conditional-permutation LISA** (9,999 permutations per location, default) to count significant H-H and L-L clusters. With 9,999 permutations, the minimum attainable p-value is 1/10,000 = 0.0001, allowing significance to be evaluated under multiple-testing correction.
- **LSAP** (the continuous LISA proxy) is framed as a **computational heuristic** optimized during search; the **final maps are strictly evaluated** using 9,999-permutation tests. No significance claims are based on analytical variance for LISA.

## What is already in place

- **LISA**: `validate_lisa_proxy.py` implements conditional permutation LISA (Anselin 1995): for each location *i*, *z*[i] is fixed, other values are permuted, and local *I*_i is recomputed to build the null. p-value = (count_extreme + 1) / (n_perms + 1) with `--n-perms 9999` (default). The optimizer uses the continuous LSAP only as a heuristic; significance is claimed only from this script.
- **Global Moran's I in the inner loop**: `FastMapState.morans_i()` returns only the scalar *I* (no z-score, no significance). It is used as a continuous objective to minimize |*I*|, not for inference.
- **Global Moran's I significance**: `validate_lisa_proxy.py` now computes a **permutation-based p-value** for global Moran's I (exact permutation of the value vector over the fixed set of locations, no bootstrapping). The script writes `global_I` and `global_I_pvalue` per map and summarizes the fraction of maps with global I significant at α = 0.05. **Manuscript claims that optimization "optimizes spatial dispersion" should cite this permutation test** (and the validation script), not analytical variance or z-scores.

## Remaining gaps and recommendations

1. **Global Moran's I**  
   Permutation-based significance for global *I* is now implemented in `validate_lisa_proxy.py`. Do not use `spatial_metrics.morans_i()` analytical variance or a 1.96 threshold for significance claims at N=31.

2. **Getis–Ord Gi\***  
   Getis-Ord $G_i^*$ is not used in this project; it has been removed from the codebase and methodology. At N ≈ 31 the asymptotic z-score is unreliable, and LISA/LSAP with conditional permutation testing are the only defensible methods for local spatial association. Do not reintroduce Gi\* for significance claims.

3. **Multiple testing (LISA)**  
   With 31 local tests per map, α = 0.05 per test does not control family-wise error. Bonferroni (α/31) assumes independent tests and is paralyzingly conservative for spatially correlated hex grids, leading to Type II errors.  
   - **Recommendation**: Use **False Discovery Rate (FDR), Benjamini–Hochberg** at q < 0.05 as the committed correction for LISA, applied per-map across the 31 locations (Caldas de Castro & Singer, 2006). Report both “per-location” and “corrected” counts so reviewers see the distinction.

4. **Power and effect size**  
   With N = 31, permutation tests are valid but have limited power against weak clustering.  
   - **Recommendation**: Report effect size and N explicitly (e.g. “number of locations with significant local clustering (permutation p < 0.05)” and “mean number of significant H-H/L-L clusters per map”). Make the “N = 31, permutation-based” and “LSAP as heuristic, significance only from validation” framing explicit in the paper.

5. **Goodhart's Law boundary for LSAP**
   The continuous LSAP proxy (sum of positive variance-normalized local Moran's $I_i$)
   is optimized without a significance threshold. A theoretical risk: the optimizer
   might exploit the proxy by generating many small, sub-threshold local statistics
   that individually escape significance under permutation testing while collectively
   lowering the objective. If that occurred, solutions with low LSAP would fail to map
   onto solutions with few permutation-significant clusters — the proxy would be *gamed*.

   We diagnose this boundary directly with three independent tests, computed by
   `scripts/validate_lisa_proxy.py` and `scripts/lsap_threshold_sensitivity.py` over
   the post-hoc validation set ($n = 120$ optimized maps; full breakdown in
   [`docs/limitations/lsap-proxy-goodhart.md`](lsap-proxy-goodhart.md)):

   1. **Per-map alignment at conventional $\alpha$.** Spearman $\rho$ between LSAP
      and the per-map count of permutation-significant LISA clusters at $\alpha = 0.05$
      does not reject the null of independence: $\rho = +0.071$, $p = 0.711$
      (Pearson $r = +0.056$; precision at proxy threshold $\tau = 1.0$ is $14.3\%$;
      $n = 30$ canonical SA-optimized maps). The proxy does not track the
      inferential quantity at headline $\alpha$. The result is structurally limited
      by the convergence floor: under canonical SA tuning, **96.7% of optimized
      maps already carry zero FDR-significant clusters** (and 86.7% carry $\geq 1$
      uncorrected-significant cluster), so the residual variance available to a
      per-map correlation test lies near the permutation test's resolution limit.

   2. **Per-map alignment under multiple-testing correction.** Re-targeted at the
      FDR-corrected count (the inferential quantity actually claimed in the
      manuscript; per-map Benjamini–Hochberg at $q < 0.05$), Spearman $\rho$
      recovers a small positive signal in the expected direction: $\rho = +0.290$
      ($p = 0.120$, $n = 30$). The directional alignment is consistent with the
      proxy tracking the FDR-corrected inferential quantity; statistical
      significance does not clear $\alpha = 0.05$ at this validation-set size,
      because only 1 of 30 maps in the canonical validation set carries any
      FDR-significant cluster, capping the achievable power for a rank correlation
      against a near-degenerate target. The legacy pre-canonical run with
      $n = 120$ (4 algorithms × 30 seeds) reached $\rho = +0.189$, $p = 0.039$ at
      a less-saturated convergence floor (85.8%); the directional finding is
      stable across regimes, the magnitude is larger under canonical, but the
      statistical significance of Test 2 alone is not load-bearing for the
      defence — the convergence-floor saturation diagnosis is.

   3. **Threshold-sensitivity ranking preservation.** Kendall $\tau$ between
      baseline LSAP rankings and rankings under the same-form thresholded variant
      `lisa_penalty_swappable_thresholded(τ = 0.05, use_local_variance=True)`
      over $n = 50$ canonical SA-optimized maps (budget 1000) is
      **$\tau = 0.5331$ ($p = 4.7 \times 10^{-8}$)**, statistically positive but
      **below the pre-registered $\tau > 0.90$ defence threshold**. This canonical
      result supersedes the legacy $\tau = 0.949$ ($p = 2.3 \times 10^{-22}$) on
      the raw-$I_i$ form, which was on a different functional. The directional
      finding (positive rank correlation) is preserved across regimes; the
      magnitude is materially smaller under the $\sqrt{k}$-stabilized form, and
      the test as pre-registered does not pass under canonical configuration.

      Two diagnostics inform interpretation. First, the $\sqrt{k}$ stabilization
      rescales each $I_i$ by $\sqrt{k_i} \in \{\sqrt{3}, \sqrt{5}, \sqrt{6}\}$,
      so the operationally equivalent noise-floor heuristic is now $\tau \approx
      \sqrt{k_i} \cdot |\mathbb{E}[I_i]| \in [0.058, 0.082]$ (per-degree; canonical
      $|\mathbb{E}[I_i]| = 1/30 \approx 0.033$). The fixed $\tau = 0.05$ used
      here is below this range for higher-degree nodes, so the test is flooring
      partial signal rather than purely noise; a properly recalibrated canonical
      Test 3 with per-node $\tau_i = \sqrt{k_i} \cdot |\mathbb{E}[I_i]|$ is a
      separate exercise. Second, the SA budget here is 1000 (the legacy reference
      regime); the higher-budget canonical SA used elsewhere in the manuscript
      drives Moran's $I$ to its null floor ($\mathbb{E}[I] \approx -0.033$), at
      which point baseline and thresholded LSAP both approach zero and the
      ranking signal degenerates.

   Honest framing of the combined evidence: Tests 1 and 2 (canonical) defend the
   proxy under their stated reading — Test 1 confirms that the post-convergence
   residual variance is below the per-map permutation test's resolution at
   $\alpha = 0.05$, and Test 2 recovers the expected positive directional
   alignment under FDR-corrected multiple-testing correction. Test 3 (canonical)
   does *not* clear its pre-registered threshold; the LSAP proxy is structurally
   sensitive to threshold choice under the canonical $\sqrt{k}$-stabilized form
   in a way that the legacy raw-$I_i$ form was not. We therefore use LSAP as a
   *ranking-preserving heuristic* inside the optimization loop, not as a
   per-location significance proxy, and we report the canonical Test 3 failure
   as part of that scope. All primary spatial claims rest on the post-hoc
   permutation tests in `validate_lisa_proxy.py`, not on LSAP magnitudes.

## Is this a fundamental limitation of TI4-style map optimization?

- **N = 31** is fixed by the canonical 6-player layout used in this study; the geometric board has 37 hex positions, of which 6 are home tiles (frozen, excluded from the spatial graph by space-type) and the remainder constitute |G|.
- The **unreliable** part is asymptotic normality/variance at this N. The **correct** approach is permutation-based (or other exact/simulation-based) inference. That is not a workaround — it is the statistically appropriate method for small N.
- So the limitation is: **any claim of statistical significance must use permutation-based (or similar) evaluation**, not analytical variance or z-scores at N = 31. The codebase already does this for LISA via `validate_lisa_proxy.py`.
- The only inherent limitation is **power**: with 31 locations, very weak clustering may be undetectable. Transparency (reporting N, permutation, and effect sizes) is the appropriate response.

**Bottom line:** Use permutation-based evaluation for all significance claims; treat LSAP as the optimization heuristic; do not use analytical variance or z-scores for significance at N=31; use FDR (Benjamini–Hochberg) for LISA multiple-testing correction; global Moran's I significance is reported via the permutation test in `validate_lisa_proxy.py`. Then the small-N issue is addressed rather than a permanent, unfixable flaw of TI4-style map optimization.

## Moran's I boundary violations on the primary n = 31 graph

In the canonical Phase 1 condition ablation (`output/paper1_canonical_20260509_134024/benchmark_20260509_191848/results.csv`), 29 of 12,000 (0.242%) optimized solutions produced Moran's I values marginally below −1.0, with a range of [−1.047, −1.001]. All 29 violations occurred under the `moran_only` (28) and `lsap_only` (1) conditions at evaluation budgets ≥ 25,000. The legacy pre-canonical run reported 21 of 12,000 (0.175%) violations in [−1.063, −1.001] under the same conditions; the qualitative pattern (rare, marginally-below-bound, confined to non-JFI-anchored conditions at high budgets) is preserved across regimes.

These violations are on the **primary `FastMapState.morans_i()` metric** computed on the spatial graph G (|G| = 31), not on the auxiliary `morans_i_swappable()` over a smaller swappable subgraph. For row-standardized adjacency matrices on small irregular graphs, the classical [−1, +1] bounds derived under uniform-weight or symmetric-W assumptions do not strictly hold; values marginally outside [−1, +1] are mathematically possible because the row-standardization renormalises lag values such that the ratio $n / W_{\text{sum}}$ in the I formula no longer factors into the bound. Anselin & Rey (2014, *Modern Spatial Econometrics in Practice*) and de Jong, Sprenger & van Veen (1984, "On extreme values of Moran's I and Geary's c") discuss the asymmetric-W case explicitly.

For reporting, values are clipped to [−1, +1]; excluding these 29 observations does not alter any reported result (medians, IQRs, and Wilcoxon test outcomes are insensitive to removal of values 0.001–0.047 below the boundary). The violations are documented here so reviewers familiar with the symmetric-W bound argument do not flag the asymmetric-W behaviour as a coding error.

## Search Space Symmetry (D₆ Dihedral Group)

The effective search space is approximately 37!/12 rather than 37! due to D₆ dihedral symmetry (6 rotations, 6 reflections on the hex grid). All algorithms may redundantly evaluate rotationally or reflectively equivalent configurations, inflating absolute convergence estimates by up to 12× while leaving relative algorithm rankings unaffected.

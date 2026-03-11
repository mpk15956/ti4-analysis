# Limitations: Small-N Problem in Spatial Statistics

## The flaw

Standard spatial statistics (Moran's I, Getis-Ord Gi*) rely on **asymptotic normality** for variance calculations and z-scores (e.g. the 1.96 threshold). A TI4 map has roughly **37 swappable system tiles**. At **N = 37**, asymptotic approximations are mathematically shaky. Top reviewers will reject papers that claim statistical significance purely from analytical variance on N=37 grids.

## The fix (what we do)

- **Exact conditional permutation testing** is required for significance.
- The codebase leans on **`scripts/validate_lisa_proxy.py`** for evaluation: it re-runs optimization for a subset of seeds, then applies **conditional-permutation LISA** (999 permutations per location, default) to count significant H-H and L-L clusters at p < 0.05.
- **LSAP** (the continuous LISA proxy) is framed as a **computational heuristic** optimized during search; the **final maps are strictly evaluated** using 999-permutation tests. No significance claims are based on analytical variance for LISA.

## What is already in place

- **LISA**: `validate_lisa_proxy.py` implements conditional permutation LISA (Anselin 1995): for each location *i*, *z*[i] is fixed, other values are permuted, and local *I*_i is recomputed to build the null. p-value = (count_extreme + 1) / (n_perms + 1) with `--n-perms 999`. The optimizer uses the continuous LSAP only as a heuristic; significance is claimed only from this script.
- **Global Moran's I in the inner loop**: `FastMapState.morans_i()` returns only the scalar *I* (no z-score, no significance). It is used as a continuous objective to minimize |*I*|, not for inference.

## Remaining gaps and recommendations

1. **Global Moran's I (if reported as “significant”)**  
   `spatial_metrics.morans_i()` returns (I, expected_I, variance_I) using Cliff–Ord analytical variance. At N=37 that variance is not reliable.  
   - **Recommendation**: In the paper, do not claim global Moran's I is “significant” via analytical variance or a 1.96 threshold. Either report global *I* only as a descriptive measure, or add a permutation test for global *I* (permute the value vector, recompute *I*, obtain p-value) and base any significance claim on that.

2. **Getis–Ord Gi\***  
   The docstring in `spatial_metrics.py` states that |Gi*| > 1.96 implies significance; that threshold is asymptotic.  
   - **Recommendation**: If Gi* is used for inference, use a permutation-based null for Gi* as well, or avoid significance claims. If it is only a continuous objective, do not attach a 1.96 significance interpretation.

3. **Multiple testing (LISA)**  
   With 37 local tests, α = 0.05 per test does not control family-wise error.  
   - **Recommendation**: Report results with a correction (e.g. Bonferroni α/37 or FDR at q < 0.05), or at least report both “per-location” and “corrected” counts so reviewers see the distinction.

4. **Power and effect size**  
   With N=37, permutation tests are valid but have limited power against weak clustering.  
   - **Recommendation**: Report effect size and N explicitly (e.g. “number of locations with significant local clustering (permutation p < 0.05)” and “mean number of significant H-H/L-L clusters per map”). Make the “N=37, permutation-based” and “LSAP as heuristic, significance only from validation” framing explicit in the paper.

## Is this a fundamental limitation of TI4-style map optimization?

- **N ≈ 37** is fixed by the game; it cannot be increased.
- The **unreliable** part is asymptotic normality/variance at this N. The **correct** approach is permutation-based (or other exact/simulation-based) inference. That is not a workaround—it is the statistically appropriate method for small N.
- So the limitation is: **any claim of statistical significance must use permutation-based (or similar) evaluation**, not analytical variance or z-scores at N=37. The codebase already does this for LISA via `validate_lisa_proxy.py`.
- The only inherent limitation is **power**: with 37 tiles, very weak clustering may be undetectable. Transparency (reporting N, permutation, and effect sizes) is the appropriate response.

**Bottom line:** Use permutation-based evaluation for all significance claims; treat LSAP as the optimization heuristic; do not use analytical variance or z-scores for significance at N=37; optionally add a permutation test for global Moran's I and a multiple-testing correction for LISA. Then the small-N issue is addressed rather than a permanent, unfixable flaw of TI4-style map optimization.

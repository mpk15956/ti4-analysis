# Goodhart's Law and the LSAP Proxy

## The flaw

The **Local Spatial Autocorrelation Penalty (LSAP)** is a continuous proxy: it sums positive variance-normalized local Moran's I values *without* permutation-test significance filtering. Because the optimization algorithms (especially SA and Tabu) aggressively minimize this proxy, they could in principle find adversarial map configurations that:

- Minimize the proxy score, but  
- Still contain strategically significant H-H or L-L clusters that a true LISA permutation test would flag.

In other words: when a measure becomes a target, it ceases to be a good measure (Goodhart's Law). If the proxy does not effectively eliminate true significant clusters, the objective function is flawed.

---

## What we already do (the fix)

**Phase 4** of the pipeline (`submit_all.sh` → `scripts/validate_lisa_proxy.py`) is the validation step. It:

1. **Correlation**  
   For a subset of seeds, re-runs each algorithm, takes the final map, and runs full conditional-permutation LISA (e.g. 999 permutations per location, \(p < 0.05\)). It then computes **Spearman ρ** and **Pearson r** between the continuous LSAP proxy and the count of significant clusters. These are printed and written to `proxy_validation_summary.json`.

2. **Precision analysis**  
   At thresholds 0.5, 1.0, 2.0, 3.0, 5.0 it reports: among maps with proxy below that threshold, what fraction have **zero** significant clusters. That answers: “Does minimizing the proxy actually eliminate significant H-H/L-L clusters?”

3. **Outputs**  
   - `validation_results.csv` — per seed, per algorithm: proxy, `total_significant`, H-H, L-L, etc.  
   - `proxy_validation_summary.json` — correlations, p-values, precision table.  
   - `proxy_validation_scatter.png` — proxy vs. significant cluster count.

So the methodological fix (report correlation and “does low proxy ⇒ zero significant clusters?”) is already in the pipeline. The remaining obligation is to **use** these results in the manuscript.

---

## What to do in the manuscript

1. **State the risk**  
   Explicitly say that LSAP is a continuous proxy for significance-tested LISA (Anselin 1995) and that minimizing it could be gamed (Goodhart’s Law).

2. **Report Phase 4**  
   Report the correlation(s) and precision (e.g. “Among maps with LSAP &lt; X, Y% had zero significant H-H/L-L clusters”).

3. **Interpret**  
   - If correlation is strong and precision is high → “Post-hoc validation supports the proxy.”  
   - If correlation is weak or many low-proxy maps still have significant clusters → state it as a **limitation**: “The objective function may not fully eliminate statistically significant local clusters; Phase 4 results indicate …”

4. **Optional**  
   From `validation_results.csv`, report per-algorithm mean (or median) significant cluster count and mean proxy. If SA/TS consistently yield both lower proxy and fewer significant clusters than HC, that supports that minimizing the proxy is aligned with the true LISA goal.

---

## Inherent limitation

- **Design trade-off:** Permutation LISA is too expensive inside the optimization loop, so we use a smooth proxy. Once that choice is made, optimizers *could* in theory find configurations that drive the proxy down while retaining some significant clusters — the proxy and true LISA are not the same quantity.

- We cannot remove that risk by switching the in-loop objective to “true” LISA without a large computational cost. What we *can* do is:
  - **Validate** (Phase 4),
  - **Report** correlation and precision in the paper,
  - **Interpret** and, if needed, state clearly that the objective may be imperfect (limitation).

**Bottom line:** Part is addressable (transparency + validation); part is inherent (the proxy is not identical to permutation LISA; we validate and report, and acknowledge the limitation where appropriate).

---

## References in codebase

- LSAP definition and composite score: `src/ti4_analysis/algorithms/spatial_optimizer.py` (e.g. `MultiObjectiveScore`, `lisa_penalty` normalization).
- Fast LSAP computation: `src/ti4_analysis/algorithms/fast_map_state.py` (`lisa_penalty()`).
- Phase 4 validation: `scripts/validate_lisa_proxy.py` (conditional permutation LISA, correlation, precision, scatter).
- Pipeline: `submit_all.sh` Phase 4.

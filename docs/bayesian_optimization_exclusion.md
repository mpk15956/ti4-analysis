# Exclusion of Bayesian Optimization (BO) from the Benchmark

## Summary

Bayesian Optimization is excluded from the primary benchmark on the grounds
of **sample efficiency mismatch**: BO's strength lies in settings where each
objective evaluation is expensive (minutes to hours), whereas TI4 map
evaluation completes in <1 ms per evaluation via vectorized numpy operations.

## Detailed Justification

### 1. Surrogate Overhead Dominates

BO maintains a probabilistic surrogate model (typically a Gaussian Process)
of the objective landscape and selects the next evaluation point by
optimizing an acquisition function (e.g., Expected Improvement).

- **GP fitting** is O(n³) in the number of observed evaluations. After 5,000
  evaluations, each GP update costs ~125 billion FLOPs — orders of magnitude
  more than the O(H × S) matmul that constitutes a map evaluation.
- **Acquisition function optimization** adds a secondary inner loop (often
  L-BFGS or CMA-ES over the acquisition surface), further inflating
  per-step wall time.

In the TI4 domain, where SA processes 50,000+ evaluations in under 30
seconds, the surrogate overhead would dominate total runtime by a factor
of 100–1000×, yielding worse anytime performance than any trajectory method.

### 2. Dimensionality of the Search Space

The TI4 optimization problem is a permutation over S ≈ 30 swappable tile
positions — a combinatorial, discrete search space of size S! ≈ 2.65 × 10³².
Standard BO operates on continuous, low-dimensional spaces (typically d < 20).
Combinatorial BO variants exist (COMBO, BOCS) but require kernel engineering
specific to the permutation group and have not been validated on spatial
equity objectives.

### 3. Sample Efficiency Analysis

SA's convergence profile shows rapid improvement in the first 500–1000
evaluations, with diminishing returns beyond 5,000. The "improvement per
evaluation" curve does not exhibit the plateaus characteristic of problems
where BO's directed exploration would outperform random-restart or
Metropolis-guided exploration.

Concretely: if the evaluation function is cheap and the improvement curve
is steep, the overhead of maintaining a surrogate is a net negative — the
same wall-time budget achieves more evaluations (and thus more improvement)
via SA.

### 4. Literature Precedent

The PCG literature for tabletop/board game map generation (Sironi et al. 2019,
Togelius et al. 2011) benchmarks trajectory methods (HC, SA, TS) and
population methods (GA, NSGA-II) but does not include BO, reflecting the
community consensus that cheap-evaluation combinatorial domains are outside
BO's sweet spot.

## Conclusion

BO is not included in the benchmark because:
1. The per-evaluation cost (<1 ms) makes surrogate overhead counterproductive.
2. The discrete permutation search space is poorly suited to GP-based surrogates.
3. SA's convergence profile shows no evidence of the exploration bottleneck
   that BO is designed to solve.

Future work could investigate combinatorial BO variants (COMBO) as the
surrogate modeling literature matures for permutation domains.

# Formal Methodology Section (Paper Draft)

*Standalone Methodology section for the academic paper, suitable for games / PCG / AI-in-Games venues. Copy-paste ready; fill in section number (e.g. 3) and cross-references as required by your target journal.*

---

## 3.1 Problem Formulation

> **Evaluation hierarchy:** Primary evaluation uses weight-independent Pareto quality indicators (Hypervolume, IGD+); the scalar weighted-sum composite serves as a secondary benchmark for single-objective comparators only. See §3.8 for Track A/B definitions.

Anomaly tiles (asteroid fields, supernovae, nebulae, gravity rifts) are assigned under a non-adjacency constraint and frozen during optimization; all algorithms operate exclusively on the blue system tile swapping problem.

Map optimization is framed as minimization of a weighted composite score over three objectives:

$$S = w_1 \cdot (1 - J_{\min}) + w_2 \cdot \max\!\left(0,\ I - \mathbb{E}[I]\right) + w_3 \cdot \frac{\text{LSAP}}{n(n-1)} \quad \text{where } \mathbb{E}[I] = -\frac{1}{n-1}$$

The one-sided hinge penalizes only positive spatial autocorrelation above the null expectation; negative autocorrelation (spatial dispersion) incurs zero penalty, reflecting the design goal of preventing resource clustering rather than enforcing spatial randomness.

In the above, $J_{\min} = \min(J_R, J_I)$ is the **Multi-Jain bottleneck** — Jain's Fairness Index computed independently on distance-weighted raw Resources and raw Influence, with the minimum (bottleneck) dimension determining the fairness term. This formulation is theoretically inspired by the bottleneck intuition in multi-resource allocation (Ghodsi et al., 2011): map fairness is limited by the *least fair* resource dimension; we do not claim the formal DRF axioms, which apply to allocation mechanisms, not fixed topologies.

Given the lack of consensus on optimal weighting for composite spatial indicators (Libório et al.), we use **equal weights 1:1:1** ($w_1 = w_2 = w_3 = 1/3$) for the full composite — a fixed target for single-objective methods (SA, TS, HC, SGA). Primary evaluation for the main experiment is condition comparison (spatial profile, JFI parity); weight-independent Pareto indicators (HV, IGD+) support methods justification; see §3.8 and Design_Rationale.md. **Weight-sensitivity analysis** (e.g. `analyze_benchmark.py --sensitivity`) tests whether algorithm rankings (e.g. SA vs HC) are robust to alternative weight configurations around 1:1:1. When superiority holds across configurations, the equal-weight choice is academically defensible. HV and IGD+ (Ishibuchi et al., 2015) are computed against an **empirical reference front** formed by merging all observed Pareto points across seeds when the true Pareto front is unknown.

**Normalization vs. Weighting (Gen-0 Standardization)**  
To resolve the mathematical heteroskedasticity inherent in the raw objectives, the engine strictly decouples *normalization* (unit equalization) from *weighting* (strategic preference). At initialization, the system samples $N=1,000$ uniformly random map configurations from the underlying permutation state space ($37! \approx 1.37 \times 10^{43}$) to compute the empirical standard deviation ($\sigma$) of each objective. These Gen-0 static $\sigma$ values are not arbitrary, human-chosen scalars; they are exact empirical properties of the unoptimized problem domain.  
Dividing each term by its Gen-0 $\sigma$ converts the objectives into comparable standard-deviation units (Z-score analogues). Without this conversion, the Moran's I hinge term—which accounts for approximately $90\%$ of total weighted variance at Gen-0 under the nominal 1:1:1 weights (see §3.4.4 for the canonical formulation, §3.8 for the variance-equalization diagnostic)—effectively dictates the gradient, reducing the search to a single-objective spatial optimizer. Gen-0 normalization forces dimensional parity. Consequently, the assigned weight vector (e.g., 1:1:1 or sensitivity variants) rigorously governs the relative importance of the objectives in the gradient calculation, ensuring the composite behaves mathematically as intended.

All three terms are normalized to $[0, 1]$ before weighting: $(1 - J_{\min}) \in [0,1]$, $\max(0,\ I - \mathbb{E}[I]) \in \left[0,\ 1 + \tfrac{1}{|G|-1}\right] \approx [0, 1.033]$ for $|G|=31$ (effectively bounded at 1 for all observed map configurations), and $\text{LSAP}/[|G|(|G|-1)] \in [0,1]$. The divisor $|G|(|G|-1)$ is the theoretical maximum of the sum of positive variance-normalized local Moran's I values under row-standardized $\mathbf{W}$. Balance gap (max − min player value) is retained as a display attribute but is excluded from the composite score and all Pareto dominance calculations.

---

## 3.2 Distance-Weighting Model

Per-player home values are computed as the weighted sum of all reachable system tile values within a hex-distance range gate of 5. We employ a **discrete step-function decay** based on terrain-cost pathfinding rather than continuous inverse-distance power laws ($w = 1/d^\gamma$), reflecting the quantized nature of board game movement and avoiding a free parameter $\gamma$ with no ground-truth calibration in the game domain.

For each (home $p$, system $s$) pair: (1) **Routing distance:** compute the shortest "modded distance" $d_m(p, s)$ via BFS pathfinding, where each hop accumulates a terrain cost from the evaluator; supernovae block the path entirely. (2) **Step-function weight lookup:** map $d_m$ to a weight via a discrete multiplier table:

$$v_p = \sum_{s \in \mathcal{R}(p)} \mathbf{w}\bigl[\lfloor d_m(p, s) \rfloor\bigr] \cdot \text{val}(s)$$

The default table (community-calibrated "Joebrew" evaluator) uses weights 6, 6, 6, 4, 4, 2, 1, 0 for $\lfloor d_m \rfloor = 0, 1, \ldots, 7+$, with a "shelf" at weight 6 for $d_m \in [0, 2]$ to capture equal accessibility within one round's movement. Terrain costs: blue/empty 0; nebula 0; gravity rift $0.3\,d_{\text{rift}} - 0.4$; asteroid/supernova block. For Multi-Jain, the same weight matrix is applied to raw Resources and raw Influence separately.

**Distance-weight sensitivity:** Six alternative weight tables (`flat_nearby`, `steep_decay`, `linear`, `inverse_distance`, `binary_reachable`, plus the baseline) are tested on a representative subset of seeds, and algorithm rankings are compared via Kendall's τ. **Mean Kendall's τ across all perturbed configurations vs. baseline = 1.000 (n = 15 pairwise comparisons, p < 0.001 by exact permutation), and the SA-vs-SGA winner is identical under every configuration.** Algorithm rankings are therefore weight-invariant, and the conclusions are not brittle to the specific distance-decay parameterization.

Source: `scripts/distance_weight_sensitivity.py` (per-config and pairwise τ in `output/saturation_20260314_205919/dist_sensitivity_20260316_100535/sensitivity_report.txt`).

---

## 3.3 Metrics

### Balance gap

$\text{gap} = \max(\mathbf{v}) - \min(\mathbf{v})$ where $\mathbf{v}$ is the per-player distance-weighted value. Retained for display only; excluded from composite and Pareto dominance.

### The spatial graph $G$

All spatial-autocorrelation metrics in this study — the optimizer's in-loop global Moran's I and LSAP, and the post-hoc conditional-permutation LISA in `scripts/validate_lisa_proxy.py` — operate on the same spatial graph $G$, defined as follows. Take the set of map spaces of type SYSTEM with non-None systems (this excludes the 6 home tiles by space type, and any empty hex positions). Construct adjacency from hex contiguity, including wormhole connections, but **excise edges through impassable systems** (Supernova) so that spatial proximity reflects navigable rather than purely geometric adjacency. Finally, **purge nodes left zero-degree** by that excision (typically a Supernova that ends up isolated). The result is the row-standardized adjacency $\mathbf{W}$ over $|G|$ surviving nodes; rows sum to 1.0 by construction.

For the canonical 6-player layout, $29 \leq |G| \leq 31$ depending on per-seed anomaly placement: the 37 hex positions minus 6 home tiles produce 31 candidate spatial nodes, of which a Supernova that ends up isolated by impassable-edge excision is then zero-degree-purged. Across the 30-seed × 4-algorithm post-hoc validation set (120 maps total in `lisa_validation_*/validation_results.csv`), $|G|$ takes values 29, 30, 31 with frequencies 52, 52, 16 respectively — the canonical $|G| = 31$ value occurs when no Supernova is purged (4 of 30 seeds), the modal $|G| = 30$ when one Supernova is purged, $|G| = 29$ when two are. All metrics use each map's own $|G|$ for its $E[I]$ and LSAP normalization (single source: `MapTopology.n_spatial`); the §3.3 formulas below are stated for general $|G|$, not for a fixed value. Implementation: [`src/ti4_analysis/algorithms/map_topology.py`](src/ti4_analysis/algorithms/map_topology.py).

> **Optimizer–validator alignment.** Both `FastMapState.morans_i()`/`lisa_penalty()` and `validate_lisa_proxy.py:conditional_permutation_lisa()` source `z` from `FastMapState.spatial_values()` and $\mathbf{W}$ from `MapTopology.spatial_W` — the same per-seed graph $G$. The Goodhart-style failure mode (heuristic optimized on one graph, significance assessed on another) is structurally precluded by this shared dependency. The auxiliary `morans_i_swappable()` over the swappable-connected subgraph is reported but not used in primary results.

### Moran's I

Global spatial autocorrelation:

$$I = \frac{|G|}{W} \cdot \frac{\sum_i \sum_j w_{ij}(x_i - \bar{x})(x_j - \bar{x})}{\sum_i (x_i - \bar{x})^2}$$

Under the null of no spatial association, $E[I] = -1/(|G|-1)$. For the canonical layout with $|G| = 31$, $E[I] = -1/30 \approx -0.033$. Interpretation: $I > E[I]$ positive autocorrelation; $I \approx E[I]$ spatial randomness; $I < E[I]$ negative autocorrelation. The 29 of 12,000 (0.242%) canonical Phase 1 solutions producing $I$ marginally below $-1.0$ (range $[-1.047, -1.001]$, confined to `moran_only` and `lsap_only` conditions at budgets $\ge 25{,}000$) are discussed in [`docs/limitations/limitations.md`](docs/limitations/limitations.md); they reflect the known property of asymmetric row-standardized $\mathbf{W}$ on small irregular graphs (de Jong, Sprenger & van Veen, 1984) and do not affect any reported median, IQR, or Wilcoxon outcome.

### Spatial weights matrix and edge effects

Standard spatial statistics are often derived under infinite or wrapped domains. On a small, bounded hex grid with $|G| = 31$, edge nodes have fewer neighbours than interior ones, which can bias global and local autocorrelation (Boots and Tiefelsdorf, 2000). The row-standardization defined above — each row of $\mathbf{W}$ sums to 1.0 — is the standard approach to mitigate edge-effect bias in bounded tessellations (Boots and Tiefelsdorf, 2000). After standardization, the spatial lag $(\mathbf{W}\mathbf{z})_i$ at each position is a proper weighted average of its neighbours regardless of neighbour count, ensuring that edge systems are not artificially penalized or advantaged. We do not apply a separate edge-correction penalty.

### Local Spatial Autocorrelation Penalty (LSAP)

The LSAP is a variance-normalized local spatial penalty that proxies significance-tested LISA (Anselin, 1995). For each node $i \in G$:

$$I_i = \frac{(x_i - \bar{x}) \sum_j w_{ij}(x_j - \bar{x})}{m_2}, \quad m_2 = \frac{\sum_{i \in G}(x_i - \bar{x})^2}{|G|}$$

Positive $I_i$ identifies H-H and L-L clusters. LSAP sums only the positive local values. The composite normalizes by $|G|(|G|-1)$, which bounds LSAP $\in [0, 1]$ because $\mathbf{W}$ is row-standardized. The LSAP serves as a **continuous heuristic** in the optimization loop (permutation LISA would be prohibitively expensive at $\sim 10^5$ evaluations per seed × 9,999 permutations × $|G|$ locations); post-hoc inferential significance is delivered by `scripts/validate_lisa_proxy.py`, which on the same graph $G$ runs 9,999-permutation conditional LISA and applies Benjamini–Hochberg FDR correction at $q < 0.05$ per-map across the $|G| = 31$ locations (Caldas de Castro & Singer, 2006). The proxy–target relationship is empirically diagnosed in `scripts/lisa_proxy_per_map_diagnostic.py`; see [`docs/limitations/lsap-proxy-goodhart.md`](docs/limitations/lsap-proxy-goodhart.md) for the validated three-test defence.

We omit Getis-Ord $G_i^*$ from the methodology because at $|G| = 31$ its asymptotic z-score is unreliable, and because LISA/LSAP with conditional-permutation inference are better suited to detecting spatial outliers; significance is claimed only via conditional permutation LISA.

### Multi-Jain Fairness Index (Bottleneck JFI)

For one dimension, $J(\mathbf{x}) = (\sum x_i)^2 / (n \sum x_i^2)$, range $[1/n, 1]$. We compute JFI on raw Resources and raw Influence separately and use $J_{\min} = \min(J_R, J_I)$ so that fairness is limited by the least-fair dimension (Jain et al., 1984; Ghodsi et al., 2011).

---

## 3.4 The canonical fitness landscape

The metric definitions in §3.1 and §3.3 are the foundational mathematical forms; the search routines of §3.5 operate not on those forms directly but on a canonical implementation in which three of the four composite terms have been replaced by their differentiable relaxations and the fourth has been variance-stabilized. Three of the four adjustments are the standard treatment of a known issue in either continuous optimization (gradient discontinuity at $\min$ and at the hinge) or spatial statistics (heteroskedasticity of local Moran's $I_i$ under row-standardized $\mathbf{W}$); the fourth is a variance-equalization choice motivated by §3.8's empirical analysis. None introduces a new mathematical object. Every reported phase of this study runs under the formulation defined in this section.

### 3.4.1 Smooth-min Jain bottleneck

The Multi-Jain bottleneck $J_{\min} = \min(J_R, J_I)$ has zero gradient with respect to the non-bottleneck dimension. Under simulated annealing's Metropolis criterion, perturbations that improve only the non-bottleneck dimension produce $\Delta S = 0$ moves, blunting the gradient signal and inducing plateau dynamics. We use the generalized power-mean (Hölder mean) relaxation of $\min$ at order $-p$ with $p = 8$:

$$J_{\min}^{(p)}(J_R, J_I) \;=\; \Bigl( \tfrac{1}{2}\bigl(J_R^{-p} + J_I^{-p}\bigr) \Bigr)^{-1/p},$$

which converges to $\min(J_R, J_I)$ as $p \to \infty$ and is everywhere differentiable for $J_R, J_I > 0$ (Hardy, Littlewood & Pólya, 1952, §2.3; Boyd & Vandenberghe, 2004, §3.1.5). One verifies $\min(J_R, J_I) \le J_{\min}^{(p)} \le \min(J_R, J_I) \cdot 2^{1/p}$, so the relaxation overestimates the true bottleneck by at most $\min(J_R, J_I) \cdot (2^{1/p} - 1)$ in JFI units; at $p = 8$ this multiplicative slack is $2^{1/8} - 1 \approx 0.091$ and is achieved only at $J_R = J_I$, decaying as $|J_R - J_I|$ grows. The relaxation preserves the location of the maximum (both $J_{\min}$ and $J_{\min}^{(p)}$ attain their upper bound at $J_R = J_I = 1$); the smoothing acts only on the gradient near the bottleneck switch, where the unsmoothed $\min$ is non-differentiable and SA's Metropolis criterion would otherwise see $\Delta S = 0$ moves.

### 3.4.2 Softplus hinge

The one-sided hinge $\max(0,\, I - \mathbb{E}[I])$ is non-differentiable at $I = \mathbb{E}[I]$, the threshold separating "no spatial penalty" from "spatial penalty applies" — precisely the most informative point on the objective surface. We use softplus at $k = 10$:

$$\mathrm{softplus}_k(I - \mathbb{E}[I]) \;=\; \tfrac{1}{k}\log\!\bigl(1 + e^{k(I - \mathbb{E}[I])}\bigr).$$

This is log-sum-exp of $\{0,\, I-\mathbb{E}[I]\}$ with one term fixed at zero, hence the same Boyd & Vandenberghe (2004, §3.1.5) reference applies. The softplus is a uniform upper bound on the hinge, $\mathrm{softplus}_k(x) \ge \max(0, x)$, with $\mathrm{softplus}_k(x) - \max(0, x) \le \log(2)/k$ achieved at $x = 0$ and decaying exponentially as $|x|$ grows. At $k = 10$ this bound is $\approx 0.069$. For $|G| \in [29, 31]$, $|\mathbb{E}[I]| = 1/(|G|-1) \in [0.033, 0.036]$, so the smoothing magnitude is comparable to $|\mathbb{E}[I]|$ in raw units. We accept this trade-off because the smoothing supplies the gradient signal SA's Metropolis criterion requires near the null threshold; the alternative (unsmoothed hinge) produces $\Delta S = 0$ moves at precisely the boundary that distinguishes "no spatial penalty" from "penalty applies." The smoothing is one-sided and bounded — it cannot produce penalties of the wrong sign — and the location of the optimum (driving $I$ as far below $\mathbb{E}[I]$ as the topology allows) is unchanged.

### 3.4.3 Variance-stabilized LSAP

The local Moran's $I_i$ defined in §3.3 has variance that scales inversely with the topological degree $k_i$ under row-standardized $\mathbf{W}$ (following the variance derivation in Anselin, 1995). On the canonical 6-player layout, $k_i \in \{3, 5, 6\}$ (12 boundary nodes at degree 3, 6 transitional nodes at degree 5, 13 interior nodes at degree 6), so without correction 12 of 31 nodes — 39% of the spatial graph — contribute to LSAP at $\sqrt{6/3} = \sqrt{2}$-inflated standard deviation relative to interior nodes, regardless of clustering structure. We apply the natural variance-stabilizing scaling

$$I_i^{\,\text{corr}} \;=\; \sqrt{k_i}\,I_i,$$

so each node contributes to LSAP at unit-standardized scale. LSAP is then the sum of positive $I_i^{\text{corr}}$ values, normalized as in §3.3. Boots & Tiefelsdorf (2000) discuss the same heteroskedasticity for bounded regular tessellations and motivate equivalent corrections in the edge-effect setting.

### 3.4.4 Static Gen-0 σ normalization

The three composite terms differ in empirical magnitude by orders of magnitude on this topology; without an equalizing divisor the composite collapses to a single-objective optimizer for whichever term carries the largest variance. The §3.1 problem formulation adopts the standard treatment: each term is divided by its empirical standard deviation under $N = 1{,}000$ uniformly random map configurations, with the divisor frozen for the entire run. The alternative — dynamic per-iteration $\sigma$ — conflates landscape geometry with optimization progress, producing a composite that is not a well-defined function of map state alone, since $S(\text{map})$ acquires a path-dependence through the running $\sigma$ estimate. The empirical magnitude of the equalization is documented in §3.8: under 1:1:1 weights, the Moran's I hinge term accounts for approximately $90\%$ of total weighted variance at Gen-0, collapsing to near-zero variance at convergence as the optimizer drives spatial autocorrelation to its null floor.

A `corrected_landscape=False` mode is retained in the codebase for sensitivity probes against the nominal published forms of the constituent metrics (raw $\min$, raw hinge, raw $I_i$) and dynamic $\sigma$ normalization; it does not appear in any reported result of this study.

---

## 3.5 Algorithms

- **HC (Greedy Hill-Climbing):** Accepts only improving moves; no memory. HC optimizes the `balance_gap` as its sole scalar objective; it is included in Phase 2 exclusively as a baseline comparator and warm-start mechanism, not as an instrument for the main multi-objective experiment.
- **SA (Simulated Annealing):** Markov chain with $P(\text{accept}) = \exp(-\Delta/T)$. Cooling rate $\alpha = (T_{\min}/T_0)^{1/N}$ over $N$ steps (Kirkpatrick et al., 1983). $T_0$ calibrated to initial acceptance rate.
- **SGA (Single-Objective GA):** Same BFS-blob crossover and swap mutation as NSGA-II; scalar tournament selection and $(\mu+\lambda)$ replacement. **Architecture control:** SA vs SGA = same scalar, different architecture; SGA vs NSGA-II = same operators, scalar vs Pareto. Recombination is localized (BFS-connected blobs + OX1) so that the geometric contiguity of the search space is preserved and epistasis is controlled; no further structural changes are required for epistatic control.
- **NSGA-II:** Three-objective Pareto (1 − JFI, |I|, LSAP). BFS-blob crossover; non-dominated sorting and crowding distance (Deb et al., 2002). For Track A we apply a posteriori scalarization (min composite over the front).
- **TS (Tabu Search):** Full-neighbourhood search; best non-tabu move applied each iteration. Default tenure scales with neighbourhood size: $\max(3, \lceil 0.05 \cdot \binom{S}{2}\rceil)$ so that at saturation budgets (e.g. 500k evaluations) TS retains meaningful memory; optional override via $\lceil k\sqrt{S}\rceil$ (Glover, 1989). **Methodological control** (cf. Terra Mystica PCG literature).
- **RS (Random Search):** Uniform random permutations; null baseline.

SA accepts equal-fitness moves with probability 1 under the Metropolis criterion; HC, TS, and SGA use lexicographic tie-breaking (primary: composite score; secondary: $-J_{\max}$) on zero-delta moves. This behavioral difference on flat fitness regions is intentional and constitutes part of the architectural comparison.

**Structural corrections (optional `--corrected-landscape`).** When enabled, the benchmark applies four corrections to the fitness landscape: (1) **Static Gen-0 normalization:** empirical standard deviations of the three objective terms are computed once from 1,000 random map permutations and used as fixed divisors for the entire run, eliminating variance dominance while keeping the landscape stationary. (2) **Smooth objectives:** the Multi-Jain bottleneck is replaced by a smooth L$_{-p}$ mean ($p=8$) and the Moran hinge by a softplus ($k=10$) to restore gradient continuity for SA. (3) **Local-variance LSAP:** each local Moran $I_i$ is scaled by $\sqrt{k_i}$ (node degree) so that edge and interior nodes have comparable scale. (4) **TS tenure** as above. These corrections are gated by the `--corrected-landscape` flag; when off, behaviour matches the nominal formulation for reproducibility.

Bayesian optimization is excluded (see `docs/bayesian_optimization_exclusion.md`).

---

## 3.6 Experimental Protocol

- **Seeds:** Benchmark seeds 0–99 (100 maps). Tuning seeds 9,000–9,099 (100). Held-out seeds 9,100–9,149 (50).
- **Budget:** 1k–500k fitness evaluations per algorithm per seed (saturation study). Budget consumed as: HC/SA by iterations; SGA/NSGA-II by generations × population; TS by full-neighbourhood iterations; RS by independent samples. Budgets are chosen on a quasi-logarithmic grid (e.g. 1k, 5k, 10k, 50k, 100k, 500k) to characterize **anytime performance**: how quickly each algorithm improves as a function of evaluations, and at what point additional budget yields diminishing returns.
- **Hyperparameter tuning:** SA, SGA, NSGA-II tuned on disjoint seeds (9,000–9,099) via Optuna TPE (50 trials each). TS tuned via **exhaustive grid search** over the tenure coefficient k (θ = max(3, ⌈k·√S⌉)) on the same seeds. Best parameters validated on a held-out set (9,100–9,149). `best_params.json` reports `cv_mean`, `cv_std` (training) and `held_out_mean`, `held_out_std` for overfitting comparison. **Held-out variance** is discussed in §3.7.
- **Design topology (randomized block / repeated measures):** The main benchmark is a **randomized block design**: the same set of 100 seeds is evaluated under every algorithm at each budget level. Observations are therefore **paired within seeds** rather than independent across algorithms.
- **Convergence:** `evals_to_best` recorded; convergence curves and budget utilization analyzed per budget. For each budget we summarize performance via medians and interquartile ranges, and use **bootstrap 95% confidence intervals on median differences** (rather than raw empirical percentiles) to obtain stable uncertainty estimates in the presence of rugged, discrete combinatorial landscapes.

---

## 3.7 Hyperparameter Validation and Held-Out Variance

**(A) Landscape ruggedness.** The variance observed on the held-out validation set is not a failure of the hyperparameters (e.g. "overfitting to the tuning set"). In combinatorial optimization, unlike predictive ML, each run starts from a different random map (seed). The TI4 map fitness landscape is highly rugged; different seeds place the search in vastly different regions of the space. Some starting maps are relatively close to a fair configuration, while others require traversing deep, unrecoverable local minima. Thus, **the spread in held-out performance reflects the inherent variability of problem difficulty across starting states**, not that the tuned settings (e.g. tabu tenure, cooling schedule) have "memorized" the tuning seeds.

**(B) Resiliency of the mean.** Empirically, the held-out distribution is comparable to (and not noisier than) the cross-validation distribution at the canonical Phase 0 SA tuning: CV mean ± std = $0.0914 \pm 0.0447$ vs held-out mean ± std = $0.0872 \pm 0.0423$ (`best_params.json`; tuning seeds 9000–9099, held-out seeds 9100–9149, disjoint from each other and from the benchmark seeds 0–99). The held-out mean falls within ${\sim}5\%$ of the CV mean — slightly *below*, not above — and the held-out standard deviation (0.0423) is smaller than the CV standard deviation (0.0447), so the directional pattern that would indicate hyperparameter overfitting to the tuning seeds is absent. The absolute coefficient of variation is high ($\approx 49\%$ on CV), but this reflects the rugged-landscape variance described in (A) — different starting seeds produce vastly different optimization trajectories — not tuning instability. The same hyperparameters that perform well on the tuning seeds perform well on average on unseen seeds, with comparable (and slightly smaller) spread.

**(C) Phase 2 as the ultimate mitigation.** This variance is precisely why the main benchmark (Phase 2: Saturation Benchmark) is designed as it is. We do not judge algorithms on a single run or on means alone. We run each algorithm on **100 seeds** (disjoint from tuning and held-out) and use **non-parametric inference**: Friedman omnibus test and pairwise Wilcoxon signed-rank tests with Holm–Bonferroni correction, plus Vargha–Delaney A and bootstrap CIs on median difference. By using repeated measures across 100 unseen seeds, we explicitly account for starting-state variance, ensuring that no algorithm is unfairly penalized by a "bad" starting seed and that claims of superiority are based on reported p-values and effect sizes.

**(D) Budget-invariance assumption (pre-registered).** Phase 0 tunes hyperparameters at a fixed evaluation budget per seed (`iter_budget = 1000` for the multi-algorithm run; `iter_budget = 1000` for the SA-only condition ablation). Phase 1 then applies those hyperparameters at every benchmark budget (1k, 5k, 10k, 25k, 50k, 100k, 200k, 500k). The implicit methodology claim is **budget-invariance**: the values that work at one tuning regime continue to work across the benchmark range. This claim is plausible for SA's regime-controlling parameters (`initial_acceptance_rate` and `min_temp` describe shape, not magnitude) and for population-genetic operators whose effect is expected to be self-similar across budgets, but it is not free. We test it via a sensitivity check (Phase 0e in `submit_paper1_multialgo.sh`): SA is re-tuned at `iter_budget = 10{,}000` (10× the primary tuning budget); the relative drift of `initial_acceptance_rate` and `min_temp` between the two tunings is reported in the run log. A drift threshold of $20\%$ for `rate` or $50\%$ for `min_temp` triggers a warning that the assumption may not hold; below the threshold, the assumption is treated as defensible for the manuscript's reported budget range. Per-budget tuning is the methodologically rigorous alternative for an extension paper, but adds an order-of-magnitude cost in Phase 0 and is out of scope for this study.

---

## 3.8 Analysis Tracks

**Track A — Scalar (methods justification).** The 1:1:1 composite is applied to every algorithm's final solution; for NSGA-II, the Pareto-front member with minimum composite is selected (a posteriori scalarization). For NSGA-II, the Track A composite score is computed from the Pareto front member with minimum composite score at the end of the run; for scalar algorithms it is the run-best composite score observed at any point during the run. All six algorithms (RS, HC, SA, SGA, NSGA-II, TS) are compared via **median and IQR**, **Friedman** test, **Wilcoxon** signed-rank pairwise with Holm–Bonferroni correction, **Vargha–Delaney A**, and **bootstrap 95% CIs** on median difference. We do not base claims on mean values; we report p-values and claim one algorithm outperforms another only when the corrected p-value is below the chosen significance level. **Empirical variance note (pre-registered, `variance_equalization_diagnostic.py`).** The two-regime story documents why 1:1:1 weights are operationally consistent under the canonical formulation (§3.4). At Gen-0 ($N = 1{,}000$ uniformly random configurations), the Moran's I hinge term carries approximately $90\%$ of total weighted empirical variance under the nominal 1:1:1 weights; JFI gap and LSAP share the remaining $\sim 10\%$. Reported in isolation that figure suggests the composite is a single-objective spatial optimizer in disguise — but the σ-shift diagnostic shows the regime is transient: by SA convergence the Moran hinge has collapsed to near-zero variance ($\sigma \approx 0$, the optimizer drives spatial autocorrelation to its null floor) while LSAP and JFI retain meaningful variance, contributing approximately $56\%$ and $44\%$ of weighted empirical variance respectively at convergence. The Gen-0 dominance is therefore an early-search transient that the static σ normalization (§3.4.4) prevents from becoming a permanent landscape distortion; the 1:1:1 weights govern the relative importance of the three objectives in the optimized subspace where the benchmark results are reported. Track B (Hypervolume, weight-independent) remains the primary evaluation regardless.

**Track B — Pareto (multi-objective quality).** NSGA-II's raw Pareto archives are evaluated with **Hypervolume (HV)**, **IGD+**, and **Spacing**. This is the primary evaluation for multi-objective performance and avoids scalarization. Scalar algorithms do not produce Pareto fronts and are compared only via Track A.

**Cross-method IGD.** Scalar-algorithm terminal states (SA, TS, HC, SGA) are projected into the 3-objective space and their IGD+ to the per-budget empirical reference front is reported. This validates that single-objective trajectories converge near the Pareto manifold and supports the justification for using SA (or another scalar algorithm) in production.

**Unified HV (objective commensurability).** To place scalar and multi-objective methods in a **single objective space**, we additionally compute Hypervolume for every algorithm using **empirical Pareto fronts** extracted from the logged run histories. For each (algorithm, seed, budget), we collect the non-dominated set of visited maps in the canonical 3D space $(1 - J_{\min}, |I|, \text{LSAP})$ and compute HV against a common nadir reference point derived from the worst observed objective values (auto: worst×1.1). This yields a weight-independent, distribution-agnostic quality indicator that is commensurable across RS, HC, SA, SGA, NSGA-II, and TS. The unified HV tables and non-parametric statistics (Friedman / Wilcoxon / Vargha–Delaney) are produced by `scripts/unified_hv_analysis.py` using the `unified_archives/` emitted by `benchmark_engine.py`.

**Methodological scope.** The present study contributes a spatial-statistics methodology — Multi-Jain bottleneck JFI, smooth-min/softplus relaxations, $\sqrt{k}$-stabilized LSAP, static Gen-0 $\sigma$ normalization, and the canonical composite of §3.4 — and an empirical demonstration on the TI4 6-player toy problem under fixed-anomaly topology that spatial metrics detect map configurations scalar fairness metrics cannot. Generalization to variable-topology spatial-allocation problems (anomaly placement as a swap variable) is the natural next direction for the methodology and is sketched in the README's *Future Work* section.

**Statistical justification.** Because the design is randomized-block / repeated-measures (same seeds under all algorithms), our data consist of **dependent paired samples** across algorithms. The **Friedman** test is therefore the correct non-parametric omnibus analogue of repeated-measures ANOVA; independent-sample tests such as Kruskal–Wallis would incorrectly treat within-seed variance as between-subject noise and inflate Type II error. Likewise, pairwise comparisons use the **Wilcoxon signed-rank** test (paired) rather than Mann–Whitney U / rank-sum (independent samples), ensuring that within-seed pairing is fully exploited in the inference.

---

## 3.9 Ablation Study: Objective-Weight Paths

To empirically validate the necessity of the multi-objective formulation and satisfy the parsimony requirement (Anselin, 1995), the methodology employs a five-condition ablation design:

| Condition | Weights ($w_1$:$w_2$:$w_3$) | Purpose |
|-----------|------------------------------|---------|
| **C0** (`jfi_only`) | 1:0:0 (JFI only) | Baseline — current state of the art |
| **C1** (`moran_only`) | 0:1:0 (Moran's I only) | Does global clustering alone change maps? |
| **C2** (`lsap_only`) | 0:0:1 (LSAP only) | Does local clustering alone change maps? |
| **C3** (`jfi_moran`) | 1:1:0 (JFI + Moran) | Is global Moran sufficient without LSAP? (parsimony test) |
| **C4** (`full_composite`) | 1:1:1 (full composite) | Does LSAP add constraint beyond C3? |

C3 is the critical parsimony test: if global Moran's I is sufficient to constrain topological exploitation while maintaining JFI parity, C4 is theoretically redundant. All conditions use Gen-0 static normalization (`--corrected-landscape`) so that objective scales are topology-agnostic.

**Empirical result.** At budget = 500k under the canonical formulation (100 seeds × 3 chains aggregated to one observation per seed via mean), the formal analysis (`scripts/analyze_phase1_conditions.py`) returns the predicted directions for all three primary metrics. The Friedman omnibus rejects the null of equal distributions across conditions for every metric ($\chi^2 = 349.82$ for Moran's I, $\chi^2 = 384.88$ for LSAP, $\chi^2 = 367.14$ for JFI; all $df = 4$, $p \ll 10^{-70}$).

- **C0 (JFI only)** produces maps that are numerically fair (median JFI $= 0.9999$) but topologically pathological. Median Moran's I $= -0.0986$ (effectively at the null floor for $|G| \approx 30$); median LSAP $= 4.4571$. Resources cluster in spatially contiguous regions, maximizing the LSAP penalty exactly as the scalar-fairness baseline allows.
- **C3 (JFI + Moran)** is the parsimony test. Global autocorrelation is suppressed (median Moran's I $= -0.5926$), but **localized clustering is not** (median LSAP $= 0.9674$). Global Moran's I is blind to local non-stationarity in this topology — exactly the failure mode the §3.3 LSAP definition is designed to detect.
- **C4 (full composite, 1:1:1)** suppresses both global and local autocorrelation (median Moran's I $= -0.6714$, median LSAP $= 0.0000$).

Pairwise C0 vs Cx Wilcoxon signed-rank paired by seed, Holm-Bonferroni corrected across all 12 simultaneous tests (4 condition pairs × 3 metrics), is significant in every cell with Vargha–Delaney $A \geq 0.718$ (large by the pre-registered $A \geq 0.64$ threshold). The headline C0$\to$C4 contrasts: $\Delta(\text{Moran's I}) = +0.5728$, 95% CI $[+0.5362, +0.6030]$, Cohen's $d_z = +5.019$ (large); $\Delta(\text{LSAP}) = +4.4571$, 95% CI $[+4.1102, +4.9822]$, Cohen's $d_z = +3.191$ (large) — both far above the $|d_z| \geq 0.8$ "large" band of Lakens (2013).

**JFI parity — statistical vs operational.** The one-sided Wilcoxon JFI parity test rejects the null for every Cx including C3 and C4: the formal test detects that C3 and C4 produce statistically lower JFI than C0. The magnitude of the sacrifice, however, separates the conditions cleanly. C1/C2 sacrifice JFI by median $\Delta \approx 0.011$ (~1.1% loss) — operationally meaningful and consistent with the §3.9 prediction that pure spatial-only conditions trade fairness for spatial structure. C3/C4 sacrifice JFI by median $\Delta = 0.00007$ and $0.00015$ respectively (~0.01–0.02% loss) — well below the operational fairness threshold for a metric bounded in $[1/n, 1]$, and three orders of magnitude smaller than the C1/C2 sacrifice. The Wilcoxon $W = 82.5$ for C3 and $W = 0.0$ for C4 reflect the consistent direction across $n = 100$ paired observations rather than a substantive magnitude (Vargha–Delaney $A = 0.718$ for C3 and $A = 0.883$ for C4 — large by VDA bands but the smallest A's in the entire pairwise panel). The operational claim is that C4 maintains JFI within $0.0002$ of C0 while suppressing LSAP by $4.46$; the formal-statistical claim is more nuanced and is reported in `output/<run>/stats/phase1_jfi_parity.csv` for reviewer inspection.

LSAP provides constraint that global autocorrelation alone cannot achieve, and the constraint is paid at a JFI cost three orders of magnitude smaller than the cost of suppressing only one of the spatial dimensions.

---

## 3.10 Null hypotheses and pre-registered results

**RQ1.** $H_0$: at the canonical hyperparameters and evaluation budget (500k), the distribution of optimized maps does not differ across the five ablation conditions (C0–C4) in the spatial metrics (Moran's I, LSAP) or in JFI. **Result:** Rejected — Friedman omnibus $\chi^2 = 349.82$ (Moran's I), $384.88$ (LSAP), $367.14$ (JFI) at $df = 4$, $p \ll 10^{-70}$ for every metric (§3.9). Headline pairwise C0$\to$C4: Vargha–Delaney $A \geq 0.88$ for all three metrics; Cohen's $d_z = +5.019$ (Moran's I), $+3.191$ (LSAP), $+1.109$ (JFI) — all "large" by Lakens (2013).

**RQ2.** $H_0$: NSGA-II's hypervolume does not exceed that of scalar algorithms (RS, HC, SA, SGA, TS) under the canonical formulation at equal total evaluation budget (Wilcoxon signed-rank, one-tailed, Holm–Bonferroni corrected). **Result:** Rejected at the canonical budget against every scalar, but the comparison against SA is a budget-dependent crossover. At $b = 500{,}000$, NSGA-II's hypervolume significantly exceeds RS, HC, SGA, and TS with large effect (Vargha–Delaney $A = 0.86$ for SGA up to $0.99$ for HC, with RS and TS between; Holm-corrected $p < 0.01$). Against SA the separation is significant but falls below the pre-registered practical-significance threshold ($A \geq 0.64$; Design_Rationale §3): $A = 0.54$, $p_{\text{Holm}} = 8.4 \times 10^{-4}$, median-hypervolume gap below $5 \times 10^{-4}$. Across budgets the SA comparison crosses over: at $b = 1{,}000$ NSGA-II does not exceed SA ($A = 0.44$, $p_{\text{Holm}} = 0.997$, n.s.), from $b = 5{,}000$ onward it does, but the effect stays below that practical-significance threshold at every budget ($A \in [0.44, 0.61]$, all $< 0.64$). A scalar SA Markov chain therefore tracks the dedicated multi-objective optimizer below the practical-significance line at every budget, which the other four scalars do not.

**RQ3.** *(Trade-offs between balance gap and spatial distribution.)* Exploratory — no directional hypothesis pre-specified. Reported as Spearman correlations between balance gap and spatial metrics across optimized solutions. **Result:**

- Within the canonical full-composite condition C4 at $b = 500{,}000$ (100 SA-optimized maps): $\rho(\text{balance\_gap}, \text{Moran's I}) = +0.176$ ($p = 0.079$, n.s.); $\rho(\text{balance\_gap}, \text{LSAP}) = +0.062$ ($p = 0.542$, n.s.); $\rho(\text{balance\_gap}, \text{JFI}) = -0.544$ ($p < 10^{-7}$). At convergence the spatial floor is saturated, leaving no residual signal in the spatial dimensions to correlate with residual balance-gap variation; the JFI–balance-gap link is recovered as expected.
- Pooled across all five conditions × eight budgets ($n = 12{,}000$ rows), the spatial trade-offs are recovered: $\rho(\text{balance\_gap}, \text{Moran's I}) = -0.415$, $\rho(\text{balance\_gap}, \text{LSAP}) = -0.436$, $\rho(\text{balance\_gap}, \text{JFI}) = -0.676$. These pooled rows count all three chains per seed as separate observations, so the $n = 12{,}000$ figure overstates the independent sample size; the coefficients are reported descriptively and no pooled $p$-value is claimed (the inferential weight rests on the within-C4 correlations above, computed on 100 independent seeds). Maps with lower balance gap also have lower spatial autocorrelation across the cross-condition cohort, consistent with the §3.9 finding that the full composite improves all three dimensions jointly.

**RQ4.** $H_0$ (registered): the time to reach a target composite score does not differ across algorithms (Friedman test, paired by seed). *Pre-specified deviation.* The registration operationalized this as wall-clock seconds; we report the Friedman omnibus on **evaluations-to-best** instead (`evals_to_best`, the evaluation index at which each run's best composite first appeared, recorded in `results.csv`). The deviation is on two counts, each adopted for construct validity: evaluation count rather than wall-clock seconds, to factor out implementation overhead and cluster-architecture heterogeneity (the confound a wall-clock metric conflates with the execution environment); and each run's own best rather than a fixed external target, so the measure is well-defined for every algorithm without choosing a threshold that privileges one. Evaluation count to a run's best is the standard machine-independent anytime-performance measure in the stochastic-local-search and metaheuristics literature (Hoos & Stützle, 2004). Wall-clock (`elapsed_sec`) is retained and reported as a secondary descriptive statistic per algorithm. **Result:** Rejected. At the canonical RQ4 budget $b = 200{,}000$ (the largest budget at which all six algorithms carry a real, non-sentinel `evals_to_best`), the six-way Friedman omnibus is $\chi^2 = 456.80$ ($df = 5$, $n = 100$, $p = 1.7 \times 10^{-96}$), and it is significant at every budget tested from $b = 1{,}000$ to $b = 200{,}000$. NSGA-II reaches its best 1:1:1 composite only late in each run: its median `evals_to_best` at $b = 200{,}000$ is 134,100, about $67\%$ of the budget. At $b = 500{,}000$ NSGA-II was not re-instrumented, so RQ4 there is reported as a five-way descriptive comparison only.

**Mechanism: depth versus breadth.** The RQ2 crossover and the RQ4 breadth tax are two views of one structural trade-off between search depth and search breadth under a fixed evaluation budget. On this rugged combinatorial landscape (the $37! \approx 1.37 \times 10^{43}$ permutation state space of §3.1), escaping local optima rewards sustained trajectory depth. A scalar Markov chain such as SA spends its entire budget extending a single trajectory, maximizing depth; NSGA-II spreads the same budget across a population front, truncating the depth of any one lineage. Under a severely restricted budget NSGA-II is therefore at a structural disadvantage, which is why at $b = 1{,}000$ it does not exceed SA on hypervolume. As the budget grows its breadth matures and the comparison crosses over (significant from $b = 5{,}000$), yet the effect stays below the pre-registered practical-significance threshold at every budget: even at $b = 500{,}000$, where the population dynamics are fully realized, the SA gap remains under that line. That a single scalar chain holds parity with NSGA-II across the entire budget range is the stronger-than-expected result.

RQ4 quantifies the same mechanism on the shared, black-box observable of anytime-composite delivery, and its interpretation requires a construct caveat. `evals_to_best` is a genuine internal-convergence measure for the five algorithms that optimize the scalar composite directly; for NSGA-II it is instead the evaluation index at which a good composite first appeared inside a population the algorithm never selects on that scalarization. NSGA-II's late median (about $67\%$ of the budget) therefore reflects the cost of extracting a single 1:1:1 optimum from a breadth-first Pareto search, not a failure of multi-objective convergence. The inferential weight for the mechanism accordingly rests on RQ2, where hypervolume compares the methods on NSGA-II's own native objective and the comparison is fair by construction; RQ4 is the corroborating operational companion, read on the question a practitioner who wants the single 1:1:1 map actually faces.

That framing also bounds the metric's reach. The anytime-composite measure privileges a single-optimum practitioner. Under a portfolio framing, where the deliverable is a diverse set of mathematically viable, Pareto-optimal map configurations representing distinct fairness trade-offs, the same population breadth that reads as a tax here is the product rather than its cost. NSGA-II's slower convergence to one scalar optimum is the price of collapsing a Pareto front to a point, and says nothing against its value when a set of alternatives, not a single map, is what is wanted.

---

## 3.11 Statistical methods and effect size

**Effect size for paired comparisons.** For paired (within-seed) comparisons we report two complementary effect sizes: **Vargha–Delaney $A_{12}$** (non-parametric stochastic-dominance probability, $A \in [0, 1]$, with bands $|A - 0.5| < 0.06$ negligible, $< 0.14$ small, $< 0.21$ medium, $\geq 0.21$ large; Vargha & Delaney, 2000) and **Cohen's $d_z$** (standardized mean difference of paired differences, $d_z = \bar{D} / s_D$ with $s_D$ at ddof=1; bands $|d_z| < 0.2$ negligible, $< 0.5$ small, $< 0.8$ medium, $\geq 0.8$ large; Lakens, 2013). VDA is the primary; Cohen's $d_z$ supplies the parametric companion. Both are produced by `scripts/analyze_phase1_conditions.py` for every pairwise contrast and emitted in `phase1_condition_pairs.csv` (`vda_A`, `vda_mag`, `cohens_dz`, `cohens_dz_mag` columns).

**Multi-objective fairness and tie-breaking.** The composite uses the **Multi-Jain bottleneck** $J_{\min} = \min(J_R, J_I)$, so the fairness term prioritizes the *most disadvantaged* resource dimension (Resources or Influence). When a move leaves the composite score unchanged (e.g. improves only the non-bottleneck dimension), **HC, TS, and SGA** use a **lexicographic tie-breaker**: they compare solutions by $(\text{composite}, -J_{\max})$ where $J_{\max} = \max(J_R, J_I)$, so the solution with higher $J_{\max}$ is preferred on the plateau. **SA** does not use this key; it accepts equal-cost moves with probability 1 and thus traverses the plateau without an explicit tie-breaker. The observed **JFI success** (good fairness on both dimensions) is therefore the result of a multi-objective search that (i) prioritizes the worst-off dimension in the scalar composite and (ii) still improves the secondary dimension on the plateau (via lex_key for HC/TS/SGA and equal-move acceptance for SA).

---

## References (Methodology)

Anselin, L. (1995). Local indicators of spatial association — LISA. *Geographical Analysis*, 27(2), 93–115.

Boots, B., & Tiefelsdorf, M. (2000). Global and local spatial autocorrelation in bounded regular tessellations. *Journal of Geographical Systems*, 2(4), 319–348.

Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, 6(2), 182–197.

Ghodsi, A., Zaharia, M., Hindman, B., Konwinski, A., Shenker, S., & Stoica, I. (2011). Dominant resource fairness: Fair allocation of multiple resource types. *Proceedings of the 8th USENIX Symposium on Networked Systems Design and Implementation (NSDI)*, 24, 323–336.

Glover, F. (1989). Tabu search — Part I. *ORSA Journal on Computing*, 1(3), 190–206.

Ishibuchi, H., Masuda, H., Tanigaki, Y., & Nojima, Y. (2015). Modified distance calculation in generational distance and inverted generational distance. *Proceedings of the International Conference on Evolutionary Multi-Criterion Optimization (EMO)*, 110–125.

Jain, R., Chiu, D. M., & Hawe, W. R. (1984). A quantitative measure of fairness and discrimination for resource allocation in shared computer systems. *DEC Research Report TR-301*.

Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by simulated annealing. *Science*, 220(4598), 671–680.

Lakens, D. (2013). Calculating and reporting effect sizes to facilitate cumulative science: a practical primer for $t$-tests and ANOVAs. *Frontiers in Psychology*, 4, 863. https://doi.org/10.3389/fpsyg.2013.00863

Libório, M. P., de Abreu, J. F., Ekel, P., & Machado, A. (2022). Effect of sub-indicator weighting schemes on the spatial dependence of multidimensional phenomena. *Journal of Geographical Systems*, 25, 185–211.

Vargha, A., & Delaney, H. D. (2000). A critique and improvement of the CL common language effect size statistics of McGraw and Wong. *Journal of Educational and Behavioral Statistics*, 25(2), 101–132.

Hoos, H. H., & Stützle, T. (2004). *Stochastic Local Search: Foundations and Applications*. Morgan Kaufmann.

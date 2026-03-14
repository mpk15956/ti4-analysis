# Design Rationale: Geographic Paper and Experiment Protocol

This document is the single source of truth for what the paper is trying to do and how the experiment is designed. It prevents future collaborators or reviewers from steering the work back toward an algorithmic-contribution framing.

---

## 1. Paper Goal

The paper’s contribution is **geographic/methodological**, not algorithmic. We show that **spatial metrics (Moran’s I, LSAP) add meaningful constraint beyond scalar distributional fairness (JFI)** on topologically embedded discrete spaces. Algorithms (e.g. SA) are **instruments** used to generate optimized maps; we report which instrument we use for reproducibility, but the finding is about spatial metrics, not about algorithm novelty. The study is scoped to small discrete bounded topologies (e.g. n=37); human validation (causal chain from spatial clustering to competitive disadvantage in play) is explicitly **future work**.

---

## 2. Five-Condition Ablation

| Condition | Weights ($w_1$:$w_2$:$w_3$) | Purpose |
|-----------|------------------------------|---------|
| **C0** (`jfi_only`) | 1:0:0 (JFI only) | Baseline — current state of the art |
| **C1** (`moran_only`) | 0:1:0 (Moran's I only) | Does global clustering alone change maps? |
| **C2** (`lsap_only`) | 0:0:1 (LSAP only) | Does local clustering alone change maps? |
| **C3** (`jfi_moran`) | 1:1:0 (JFI + Moran) | Is global Moran sufficient without LSAP? (parsimony test) |
| **C4** (`full_composite`) | 1:1:1 (full composite) | Does LSAP add constraint beyond C3? |

C3 is the critical parsimony test per Anselin (1995): if global Moran's I is sufficient to constrain topological exploitation, C4 is theoretically redundant. Only by empirically demonstrating that C3 fails to suppress localized clustering can the paper justify the full three-term composite. All five conditions use **SA** as the instrument; same seeds, same budget per condition.

**Lex_key note:** The `--conditions` flag sets the **primary composite weights**. HC, TS, and SGA use a lexicographic tie-breaker with $J_{\max}$ as the secondary dimension regardless of condition, so under C1/C2 they would still apply secondary JFI pressure. **SA is the chosen instrument** for the main experiment because its Metropolis criterion does not impose that secondary optimization; the declared condition is then consistent with what is actually optimized.

---

## 3. Statistical Protocol

**Design:** For each of 100 seeds, run SA under each condition at the same budget. Collect final maps and **raw objective vectors** $(1-J_{\min}, I, \text{LSAP})$. **Cross-condition comparison uses raw objective vectors only**; composite score is **not** comparable across conditions (when one weight is 1.0 the composite is that term only). Composite score is only comparable within a condition.

**Primary tests (two pre-specified):**
1. Wilcoxon signed-rank on **LSAP** of final solutions, paired by seed.
2. Wilcoxon signed-rank on **Moran's I hinge** $\max(0, I - E[I])$ of final solutions, paired by seed.

α = 0.05. **Holm–Bonferroni correction is applied across all 8 primary spatial tests simultaneously (2 spatial metrics × 4 condition pairs: C0 vs C1, C0 vs C2, C0 vs C3, C0 vs C4), not within each pair independently.**

Pre-specified effect size: Vargha–Delaney A ≥ 0.64 for practical significance.

**Interpretation rule:** “Spatial metrics add constraint” is **supported if either** spatial metric (LSAP or Moran's I hinge) shows A ≥ 0.64 in the relevant comparison **and** the JFI parity check passes for that condition pair.

**JFI parity check (all four pairs C0 vs C1, C0 vs C2, C0 vs C3, C0 vs C4):**
Wilcoxon signed-rank on $J_{\min}$, paired by seed. **H0: median $J_{\min}$ in Cx ≥ median $J_{\min}$ in C0 (one-sided, lower tail).** Rejection indicates Cx achieves lower JFI than the baseline — interpreted as **fairness sacrifice**, not spatial improvement. **Note that higher $J_{\min}$ is better (JFI = 1 is perfect equality); lower $J_{\min}$ in Cx than C0 means the condition sacrificed distributional fairness.**

**C1 and C2 interpretation (pre-specified):**
C1 (Moran's I only) and C2 (LSAP only) are **expected to fail JFI parity** — SA will drive the single spatial objective with no JFI pressure, so JFI will likely be compromised. This is **not** a failure of the experiment; it demonstrates that single spatial objectives cannot replace the composite. **C3 and C4 are both predicted to pass JFI parity.** The critical pre-specified prediction for C3 is: C3 passes JFI parity but produces maps with elevated LSAP relative to C4, demonstrating that global Moran's I is blind to local non-stationarity in this grid topology and cannot substitute for LSAP.

**Multi-budget prediction:**
We predict the spatial profile difference between C0 and C4 will be detectable at all budgets ≥ 10k evaluations. Additionally, we predict C3 will pass JFI parity at all budgets but show elevated LSAP relative to C4, confirming LSAP adds constraint independent of budget level. If the C3 vs C4 LSAP difference only emerges at budgets ≥ 100k, we interpret this as evidence that local clustering suppression requires sustained search pressure and report the minimum effective budget as a secondary finding.

---

## 4. Primary Figure

**One primary figure:** $(1-J_{\min}, I, \text{LSAP})$ for **all five conditions** (C0, C1, C2, C3, C4) as grouped boxplots. This is the paper’s central empirical contribution — readers see C0 baseline, C1 moving Moran’s I but not necessarily LSAP, C2 moving LSAP but not necessarily Moran’s I, C3 moving Moran’s I while maintaining JFI parity but failing to suppress LSAP, and C4 suppressing both spatial metrics while maintaining JFI parity. Not only C0 vs C4.

---

## 5. Equal Weights and Gen-0 Decision

**Weights:** The full composite (C4) uses **equal weights 1:1:1** (i.e. $w_1 = w_2 = w_3 = 1/3$). This is self-justifying and consistent with the five-condition scheme; we do not optimize or defend a particular weight vector.

**Gen-0 equalizer:** We use **Option B — Gen-0 static normalization enabled (`--corrected-landscape`).** The empirical standard deviations of the three objective terms are computed from 1,000 random permutations (disjoint seed offset +99999) prior to optimization, and applied uniformly across all optimizers including NSGA-II. This renders the methodology topology-agnostic: a researcher applying this framework to a different grid computes Gen-0 sigma from their own permutation space without re-deriving any domain-specific bounds. The $\sqrt{k_i}$ LSAP scaling (`use_local_variance_lisa=True`) is also active; Gen-0 sigma is sampled from the same $\sqrt{k_i}$-scaled distribution, so normalization and scaling operate at a consistent scale with no compound distortion.

---

## 6. Human Validation Limitation

This study demonstrates computationally that spatial autocorrelation metrics detect map configurations that scalar fairness metrics cannot optimize toward. The causal chain from spatial clustering to competitive disadvantage in human play is not tested here and constitutes the **primary empirical question for subsequent work**, addressable through telemetry from the companion application or controlled play experiments. This sentence appears in the abstract and in limitations.

---

## 7. Pipeline and Minimum Viable Run

- **Phase 0:** SA hyperparameter tuning (disjoint seeds, e.g. 9000–9149).
- **Phase 1 (primary):** Five-condition SA run using **tuned SA parameters** — same seeds, same budgets (e.g. 10k, 50k, 100k, 500k). Invoke with `--conditions jfi_only,moran_only,lsap_only,jfi_moran,full_composite --corrected-landscape --algorithms sa`.
- **Phase 2 (methods justification):** Algorithm benchmarking — all six algorithms, same budgets, using all tuned parameters. This run is **not** the primary result; it justifies the choice of SA as the instrument.

**Pre-submission gate:** Run `scripts/pre_submission_check.sh` from the project root; it must report **zero matches**.

**Mutual exclusivity:** `--conditions` and `--weight-grid-step > 0` cannot both be set; the script exits with an explicit error if they are.

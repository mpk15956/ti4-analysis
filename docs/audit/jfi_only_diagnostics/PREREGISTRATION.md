# Pre-registration: JFI-only basin-structure validation

**Written:** 2026-05-10, before the HC/TS/RS basin-validation run is submitted.
**Purpose:** lock the diagnostic criteria for "bimodality is a fitness-landscape
property of JFI" vs. "bimodality is SA-specific" *before* seeing the data,
so the verdict is read off rules instead of fitted to results.

## Background — what we know already (SA, n=300)

From the canonical SA run on `jfi_only` (final budget = 500k evals,
100 seeds × 3 chains, see [`jfi_only_diagnostics.png`](jfi_only_diagnostics.png)):

- **|Moran's I| is bimodal** under SA. Sarle's BC = **0.552** (at the
  conventional 0.555 threshold); GMM ΔBIC for k=2 over k=1 is **48.8**
  (very strong by Kass-Raftery). Two components: 70% mass at |I| ≈ 0.107,
  30% at |I| ≈ 0.315.
- The bimodality is **chain-driven, not seed-driven**. The per-seed
  distribution of "chains in upper |I| mode" is {0:35, 1:45, 2:20, 3:0},
  matching a chain-level Bernoulli, *not* a seed-level cluster (which
  would concentrate at 0 and 3). This says: under SA, the bimodality
  arises from convergence-basin selection during the search, not from
  the problem instance.
- **LSAP is unimodal-with-right-skew** (BC = 0.411, ΔBIC = 8.0; symmetric
  components heavily overlap). Mean 4.71, median 4.52, IQR [3.0, 6.0].
- LSAP and |I| are **moderately negatively coupled** within JFI-only:
  Spearman ρ = −0.425 (p = 1.4e-14). The two failure modes anti-correlate
  through the basin landscape.

The chain-level driver is consistent with two readings, which this run
will discriminate:

- **Reading A — fitness-landscape property:** JFI's level sets in
  tile-permutation space contain multiple distinct configuration regions
  (basins) with anti-correlated spatial profiles. Any local-improvement
  search will fall into one or the other depending on its starting
  point. *Predicts:* HC and TS (different local-improvement dynamics)
  also exhibit bimodal |I|; RS (random sampling, no dynamics) reflects
  the underlying geometry of the |I| distribution over the JFI level set.
- **Reading B — search-dynamics artifact:** the bimodality is specific
  to SA's temperature-driven trajectory (e.g., low-temperature trapping
  in particular regions) and does not reflect intrinsic landscape
  geometry. *Predicts:* HC, TS, or RS produce a unimodal |I| distribution
  even with chain-level variation in starting point.

## Pre-registered decision rules

The new run produces final-budget |I| samples for HC, TS, and RS at
matched seeds (n=300 per algorithm). Apply the same diagnostics used
on the SA data:

### Per-algorithm bimodality verdict (|Moran's I|)

For each of HC, TS, RS independently:

| Sarle's BC | GMM ΔBIC (k=2 over k=1) | Verdict |
|---|---|---|
| BC ≥ 0.555 | ΔBIC > 10 | **Bimodal (robust)** |
| BC ≥ 0.555 | 2 ≤ ΔBIC ≤ 10 | **Bimodal (soft)** |
| BC < 0.555 | ΔBIC > 10 | **Bimodal (soft)** — moments and GMM disagree, treat as soft |
| BC < 0.555 | 2 ≤ ΔBIC ≤ 10 | **Inconclusive** |
| BC < 0.555 | ΔBIC < 2 | **Unimodal** |

The Kass-Raftery thresholds (1995): ΔBIC < 2 = no support, 2-6 = positive,
6-10 = strong, >10 = very strong. We collapse 6-10 and >10 into "robust"
because n=300 is large enough that strong effects produce ΔBIC > 10
trivially; the meaningful break is between "noticeable" and "trivial."

### Joint verdict (the load-bearing claim)

The §3 framing depends on the *joint* verdict across HC, TS, and RS:

| HC | TS | RS | Verdict for §3 |
|---|---|---|---|
| Bimodal (any) | Bimodal (any) | Bimodal (any) | **Reading A — fitness-landscape property.** Strongest framing: "JFI's solution geometry contains anti-correlated basins; any search method, including random sampling, exhibits the bimodality." |
| Bimodal (any) | Bimodal (any) | Unimodal | **Reading A′ — local-search property of JFI.** "Any local-improvement search on JFI exhibits the basin structure; uniform sampling does not, because the basins are over-represented near the JFI optimum and RS does not concentrate there." Still publishable as a fitness-landscape claim *conditional on local-improvement dynamics*. |
| Bimodal in only one of HC/TS | (either) | (either) | **Reading B′ — algorithm-specific.** Soften §3 to "in our SA experiments, the optimizer converges into two basins" with explicit scope-limit. The cross-algorithm framing fails. |
| HC unimodal AND TS unimodal | — | — | **Reading B — SA-specific.** The bimodality is a property of SA's trajectory, not of JFI's landscape. §3 must drop the basin framing entirely; the load-bearing claim becomes the magnitude gaps (1500× LSAP, 4.6× LSAP reduction under jfi_moran). |

The "bimodal in any" cells aggregate "robust" and "soft" verdicts.
A "soft" bimodality verdict in a single algorithm should be reported
as such in the manuscript; we don't want to launder soft evidence
into a confident claim, but a soft + robust pair across HC and TS
is sufficient for Reading A′ (since the worry being addressed is
"is this a property of search dynamics on JFI" — two different
local-improvement dynamics independently confirming is the
relevant evidence).

### What about LSAP?

LSAP was unimodal-with-right-skew under SA. The pre-registered
prediction is that LSAP remains unimodal under HC, TS, and RS
(the |I| bimodality is the load-bearing finding; LSAP shape is
descriptive). If LSAP becomes bimodal under HC or TS, that's a
new finding worth reporting but it doesn't change the §3 framing
of the basin claim.

### What about the negative LSAP/|I| coupling?

Pre-registered: under each algorithm independently, compute Spearman
ρ between LSAP and |I| at final budget. Report all values.

- If ρ is consistently negative (≤ −0.2) across HC, TS, and (where
  applicable) RS, the "anti-correlated basins" framing is robust.
- If sign or magnitude varies substantially across algorithms,
  the coupling is search-dependent and we report it as such.

This is descriptive, not a verdict criterion — the joint verdict
above governs the §3 framing.

## What the new run does NOT establish

This run validates *whether the bimodality replicates* across optimizers.
It does **not** validate:

- Whether the basins correspond to qualitatively different *map structures*
  in any sense beyond their summary spatial statistics. (This would
  require visualizing tile placements and analyzing them by some
  structural typology — separate work.)
- Whether the basin behavior holds at other player counts, board
  topologies, or tile inventories. (The whole experiment is on the
  6-player canonical layout with |G|=31.)
- Whether the basin behavior depends on the corrected-landscape
  formulation. (We run with `--corrected-landscape` to match the
  canonical SA results.)

These are scope limits to disclose explicitly when reporting the
finding.

## Configuration that will be used

- Algorithms: `hc,ts,rs`
- Condition: `jfi_only` only (no other conditions)
- Seeds: `100`, base seed `0` — same as canonical SA run, so HC/TS/RS
  see the same 100 problem instances SA saw.
- Chains: `3` (matches canonical SA).
- Budgets: `1000,5000,10000,25000,50000,100000,200000,500000` (matches
  canonical).
- TS hyperparameters: `ts_k` from this run's own Optuna grid search
  (Phase 0). HC and RS are parameter-free.
- `--corrected-landscape`, `use_local_variance_lisa=true` (matches
  canonical).
- Workers: 16.

## Outputs

The run will produce:

- `output/basin_validation_*/benchmark_*/results.csv` — the new data.
- `output/basin_validation_*/basin_diagnostic.txt` — the formal verdict
  per the rules above. Run by `scripts/basin_diagnostic.py` against
  the new results.csv at run completion.
- `output/basin_validation_*/basin_diagnostic.png` — figures for
  HC, TS, RS analogous to the SA figure.

The verdict file is committed to the repo as evidence of what was
decided *before* drafting §3.

---

*Maintainer note: do not edit this file post-run. If subsequent analyses
suggest the rules above are wrong, write a new file documenting the
revision and the reason; the original pre-registration stays intact.*

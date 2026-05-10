"""
Manuscript-values validator. Asserts that every numerical value cited in the
manuscript matches the artifact it came from, AND appears in the manuscript
text where the manifest says it should.

CONTEXT (May 2026 manuscript rewrite). The manifest at
`tests/manuscript_values.yaml` enumerates the contract between
artifact-derived values and the manuscript prose. Without an active check at
the boundary, a CSV update silently drifts from the manuscript (per
`feedback_storage_passive_verification_active.md` — storage is passive,
verification is active).

FOUR PASSES:
    1. Schema + anchor resolution: validate manifest entry shape; for Tier 2
       entries, confirm section_anchor resolves to exactly one section header
       in the target file. Catches author error before any I/O on manuscript
       text or artifacts.
    2. Collision detection: build a (file, anchor, rendered_value) disambig
       index and flag any key that resolves to >1 manifest entry. Forces
       escalation Tier 1 → Tier 2 (or Tier 2 → Tier 3) at manifest-load time.
    3. Existence check: for each entry, count occurrences of rendered_value
       in the appropriate scope (whole file / section window / regex match).
       Fail if count < min_count. Token matching uses a custom-boundary regex
       that rejects [\\d.] on either side, so "0.05" does not match "0.057".
    4. Artifact provenance: each entry names a verifier function that reads
       the source artifact and confirms the raw_value the manifest claims
       matches what's actually in the artifact. Catches "the CSV updated but
       the manifest didn't."

HIERARCHY-AWARE SECTION INDEX. `build_section_index` parses markdown headers
and computes each section's end_line as the first subsequent header at level
≤ the current header's level. So:
    Anchor `3.4`   (level 2) → window includes all of 3.4.1–3.4.4 (until 3.5)
    Anchor `3.4.2` (level 3) → window is just §3.4.2's body (until next ###/##/#)

FAILURE ACCUMULATION. All four passes accumulate failures into one list and
report all-at-once. Fixing one entry and re-running surfaces the next failure
without bailing — needed for batch authoring.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = Path(__file__).parent / "manuscript_values.yaml"


# ── Verifier registry ─────────────────────────────────────────────────────
#
# Each verifier reads one source artifact and asserts the manifest's raw_value
# matches what's there, within tolerance. New manifest entries register a new
# verifier here. The signature is (raw_value) -> (ok: bool, detail: str).

VERIFIERS: dict[str, Callable[[Any], tuple[bool, str]]] = {}


def verifier(name: str):
    """Decorator: register a verifier function under `name`."""
    def deco(fn: Callable[[Any], tuple[bool, str]]):
        VERIFIERS[name] = fn
        return fn
    return deco


@verifier("g_size_from_canonical_topology")
def _verify_g_size(raw_value: int) -> tuple[bool, str]:
    """|G| = 31 on the canonical 6p layout, recovered from MapTopology."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))
    from tests.test_metric_parity import _canonical_6p_state  # type: ignore[import-not-found]
    topo, _ = _canonical_6p_state()
    actual = int(topo.n_spatial)
    return (actual == raw_value,
            f"MapTopology.n_spatial = {actual}; manifest raw_value = {raw_value}")


@verifier("gen0_moran_share_under_1_1_1")
def _verify_gen0_share(raw_value: float | int) -> tuple[bool, str]:
    """Gen-0 weighted-variance share of the Moran's I hinge term under 1:1:1
    weights, computed from the post-fix variance_equalization CSV. Matches
    the canonical (May 6 post-audit-fix) run, not the legacy 5:5:3 92%."""
    csv_path = ROOT / "output/verify_pixi_apptainer_20260506_205045/variance_equalization_20260506_224438.csv"
    if not csv_path.exists():
        return False, f"Artifact missing: {csv_path}"
    import pandas as pd
    import numpy as np
    df = pd.read_csv(csv_path)
    gen0 = df[df["phase"] == "gen0"]
    sigmas = gen0[["hinge_morans_i", "jfi_gap", "lisa_norm"]].std(ddof=1).values
    weights = np.array([1/3, 1/3, 1/3])
    weighted = sigmas * weights
    hinge_share_pct = (weighted[0] / weighted.sum()) * 100
    # raw_value is 90 (an integer percent rounded from the actual share).
    # Tolerance ±1 percentage point absorbs any small re-computation drift
    # while still catching a meaningful regime shift.
    rounded = round(hinge_share_pct)
    return (abs(rounded - raw_value) <= 1,
            f"variance_equalization_*.csv Gen-0 hinge share = {hinge_share_pct:.1f}% "
            f"(rounds to {rounded}); manifest raw_value = {raw_value}")


def _convergence_share_under_1_1_1(component_idx: int, raw_value: int | float) -> tuple[bool, str]:
    """Shared backend for convergence-phase variance shares. component_idx
    selects the component: 0=hinge_morans_i, 1=jfi_gap, 2=lisa_norm. Returns
    the share of total weighted variance under 1:1:1 weights at the
    convergence sample, matching the §3.8 reported numbers."""
    csv_path = ROOT / "output/verify_pixi_apptainer_20260506_205045/variance_equalization_20260506_224438.csv"
    if not csv_path.exists():
        return False, f"Artifact missing: {csv_path}"
    import pandas as pd
    import numpy as np
    df = pd.read_csv(csv_path)
    conv = df[df["phase"] == "convergence"]
    sigmas = conv[["hinge_morans_i", "jfi_gap", "lisa_norm"]].std(ddof=1).values
    weights = np.array([1/3, 1/3, 1/3])
    weighted = sigmas * weights
    share_pct = (weighted[component_idx] / weighted.sum()) * 100
    rounded = round(share_pct)
    return (abs(rounded - raw_value) <= 1,
            f"variance_equalization_*.csv convergence component[{component_idx}] share = "
            f"{share_pct:.1f}% (rounds to {rounded}); manifest raw_value = {raw_value}")


@verifier("convergence_lsap_share_under_1_1_1")
def _verify_convergence_lsap(raw_value: int | float) -> tuple[bool, str]:
    """LSAP share of weighted variance at convergence under 1:1:1 (component_idx=2)."""
    return _convergence_share_under_1_1_1(2, raw_value)


@verifier("convergence_jfi_share_under_1_1_1")
def _verify_convergence_jfi(raw_value: int | float) -> tuple[bool, str]:
    """JFI gap share of weighted variance at convergence under 1:1:1 (component_idx=1)."""
    return _convergence_share_under_1_1_1(1, raw_value)


@verifier("distance_weight_tau_from_sensitivity_report")
def _verify_distance_tau(raw_value: float) -> tuple[bool, str]:
    """LEGACY Mean Kendall's τ across distance-weight perturbations from the
    pre-canonical Phase 5 sensitivity report. Tolerance 1e-3 (the report rounds
    to 3 decimal places). Retained as backstop for legacy-cite contexts."""
    path = ROOT / "output/saturation_20260314_205919/dist_sensitivity_20260316_100535/sensitivity_report.txt"
    if not path.exists():
        return False, f"Artifact missing: {path}"
    text = path.read_text()
    m = re.search(r"Mean Kendall's tau across config pairs:\s*([\d.]+)", text)
    if not m:
        return False, f"Pattern 'Mean Kendall's tau across config pairs: <val>' not found in {path}"
    actual = float(m.group(1))
    return (abs(actual - raw_value) <= 1e-3,
            f"sensitivity_report.txt mean τ = {actual}; manifest raw_value = {raw_value}")


@verifier("distance_weight_tau_canonical_from_csv")
def _verify_distance_tau_canonical(raw_value: float) -> tuple[bool, str]:
    """Canonical Mean Kendall's τ across distance-weight perturbations,
    computed directly from sensitivity_results.csv. The canonical Phase 5 did
    not write a sensitivity_report.txt, so we re-derive: for each pair of
    weight configs, rank seeds by composite_score within each, compute
    Kendall τ on the rankings; report mean τ across all C(N_configs, 2) pairs.
    Tolerance 1e-3."""
    path = ROOT / "output/paper1_canonical_20260509_134024/dist_sensitivity_20260509_204651/sensitivity_results.csv"
    if not path.exists():
        return False, f"Artifact missing: {path}"
    import pandas as pd
    import itertools
    from scipy.stats import kendalltau as _kt
    df = pd.read_csv(path)
    configs = sorted(df["weight_config"].unique())
    taus = []
    for c1, c2 in itertools.combinations(configs, 2):
        s1 = df[df["weight_config"] == c1].set_index("seed")["composite_score"]
        s2 = df[df["weight_config"] == c2].set_index("seed")["composite_score"]
        common = s1.index.intersection(s2.index)
        if len(common) < 3:
            continue
        tau, _ = _kt(s1.loc[common], s2.loc[common])
        taus.append(tau)
    if not taus:
        return False, f"No pairwise comparisons computable from {path}"
    actual = sum(taus) / len(taus)
    return (abs(actual - raw_value) <= 1e-3,
            f"canonical sensitivity_results.csv mean τ across {len(taus)} pairwise = {actual:.6f}; "
            f"manifest raw_value = {raw_value}")


@verifier("goodhart_rho_alpha05_from_summary_json")
def _verify_goodhart_rho(raw_value: float) -> tuple[bool, str]:
    """Goodhart Spearman ρ vs α=0.05 cluster count, from the legacy LISA
    proxy validation summary. Tolerance 1e-3 (manifest rounds to 3 decimals
    via {:+.3f}, artifact has full precision)."""
    path = ROOT / "output/saturation_20260314_205919/lisa_validation_20260316_100413/proxy_validation_summary.json"
    if not path.exists():
        return False, f"Artifact missing: {path}"
    data = json.loads(path.read_text())
    actual = data.get("spearman_rho")
    if actual is None:
        return False, f"Key 'spearman_rho' missing from {path}"
    return (abs(actual - raw_value) <= 1e-3,
            f"proxy_validation_summary.json spearman_rho = {actual}; manifest raw_value = {raw_value}")


@verifier("goodhart_rho_fdr_from_diagnostic")
def _verify_goodhart_rho_fdr(raw_value: float) -> tuple[bool, str]:
    """Goodhart Spearman ρ under FDR-corrected target, from the LEGACY per-map
    diagnostic (n=120, 4 algorithms × 30 seeds). Tolerance 1e-3."""
    path = ROOT / "output/saturation_20260314_205919/lisa_validation_20260316_100413/lisa_proxy_per_map_diagnostic.json"
    if not path.exists():
        return False, f"Artifact missing: {path}"
    data = json.loads(path.read_text())
    try:
        actual = data["tests"]["per_map_rank_total_significant_fdr"]["spearman_rho"]
    except KeyError as e:
        return False, f"Path tests.per_map_rank_total_significant_fdr.spearman_rho missing from {path}: {e}"
    return (abs(actual - raw_value) <= 1e-3,
            f"legacy lisa_proxy_per_map_diagnostic.json FDR ρ = {actual}; manifest raw_value = {raw_value}")


@verifier("goodhart_rho_alpha05_canonical_from_summary_json")
def _verify_goodhart_rho_alpha05_canonical(raw_value: float) -> tuple[bool, str]:
    """Canonical Goodhart Spearman ρ vs α=0.05 cluster count, from canonical
    Phase 4 (SA only, n=30 maps). Tolerance 1e-3."""
    path = ROOT / "output/paper1_canonical_20260509_134024/lisa_validation_20260509_204628/proxy_validation_summary.json"
    if not path.exists():
        return False, f"Artifact missing: {path}"
    data = json.loads(path.read_text())
    actual = data.get("spearman_rho")
    if actual is None:
        return False, f"Key 'spearman_rho' missing from {path}"
    return (abs(actual - raw_value) <= 1e-3,
            f"canonical proxy_validation_summary.json spearman_rho = {actual}; manifest raw_value = {raw_value}")


@verifier("goodhart_rho_fdr_canonical_from_diagnostic")
def _verify_goodhart_rho_fdr_canonical(raw_value: float) -> tuple[bool, str]:
    """Canonical Goodhart Spearman ρ under FDR-corrected target, from canonical
    Phase 4 per-map diagnostic (SA only, n=30). Tolerance 1e-3."""
    path = ROOT / "output/paper1_canonical_20260509_134024/lisa_validation_20260509_204628/lisa_proxy_per_map_diagnostic.json"
    if not path.exists():
        return False, f"Artifact missing: {path}"
    data = json.loads(path.read_text())
    try:
        actual = data["tests"]["per_map_rank_total_significant_fdr"]["spearman_rho"]
    except KeyError as e:
        return False, f"Path tests.per_map_rank_total_significant_fdr.spearman_rho missing from {path}: {e}"
    return (abs(actual - raw_value) <= 1e-3,
            f"canonical lisa_proxy_per_map_diagnostic.json FDR ρ = {actual}; manifest raw_value = {raw_value}")


@verifier("lsap_threshold_tau_from_summary_json")
def _verify_lsap_threshold_tau(raw_value: float) -> tuple[bool, str]:
    """Legacy threshold-sensitivity Kendall τ on the raw I_i form (60-seed × 4-algo
    set). Source: scripts/lsap_threshold_sensitivity.py summary JSON. Tolerance 1e-3."""
    path = ROOT / "output/saturation_20260314_205919/lsap_threshold_20260316_194325/lsap_threshold_summary.json"
    if not path.exists():
        return False, f"Artifact missing: {path}"
    data = json.loads(path.read_text())
    actual = data.get("kendalls_tau")
    if actual is None:
        return False, f"Key 'kendalls_tau' missing from {path}"
    return (abs(actual - raw_value) <= 1e-3,
            f"lsap_threshold_summary.json kendalls_tau = {actual}; manifest raw_value = {raw_value}")


@verifier("lsap_threshold_tau_canonical_from_summary_json")
def _verify_lsap_threshold_tau_canonical(raw_value: float) -> tuple[bool, str]:
    """Canonical same-form threshold-sensitivity Kendall τ:
    `lisa_penalty_swappable(use_local_variance=True)` baseline vs
    `lisa_penalty_swappable_thresholded(τ=0.05, use_local_variance=True)`
    thresholded comparator (50 SA seeds, budget 1000, --corrected-landscape).
    Source: lsap_threshold_summary.json from the May 2026 canonical run.
    Tolerance 1e-3."""
    path = ROOT / "output/paper1_canonical_20260509_134024/lsap_threshold_20260509_231008/lsap_threshold_summary.json"
    if not path.exists():
        return False, f"Artifact missing: {path}"
    data = json.loads(path.read_text())
    actual = data.get("kendalls_tau")
    if actual is None:
        return False, f"Key 'kendalls_tau' missing from {path}"
    return (abs(actual - raw_value) <= 1e-3,
            f"canonical lsap_threshold_summary.json kendalls_tau = {actual}; manifest raw_value = {raw_value}")


# ── Phase 0 canonical hyperparameter-tuning verifiers (May 9 2026) ──
# Shared backend reads best_params.json once; the four wrapper verifiers
# select the specific field. Tolerance 1e-5 (best_params.json values are
# rounded to 6 decimal places; the manifest renders to 4 decimal places, so
# the verifier compares raw artifact value to manifest raw_value at 1e-5).

PHASE0_BEST_PARAMS = ROOT / "output/paper1_canonical_20260509_134024/optuna_20260509_134028/best_params.json"


def _phase0_field(field: str, raw_value: float) -> tuple[bool, str]:
    if not PHASE0_BEST_PARAMS.exists():
        return False, f"Phase 0 artifact missing: {PHASE0_BEST_PARAMS} (canonical run not yet on disk)"
    data = json.loads(PHASE0_BEST_PARAMS.read_text())
    actual = data.get(field)
    if actual is None:
        return False, f"Key {field!r} missing from {PHASE0_BEST_PARAMS}"
    return (abs(actual - raw_value) <= 1e-5,
            f"best_params.json {field} = {actual}; manifest raw_value = {raw_value}")


@verifier("phase0_sa_cv_mean_from_best_params")
def _verify_phase0_cv_mean(raw_value: float) -> tuple[bool, str]:
    return _phase0_field("cv_mean", raw_value)


@verifier("phase0_sa_cv_std_from_best_params")
def _verify_phase0_cv_std(raw_value: float) -> tuple[bool, str]:
    return _phase0_field("cv_std", raw_value)


@verifier("phase0_sa_held_out_mean_from_best_params")
def _verify_phase0_held_out_mean(raw_value: float) -> tuple[bool, str]:
    return _phase0_field("held_out_mean", raw_value)


@verifier("phase0_sa_held_out_std_from_best_params")
def _verify_phase0_held_out_std(raw_value: float) -> tuple[bool, str]:
    return _phase0_field("held_out_std", raw_value)


@verifier("phase0_sa_cv_coefficient_of_variation_pct")
def _verify_phase0_cv_cov_pct(raw_value: float) -> tuple[bool, str]:
    """Coefficient of variation (cv_std / cv_mean) for Phase 0 SA tuning,
    rounded to integer percent. Cited in §3.7(B) as ~49% characterizing
    rugged-landscape variance. Tolerance 1pp."""
    if not PHASE0_BEST_PARAMS.exists():
        return False, f"Phase 0 artifact missing: {PHASE0_BEST_PARAMS}"
    data = json.loads(PHASE0_BEST_PARAMS.read_text())
    cv_mean = data.get("cv_mean")
    cv_std = data.get("cv_std")
    if cv_mean is None or cv_std is None or cv_mean == 0:
        return False, f"cv_mean/cv_std missing or zero in {PHASE0_BEST_PARAMS}"
    actual_pct = round(100.0 * cv_std / cv_mean)
    return (abs(actual_pct - raw_value) <= 1.0,
            f"CV(cv_std/cv_mean) = {actual_pct}%; manifest raw_value = {raw_value}%")


# ── Phase 1 canonical results.csv verifiers (5-condition ablation, May 9 2026) ──
# Shared backend reads results.csv once per call, filters to budget=500000,
# and computes the median of the requested metric within the requested
# condition. Tolerance 1e-3 (manuscript renders to 4 decimal places via
# {:+.4f} or {:.4f}; the underlying CSV stores 4-decimal-rounded values
# already, so the precision contract is "the median matches what's in the
# CSV at the cited budget cell").

PHASE1_RESULTS = ROOT / "output/paper1_canonical_20260509_134024/benchmark_20260509_191848/results.csv"


def _phase1_condition_metric_median(condition: str, metric: str, raw_value: float,
                                     budget: int = 500000) -> tuple[bool, str]:
    """Chain-aggregated per-seed median: aggregate the 3 chains per (seed,
    condition, budget) cell to one observation, then take median across 100
    seeds. This matches what scripts/analyze_phase1_conditions.py reports
    and what the §3.9 prose cites — pairwise Wilcoxon needs one observation
    per seed, so chain aggregation is methodologically required."""
    if not PHASE1_RESULTS.exists():
        return False, f"Phase 1 artifact missing: {PHASE1_RESULTS}"
    import pandas as pd
    df = pd.read_csv(PHASE1_RESULTS)
    sub = df[(df["condition"] == condition) & (df["budget"] == budget)]
    if len(sub) == 0:
        return False, f"No rows for condition={condition} metric={metric} budget={budget} in {PHASE1_RESULTS}"
    # Chain aggregation: MEAN across chains, matching what
    # analyze_phase1_conditions.py does at line 99 (groupby [...].mean()).
    # Then take median across seeds (the descriptive median the analyzer reports).
    per_seed = sub.groupby("seed")[metric].mean()
    actual = float(per_seed.median())
    return (abs(actual - raw_value) <= 1e-3,
            f"results.csv ({condition}, {metric}, b={budget}) "
            f"chain-mean / seed-median = {actual:.6f}; "
            f"manifest raw_value = {raw_value:.6f} (n_seeds={len(per_seed)})")


@verifier("phase1_morans_i_lt_neg1_count")
def _v_phase1_boundary_count(raw_value: int) -> tuple[bool, str]:
    """Count of canonical Phase 1 rows with morans_i < -1.0 (boundary
    violations on row-standardized W; documented in limitations.md and
    cited as a count in §3.3). Exact equality required."""
    if not PHASE1_RESULTS.exists():
        return False, f"Phase 1 artifact missing: {PHASE1_RESULTS}"
    import pandas as pd
    df = pd.read_csv(PHASE1_RESULTS)
    actual = int((df["morans_i"] < -1.0).sum())
    return (actual == int(raw_value),
            f"results.csv #(morans_i < -1.0) = {actual}; manifest raw_value = {raw_value}")


@verifier("phase1_c0_morans_i_b500k")
def _v_c0_morans(rv): return _phase1_condition_metric_median("jfi_only", "morans_i", rv)


@verifier("phase1_c4_morans_i_b500k")
def _v_c4_morans(rv): return _phase1_condition_metric_median("full_composite", "morans_i", rv)


@verifier("phase1_c3_morans_i_b500k")
def _v_c3_morans(rv): return _phase1_condition_metric_median("jfi_moran", "morans_i", rv)


@verifier("phase1_c0_lisa_penalty_b500k")
def _v_c0_lisa(rv): return _phase1_condition_metric_median("jfi_only", "lisa_penalty", rv)


@verifier("phase1_c3_lisa_penalty_b500k")
def _v_c3_lisa(rv): return _phase1_condition_metric_median("jfi_moran", "lisa_penalty", rv)


@verifier("phase1_c4_lisa_penalty_b500k")
def _v_c4_lisa(rv): return _phase1_condition_metric_median("full_composite", "lisa_penalty", rv)


# ── Phase 1 ablation §3.9 stats panel: Friedman χ² + C0→C4 paired deltas ──
# Re-computed from raw results.csv on each verify pass — same chain-mean /
# seed-aggregation as analyze_phase1_conditions.py — so the manuscript values
# are pinned to the artifact, not to the derived report.txt.

CONDITION_ORDER = ["jfi_only", "moran_only", "lsap_only", "jfi_moran", "full_composite"]


def _phase1_per_seed_panel(metric: str, budget: int = 500000):
    """Build the (n_seeds × 5) per-seed × condition matrix used by Friedman /
    Wilcoxon in analyze_phase1_conditions.py. Aggregates 3 chains per
    (seed, condition, budget) cell to one observation via mean."""
    import pandas as pd
    df = pd.read_csv(PHASE1_RESULTS)
    sub = df[df["budget"] == budget]
    per_seed_cond = (
        sub.groupby(["seed", "condition"])[metric].mean().unstack("condition")
    )
    per_seed_cond = per_seed_cond[CONDITION_ORDER].dropna()
    return per_seed_cond


def _phase1_friedman_chi2(metric: str, raw_value: float) -> tuple[bool, str]:
    if not PHASE1_RESULTS.exists():
        return False, f"Phase 1 artifact missing: {PHASE1_RESULTS}"
    from scipy import stats
    panel = _phase1_per_seed_panel(metric)
    chi2, _ = stats.friedmanchisquare(*[panel[c].values for c in CONDITION_ORDER])
    return (abs(chi2 - raw_value) <= 1e-2,
            f"Friedman χ²({metric}, df=4) = {chi2:.4f}; manifest raw_value = {raw_value}")


@verifier("phase1_friedman_chi2_morans_i")
def _v_phase1_friedman_morans(rv): return _phase1_friedman_chi2("morans_i", rv)


@verifier("phase1_friedman_chi2_lisa_penalty")
def _v_phase1_friedman_lisa(rv): return _phase1_friedman_chi2("lisa_penalty", rv)


@verifier("phase1_friedman_chi2_jains_index")
def _v_phase1_friedman_jfi(rv): return _phase1_friedman_chi2("jains_index", rv)


def _phase1_c0_vs_c4_median_diff(metric: str, raw_value: float) -> tuple[bool, str]:
    """C0→C4 difference of medians: median(C0) − median(C4) at b=500k. This is
    the form analyze_phase1_conditions.py reports (and §3.9 cites); not the
    median-of-paired-differences (which differs because median is not a
    linear functional). Sign matches §3.9 (positive = C4 has lower
    morans_i / lisa_penalty than C0)."""
    if not PHASE1_RESULTS.exists():
        return False, f"Phase 1 artifact missing: {PHASE1_RESULTS}"
    import numpy as np
    panel = _phase1_per_seed_panel(metric)
    actual = float(np.median(panel["jfi_only"].values) - np.median(panel["full_composite"].values))
    return (abs(actual - raw_value) <= 1e-3,
            f"C0-C4 median difference ({metric}) = {actual:.6f}; manifest raw_value = {raw_value}")


@verifier("phase1_c0_vs_c4_morans_i_median_diff")
def _v_c0c4_morans_diff(rv): return _phase1_c0_vs_c4_median_diff("morans_i", rv)


@verifier("phase1_c0_vs_c4_lisa_penalty_median_diff")
def _v_c0c4_lisa_diff(rv): return _phase1_c0_vs_c4_median_diff("lisa_penalty", rv)


def _phase1_c0_vs_c4_cohens_dz(metric: str, raw_value: float) -> tuple[bool, str]:
    """Cohen's d_z = mean(diff) / std(diff, ddof=1) on the per-seed paired
    panel C0 (jfi_only) − C4 (full_composite). Same definition as
    analyze_phase1_conditions.py::cohens_dz."""
    if not PHASE1_RESULTS.exists():
        return False, f"Phase 1 artifact missing: {PHASE1_RESULTS}"
    import numpy as np
    panel = _phase1_per_seed_panel(metric)
    diffs = panel["jfi_only"].values - panel["full_composite"].values
    sd = float(np.std(diffs, ddof=1))
    if sd == 0:
        return False, "std(diff) = 0; d_z undefined"
    actual = float(np.mean(diffs) / sd)
    return (abs(actual - raw_value) <= 1e-2,
            f"Cohen's d_z (C0-C4, {metric}) = {actual:.4f}; manifest raw_value = {raw_value}")


@verifier("phase1_c0_vs_c4_morans_i_cohens_dz")
def _v_c0c4_morans_dz(rv): return _phase1_c0_vs_c4_cohens_dz("morans_i", rv)


@verifier("phase1_c0_vs_c4_lisa_penalty_cohens_dz")
def _v_c0c4_lisa_dz(rv): return _phase1_c0_vs_c4_cohens_dz("lisa_penalty", rv)


def _phase1_rq3_spearman_c4_b500k(metric: str, raw_value: float) -> tuple[bool, str]:
    """RQ3: Spearman ρ between balance_gap and `metric` within canonical
    full-composite C4 at b=500k, chain-mean per-seed (n=100). Tolerance 1e-3."""
    if not PHASE1_RESULTS.exists():
        return False, f"Phase 1 artifact missing: {PHASE1_RESULTS}"
    import pandas as pd
    from scipy import stats
    df = pd.read_csv(PHASE1_RESULTS)
    sub = df[(df["condition"] == "full_composite") & (df["budget"] == 500000)]
    per_seed = sub.groupby("seed")[["balance_gap", metric]].mean()
    rho, _ = stats.spearmanr(per_seed["balance_gap"], per_seed[metric])
    return (abs(rho - raw_value) <= 1e-3,
            f"RQ3 ρ(balance_gap, {metric}) = {rho:.4f}; manifest raw_value = {raw_value}")


@verifier("phase1_rq3_spearman_c4_b500k_morans_i")
def _v_rq3_morans(rv): return _phase1_rq3_spearman_c4_b500k("morans_i", rv)


@verifier("phase1_rq3_spearman_c4_b500k_lisa_penalty")
def _v_rq3_lisa(rv): return _phase1_rq3_spearman_c4_b500k("lisa_penalty", rv)


@verifier("phase1_rq3_spearman_c4_b500k_jains_index")
def _v_rq3_jfi(rv): return _phase1_rq3_spearman_c4_b500k("jains_index", rv)


def _phase1_c0_vs_cx_vda(metric: str, condition_x: str, raw_value: float) -> tuple[bool, str]:
    """Vargha-Delaney A12 between C0 (jfi_only) and Cx for metric. Sign and
    direction match analyze_phase1_conditions.py: A12 = P(C0 > Cx) + 0.5 *
    P(C0 == Cx) on per-seed paired observations (chain-mean aggregated).
    For metrics where C4 is expected lower than C0 (morans_i, lisa_penalty)
    or higher (jains_index reversed sense), A12 captures the rank concordance."""
    if not PHASE1_RESULTS.exists():
        return False, f"Phase 1 artifact missing: {PHASE1_RESULTS}"
    panel = _phase1_per_seed_panel(metric)
    a = panel["jfi_only"].values
    b = panel[condition_x].values
    n = len(a)
    # A12 = (#wins + 0.5 * #ties) / (n_a * n_b) over all paired comparisons
    import numpy as np
    A_grid = a[:, None] - b[None, :]
    wins = float((A_grid > 0).sum())
    ties = float((A_grid == 0).sum())
    a12 = (wins + 0.5 * ties) / (n * n)
    return (abs(a12 - raw_value) <= 1e-3,
            f"VDA A12(C0 vs {condition_x}, {metric}) = {a12:.4f}; manifest raw_value = {raw_value}")


@verifier("phase1_vda_c0_vs_c3_jains_index")
def _v_vda_c3_jfi(rv): return _phase1_c0_vs_cx_vda("jains_index", "jfi_moran", rv)


@verifier("phase1_vda_c0_vs_c4_jains_index")
def _v_vda_c4_jfi(rv): return _phase1_c0_vs_cx_vda("jains_index", "full_composite", rv)


def _phase1_jfi_parity_wilcoxon(condition_x: str, raw_value: float) -> tuple[bool, str]:
    """Wilcoxon signed-rank W statistic for one-sided JFI parity test:
    H0: jains_index(Cx) >= jains_index(C0). The chain-mean per-seed panel
    is the input; W is the smaller sum of signed ranks (scipy default)."""
    if not PHASE1_RESULTS.exists():
        return False, f"Phase 1 artifact missing: {PHASE1_RESULTS}"
    from scipy import stats
    panel = _phase1_per_seed_panel("jains_index")
    diffs = panel["jfi_only"].values - panel[condition_x].values
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return False, "All paired differences are zero"
    res = stats.wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox")
    return (abs(float(res.statistic) - raw_value) <= 1e-1,
            f"Wilcoxon W (C0 vs {condition_x}, jains_index) = {res.statistic}; "
            f"manifest raw_value = {raw_value}")


@verifier("phase1_jfi_parity_w_c3")
def _v_jfi_w_c3(rv): return _phase1_jfi_parity_wilcoxon("jfi_moran", rv)


@verifier("phase1_jfi_parity_w_c4")
def _v_jfi_w_c4(rv): return _phase1_jfi_parity_wilcoxon("full_composite", rv)


# ── Data structures ──────────────────────────────────────────────────────


@dataclass
class ManifestEntry:
    key: str
    raw_value: Any
    format: str
    files: list[str]
    match: str
    section_anchor: str | None = None
    pattern: str | None = None
    source_artifact: str = ""
    verifier: str | None = None
    min_count: int = 1

    @property
    def rendered_value(self) -> str:
        return self.format.format(self.raw_value)


@dataclass
class Section:
    level: int       # 2 for ##, 3 for ###, etc.
    number: str      # "3.4.2" — used as anchor lookup key
    title: str       # heading text after the number
    start_line: int  # line index of the header
    end_line: int    # exclusive; first subsequent header at level <= self.level


# ── Section index builder (hierarchy-aware) ──────────────────────────────


_HEADER_PAT = re.compile(r"^(#+)\s+(\d+(?:\.\d+)*)\s+(.*?)\s*$")


def build_section_index(text: str) -> dict[str, Section]:
    """Parse a markdown document into a {section_number: Section} index.

    Hierarchical scope rule: end_line for a header at level L is the line of
    the first subsequent header at level <= L. So a `## 3.4` anchor's window
    includes all `### 3.4.X` subsections; a `### 3.4.2` anchor's window is
    just that subsection's own body.
    """
    lines = text.splitlines()
    headers: list[tuple[int, str, str, int]] = []
    for i, line in enumerate(lines):
        m = _HEADER_PAT.match(line)
        if m:
            headers.append((len(m.group(1)), m.group(2), m.group(3), i))
    index: dict[str, Section] = {}
    for i, (level, number, title, start) in enumerate(headers):
        end = len(lines)
        for j in range(i + 1, len(headers)):
            if headers[j][0] <= level:
                end = headers[j][3]
                break
        index[number] = Section(level, number, title, start, end)
    return index


# ── Token matching with custom boundary ──────────────────────────────────


def count_token_matches(haystack: str, token: str) -> int:
    """Count occurrences of `token` in `haystack` with custom word boundaries.

    Rejects matches where the token is part of a longer numeric run:
        - Preceded by a digit or dot ([\\d.]):  `0.31` blocks matching `31`
          inside it; `1.05` blocks `0.05`.
        - Followed by a digit:                  `315` blocks matching `31`.
        - Followed by `.<digit>`:               `31.5` blocks matching `31`,
          but `31.` (sentence-ending) does NOT block — periods at end-of-token
          are commonly punctuation in prose, so the trailing boundary must
          distinguish "decimal continuation" from "sentence break."

    Same two-clause trailing boundary as the cross-ref test in
    tests/test_methodology_cross_refs.py (which originally had `(?![\\d.])` and
    falsely rejected `§3.7.` at sentence end). Same fix recipe applies here.
    """
    pat = re.compile(rf"(?<![\d.]){re.escape(token)}(?![0-9])(?!\.[0-9])")
    return len(pat.findall(haystack))


# ── Validator ────────────────────────────────────────────────────────────


class ManifestValidator:
    def __init__(self, manifest_path: Path, repo_root: Path):
        with open(manifest_path) as f:
            data = yaml.safe_load(f) or {}
        raw_entries = data.get("entries", [])
        self.entries: list[ManifestEntry] = [ManifestEntry(**e) for e in raw_entries]
        self.repo_root = repo_root
        self.failures: list[str] = []
        self._section_indices: dict[str, dict[str, Section]] = {}
        self._file_texts: dict[str, str] = {}

    def _file_text(self, rel: str) -> str:
        if rel not in self._file_texts:
            path = self.repo_root / rel
            if not path.exists():
                raise FileNotFoundError(f"Manifest file missing on disk: {rel}")
            self._file_texts[rel] = path.read_text()
        return self._file_texts[rel]

    def _section_index_for(self, rel: str) -> dict[str, Section]:
        if rel not in self._section_indices:
            self._section_indices[rel] = build_section_index(self._file_text(rel))
        return self._section_indices[rel]

    def _fail(self, msg: str) -> None:
        self.failures.append(msg)

    # Pass 1 — Schema + anchor resolution -------------------------------------
    def pass1_schema_and_anchor_resolution(self) -> None:
        seen_keys: set[str] = set()
        for entry in self.entries:
            if entry.key in seen_keys:
                self._fail(f"[{entry.key}] duplicate manifest key")
            seen_keys.add(entry.key)
            if entry.match not in ("token", "section", "regex"):
                self._fail(f"[{entry.key}] invalid match type: {entry.match!r}")
                continue
            if entry.match == "section" and not entry.section_anchor:
                self._fail(f"[{entry.key}] match=section requires section_anchor")
                continue
            if entry.match == "regex" and not entry.pattern:
                self._fail(f"[{entry.key}] match=regex requires pattern")
                continue
            for rel in entry.files:
                try:
                    text = self._file_text(rel)
                except FileNotFoundError as e:
                    self._fail(f"[{entry.key}] {e}")
                    continue
                if entry.match == "section":
                    idx = self._section_index_for(rel)
                    if entry.section_anchor not in idx:
                        available = sorted(idx.keys())[:12]
                        self._fail(
                            f"[{entry.key}] anchor {entry.section_anchor!r} resolves to 0 "
                            f"sections in {rel}; first available: {available}"
                        )

    # Pass 2 — Collision detection --------------------------------------------
    def pass2_collision_detection(self) -> None:
        disambig: dict[tuple[str, str | None, str], list[str]] = {}
        entries_by_key = {e.key: e for e in self.entries}
        for entry in self.entries:
            if entry.match == "regex":
                continue  # Tier 3 entries skip token-collision check
            try:
                rendered = entry.rendered_value
            except Exception as e:
                self._fail(f"[{entry.key}] format/raw_value error: {e}")
                continue
            for rel in entry.files:
                anchor = entry.section_anchor if entry.match == "section" else None
                key = (rel, anchor, rendered)
                disambig.setdefault(key, []).append(entry.key)
        for (rel, anchor, token), entry_keys in disambig.items():
            if len(entry_keys) <= 1:
                continue
            tiers = {entries_by_key[k].match for k in entry_keys}
            if "section" in tiers and len(tiers) == 1:
                advice = "All colliding entries are already at Tier 2; escalate one to Tier 3 with explicit regex."
            else:
                advice = "Tier 1 collision; escalate at least one entry to Tier 2 with section_anchor."
            self._fail(
                f"COLLISION at ({rel}, anchor={anchor!r}, token={token!r}): "
                f"entries {sorted(entry_keys)}. {advice}"
            )

    # Pass 3 — Existence check ------------------------------------------------
    def pass3_existence_check(self) -> None:
        for entry in self.entries:
            for rel in entry.files:
                try:
                    text = self._file_text(rel)
                except FileNotFoundError:
                    continue  # Already reported in Pass 1
                if entry.match == "token":
                    haystack = text
                    count = count_token_matches(haystack, entry.rendered_value)
                elif entry.match == "section":
                    idx = self._section_index_for(rel)
                    sec = idx.get(entry.section_anchor)
                    if sec is None:
                        continue  # Already reported in Pass 1
                    haystack = "\n".join(text.splitlines()[sec.start_line:sec.end_line])
                    count = count_token_matches(haystack, entry.rendered_value)
                else:  # regex
                    if entry.pattern is None:
                        continue
                    count = len(re.findall(entry.pattern, text))
                if count < entry.min_count:
                    rendered = entry.rendered_value if entry.match != "regex" else f"pattern={entry.pattern!r}"
                    src = entry.source_artifact or "(no source recorded)"
                    self._fail(
                        f"[{entry.key}] expected '{rendered}' to appear "
                        f">= {entry.min_count} time(s) in {rel} (match={entry.match}"
                        + (f", anchor={entry.section_anchor!r}" if entry.match == "section" else "")
                        + f"); found {count}. Source: {src}"
                    )

    # Pass 4 — Artifact provenance --------------------------------------------
    def pass4_artifact_provenance(self) -> None:
        for entry in self.entries:
            if not entry.verifier:
                continue
            verifier_fn = VERIFIERS.get(entry.verifier)
            if verifier_fn is None:
                self._fail(f"[{entry.key}] verifier {entry.verifier!r} not registered in VERIFIERS")
                continue
            try:
                ok, detail = verifier_fn(entry.raw_value)
            except Exception as e:
                self._fail(
                    f"[{entry.key}] verifier {entry.verifier!r} raised {type(e).__name__}: {e}"
                )
                continue
            if not ok:
                self._fail(f"[{entry.key}] artifact provenance failed: {detail}")

    def validate(self) -> list[str]:
        self.pass1_schema_and_anchor_resolution()
        # If schema is malformed, Pass 2/3/4 may explode on rendered_value.
        # Filter to schema-valid entries before proceeding.
        schema_failures = [f for f in self.failures
                           if "invalid match type" in f or "requires" in f]
        if not schema_failures:
            self.pass2_collision_detection()
            self.pass3_existence_check()
            self.pass4_artifact_provenance()
        return self.failures


# ── Pytest entry point ────────────────────────────────────────────────────


def test_manuscript_values():
    """All-passes-accumulate validator: report every failure, not just the first."""
    if not MANIFEST_PATH.exists():
        pytest.skip(f"Manifest not present: {MANIFEST_PATH}")
    validator = ManifestValidator(MANIFEST_PATH, ROOT)
    failures = validator.validate()
    if failures:
        msg = ["MANIFEST validation failed:", ""]
        for i, f in enumerate(failures, 1):
            msg.append(f"  {i}. {f}")
        msg.append("")
        msg.append(f"Total failures: {len(failures)}")
        msg.append("Manifest: " + str(MANIFEST_PATH))
        msg.append("Fix the manifest (or the manuscript / artifact it points at) and re-run.")
        pytest.fail("\n".join(msg))


def test_section_index_hierarchy():
    """Pin the hierarchical scope rule end_line = first subsequent header at level <= L.

    Constructs a synthetic markdown with nested sections and asserts:
      - Parent anchor `3.4` window includes all child subsections
      - Child anchor `3.4.2` window is just that subsection
      - No bleed across parent boundaries (3.4's window does not include 3.5)
    """
    md = "\n".join([
        "## 3.4 Parent",       # line 0
        "Parent preamble.",    # line 1
        "",                    # line 2
        "### 3.4.1 First",     # line 3
        "First body.",         # line 4
        "",                    # line 5
        "### 3.4.2 Second",    # line 6
        "Second body.",        # line 7
        "",                    # line 8
        "## 3.5 Sibling",      # line 9
        "Sibling body.",       # line 10
    ])
    idx = build_section_index(md)
    assert "3.4" in idx and "3.4.1" in idx and "3.4.2" in idx and "3.5" in idx

    # Parent window covers preamble + both subsections, ends at sibling.
    assert idx["3.4"].start_line == 0
    assert idx["3.4"].end_line == 9, f"parent should end at sibling header (line 9), got {idx['3.4'].end_line}"

    # Child window is bounded by the next H3.
    assert idx["3.4.1"].start_line == 3
    assert idx["3.4.1"].end_line == 6, f"3.4.1 should end at 3.4.2 (line 6), got {idx['3.4.1'].end_line}"

    assert idx["3.4.2"].start_line == 6
    assert idx["3.4.2"].end_line == 9, f"3.4.2 should end at sibling H2 (line 9), got {idx['3.4.2'].end_line}"

    # Sibling window is independent of parent.
    assert idx["3.5"].start_line == 9
    assert idx["3.5"].end_line == 11  # EOF


def test_count_token_matches_boundary():
    """Custom boundary regex must reject [\\d.] on either side."""
    assert count_token_matches("the value is 31 today", "31") == 1
    assert count_token_matches("0.31 percent", "31") == 0, "31 should not match inside 0.31"
    assert count_token_matches("ratio 31.5", "31") == 0, "31 should not match inside 31.5"
    assert count_token_matches("31, 31, and 31 again", "31") == 3
    # Whole numbers vs decimals
    assert count_token_matches("0.05 alpha and 0.057 elsewhere", "0.05") == 1
    assert count_token_matches("alpha 0.05.", "0.05") == 1, "trailing period should not block match"

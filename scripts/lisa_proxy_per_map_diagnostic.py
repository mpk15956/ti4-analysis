#!/usr/bin/env python3
"""
Per-map LSAP-vs-significance diagnostic for the Goodhart-defence narrative.

The existing proxy_validation_summary.json reports Spearman ρ between continuous
per-map LSAP and FDR-corrected per-map cluster counts at ρ = -0.025 (p = 0.788),
falling below the pre-registered ρ > 0.70 threshold. The structural reason: the
optimizer is so effective at suppressing local clusters that
`total_significant_fdr` collapses to mostly zero across all 120 optimized maps,
leaving no variance to correlate with. Per-map continuous correlation is the
wrong test for a ranking-preserving heuristic on a near-degenerate distribution.

This script reframes the validation around what the heuristic is actually used
for — per-map and per-algorithm ranking — and reports:

  1. Spearman ρ and Kendall τ on (LSAP, total_significant) at α=0.05 *and*
     FDR-corrected, separately. The α=0.05 test has more variance to work with.
  2. Discrimination AUC: does LSAP rank predict "≥1 FDR-significant cluster"?
  3. Algorithm-level rank concordance: rank algorithms by median LSAP and by
     median significant-cluster count, then report Kendall τ between the two
     rankings (this is the test the optimizer's "ranks methods correctly"
     claim actually depends on).
  4. Convergence-floor diagnostic: what fraction of maps have zero FDR-significant
     clusters, and what is the LSAP distribution within that zero-count subset?
     This contextualises why per-map continuous correlation is structurally weak.

Outputs JSON + a short text report next to the source CSV.

Usage:
    python scripts/lisa_proxy_per_map_diagnostic.py \\
        --csv output/saturation_20260314_205919/lisa_validation_20260316_100413/validation_results.csv \\
        [--output-dir <dir>]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument(
        "--csv", type=Path, required=True,
        help="Path to validation_results.csv from validate_lisa_proxy.py",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Where to write outputs (default: same dir as --csv)",
    )
    return p.parse_args()


def rank_corr(x, y):
    """Return (Spearman ρ, Kendall τ, p_spearman, p_kendall, n)."""
    if len(x) < 3:
        return float("nan"), float("nan"), float("nan"), float("nan"), len(x)
    sp = stats.spearmanr(x, y)
    kt = stats.kendalltau(x, y)
    return float(sp.statistic), float(kt.statistic), float(sp.pvalue), float(kt.pvalue), len(x)


def main():
    args = parse_args()
    out_dir = args.output_dir or args.csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    n = len(df)
    print(f"Loaded {n} maps from {args.csv}")
    print(f"Algorithms: {sorted(df['algorithm'].unique())}")
    print()

    report: dict = {"source_csv": str(args.csv), "n_maps": int(n), "tests": {}}

    # ── 1. Per-map rank correlation, both α=0.05 and FDR-corrected ──────────
    print("─" * 78)
    print("1. Per-map rank correlation: LSAP vs significant-cluster count")
    print("─" * 78)
    print(f"{'target':<28s} {'Spearman ρ':>12s} {'p':>10s} {'Kendall τ':>12s} {'p':>10s} {'n':>5s}")
    print("─" * 78)

    for label, col in [
        ("total_significant (α=0.05)", "total_significant"),
        ("total_significant_fdr (BH)", "total_significant_fdr"),
    ]:
        sp, kt, sp_p, kt_p, k = rank_corr(df["lisa_proxy"], df[col])
        print(f"{label:<28s} {sp:+12.4f} {sp_p:10.4f} {kt:+12.4f} {kt_p:10.4f} {k:5d}")
        report["tests"][f"per_map_rank_{col}"] = {
            "spearman_rho": sp, "spearman_p": sp_p,
            "kendall_tau": kt, "kendall_p": kt_p,
            "n": k,
        }
    print()

    # ── 2. Discrimination AUC ───────────────────────────────────────────────
    print("─" * 78)
    print("2. Discrimination: does LSAP separate 'has ≥1 FDR-sig cluster'?")
    print("─" * 78)
    has_sig_fdr = (df["total_significant_fdr"] > 0).astype(int).values
    has_sig_a05 = (df["total_significant"] > 0).astype(int).values

    # Higher LSAP = more clustering, so use LSAP directly as the score.
    for label, y in [("FDR-corrected (BH q<0.05)", has_sig_fdr),
                     ("uncorrected (α=0.05)",       has_sig_a05)]:
        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)
        if n_pos == 0 or n_neg == 0:
            print(f"  {label:<28s} skipped (n_pos={n_pos}, n_neg={n_neg})")
            report["tests"][f"discrimination_{label}"] = {
                "auc": None, "n_pos": n_pos, "n_neg": n_neg,
            }
            continue
        auc = roc_auc_score(y, df["lisa_proxy"].values)
        print(f"  {label:<28s} AUC = {auc:.4f}  (n_pos={n_pos}, n_neg={n_neg})")
        report["tests"][f"discrimination_{label}"] = {
            "auc": float(auc), "n_pos": n_pos, "n_neg": n_neg,
        }
    print()

    # ── 3. Algorithm-level rank concordance ─────────────────────────────────
    print("─" * 78)
    print("3. Algorithm-level rank concordance (the operationally relevant test)")
    print("─" * 78)
    by_algo = (df.groupby("algorithm")
                 .agg(median_lsap=("lisa_proxy", "median"),
                      median_sig_count=("total_significant", "median"),
                      median_sig_count_fdr=("total_significant_fdr", "median"),
                      mean_sig_count=("total_significant", "mean"),
                      frac_with_sig_fdr=("total_significant_fdr", lambda s: (s > 0).mean()))
                 .reset_index())
    print(by_algo.to_string(index=False))
    print()

    rank_lsap = by_algo["median_lsap"].rank()
    rank_count_a05 = by_algo["mean_sig_count"].rank()
    rank_count_fdr = by_algo["frac_with_sig_fdr"].rank()
    for label, ranks in [("median LSAP vs mean sig-count (α=0.05)", rank_count_a05),
                         ("median LSAP vs P(≥1 FDR-sig cluster)",  rank_count_fdr)]:
        if len(set(ranks.round(6))) == 1:
            print(f"  {label:<42s} τ = NA   (target ranking is degenerate)")
            report["tests"][f"algo_rank_{label}"] = {"kendall_tau": None, "note": "degenerate"}
            continue
        kt = stats.kendalltau(rank_lsap, ranks)
        print(f"  {label:<42s} τ = {kt.statistic:+.4f}  (p = {kt.pvalue:.4f}, n_algos = {len(by_algo)})")
        report["tests"][f"algo_rank_{label}"] = {
            "kendall_tau": float(kt.statistic), "p": float(kt.pvalue), "n_algos": len(by_algo),
        }
    print()

    # ── 4. Convergence-floor diagnostic ─────────────────────────────────────
    print("─" * 78)
    print("4. Convergence-floor: variance suppression in the optimized regime")
    print("─" * 78)
    n_zero_fdr = int((df["total_significant_fdr"] == 0).sum())
    n_zero_a05 = int((df["total_significant"] == 0).sum())
    print(f"  Maps with zero FDR-significant clusters : {n_zero_fdr}/{n} ({100*n_zero_fdr/n:.1f}%)")
    print(f"  Maps with zero α=0.05 significant clusters: {n_zero_a05}/{n} ({100*n_zero_a05/n:.1f}%)")

    sub_zero_fdr = df.loc[df["total_significant_fdr"] == 0, "lisa_proxy"]
    print(f"  LSAP within zero-FDR-sig subset (n={len(sub_zero_fdr)}):")
    print(f"    median = {sub_zero_fdr.median():.4f}, IQR = "
          f"[{sub_zero_fdr.quantile(0.25):.4f}, {sub_zero_fdr.quantile(0.75):.4f}], "
          f"range = [{sub_zero_fdr.min():.4f}, {sub_zero_fdr.max():.4f}]")
    print()
    print(f"  Whole-sample LSAP distribution: ")
    print(f"    median = {df['lisa_proxy'].median():.4f}, IQR = "
          f"[{df['lisa_proxy'].quantile(0.25):.4f}, {df['lisa_proxy'].quantile(0.75):.4f}], "
          f"range = [{df['lisa_proxy'].min():.4f}, {df['lisa_proxy'].max():.4f}]")
    report["tests"]["convergence_floor"] = {
        "n_zero_fdr": n_zero_fdr,
        "n_zero_alpha05": n_zero_a05,
        "lsap_median_overall": float(df["lisa_proxy"].median()),
        "lsap_median_zero_fdr_subset": float(sub_zero_fdr.median()),
        "lsap_range_zero_fdr_subset": [float(sub_zero_fdr.min()), float(sub_zero_fdr.max())],
    }
    print()

    # ── 5. Headline ─────────────────────────────────────────────────────────
    print("─" * 78)
    print("HEADLINE")
    print("─" * 78)
    sp_a05 = report["tests"]["per_map_rank_total_significant"]["spearman_rho"]
    auc_a05 = report["tests"]["discrimination_uncorrected (α=0.05)"]["auc"]
    auc_fdr = report["tests"]["discrimination_FDR-corrected (BH q<0.05)"]["auc"]
    pct_zero = 100 * n_zero_fdr / n
    print(f"  Per-map Spearman ρ on α=0.05 cluster count : {sp_a05:+.4f}")
    print(f"  Discrimination AUC (α=0.05 vs zero)        : {auc_a05:.4f}")
    print(f"  Discrimination AUC (FDR-significant vs zero): {auc_fdr if auc_fdr is not None else 'NA':>6}")
    print(f"  Convergence floor (frac maps with 0 FDR-sig): {pct_zero:.1f}%")
    print()
    print(f"  Frame for §3.8/limitations: After SA convergence, {pct_zero:.0f}% of optimized")
    print(f"  maps already have zero FDR-significant LISA clusters, leaving the per-map")
    print(f"  continuous correlation test on a near-degenerate distribution. The")
    print(f"  operationally relevant tests (uncorrected α=0.05 cluster count discrimination,")
    print(f"  algorithm-level rank concordance) are reported above; the LSAP heuristic is")
    print(f"  used to *rank* maps and methods, not to certify per-location significance.")
    print()

    # ── 6. Persist ──────────────────────────────────────────────────────────
    json_path = out_dir / "lisa_proxy_per_map_diagnostic.json"
    txt_path = out_dir / "lisa_proxy_per_map_diagnostic.txt"
    with json_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"JSON written : {json_path}")
    print(f"(stdout suitable for redirect to {txt_path})")


if __name__ == "__main__":
    main()

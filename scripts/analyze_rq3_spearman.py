#!/usr/bin/env python3
"""
RQ3 closer: Spearman correlations between balance gap and the spatial metrics
across optimized solutions.

§3.10 RQ3:
    Trade-offs between balance gap and spatial distribution. Exploratory —
    no directional hypothesis pre-specified. Reported as Spearman
    correlations between balance gap and spatial metrics across optimized
    solutions.

This script consumes the multi-algorithm canonical results.csv and computes:
  - Per-(algorithm, budget) Spearman ρ between balance_gap and {morans_i,
    lisa_penalty, jains_index} on the per-seed terminal-state distribution
    (with chain-mean aggregation if `chain_id` is present).
  - The pooled correlation across all (algorithm, budget) cells, for the
    aggregate trade-off pattern.

Output: rq3_spearman.csv (one row per algorithm × budget × spatial-metric)
        plus a brief stdout summary.

Usage:
    python scripts/analyze_rq3_spearman.py output/run_tag/benchmark_dir/results.csv \\
        --output output/run_tag/rq3_spearman.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy import stats

SPATIAL_METRICS = ["morans_i", "lisa_penalty", "jains_index"]


def per_seed_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate chains to per-(seed, algorithm, budget) means if `chain_id`
    is present; pass-through otherwise."""
    if "chain_id" in df.columns and df["chain_id"].nunique() > 1:
        group = ["seed", "algorithm", "budget"]
        if "condition" in df.columns:
            group.append("condition")
        keep = ["balance_gap"] + SPATIAL_METRICS
        keep = [c for c in keep if c in df.columns]
        df = df.groupby(group)[keep].mean().reset_index()
    return df


def spearman_block(df: pd.DataFrame) -> List[dict]:
    """Per-(algorithm, budget) Spearman ρ(balance_gap, metric) for each
    spatial metric. Skips cells with n < 5."""
    rows: List[dict] = []
    for (algo, budget), sub in df.groupby(["algorithm", "budget"]):
        for metric in SPATIAL_METRICS:
            if metric not in sub.columns:
                continue
            x = sub["balance_gap"].values
            y = sub[metric].values
            n = len(sub)
            if n < 5:
                rows.append({"algorithm": algo, "budget": int(budget),
                             "metric": metric, "n": n,
                             "spearman_rho": np.nan, "p_value": np.nan})
                continue
            try:
                rho, p = stats.spearmanr(x, y)
            except Exception:
                rho, p = float("nan"), float("nan")
            rows.append({"algorithm": algo, "budget": int(budget),
                         "metric": metric, "n": n,
                         "spearman_rho": float(rho), "p_value": float(p)})
    return rows


def pooled_block(df: pd.DataFrame) -> List[dict]:
    """Aggregate Spearman across all algorithms × budgets pooled — the
    cross-cohort correlation cited in §3.10 alongside the per-cell ones."""
    rows: List[dict] = []
    for metric in SPATIAL_METRICS:
        if metric not in df.columns:
            continue
        x = df["balance_gap"].values
        y = df[metric].values
        rho, p = stats.spearmanr(x, y)
        rows.append({"algorithm": "POOLED", "budget": -1,
                     "metric": metric, "n": len(df),
                     "spearman_rho": float(rho), "p_value": float(p)})
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv", type=Path, help="Path to the multi-algorithm benchmark results.csv")
    p.add_argument("--output", type=Path, required=True, help="Output rq3_spearman.csv")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.csv.exists():
        print(f"ERROR: {args.csv} not found", file=sys.stderr)
        return 1
    df = pd.read_csv(args.csv)
    df = per_seed_panel(df)

    rows = spearman_block(df) + pooled_block(df)
    out_df = pd.DataFrame(rows, columns=["algorithm", "budget", "metric", "n",
                                          "spearman_rho", "p_value"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"RQ3 Spearman written to {args.output}")
    print(out_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

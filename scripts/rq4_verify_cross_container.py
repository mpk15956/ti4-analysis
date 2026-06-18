#!/usr/bin/env python
"""Cross-container seam verification for the RQ4 NSGA-II evals_to_best fill.

The targeted fill (Option 1) re-ran NSGA-II in a freshly rebuilt .sif solely to
RECORD evals_to_best, which the banked six-algo run left at the -1 sentinel
(never instrumented). Splicing the fresh evals_to_best onto the banked dataset
is only valid if the fresh .sif reproduces the banked NSGA-II RESULT (same maps,
same derived metrics) at every matched [seed, budget, chain_id]. If it does, the
only thing the splice changes is the column that was never instrumented; every
other manuscript number is untouched.

Strongest signal: final_tile_layout exact-match rate (the integer assignment the
search converged to). If layouts match, every derived metric is identical by
construction regardless of BLAS/threading. Metric columns are diffed as a
secondary quantitative check. Exits non-zero (fail-loud) if the seam does not
close, so a CI/driver step refuses to splice on a drifted environment.

Usage:
    python scripts/rq4_verify_cross_container.py FRESH.csv BANKED.csv [--le-budget 200000]
"""
import argparse
import sys
import numpy as np
import pandas as pd

KEY = ["seed", "budget", "chain_id"]
# Deterministic functions of the converged layout — must agree if the search
# reproduced. elapsed_sec is wall-clock; evals_to_best is the fill's whole point
# (banked = -1, fresh = real). Both excluded from the equality check.
METRIC_COLS = [
    "balance_gap", "morans_i", "jains_index", "jfi_resources",
    "jfi_influence", "lisa_penalty", "composite_score", "front_size",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fresh_csv", help="fresh NSGA-II fill results.csv")
    ap.add_argument("banked_csv", help="banked six-algo results.csv")
    ap.add_argument("--le-budget", type=int, default=200000,
                    help="compare nsga2 rows at budget <= this (the fill's coverage)")
    args = ap.parse_args()

    fresh = pd.read_csv(args.fresh_csv)
    banked = pd.read_csv(args.banked_csv)

    fresh = fresh[(fresh.algorithm == "nsga2") & (fresh.budget <= args.le_budget)].copy()
    bank_n = banked[(banked.algorithm == "nsga2") & (banked.budget <= args.le_budget)].copy()
    print(f"fresh  nsga2 <={args.le_budget} rows: {len(fresh)}")
    print(f"banked nsga2 <={args.le_budget} rows: {len(bank_n)}")

    merged = fresh.merge(bank_n, on=KEY, how="outer", suffixes=("_f", "_b"), indicator=True)
    only_fresh = int((merged["_merge"] == "left_only").sum())
    only_bank = int((merged["_merge"] == "right_only").sum())
    if only_fresh or only_bank or len(fresh) != len(bank_n) or len(merged) != len(fresh):
        print(f"FAIL: join not 1:1 — only_fresh={only_fresh}, only_banked={only_bank}, "
              f"fresh={len(fresh)}, banked={len(bank_n)}, merged={len(merged)}")
        return 1
    print(f"join: 1:1 perfect on {KEY} ({len(merged)} rows)\n")

    layout_match = int((merged["final_tile_layout_f"] == merged["final_tile_layout_b"]).sum())
    layout_rate = layout_match / len(merged)
    print(f"final_tile_layout exact-match: {layout_match}/{len(merged)} = {layout_rate:.4%}")

    print("\nmetric column max|fresh-banked| (and exact-match count):")
    worst, all_exact = 0.0, True
    for c in METRIC_COLS:
        d = np.abs(merged[f"{c}_f"].astype(float).values - merged[f"{c}_b"].astype(float).values)
        exact = int((d == 0).sum())
        mx = float(np.nanmax(d))
        worst = max(worst, mx)
        all_exact &= (exact == len(merged))
        print(f"  {c:16s} max|Δ|={mx:.3e}  exact={exact}/{len(merged)}")

    banked_sentinel = bool((merged["evals_to_best_b"] == -1).all())
    fresh_real = bool((merged["evals_to_best_f"] >= 0).all())
    print(f"\nevals_to_best: banked all -1 sentinel? {banked_sentinel} | fresh all >=0? {fresh_real}")

    print("\n=== VERDICT ===")
    ok = banked_sentinel and fresh_real
    if layout_rate == 1.0 and all_exact:
        print("SEAM CLOSED (exact): fresh .sif reproduces banked NSGA-II maps bit-identically.")
        print("Splice is valid; only evals_to_best changes (sentinel -> real).")
    elif layout_rate == 1.0:
        print(f"SEAM CLOSED (layouts identical; metric Δ<= {worst:.3e} = float-repr noise).")
    else:
        print(f"SEAM OPEN: layouts/metrics diverge (worst {worst:.3e}). Splice NOT valid.")
        ok = False
    if not (banked_sentinel and fresh_real):
        print("WARNING: evals_to_best precondition not as expected — inspect before merge.")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())

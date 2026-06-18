#!/usr/bin/env python
"""Build the six-way RQ4 results.csv by splicing the fresh NSGA-II evals_to_best
onto the banked six-algo dataset (Option 1: targeted NSGA-II fill, budgets <=200k).

Precondition (proven by scripts/rq4_verify_cross_container.py): for nsga2 at
budget <=200k the fresh .sif reproduces every banked map bit-identically. So the
merge is the minimal possible edit — for each banked nsga2 row at budget <=200k,
overwrite ONLY evals_to_best (-1 sentinel) with the fresh real value, matched on
[seed, budget, chain_id]. Every other cell, every other algorithm, and nsga2 at
500k (five-way descriptive, left at sentinel) are untouched.

Fail-loud: every targeted banked row must match exactly one fresh value, the edit
count must equal the fresh row count, and the output shape must be unchanged.

Usage:
    python scripts/rq4_build_merged_results.py BANKED.csv FRESH.csv OUT.csv [--le-budget 200000]
"""
import argparse
import sys
import pandas as pd

KEY = ["seed", "budget", "chain_id"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("banked_csv", help="banked six-algo results.csv (canonical)")
    ap.add_argument("fresh_csv", help="fresh NSGA-II fill results.csv (real evals_to_best)")
    ap.add_argument("out_csv", help="output merged six-way results.csv")
    ap.add_argument("--le-budget", type=int, default=200000)
    args = ap.parse_args()

    banked = pd.read_csv(args.banked_csv)
    fresh = pd.read_csv(args.fresh_csv)
    n_in, col_order = len(banked), list(banked.columns)

    fresh_n = fresh[(fresh.algorithm == "nsga2") & (fresh.budget <= args.le_budget)].copy()
    lut = fresh_n.set_index(KEY)["evals_to_best"]
    if lut.index.duplicated().any():
        print("FAIL: duplicate keys in fresh nsga2 lookup"); return 1
    if (lut < 0).any():
        print("FAIL: fresh evals_to_best contains a sentinel/negative"); return 1

    target = (banked.algorithm == "nsga2") & (banked.budget <= args.le_budget)
    n_target = int(target.sum())
    if n_target != len(fresh_n):
        print(f"FAIL: target banked rows {n_target} != fresh rows {len(fresh_n)}"); return 1

    keys = list(zip(*[banked.loc[target, k] for k in KEY]))
    mapped = pd.Series(keys).map(lut)
    if mapped.isna().any():
        print(f"FAIL: {int(mapped.isna().sum())} targeted banked rows had no fresh match"); return 1

    pre_sentinel = bool((banked.loc[target, "evals_to_best"] == -1).all())
    banked.loc[target, "evals_to_best"] = mapped.astype(int).values

    if len(banked) != n_in or list(banked.columns) != col_order:
        print("FAIL: output shape/columns changed during merge"); return 1
    nsga_le = banked[(banked.algorithm == "nsga2") & (banked.budget <= args.le_budget)]
    nsga_hi = banked[(banked.algorithm == "nsga2") & (banked.budget > args.le_budget)]
    if not (nsga_le["evals_to_best"] >= 0).all():
        print("FAIL: some nsga2 <=le-budget still sentinel after merge"); return 1
    if not (nsga_hi["evals_to_best"] == -1).all():
        print("FAIL: nsga2 >le-budget unexpectedly modified (should stay -1)"); return 1

    banked.to_csv(args.out_csv, index=False)
    print(f"pre-merge target rows all -1 sentinel: {pre_sentinel}")
    print(f"edited {n_target} nsga2 evals_to_best cells (budget <={args.le_budget}): -1 -> real")
    print(f"nsga2 >{args.le_budget} left five-way-descriptive: {len(nsga_hi)} rows")
    print(f"wrote {args.out_csv}: {len(banked)} rows, {len(banked.columns)} cols (unchanged shape)")
    med = (banked[banked.algorithm == "nsga2"].groupby("budget")["evals_to_best"].median())
    print("\nnsga2 evals_to_best median by budget (post-merge):")
    print(med.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())

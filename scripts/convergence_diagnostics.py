#!/usr/bin/env python3
"""
R-hat (Potential Scale Reduction Factor) convergence diagnostic for multi-chain runs.

Loads trajectory files produced by benchmark_engine.py (with --chains >= 3),
discards the first 50% of each chain (burn-in), then computes the Gelman–Rubin
R-hat from the last 50%. Writes rhat_report.csv (algo, seed, budget, r_hat, converged).

Usage:
    python scripts/convergence_diagnostics.py --trajectory-dir output/benchmark_*/trajectories
    python scripts/convergence_diagnostics.py --trajectory-dir output/benchmark_*/trajectories --output-dir output/benchmark_*
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np


def parse_trajectory_file(path: Path) -> Tuple[str, int, int, List[Tuple[int, int, float]]]:
    """Parse trajectories_algo{X}_seed{Y}_b{Z}.csv. Returns (algo, seed, budget, [(chain_id, eval_count, value), ...])."""
    name = path.stem
    m = re.match(r"trajectories_algo(\w+)_seed(\d+)_b(\d+)", name)
    if not m:
        raise ValueError(f"Unexpected filename: {path.name}")
    algo, seed, budget = m.group(1), int(m.group(2)), int(m.group(3))
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append((int(row["chain_id"]), int(row["eval_count"]), float(row["trajectory_value"])))
    return algo, seed, budget, rows


def chains_to_matrix(rows: List[Tuple[int, int, float]]) -> np.ndarray:
    """Convert (chain_id, eval_count, value) to list of per-chain arrays (post-burn-in only).
    Burn-in: discard first 50% of each chain. Returns (n_chains, n_draws) array of last-50% values.
    """
    by_chain = {}
    for cid, ev, val in rows:
        by_chain.setdefault(cid, []).append((ev, val))
    # Sort each chain by eval_count and take last 50%
    chains = []
    for cid in sorted(by_chain.keys()):
        pts = sorted(by_chain[cid], key=lambda x: x[0])
        n = len(pts)
        start = max(1, n // 2)  # discard first 50%
        post = [v for _, v in pts[start:]]
        chains.append(post)
    if not chains:
        return np.array([]).reshape(0, 0)
    # Pad to same length (use min length so we have balanced draws)
    min_len = min(len(c) for c in chains)
    return np.array([c[:min_len] for c in chains])


def r_hat(draws: np.ndarray) -> float:
    """
    Gelman–Rubin R-hat (PSRF). draws shape (n_chains, n_draws).
    R_hat = sqrt( (1 - 1/L)*W + (1/L)*B ) / sqrt(W)
    W = within-chain variance (mean of per-chain variances), B = between-chain variance (variance of chain means).
    """
    if draws.size == 0 or draws.shape[0] < 2 or draws.shape[1] < 2:
        return float("nan")
    L, n = draws.shape[0], draws.shape[1]
    chain_means = draws.mean(axis=1)
    chain_vars = draws.var(axis=1, ddof=1)
    W = float(np.mean(chain_vars))
    B = float(n * np.var(chain_means, ddof=1))
    if W <= 0:
        return float("nan")
    var_plus = (1 - 1 / n) * W + (1 / n) * B
    return float(np.sqrt(var_plus / W))


def main() -> int:
    p = argparse.ArgumentParser(description="R-hat convergence diagnostic from trajectory files")
    p.add_argument("--trajectory-dir", type=str, required=True,
                  help="Directory containing trajectories_algo*_seed*_b*.csv")
    p.add_argument("--output-dir", type=str, default=None,
                  help="Directory to write rhat_report.csv (default: same as trajectory-dir parent)")
    p.add_argument("--r-hat-threshold", type=float, default=1.1,
                  help="Converged if R_hat <= this (default: 1.1)")
    args = p.parse_args()

    traj_dir = Path(args.trajectory_dir)
    if not traj_dir.is_dir():
        print(f"ERROR: Not a directory: {traj_dir}", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir) if args.output_dir else traj_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern = "trajectories_algo*_seed*_b*.csv"
    files = sorted(traj_dir.glob(pattern))
    if not files:
        print(f"No trajectory files matching {pattern} in {traj_dir}", file=sys.stderr)
        return 1

    report_path = out_dir / "rhat_report.csv"
    report_rows = []

    for path in files:
        try:
            algo, seed, budget, rows = parse_trajectory_file(path)
        except Exception as e:
            print(f"Skip {path.name}: {e}", file=sys.stderr)
            continue
        if not rows:
            continue
        draws = chains_to_matrix(rows)
        if draws.shape[0] < 2:
            report_rows.append({
                "algorithm": algo, "seed": seed, "budget": budget,
                "r_hat": float("nan"), "converged": False,
            })
            continue
        r = r_hat(draws)
        report_rows.append({
            "algorithm": algo, "seed": seed, "budget": budget,
            "r_hat": round(r, 6), "converged": r <= args.r_hat_threshold,
        })

    with open(report_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["algorithm", "seed", "budget", "r_hat", "converged"])
        w.writeheader()
        w.writerows(report_rows)

    print(f"R-hat report: {report_path} ({len(report_rows)} rows)")
    n_converged = sum(1 for r in report_rows if r["converged"])
    print(f"Converged (R_hat <= {args.r_hat_threshold}): {n_converged}/{len(report_rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

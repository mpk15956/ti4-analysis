#!/usr/bin/env python3
"""
LSAP outcome ablation: compare Structural Parity (σ_slice) with vs without LSAP.

Runs the optimizer (default SA) with full composite (JFI + Moran's I + LSAP) and
with LSAP weight set to 0 (JFI + Moran only) on the same seeds. Records
Structural Parity = std(distance-weighted resource totals per player slice) on
the final map. Lower σ_slice = more parity. Reports Wilcoxon signed-rank and
Vargha–Delaney A. A < 0.5 indicates with-LSAP yields lower σ_slice (better outcome).

Usage:
    python scripts/ablation_lsap_outcome.py --seeds 50 --budget 1000
    python scripts/ablation_lsap_outcome.py --seeds 100 --budget 2000 --algorithm sa --output-dir output
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats


def vargha_delaney_a(x: np.ndarray, y: np.ndarray) -> float:
    """A = P(X < Y) + 0.5*P(X==Y). For Structural Parity (lower=better), A < 0.5 means X (with-LSAP) tends lower."""
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return 0.5
    count_less = sum(np.sum(xi < y) for xi in x)
    count_eq = sum(np.sum(xi == y) for xi in x)
    return (count_less + 0.5 * count_eq) / (n * m)


def main() -> int:
    p = argparse.ArgumentParser(description="LSAP ablation: Structural Parity with vs without LSAP")
    p.add_argument("--seeds", type=int, default=50, help="Number of seeds")
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--budget", type=int, default=1000, help="SA iterations per run")
    p.add_argument("--algorithm", type=str, default="sa", help="Algorithm (sa only supported for ablation)")
    p.add_argument("--output-dir", type=str, default="output")
    p.add_argument("--players", type=int, default=6)
    p.add_argument("--sa-rate", type=float, default=0.80)
    p.add_argument("--sa-min-temp", type=float, default=0.01)
    args = p.parse_args()

    if args.algorithm != "sa":
        print("Only SA is supported for this ablation (same budget, same logic).", file=sys.stderr)
        return 1

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    from ti4_analysis.evaluation.batch_experiment import create_joebrew_evaluator
    from ti4_analysis.algorithms.map_generator import generate_random_map
    from ti4_analysis.algorithms.spatial_optimizer import improve_balance_spatial, evaluate_map_multiobjective
    from ti4_analysis.algorithms.map_topology import MapTopology
    from ti4_analysis.algorithms.fast_map_state import FastMapState

    evaluator = create_joebrew_evaluator()

    # Weights: with LSAP (default) vs without LSAP
    weights_with = None  # use default (includes lisa_penalty)
    weights_without = {"morans_i": 0.5, "jains_index": 0.5, "lisa_penalty": 0.0}

    rows = []
    for seed in range(args.base_seed, args.base_seed + args.seeds):
        initial_map = generate_random_map(
            player_count=args.players, template_name="normal",
            include_pok=True, random_seed=seed,
        )
        # With LSAP
        m1 = initial_map.copy()
        improve_balance_spatial(
            m1, evaluator, iterations=args.budget,
            initial_acceptance_rate=args.sa_rate, min_temp=args.sa_min_temp,
            random_seed=seed, verbose=False, weights=weights_with,
        )
        topo1 = MapTopology.from_ti4_map(m1, evaluator)
        fs1 = FastMapState.from_ti4_map(topo1, m1, evaluator)
        sigma_with = fs1.structural_parity()
        # Without LSAP (same seed, fresh copy of initial map)
        m2 = initial_map.copy()
        improve_balance_spatial(
            m2, evaluator, iterations=args.budget,
            initial_acceptance_rate=args.sa_rate, min_temp=args.sa_min_temp,
            random_seed=seed, verbose=False, weights=weights_without,
        )
        topo2 = MapTopology.from_ti4_map(m2, evaluator)
        fs2 = FastMapState.from_ti4_map(topo2, m2, evaluator)
        sigma_without = fs2.structural_parity()
        rows.append({"seed": seed, "structural_parity_with_lsap": sigma_with, "structural_parity_without_lsap": sigma_without})

    with_lsap = np.array([r["structural_parity_with_lsap"] for r in rows])
    without_lsap = np.array([r["structural_parity_without_lsap"] for r in rows])

    # Wilcoxon signed-rank (paired)
    try:
        stat, p_value = stats.wilcoxon(with_lsap, without_lsap, alternative="two-sided")
    except Exception:
        stat, p_value = float("nan"), float("nan")
    # Vargha–Delaney A: with_LSAP vs without_LSAP. Lower σ = better; A < 0.5 means with_LSAP tends lower (better)
    a = vargha_delaney_a(with_lsap, without_lsap)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ablation_lsap_outcome.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "structural_parity_with_lsap", "structural_parity_without_lsap"])
        w.writeheader()
        w.writerows(rows)

    report_path = out_dir / "ablation_lsap_report.txt"
    with open(report_path, "w") as f:
        f.write("LSAP Ablation: Structural Parity (σ_slice, lower = better)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Seeds: {args.seeds}, Budget: {args.budget}, Algorithm: {args.algorithm}\n\n")
        f.write(f"Mean σ_slice (with LSAP):    {np.mean(with_lsap):.6f}\n")
        f.write(f"Mean σ_slice (without LSAP): {np.mean(without_lsap):.6f}\n")
        f.write(f"Median σ_slice (with LSAP):    {np.median(with_lsap):.6f}\n")
        f.write(f"Median σ_slice (without LSAP): {np.median(without_lsap):.6f}\n\n")
        f.write(f"Wilcoxon signed-rank p-value (two-sided): {p_value}\n")
        f.write(f"Vargha–Delaney A (with vs without): {a:.4f}\n")
        f.write("  (A < 0.5: with-LSAP tends to have lower σ_slice / better parity)\n")
        if p_value < 0.05:
            f.write("\nConclusion: Significant difference (p < 0.05). ")
            if a < 0.5:
                f.write("LSAP improves Structural Parity (lower σ_slice).\n")
            else:
                f.write("Without-LSAP has lower σ_slice on average; report as limitation or discuss.\n")
        else:
            f.write("\nConclusion: No significant difference; report as limitation.\n")

    print(f"CSV: {csv_path}")
    print(f"Report: {report_path}")
    print(f"Wilcoxon p = {p_value:.4f}, Vargha–Delaney A = {a:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

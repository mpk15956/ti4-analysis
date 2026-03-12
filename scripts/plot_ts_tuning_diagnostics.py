#!/usr/bin/env python3
"""
Plot TS tuning variance (violin + swarm) and optional Optuna convergence.

Usage:
  python scripts/plot_ts_tuning_diagnostics.py --run-dir output/ts_retune_highbudget/optuna_YYYYMMDD_HHMMSS
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Plot TS tuning variance and convergence"
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        type=str,
        help="Path to the optuna run directory (e.g. output/ts_retune_highbudget/optuna_YYYYMMDD_HHMMSS)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    best_params_path = run_dir / "best_params.json"

    if not best_params_path.exists():
        print(f"Error: Could not find {best_params_path}")
        return 1

    with open(best_params_path, "r") as f:
        bp = json.load(f)

    # 1. Variance Plot (Violin + Swarm) — raw CV vs raw held-out
    if "cv_raw_scores" in bp and "held_out_scores" in bp:
        df = pd.DataFrame({
            "Score": bp["cv_raw_scores"] + bp["held_out_scores"],
            "Validation Phase": (
                ["Cross-Validation (100 seeds)"] * len(bp["cv_raw_scores"])
                + ["Held-Out (50 seeds)"] * len(bp["held_out_scores"])
            ),
        })

        plt.figure(figsize=(8, 5))
        sns.set_theme(style="whitegrid")

        sns.violinplot(
            data=df,
            x="Validation Phase",
            y="Score",
            inner=None,
            cut=0,
            palette="muted",
            linewidth=1.5,
            alpha=0.6,
        )
        sns.swarmplot(
            data=df,
            x="Validation Phase",
            y="Score",
            color="k",
            size=4,
            alpha=0.7,
        )

        plt.title(
            f"Tabu Search Performance Variance (Budget: {bp.get('iter_budget', 'N/A')})",
            pad=15,
        )
        plt.ylabel("Composite Score (Lower is Better)")
        plt.xlabel("")
        plt.tight_layout()

        plot_path = run_dir / "ts_variance_violin_swarm.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved variance plot to {plot_path}")
    else:
        print(
            "Could not find 'cv_raw_scores' or 'held_out_scores' in JSON. "
            "Cannot plot variance."
        )

    # 2. Optional: Optuna convergence (trial vs best_so_far)
    convergence_path = run_dir / "optuna_convergence.csv"
    if convergence_path.exists():
        conv = pd.read_csv(convergence_path)
        plt.figure(figsize=(7, 4))
        plt.plot(conv["trial"], conv["best_so_far"], label="Best so far")
        plt.xlabel("Trial")
        plt.ylabel("Best objective (composite score)")
        plt.title("Optuna convergence (TS tuning)")
        plt.legend()
        plt.tight_layout()
        conv_plot_path = run_dir / "optuna_convergence.png"
        plt.savefig(conv_plot_path, dpi=300)
        plt.close()
        print(f"Saved convergence plot to {conv_plot_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

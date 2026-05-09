#!/usr/bin/env python3
"""
Distance-weight sensitivity analysis: verify that algorithm rankings are
invariant to perturbations of the DISTANCE_MULTIPLIER table.

The default table [6,6,6,4,4,2,1] is community-calibrated (Joebrew evaluator).
This script re-runs a representative subset of seeds under alternative weight
tables and tests whether algorithm rankings change.

Usage:
    python scripts/distance_weight_sensitivity.py --seeds 50 --workers 16
    python scripts/distance_weight_sensitivity.py --algorithms sa,sga --seeds 30

Outputs (inside --output-dir / dist_sensitivity_YYYYMMDD_HHMMSS/):
    sensitivity_results.csv  — one row per (seed, algorithm, weight_config)
    sensitivity_summary.csv  — per-pair ranking stability across configs
    sensitivity_report.txt   — human-readable summary
"""

import argparse
import csv
import gc
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import numpy as np


WEIGHT_CONFIGS = {
    "current":          [6, 6, 6, 4, 4, 2, 1],
    "flat_nearby":      [6, 6, 6, 6, 4, 2, 1],
    "steep_decay":      [6, 6, 5, 3, 3, 2, 1],
    "linear":           [6, 5, 4, 3, 2, 1, 0],
    "inverse_distance": [6, 6, 3, 2, 2, 1, 1],
    "binary_reachable": [1, 1, 1, 1, 1, 0, 0],
}

CSV_FIELDS = [
    "seed", "algorithm", "weight_config",
    "composite_score", "jains_index", "morans_i", "lisa_penalty",
    "balance_gap", "elapsed_sec",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distance-weight sensitivity analysis")
    p.add_argument("--seeds", type=int, default=50,
                   help="Number of random seeds (default: 50)")
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--players", type=int, default=6)
    p.add_argument("--budget", type=int, default=10000,
                   help="Evaluation budget per algorithm (default: 10000)")
    p.add_argument("--algorithms", type=str, default="sa,sga",
                   help="Comma-separated algorithms (default: sa,sga)")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--output-dir", type=str, default="output")
    p.add_argument("--sa-rate", type=float, default=0.80)
    p.add_argument("--sa-min-temp", type=float, default=0.01)
    p.add_argument("--sga-blob", type=float, default=0.5)
    p.add_argument("--sga-mut", type=float, default=0.05)
    p.add_argument("--sga-warm", type=float, default=0.10)
    p.add_argument("--corrected-landscape", action="store_true",
                   help="Run sensitivity under the canonical (corrected) fitness "
                        "landscape: Gen-0 σ normalization, smooth-min Jain (p=8), "
                        "softplus hinge (k=10), √k-LSAP local-variance correction. "
                        "Mirrors benchmark_engine.py so the τ values measured here "
                        "are sensitivity-of-the-canonical-objective, not "
                        "sensitivity-of-the-legacy-uncorrected-objective.")
    return p.parse_args()


_evaluator_cache = {}


def _init_pool():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def _run_seed_config(job):
    """Run all algorithms for one (seed, weight_config) pair."""
    (seed, algos_list, weight_config_name, weight_table,
     budget, players,
     sa_rate, sa_min_temp,
     sga_blob, sga_mut, sga_warm,
     corrected_landscape) = job

    from ti4_analysis.algorithms.map_generator import generate_random_map
    from ti4_analysis.algorithms.spatial_optimizer import (
        improve_balance_spatial, evaluate_map_multiobjective,
        compute_gen0_sigma,
    )
    from ti4_analysis.algorithms.sga_optimizer import sga_optimize
    from ti4_analysis.algorithms.map_topology import MapTopology
    from ti4_analysis.algorithms.fast_map_state import FastMapState
    from ti4_analysis.evaluation.batch_experiment import create_joebrew_evaluator
    from ti4_analysis.data.map_structures import Evaluator
    import copy
    import math

    evaluator = create_joebrew_evaluator()
    evaluator.DISTANCE_MULTIPLIER = list(weight_table)

    algos = set(algos_list)
    rows = []

    try:
        initial_map = generate_random_map(
            player_count=players, template_name="normal",
            include_pok=True, random_seed=seed,
        )

        # Build the corrections kwargs once per seed; mirrors the values
        # benchmark_engine.py uses (n_samples=1000, seed offset +99999,
        # smooth_p=8, smooth_k=10, use_local_variance_lisa=True). When
        # corrected_landscape=False, eval_kw stays empty and every optimizer
        # call uses default (uncorrected) behavior.
        eval_kw = {}
        if corrected_landscape:
            topo = MapTopology.from_ti4_map(initial_map, evaluator)
            sigma = compute_gen0_sigma(
                topo, evaluator, initial_map.copy(),
                n_samples=1000, random_seed=seed + 99999,
                use_local_variance_lisa=True,
            )
            eval_kw = dict(
                normalizer_sigma=sigma,
                use_smooth_objectives=True,
                smooth_p=8.0,
                smooth_k=10.0,
                use_local_variance_lisa=True,
            )

        if "sa" in algos:
            sa_map = initial_map.copy()
            t0 = time.time()
            sa_score, _, _ = improve_balance_spatial(
                sa_map, evaluator, iterations=budget,
                initial_acceptance_rate=sa_rate, min_temp=sa_min_temp,
                random_seed=seed, verbose=False,
                **eval_kw,
            )
            rows.append({
                "seed": seed, "algorithm": "sa",
                "weight_config": weight_config_name,
                "composite_score": round(float(sa_score.composite_score()), 4),
                "jains_index": round(float(sa_score.jains_index), 4),
                "morans_i": round(float(sa_score.morans_i), 4),
                "lisa_penalty": round(float(sa_score.lisa_penalty), 4),
                "balance_gap": round(float(sa_score.balance_gap), 4),
                "elapsed_sec": round(time.time() - t0, 2),
            })
            del sa_map

        if "sga" in algos:
            sga_map = initial_map.copy()
            t0 = time.time()
            pop_size = max(10, int(math.sqrt(budget)))
            sga_gen = max(1, budget // pop_size)
            sga_score, _ = sga_optimize(
                sga_map, evaluator, generations=sga_gen,
                population_size=pop_size,
                blob_fraction=sga_blob, mutation_rate=sga_mut,
                warm_fraction=sga_warm,
                random_seed=seed, verbose=False,
                **eval_kw,
            )
            rows.append({
                "seed": seed, "algorithm": "sga",
                "weight_config": weight_config_name,
                "composite_score": round(float(sga_score.composite_score()), 4),
                "jains_index": round(float(sga_score.jains_index), 4),
                "morans_i": round(float(sga_score.morans_i), 4),
                "lisa_penalty": round(float(sga_score.lisa_penalty), 4),
                "balance_gap": round(float(sga_score.balance_gap), 4),
                "elapsed_sec": round(time.time() - t0, 2),
            })
            del sga_map

        return True, rows

    except Exception as exc:
        print(f"seed={seed} config={weight_config_name}: ERROR {exc}", file=sys.stderr)
        return False, []

    finally:
        gc.collect()


def main() -> int:
    args = parse_args()
    algos = sorted({a.strip().lower() for a in args.algorithms.split(",")})

    _init_pool()

    run_name = datetime.now().strftime("dist_sensitivity_%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "sensitivity_results.csv"

    # Audit follow-up: previously this phase wrote no run_config.json or even
    # a sidecar JSON of the WEIGHT_CONFIGS perturbation grid. Without these,
    # the artifact was unbound to any code state. The helper records git_hash,
    # env, per-file metric hashes, and the perturbation grid the run used.
    from ti4_analysis.utils.run_config import write_run_config
    write_run_config(
        run_dir, args=args,
        extra={
            "phase": "5_dist_sensitivity",
            "run_name": run_name,
            "weight_configs": {name: list(weights) for name, weights in WEIGHT_CONFIGS.items()},
        },
    )

    print(f"Distance-Weight Sensitivity Analysis")
    print(f"Seeds          : {args.seeds} (base={args.base_seed})")
    print(f"Algorithms     : {', '.join(algos)}")
    print(f"Budget         : {args.budget}")
    print(f"Weight configs : {len(WEIGHT_CONFIGS)}")
    print(f"Workers        : {args.workers}")
    print(f"Output         : {run_dir}")
    print()

    jobs = []
    for config_name, weight_table in WEIGHT_CONFIGS.items():
        for seed in range(args.base_seed, args.base_seed + args.seeds):
            jobs.append((
                seed, algos, config_name, weight_table,
                args.budget, args.players,
                args.sa_rate, args.sa_min_temp,
                args.sga_blob, args.sga_mut, args.sga_warm,
                args.corrected_landscape,
            ))

    pool = None
    if args.workers > 1:
        from multiprocessing import Pool
        pool = Pool(args.workers, initializer=_init_pool)

    run_start = time.time()
    all_rows = []

    try:
        mapper = pool.imap_unordered if pool else map
        done = 0
        for ok, rows in mapper(_run_seed_config, jobs):
            all_rows.extend(rows)
            done += 1
            if done % 50 == 0:
                elapsed = time.time() - run_start
                eta = (len(jobs) - done) * (elapsed / done)
                print(f"  {done}/{len(jobs)} jobs  eta={eta/60:.1f}min")
    finally:
        if pool:
            pool.close()
            pool.join()

    all_rows.sort(key=lambda r: (r["weight_config"], r["seed"], r["algorithm"]))
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    total_time = time.time() - run_start
    print(f"\nTotal time: {total_time/60:.1f} min")
    print(f"Results: {csv_path}")

    # ── Ranking stability analysis ────────────────────────────────────────
    try:
        import pandas as pd
        from scipy import stats as scipy_stats

        df = pd.read_csv(csv_path)
        if len(algos) < 2:
            print("\nSkipping ranking analysis (need >= 2 algorithms)")
            return 0

        print("\n── Ranking Stability ──")
        report_lines = ["DISTANCE-WEIGHT SENSITIVITY ANALYSIS", "=" * 60, ""]

        pair_winners = defaultdict(list)
        for config_name in WEIGHT_CONFIGS:
            sub = df[df["weight_config"] == config_name]
            wide = sub.pivot(index="seed", columns="algorithm",
                             values="composite_score").dropna()
            avail = [a for a in algos if a in wide.columns]
            for a, b in combinations(avail, 2):
                x, y = wide[a].values, wide[b].values
                med_a, med_b = np.median(x), np.median(y)
                winner = a if med_a < med_b else b if med_b < med_a else "tie"
                pair_winners[(a, b)].append((config_name, winner))

                try:
                    _, p = scipy_stats.wilcoxon(x, y, alternative="two-sided")
                except ValueError:
                    p = 1.0
                line = (f"  {config_name:<20s}  {a} vs {b}: "
                        f"med({a})={med_a:.4f}  med({b})={med_b:.4f}  "
                        f"winner={winner}  p={p:.4f}")
                print(line)
                report_lines.append(line)

        print("\n── Summary ──")
        report_lines.append("\n\nSUMMARY")
        report_lines.append("-" * 60)

        all_stable = True
        for (a, b), results in pair_winners.items():
            winners = [w for _, w in results]
            unique_winners = set(winners)
            n_configs = len(results)
            if len(unique_winners) == 1:
                msg = (f"  {a} vs {b}: STABLE — {winners[0]} wins "
                       f"across all {n_configs} configs")
            else:
                all_stable = False
                counts = {w: winners.count(w) for w in unique_winners}
                msg = (f"  {a} vs {b}: UNSTABLE — {counts} "
                       f"across {n_configs} configs")
            print(msg)
            report_lines.append(msg)

        # Kendall's tau rank correlation across configs
        if len(algos) >= 2 and len(WEIGHT_CONFIGS) >= 2:
            config_names = list(WEIGHT_CONFIGS.keys())
            algo_medians = {}
            for config_name in config_names:
                sub = df[df["weight_config"] == config_name]
                medians = sub.groupby("algorithm")["composite_score"].median()
                algo_medians[config_name] = medians

            if len(config_names) >= 2:
                tau_pairs = []
                for i in range(len(config_names)):
                    for j in range(i + 1, len(config_names)):
                        c1, c2 = config_names[i], config_names[j]
                        shared = sorted(set(algo_medians[c1].index) &
                                        set(algo_medians[c2].index))
                        if len(shared) >= 2:
                            r1 = [algo_medians[c1][a] for a in shared]
                            r2 = [algo_medians[c2][a] for a in shared]
                            tau, p = scipy_stats.kendalltau(r1, r2)
                            tau_pairs.append(tau)

                if tau_pairs:
                    mean_tau = statistics.mean(tau_pairs)
                    msg = (f"\n  Mean Kendall's tau across config pairs: "
                           f"{mean_tau:.3f} (1.0 = perfectly stable)")
                    print(msg)
                    report_lines.append(msg)

        verdict = ("WEIGHT-INVARIANT" if all_stable
                    else "WEIGHT-SENSITIVE — inspect per-config results")
        msg = f"\n  Verdict: Algorithm rankings are {verdict}"
        print(msg)
        report_lines.append(msg)

        report_path = run_dir / "sensitivity_report.txt"
        report_path.write_text("\n".join(report_lines))
        print(f"\n  Report → {report_path}")

    except ImportError:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())

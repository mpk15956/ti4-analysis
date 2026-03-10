#!/usr/bin/env python3
"""
Post-hoc LISA validation: verify that minimising the continuous LISA proxy
successfully eliminates *statistically significant* spatial clusters.

For a subset of seeds, re-runs each optimisation algorithm, captures the
final map state, and applies conditional-permutation LISA to count
significant H-H and L-L clusters at p < 0.05.

If SA's continuous penalty eliminates more significant clusters than HC,
the proxy is empirically justified.

Usage:
    python scripts/validate_lisa_proxy.py --seeds 30 --output-dir output/lisa_validation/
    python scripts/validate_lisa_proxy.py --seeds 30 --algorithms sa,hc --n-perms 999
"""

import argparse
import csv
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Post-hoc LISA validation via conditional permutation tests")
    p.add_argument("--seeds", type=int, default=30, help="Number of random seeds (default: 30)")
    p.add_argument("--base-seed", type=int, default=0, help="First random seed")
    p.add_argument("--players", type=int, default=6, help="Number of players")
    p.add_argument("--sa-iter", type=int, default=1000, help="SA iterations per seed")
    p.add_argument("--hc-iter", type=int, default=1000, help="HC iterations per seed")
    p.add_argument("--algorithms", type=str, default="hc,sa",
                   help="Comma-separated algorithms (hc, sa, nsga2, ts)")
    p.add_argument("--n-perms", type=int, default=999,
                   help="Permutations per location for significance test (default: 999)")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Significance threshold (default: 0.05)")
    p.add_argument("--output-dir", type=str, default="output",
                   help="Root output directory")
    # SA hyperparameters
    p.add_argument("--sa-rate", type=float, default=0.80)
    p.add_argument("--sa-min-temp", type=float, default=0.01)
    # NSGA-II hyperparameters
    p.add_argument("--nsga-gen", type=int, default=50)
    p.add_argument("--nsga-pop", type=int, default=20)
    p.add_argument("--nsga-blob", type=float, default=0.5)
    p.add_argument("--nsga-mut", type=float, default=0.05)
    p.add_argument("--nsga-warm", type=float, default=0.10)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Permutation-tested LISA
# ---------------------------------------------------------------------------

def conditional_permutation_lisa(
    z: np.ndarray,
    W,
    n_perms: int = 999,
    alpha: float = 0.05,
    rng: np.random.Generator = None,
) -> Dict:
    """
    Conditional permutation test for local Moran's I (Anselin, 1995).

    For each location i, holds z[i] fixed, randomly permutes all other
    values, and recomputes local_I[i] under each permutation to build
    a reference distribution.

    Returns dict with counts of significant H-H, L-L, H-L, L-H clusters.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(z)
    if n < 3:
        return {"n_sig_HH": 0, "n_sig_LL": 0, "n_sig_HL": 0, "n_sig_LH": 0,
                "total_significant": 0, "n_positions": n}

    z_mean = z.mean()
    z_dev = z - z_mean
    m2 = float(z_dev @ z_dev) / n
    if m2 == 0.0:
        return {"n_sig_HH": 0, "n_sig_LL": 0, "n_sig_HL": 0, "n_sig_LH": 0,
                "total_significant": 0, "n_positions": n}

    Wz = np.asarray(W @ z_dev).ravel()
    observed_local_I = z_dev * Wz / m2

    p_values = np.ones(n)
    for i in range(n):
        obs_I = observed_local_I[i]
        count_extreme = 0
        others = np.concatenate([z_dev[:i], z_dev[i+1:]])
        W_row = np.asarray(W[i].toarray()).ravel() if hasattr(W, 'toarray') else W[i]

        for _ in range(n_perms):
            perm = rng.permutation(others)
            perm_full = np.empty(n)
            perm_full[i] = z_dev[i]
            perm_full[:i] = perm[:i]
            perm_full[i+1:] = perm[i:]

            perm_m2 = float(perm_full @ perm_full) / n
            if perm_m2 == 0.0:
                continue
            perm_lag = float(W_row @ perm_full)
            perm_I = perm_full[i] * perm_lag / perm_m2

            if abs(perm_I) >= abs(obs_I):
                count_extreme += 1

        p_values[i] = (count_extreme + 1) / (n_perms + 1)

    significant = p_values < alpha
    n_sig_HH = int(np.sum(significant & (z_dev > 0) & (Wz > 0)))
    n_sig_LL = int(np.sum(significant & (z_dev < 0) & (Wz < 0)))
    n_sig_HL = int(np.sum(significant & (z_dev > 0) & (Wz < 0)))
    n_sig_LH = int(np.sum(significant & (z_dev < 0) & (Wz > 0)))

    return {
        "n_sig_HH": n_sig_HH,
        "n_sig_LL": n_sig_LL,
        "n_sig_HL": n_sig_HL,
        "n_sig_LH": n_sig_LH,
        "total_significant": int(significant.sum()),
        "n_positions": n,
        "lisa_proxy": float(observed_local_I[observed_local_I > 0].sum()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "seed", "algorithm", "n_sig_HH", "n_sig_LL", "n_sig_HL", "n_sig_LH",
    "total_significant", "n_positions", "lisa_proxy", "composite_score",
    "elapsed_sec",
]


def main() -> int:
    args = parse_args()
    algos = {a.strip().lower() for a in args.algorithms.split(",")}

    from ti4_analysis.algorithms.map_generator import generate_random_map
    from ti4_analysis.algorithms.balance_engine import improve_balance
    from ti4_analysis.algorithms.spatial_optimizer import (
        improve_balance_spatial, evaluate_map_multiobjective,
    )
    from ti4_analysis.algorithms.nsga2_optimizer import nsga2_optimize
    from ti4_analysis.algorithms.tabu_search_optimizer import improve_balance_tabu
    from ti4_analysis.algorithms.map_topology import MapTopology
    from ti4_analysis.algorithms.fast_map_state import FastMapState
    from ti4_analysis.evaluation.batch_experiment import create_joebrew_evaluator

    evaluator = create_joebrew_evaluator()

    run_name = datetime.now().strftime("lisa_validation_%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "validation_results.csv"

    print(f"LISA Proxy Validation")
    print(f"Seeds       : {args.seeds} (base={args.base_seed})")
    print(f"Algorithms  : {', '.join(sorted(algos))}")
    print(f"Permutations: {args.n_perms}")
    print(f"Output      : {run_dir}")
    print()

    rng = np.random.default_rng(42)
    run_start = time.time()

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for seed_offset in range(args.seeds):
            seed = args.base_seed + seed_offset
            seed_start = time.time()

            try:
                initial_map = generate_random_map(
                    player_count=args.players, template_name="normal",
                    include_pok=True, random_seed=seed,
                )
            except Exception as exc:
                print(f"  seed={seed}: map generation failed: {exc}", file=sys.stderr)
                continue

            for algo in sorted(algos):
                t0 = time.time()
                try:
                    if algo == "hc":
                        m = initial_map.copy()
                        improve_balance(m, evaluator, iterations=args.hc_iter, random_seed=seed)
                        topo = MapTopology.from_ti4_map(m, evaluator)
                        fs = FastMapState.from_ti4_map(topo, m, evaluator)
                    elif algo == "sa":
                        m = initial_map.copy()
                        _, _ = improve_balance_spatial(
                            m, evaluator, iterations=args.sa_iter,
                            initial_acceptance_rate=args.sa_rate,
                            min_temp=args.sa_min_temp,
                            random_seed=seed, verbose=False,
                        )
                        topo = MapTopology.from_ti4_map(m, evaluator)
                        fs = FastMapState.from_ti4_map(topo, m, evaluator)
                    elif algo == "nsga2":
                        m = initial_map.copy()
                        front = nsga2_optimize(
                            m, evaluator, generations=args.nsga_gen,
                            population_size=args.nsga_pop,
                            blob_fraction=args.nsga_blob,
                            mutation_rate=args.nsga_mut,
                            warm_fraction=args.nsga_warm,
                            random_seed=seed, verbose=False,
                        )
                        best_map, best_score = min(front, key=lambda x: x[1].composite_score())
                        topo = MapTopology.from_ti4_map(best_map, evaluator)
                        fs = FastMapState.from_ti4_map(topo, best_map, evaluator)
                    elif algo == "ts":
                        m = initial_map.copy()
                        _, _ = improve_balance_tabu(
                            m, evaluator, max_evaluations=args.sa_iter,
                            random_seed=seed, verbose=False,
                        )
                        topo = MapTopology.from_ti4_map(m, evaluator)
                        fs = FastMapState.from_ti4_map(topo, m, evaluator)
                    else:
                        continue

                    score = evaluate_map_multiobjective(m if algo != "nsga2" else best_map,
                                                       evaluator, fast_state=fs)
                    z = fs.spatial_values()
                    W = topo.spatial_W

                    result = conditional_permutation_lisa(z, W, args.n_perms, args.alpha, rng)
                    elapsed = time.time() - t0

                    row = {
                        "seed": seed,
                        "algorithm": algo,
                        "elapsed_sec": round(elapsed, 2),
                        "composite_score": round(float(score.composite_score()), 4),
                        **{k: result[k] for k in CSV_FIELDS if k in result},
                    }
                    writer.writerow(row)
                    f.flush()

                    print(f"  seed={seed:4d}  {algo:<6s}  "
                          f"sig={result['total_significant']}/{result['n_positions']}  "
                          f"HH={result['n_sig_HH']} LL={result['n_sig_LL']}  "
                          f"proxy={result['lisa_proxy']:.3f}  "
                          f"t={elapsed:.1f}s")

                except Exception as exc:
                    print(f"  seed={seed} {algo}: ERROR {exc}", file=sys.stderr)

    total_time = time.time() - run_start
    print(f"\nTotal time: {total_time/60:.1f} min")
    print(f"Results: {csv_path}")

    # Summary
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        print("\n── Summary ──")
        summary = df.groupby("algorithm").agg(
            mean_significant=("total_significant", "mean"),
            mean_HH=("n_sig_HH", "mean"),
            mean_LL=("n_sig_LL", "mean"),
            mean_proxy=("lisa_proxy", "mean"),
        ).round(3)
        print(summary)
    except ImportError:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())

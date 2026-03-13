#!/usr/bin/env python3
"""
LISA cluster map visualization for the "pathological fairness" case: maps that are
perfectly fair by JFI (JFI > 0.99) but spatially broken (max Z_LISA > 3).

Produces one figure: LISA cluster map (HH, LL, LH, HL) for the given (seed, algorithm),
showing how JFI ignores spatial clustering. Use after validate_lisa_proxy has identified
an example (pathological_fairness_example.json) or pass --seed and --algorithm directly.

Usage:
    python scripts/viz_pathological_fairness.py --seed 42 --algorithm sa
    python scripts/viz_pathological_fairness.py --example output/.../pathological_fairness_example.json
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

def _load_validate_module():
    script_dir = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "validate_lisa_proxy", script_dir / "validate_lisa_proxy.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="LISA cluster map for pathological fairness example")
    p.add_argument("--seed", type=int, default=None, help="Random seed (or use --example)")
    p.add_argument("--algorithm", type=str, default=None, help="Algorithm: hc, sa, nsga2, ts")
    p.add_argument("--example", type=str, default=None,
                   help="Path to pathological_fairness_example.json (overrides --seed/--algorithm)")
    p.add_argument("--output-dir", type=str, default="output",
                   help="Directory for output figure")
    p.add_argument("--players", type=int, default=6)
    p.add_argument("--sa-iter", type=int, default=1000)
    p.add_argument("--hc-iter", type=int, default=1000)
    p.add_argument("--ts-iter", type=int, default=None)
    p.add_argument("--nsga-gen", type=int, default=50)
    p.add_argument("--nsga-pop", type=int, default=20)
    p.add_argument("--n-perms", type=int, default=999)
    return p.parse_args()


def run_one_map(seed: int, algo: str, evaluator, args) -> tuple:
    """Run optimizer for one (seed, algo); return (final_map, topo, score, result with cluster_labels)."""
    import numpy as np
    from ti4_analysis.algorithms.map_generator import generate_random_map
    from ti4_analysis.algorithms.balance_engine import improve_balance
    from ti4_analysis.algorithms.spatial_optimizer import (
        improve_balance_spatial, evaluate_map_multiobjective,
    )
    from ti4_analysis.algorithms.nsga2_optimizer import nsga2_optimize
    from ti4_analysis.algorithms.tabu_search_optimizer import improve_balance_tabu
    from ti4_analysis.algorithms.map_topology import MapTopology
    from ti4_analysis.algorithms.fast_map_state import FastMapState

    _val = _load_validate_module()
    conditional_permutation_lisa = _val.conditional_permutation_lisa

    initial_map = generate_random_map(
        player_count=args.players, template_name="normal",
        include_pok=True, random_seed=seed,
    )
    ts_iter = args.ts_iter if getattr(args, "ts_iter", None) is not None else getattr(args, "sa_iter", 1000)
    rng = np.random.default_rng(seed)

    if algo == "hc":
        m = initial_map.copy()
        improve_balance(m, evaluator, iterations=args.hc_iter, random_seed=seed)
    elif algo == "sa":
        m = initial_map.copy()
        improve_balance_spatial(
            m, evaluator, iterations=args.sa_iter,
            initial_acceptance_rate=0.80, min_temp=0.01,
            random_seed=seed, verbose=False,
        )
    elif algo == "nsga2":
        m = initial_map.copy()
        front = nsga2_optimize(
            m, evaluator, generations=args.nsga_gen,
            population_size=args.nsga_pop,
            blob_fraction=0.5, mutation_rate=0.05, warm_fraction=0.10,
            random_seed=seed, verbose=False,
        )
        m, _ = min(front, key=lambda x: x[1].composite_score())
    elif algo == "ts":
        m = initial_map.copy()
        improve_balance_tabu(
            m, evaluator, max_evaluations=ts_iter,
            random_seed=seed, verbose=False,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    topo = MapTopology.from_ti4_map(m, evaluator)
    fs = FastMapState.from_ti4_map(topo, m, evaluator)
    score = evaluate_map_multiobjective(m, evaluator, fast_state=fs)
    z = fs.spatial_values()
    W = topo.spatial_W
    result = conditional_permutation_lisa(
        z, W, getattr(args, "n_perms", 999), 0.05, 0.05, rng,
    )
    return m, topo, score, result


def main() -> int:
    args = parse_args()
    if args.example:
        with open(args.example) as f:
            ex = json.load(f)
        seed = ex["seed"]
        algo = ex["algorithm"]
        jfi = ex.get("jains_index")
        max_z = ex.get("max_z_lisa")
    else:
        if args.seed is None or args.algorithm is None:
            print("Provide --seed and --algorithm, or --example JSON", file=sys.stderr)
            return 1
        seed = args.seed
        algo = args.algorithm.strip().lower()
        jfi = max_z = None

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    from ti4_analysis.evaluation.batch_experiment import create_joebrew_evaluator
    evaluator = create_joebrew_evaluator()

    print(f"Running seed={seed}, algorithm={algo}...")
    m, topo, score, result = run_one_map(seed, algo, evaluator, args)
    if jfi is None:
        jfi = float(score.jains_index)
    if max_z is None:
        max_z = result["max_z_lisa"]
    cluster_labels = result["cluster_labels"]
    n = len(cluster_labels)

    # Coordinates: topo.spatial_indices are indices into m.spaces
    coords = []
    for i in range(n):
        idx = int(topo.spatial_indices[i])
        c = m.spaces[idx].coord
        coords.append((c.x, c.y))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for figure", file=sys.stderr)
        return 1

    color_map = {"HH": "red", "LL": "blue", "HL": "orange", "LH": "green"}
    colors = [color_map.get(lbl, "gray") for lbl in cluster_labels]
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]

    fig, ax = plt.subplots(figsize=(10, 8))
    for lbl in ["HH", "LL", "HL", "LH"]:
        mask = [l == lbl for l in cluster_labels]
        if not any(mask):
            continue
        ax.scatter(
            [xs[i] for i in range(n) if mask[i]],
            [ys[i] for i in range(n) if mask[i]],
            c=color_map[lbl], label=lbl, s=80, alpha=0.8, edgecolors="black",
        )
    ax.set_xlabel("Hex x")
    ax.set_ylabel("Hex y")
    ax.set_title(f"LISA cluster map (seed={seed}, {algo})\nJFI = {jfi:.4f}, max(Z_LISA) = {max_z:.3f}")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pathological_fairness_lisa_cluster_seed{seed}_{algo}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

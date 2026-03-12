#!/usr/bin/env python3
"""
Optuna hyperparameter tuning for the TI4 map optimization algorithms.

Tunes the free parameters of SA, NSGA-II, SGA, or Tabu Search. SA/NSGA-II/SGA
use Bayesian optimization (TPE sampler). TS uses exhaustive grid search over
the tenure coefficient k with θ = max(3, ⌈k·√S⌉). Evaluation seeds are
deliberately disjoint from benchmark seeds (default base_seed=9000 vs benchmark 0-999)
to prevent overfitting the hyperparameters to specific map layouts.

Usage:
    python scripts/optimize_hyperparameters.py --algo sa    [options]
    python scripts/optimize_hyperparameters.py --algo nsga2  [options]
    python scripts/optimize_hyperparameters.py --algo ts     [options]

SA search space:
    initial_acceptance_rate  [0.50, 0.99]   — controls T₀ magnitude
    min_temp                 [0.001, 0.10]  — controls exploitation depth

NSGA-II search space:
    blob_fraction            [0.25, 0.75]   — BFS crossover blob size
    mutation_rate            [0.01, 0.30]   — per-position swap probability
    (warm_fraction fixed to 0 — cold-start only for Pareto diversity)

TS search space (grid, not Optuna):
    tabu_tenure_coefficient k  — grid over k; θ = max(3, ⌈k·√S⌉) per map

Outputs (inside --output-dir / optuna_YYYYMMDD_HHMMSS/):
    best_params.json  — optimal hyperparameters + cv_mean, cv_std, held_out_*
    trials.csv        — all trial results (Optuna algos only)
    grid_trials.csv   — TS only: k, mean, std, se, ci_lower, ci_upper, cv
    run_config.json   — CLI params + git hash for reproducibility

Validation: fixed train/test split. Training on eval_seeds; 50 held-out seeds.
best_params.json includes cv_mean and cv_std for overfitting comparison.
"""

import argparse
import csv
import gc
import json
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna hyperparameter tuning for TI4 optimizers")
    p.add_argument("--algo",        required=True, choices=["sa", "nsga2", "sga", "ts"],
                   help="Algorithm to tune")
    p.add_argument("--trials",      type=int, default=50,
                   help="Number of Optuna trials (default: 50)")
    p.add_argument("--eval-seeds",  type=int, default=100,
                   help="Seeds per trial for mean estimation (default: 100)")
    p.add_argument("--iter-budget", type=int, default=200,
                   help="Iteration budget per seed: SA=sa_iter, NSGA2=gen×pop (default: 200)")
    p.add_argument("--base-seed",   type=int, default=9000,
                   help="Base random seed for tuning — keep disjoint from benchmark seeds (default: 9000)")
    p.add_argument("--players",     type=int, default=6,
                   help="Number of players (default: 6)")
    p.add_argument("--output-dir",  type=str, default="output",
                   help="Root output directory (default: output)")
    p.add_argument("--workers",     type=int, default=1,
                   help="Optuna n_jobs for parallel trial evaluation (default: 1)")
    p.add_argument("--study-name",  type=str, default=None,
                   help="Optuna study name (enables resume via --storage)")
    p.add_argument("--storage",     type=str, default=None,
                   help="Optuna storage URL, e.g. sqlite:///optuna.db")
    return p.parse_args()


# ---------------------------------------------------------------------------
# NSGA-II budget factorization
# ---------------------------------------------------------------------------

def _nsga2_budget(iter_budget: int):
    """
    Factorize iter_budget into (generations, pop_size) with pop ≥ 10, gen ≥ 5.

    Example: iter_budget=200 → pop=14, gen=14 (196 evals).
    """
    pop_size    = max(10, int(math.sqrt(iter_budget)))
    generations = max(5,  iter_budget // pop_size)
    return generations, pop_size


# ---------------------------------------------------------------------------
# Objective functions
# ---------------------------------------------------------------------------

def objective_sa(trial, args, evaluator, generate_random_map, improve_balance_spatial):
    """Objective for SA: mean composite_score over eval_seeds random maps."""
    rate  = trial.suggest_float("initial_acceptance_rate", 0.50, 0.99)
    min_t = trial.suggest_float("min_temp", 0.001, 0.10, log=True)

    scores = []
    for i in range(args.eval_seeds):
        seed = args.base_seed + i
        m = generate_random_map(
            player_count=args.players,
            template_name="normal",
            include_pok=True,
            random_seed=seed,
        )
        score, _, _ = improve_balance_spatial(
            m, evaluator,
            iterations=args.iter_budget,
            initial_acceptance_rate=rate,
            min_temp=min_t,
            random_seed=seed,
            verbose=False,
        )
        scores.append(score.composite_score())
        del m
        gc.collect()

    return statistics.mean(scores)


def objective_nsga2(trial, args, evaluator, generate_random_map, nsga2_optimize):
    """Objective for NSGA-II: mean best-composite over eval_seeds random maps."""
    blob_frac  = trial.suggest_float("blob_fraction",  0.25, 0.75)
    mut_rate   = trial.suggest_float("mutation_rate",  0.01, 0.30, log=True)
    # warm_fraction fixed to 0 for NSGA-II (cold-start only for Pareto diversity)

    generations, pop_size = _nsga2_budget(args.iter_budget)

    scores = []
    for i in range(args.eval_seeds):
        seed = args.base_seed + i
        m = generate_random_map(
            player_count=args.players,
            template_name="normal",
            include_pok=True,
            random_seed=seed,
        )
        front = nsga2_optimize(
            m, evaluator,
            generations=generations,
            population_size=pop_size,
            blob_fraction=blob_frac,
            mutation_rate=mut_rate,
            warm_fraction=0,
            random_seed=seed,
            verbose=False,
        )
        best = min(front, key=lambda x: x[1].composite_score())[1]
        scores.append(best.composite_score())
        del m, front
        gc.collect()

    return statistics.mean(scores)


def objective_sga(trial, args, evaluator, generate_random_map, sga_optimize):
    """Objective for SGA: mean composite_score over eval_seeds random maps."""
    blob_frac  = trial.suggest_float("blob_fraction",  0.25, 0.75)
    mut_rate   = trial.suggest_float("mutation_rate",  0.01, 0.30, log=True)
    warm_frac  = trial.suggest_float("warm_fraction",  0.00, 0.30)

    generations, pop_size = _nsga2_budget(args.iter_budget)

    scores = []
    for i in range(args.eval_seeds):
        seed = args.base_seed + i
        m = generate_random_map(
            player_count=args.players,
            template_name="normal",
            include_pok=True,
            random_seed=seed,
        )
        score, _ = sga_optimize(
            m, evaluator,
            generations=generations,
            population_size=pop_size,
            blob_fraction=blob_frac,
            mutation_rate=mut_rate,
            warm_fraction=warm_frac,
            random_seed=seed,
            verbose=False,
        )
        scores.append(score.composite_score())
        del m
        gc.collect()

    return statistics.mean(scores)


# TS grid: k in [0.5, 4.0] step 0.2 (θ = max(3, ceil(k·√S)) for typical S≈20–40)
TS_GRID_K = [round(0.5 + i * 0.2, 2) for i in range(18)]


def _stats_from_scores(scores: list) -> dict:
    """Compute mean, std, SE, 95% CI, and CV from a list of scores."""
    n = len(scores)
    mean = statistics.mean(scores)
    std = statistics.stdev(scores) if n > 1 else 0.0
    se = std / (n ** 0.5) if n > 0 else 0.0
    ci_half = 1.96 * se
    cv = (std / mean) if mean and mean != 0 else 0.0
    return {
        "mean": mean,
        "std": std,
        "se": se,
        "ci_lower": mean - ci_half,
        "ci_upper": mean + ci_half,
        "cv": cv,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run_ts_grid(args, run_dir: Path, evaluator, generate_random_map, improve_balance_tabu) -> tuple:
    """Run grid search over k for TS. Returns (best_k, best_value, grid_rows, run_start)."""
    run_start = time.time()
    grid_rows = []
    for idx, k in enumerate(TS_GRID_K):
        scores = []
        for i in range(args.eval_seeds):
            seed = args.base_seed + i
            m = generate_random_map(
                player_count=args.players,
                template_name="normal",
                include_pok=True,
                random_seed=seed,
            )
            score, _, _, _ = improve_balance_tabu(
                m, evaluator,
                max_evaluations=args.iter_budget,
                tabu_tenure_coefficient=k,
                random_seed=seed,
                verbose=False,
            )
            scores.append(score.composite_score())
            del m
            gc.collect()
        st = _stats_from_scores(scores)
        grid_rows.append({
            "k": k,
            "mean": st["mean"],
            "std": st["std"],
            "se": st["se"],
            "ci_lower": st["ci_lower"],
            "ci_upper": st["ci_upper"],
            "cv": st["cv"],
            "scores": scores,
        })
        best_so_far = min(r["mean"] for r in grid_rows)
        elapsed = time.time() - run_start
        print(
            f"  grid {idx + 1:3d}/{len(TS_GRID_K)}  k={k:.2f}  "
            f"mean={st['mean']:.4f}  std={st['std']:.4f}  "
            f"best={best_so_far:.4f}  elapsed={elapsed:.0f}s"
        )
    best_row = min(grid_rows, key=lambda r: r["mean"])
    return best_row["k"], best_row["mean"], grid_rows, run_start


def main() -> int:
    args = parse_args()

    from ti4_analysis.algorithms.map_generator import generate_random_map
    from ti4_analysis.algorithms.spatial_optimizer import improve_balance_spatial
    from ti4_analysis.algorithms.nsga2_optimizer import nsga2_optimize
    from ti4_analysis.algorithms.sga_optimizer import sga_optimize
    from ti4_analysis.algorithms.tabu_search_optimizer import improve_balance_tabu
    from ti4_analysis.evaluation.batch_experiment import create_joebrew_evaluator

    evaluator = create_joebrew_evaluator()

    # Create run directory
    run_name = datetime.now().strftime("optuna_%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    gen, pop = _nsga2_budget(args.iter_budget)
    config = {
        "run_name": run_name,
        "algo": args.algo,
        "trials": args.trials,
        "eval_seeds": args.eval_seeds,
        "iter_budget": args.iter_budget,
        "nsga2_gen": gen,
        "nsga2_pop": pop,
        "base_seed": args.base_seed,
        "players": args.players,
        "started_at": datetime.now().isoformat(),
    }
    try:
        import subprocess
        config["git_hash"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        config["git_hash"] = "unknown"

    with open(run_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Run directory : {run_dir}")
    print(f"Algorithm     : {args.algo.upper()}")
    print(f"Eval seeds   : {args.eval_seeds}  (base_seed={args.base_seed})")
    print(f"Iter budget  : {args.iter_budget}")
    print()

    held_out_base = args.base_seed + args.eval_seeds
    held_out_size = 50
    held_out_seeds = list(range(held_out_base, held_out_base + held_out_size))

    # ── TS: grid search over k (no Optuna) ───────────────────────────────────
    if args.algo == "ts":
        print(f"TS grid: {len(TS_GRID_K)} values of k (θ = max(3, ceil(k·√S)))")
        best_k, best_value, grid_rows, run_start = _run_ts_grid(
            args, run_dir, evaluator, generate_random_map, improve_balance_tabu
        )
        grid_path = run_dir / "grid_trials.csv"
        with open(grid_path, "w", newline="") as gf:
            gw = csv.DictWriter(
                gf,
                fieldnames=["k", "mean", "std", "se", "ci_lower", "ci_upper", "cv"],
            )
            gw.writeheader()
            for r in grid_rows:
                gw.writerow({k: round(r[k], 6) if isinstance(r[k], float) else r[k]
                             for k in gw.fieldnames})
        print(f"\nGrid trials → {grid_path}")

        best_row = min(grid_rows, key=lambda r: r["mean"])
        cv_mean = best_row["mean"]
        cv_std = best_row["std"]
        cv_st = _stats_from_scores(best_row["scores"])

        print(f"\n--- Held-out validation ({held_out_size} seeds: "
              f"{held_out_base}–{held_out_base + held_out_size - 1}) ---")
        held_out_scores = []
        for seed in held_out_seeds:
            m = generate_random_map(
                player_count=args.players, template_name="normal",
                include_pok=True, random_seed=seed,
            )
            s, _, _, _ = improve_balance_tabu(
                m, evaluator, max_evaluations=args.iter_budget,
                tabu_tenure_coefficient=best_k,
                random_seed=seed, verbose=False,
            )
            held_out_scores.append(s.composite_score())
            del m
            gc.collect()
        held_out_mean = statistics.mean(held_out_scores)
        held_out_std = statistics.stdev(held_out_scores) if len(held_out_scores) > 1 else 0.0
        print(f"  Held-out mean: {held_out_mean:.4f} ± {held_out_std:.4f}")

        best_params = {
            "algorithm": args.algo,
            "best_value": round(best_value, 6),
            "best_params": {"tabu_tenure_coefficient": best_k},
            "n_trials": len(TS_GRID_K),
            "eval_seeds": args.eval_seeds,
            "iter_budget": args.iter_budget,
            "train_test_split": "train=eval_seeds, test=50 held-out",
            "cv_mean": round(cv_mean, 6),
            "cv_std": round(cv_std, 6),
            "cv_se": round(cv_st["se"], 6),
            "cv_ci_lower": round(cv_st["ci_lower"], 6),
            "cv_ci_upper": round(cv_st["ci_upper"], 6),
            "held_out_seeds": held_out_size,
            "held_out_base_seed": held_out_base,
            "held_out_mean": round(held_out_mean, 6),
            "held_out_std": round(held_out_std, 6),
            "held_out_scores": [round(s, 6) for s in held_out_scores],
        }
        best_params_path = run_dir / "best_params.json"
        with open(best_params_path, "w") as f:
            json.dump(best_params, f, indent=2)
        total_time = time.time() - run_start
        print()
        print(f"Tuning complete in {total_time/60:.1f} min")
        print(f"Best k       : {best_k}")
        print(f"Best value   : {best_value:.4f}")
        print(f"CV mean±std  : {cv_mean:.4f} ± {cv_std:.4f}")
        print(f"Held-out val : {held_out_mean:.4f} ± {held_out_std:.4f}")
        print(f"Best params  → {best_params_path}")
        return 0

    # ── Optuna path (SA, NSGA-II, SGA) ───────────────────────────────────────
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print(
            "ERROR: Optuna is not installed.\n"
            "Install it with:  pip install optuna\n"
            "Then re-run this script.",
            file=sys.stderr,
        )
        return 1

    csv_path = run_dir / "trials.csv"
    print(f"Trials       : {args.trials}")

    with open(csv_path, "w", newline="") as csv_file:
        writer_ref: list = [None]

        def csv_callback(study, trial):
            if trial.value is None:
                return
            row = {"trial": trial.number, "value": round(trial.value, 6), **trial.params}
            if writer_ref[0] is None:
                writer_ref[0] = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
                writer_ref[0].writeheader()
            writer_ref[0].writerow(row)
            csv_file.flush()
            best = study.best_value
            elapsed = time.time() - run_start
            print(
                f"  trial {trial.number:3d}/{args.trials}  "
                f"value={trial.value:.4f}  best={best:.4f}  "
                f"params={trial.params}  elapsed={elapsed:.0f}s"
            )

        study_name = args.study_name or run_name
        study = optuna.create_study(
            direction="minimize",
            study_name=study_name,
            storage=args.storage,
            sampler=optuna.samplers.TPESampler(seed=42),
            load_if_exists=bool(args.storage),
        )

        if args.algo == "sa":
            obj = lambda trial: objective_sa(
                trial, args, evaluator, generate_random_map, improve_balance_spatial
            )
        elif args.algo == "nsga2":
            obj = lambda trial: objective_nsga2(
                trial, args, evaluator, generate_random_map, nsga2_optimize
            )
        else:
            obj = lambda trial: objective_sga(
                trial, args, evaluator, generate_random_map, sga_optimize
            )

        run_start = time.time()
        study.optimize(obj, n_trials=args.trials, callbacks=[csv_callback],
                       n_jobs=args.workers)

    convergence_path = run_dir / "optuna_convergence.csv"
    trials_sorted = sorted(study.trials, key=lambda t: t.number)
    best_so_far = float('inf')
    with open(convergence_path, "w", newline="") as cf:
        cw = csv.DictWriter(cf, fieldnames=["trial", "value", "best_so_far"])
        cw.writeheader()
        for t in trials_sorted:
            if t.value is not None and t.value < best_so_far:
                best_so_far = t.value
            cw.writerow({"trial": t.number,
                         "value": round(t.value, 6) if t.value is not None else "",
                         "best_so_far": round(best_so_far, 6)})
    print(f"\nOptuna convergence → {convergence_path}")

    best_p = study.best_params

    # Re-run best trial to get per-seed scores for cv_mean, cv_std
    cv_scores = []
    for i in range(args.eval_seeds):
        seed = args.base_seed + i
        m = generate_random_map(
            player_count=args.players, template_name="normal",
            include_pok=True, random_seed=seed,
        )
        if args.algo == "sa":
            s, _, _ = improve_balance_spatial(
                m, evaluator, iterations=args.iter_budget,
                initial_acceptance_rate=best_p["initial_acceptance_rate"],
                min_temp=best_p["min_temp"],
                random_seed=seed, verbose=False,
            )
            cv_scores.append(s.composite_score())
        elif args.algo == "nsga2":
            gens, pop = _nsga2_budget(args.iter_budget)
            front = nsga2_optimize(
                m, evaluator, generations=gens, population_size=pop,
                blob_fraction=best_p["blob_fraction"],
                mutation_rate=best_p["mutation_rate"],
                warm_fraction=0,
                random_seed=seed, verbose=False,
            )
            cv_scores.append(min(f[1].composite_score() for f in front))
        else:
            gens, pop = _nsga2_budget(args.iter_budget)
            s, _ = sga_optimize(
                m, evaluator, generations=gens, population_size=pop,
                blob_fraction=best_p["blob_fraction"],
                mutation_rate=best_p["mutation_rate"],
                warm_fraction=best_p["warm_fraction"],
                random_seed=seed, verbose=False,
            )
            cv_scores.append(s.composite_score())
        del m
        gc.collect()
    cv_st = _stats_from_scores(cv_scores)

    print(f"\n--- Held-out validation ({held_out_size} seeds: "
          f"{held_out_base}–{held_out_base + held_out_size - 1}) ---")
    held_out_scores = []
    for seed in held_out_seeds:
        m = generate_random_map(
            player_count=args.players, template_name="normal",
            include_pok=True, random_seed=seed,
        )
        if args.algo == "sa":
            s, _, _ = improve_balance_spatial(
                m, evaluator, iterations=args.iter_budget,
                initial_acceptance_rate=best_p["initial_acceptance_rate"],
                min_temp=best_p["min_temp"],
                random_seed=seed, verbose=False,
            )
            held_out_scores.append(s.composite_score())
        elif args.algo in ("nsga2", "sga"):
            gens, pop = _nsga2_budget(args.iter_budget)
            if args.algo == "nsga2":
                front = nsga2_optimize(
                    m, evaluator, generations=gens, population_size=pop,
                    blob_fraction=best_p["blob_fraction"],
                    mutation_rate=best_p["mutation_rate"],
                    warm_fraction=0,
                    random_seed=seed, verbose=False,
                )
                held_out_scores.append(min(f[1].composite_score() for f in front))
            else:
                s, _ = sga_optimize(
                    m, evaluator, generations=gens, population_size=pop,
                    blob_fraction=best_p["blob_fraction"],
                    mutation_rate=best_p["mutation_rate"],
                    warm_fraction=best_p["warm_fraction"],
                    random_seed=seed, verbose=False,
                )
                held_out_scores.append(s.composite_score())
        del m
        gc.collect()

    held_out_mean = statistics.mean(held_out_scores)
    held_out_std = statistics.stdev(held_out_scores) if len(held_out_scores) > 1 else 0.0
    print(f"  Held-out mean: {held_out_mean:.4f} ± {held_out_std:.4f}")

    best_params_path = run_dir / "best_params.json"
    best_params = {
        "algorithm": args.algo,
        "best_value": round(study.best_value, 6),
        "best_params": {k: round(v, 6) for k, v in study.best_params.items()},
        "n_trials": args.trials,
        "eval_seeds": args.eval_seeds,
        "iter_budget": args.iter_budget,
        "train_test_split": "train=eval_seeds, test=50 held-out",
        "cv_mean": round(cv_st["mean"], 6),
        "cv_std": round(cv_st["std"], 6),
        "cv_se": round(cv_st["se"], 6),
        "cv_ci_lower": round(cv_st["ci_lower"], 6),
        "cv_ci_upper": round(cv_st["ci_upper"], 6),
        "held_out_seeds": held_out_size,
        "held_out_base_seed": held_out_base,
        "held_out_mean": round(held_out_mean, 6),
        "held_out_std": round(held_out_std, 6),
        "held_out_scores": [round(s, 6) for s in held_out_scores],
    }
    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=2)

    total_time = time.time() - run_start
    print()
    print(f"Tuning complete in {total_time/60:.1f} min")
    print(f"Best value    : {study.best_value:.4f}")
    print(f"Best params   : {study.best_params}")
    print(f"CV mean±std   : {cv_st['mean']:.4f} ± {cv_st['std']:.4f}")
    print(f"Held-out val  : {held_out_mean:.4f} ± {held_out_std:.4f} (train/test split, no CV)")
    print(f"Best params   → {best_params_path}")
    print(f"Trial history → {csv_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

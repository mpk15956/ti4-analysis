"""
Batch experiment runner for comparing basic vs spatial optimization.

This module implements the core experimental loop:
1. Generate N random maps
2. Measure initial (naive) balance and spatial metrics
3. Optimize each map using the basic balance optimizer
4. Measure final (optimized) balance and spatial metrics
5. Save results for statistical analysis
"""

import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np

from ..algorithms.balance_engine import TI4Map, improve_balance, analyze_balance
from ..algorithms.map_generator import generate_random_map
from ..spatial_stats.spatial_metrics import comprehensive_spatial_analysis
from ..data.map_structures import Evaluator, PlanetEvalStrategy


def create_joebrew_evaluator() -> Evaluator:
    """Create the standard 'Joebrew' evaluator used in experiments."""
    return Evaluator(
        name="Joebrew",
        PLANET_STRATEGY=PlanetEvalStrategy.GREATEST,
        BASE_PLANET_MOD=-1.0,
        RESOURCES_MULTIPLIER=3.0,
        INFLUENCE_MULTIPLIER=2.0,
        TECH_MOD=5.0,
        NONZERO_RESOURCES_MOD=1.0,
        NONZERO_INFLUENCE_MOD=1.0,
        MULTI_PLANET_MOD=1.0,
        MATCHING_PLANETS_MOD=1.0,
        MECATOL_REX_SYS_MOD=6.0,
        LEGENDARY_PLANET_SYS_MOD=6.0,
        SPACE_STATION_SYS_MOD=5.0,
        DISTANCE_MULTIPLIER=[6.0, 6.0, 6.0, 4.0, 4.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    )


def run_single_experiment(
    map_id: int,
    evaluator: Evaluator,
    player_count: int = 6,
    template_name: str = "normal",
    include_pok: bool = True,
    optimization_iterations: int = 200,
    random_seed: Optional[int] = None,
    verbose: bool = False
) -> Dict:
    """
    Run a single experimental trial: generate map, optimize, measure metrics.

    Args:
        map_id: Unique identifier for this map
        evaluator: Balance evaluator to use
        player_count: Number of players
        template_name: Board template variant
        include_pok: Include PoK expansion
        optimization_iterations: Number of iterations for improve_balance
        random_seed: Random seed for map generation
        verbose: Print progress messages

    Returns:
        Dictionary with all metrics before and after optimization
    """
    if verbose:
        print(f"[Map {map_id}] Generating random map...")

    # Generate naive (random) map
    naive_map = generate_random_map(
        player_count=player_count,
        template_name=template_name,
        include_pok=include_pok,
        random_seed=random_seed
    )

    # Measure naive metrics
    if verbose:
        print(f"[Map {map_id}] Analyzing naive map...")

    naive_balance = analyze_balance(naive_map, evaluator)
    naive_spatial = comprehensive_spatial_analysis(naive_map, evaluator)

    # Calculate accessibility variance (not included in comprehensive_spatial_analysis)
    naive_accessibility_variance = float(np.var(naive_spatial['home_accessibilities']))

    # Optimize map (basic balance only)
    if verbose:
        print(f"[Map {map_id}] Optimizing map (basic balance)...")

    optimized_map = naive_map.copy()
    start_time = time.time()
    final_gap, history = improve_balance(
        optimized_map,
        evaluator,
        iterations=optimization_iterations,
        random_seed=random_seed
    )
    optimization_time = time.time() - start_time

    # Measure optimized metrics
    if verbose:
        print(f"[Map {map_id}] Analyzing optimized map...")

    optimized_balance = analyze_balance(optimized_map, evaluator)
    optimized_spatial = comprehensive_spatial_analysis(optimized_map, evaluator)

    optimized_accessibility_variance = float(np.var(optimized_spatial['home_accessibilities']))

    # Compile results
    result = {
        'map_id': map_id,
        'random_seed': random_seed,

        # Configuration
        'player_count': player_count,
        'template_name': template_name,
        'optimization_iterations': optimization_iterations,
        'optimization_time_sec': optimization_time,

        # Naive (before) metrics - Basic
        'naive_balance_gap': naive_balance['balance_gap'],
        'naive_mean_value': naive_balance['mean'],
        'naive_std_value': naive_balance['std'],
        'naive_fairness_index': naive_balance.get('fairness_index', None),

        # Naive (before) metrics - Spatial
        'naive_morans_i': naive_spatial['resource_clustering_morans_i'],
        'naive_jains_index': naive_spatial['jains_fairness_index'],
        'naive_gini_coefficient': naive_spatial['gini_coefficient'],
        'naive_num_hotspots': naive_spatial['num_hotspots'],
        'naive_num_coldspots': naive_spatial['num_coldspots'],
        'naive_accessibility_variance': naive_accessibility_variance,

        # Optimized (after) metrics - Basic
        'optimized_balance_gap': optimized_balance['balance_gap'],
        'optimized_mean_value': optimized_balance['mean'],
        'optimized_std_value': optimized_balance['std'],
        'optimized_fairness_index': optimized_balance.get('fairness_index', None),

        # Optimized (after) metrics - Spatial
        'optimized_morans_i': optimized_spatial['resource_clustering_morans_i'],
        'optimized_jains_index': optimized_spatial['jains_fairness_index'],
        'optimized_gini_coefficient': optimized_spatial['gini_coefficient'],
        'optimized_num_hotspots': optimized_spatial['num_hotspots'],
        'optimized_num_coldspots': optimized_spatial['num_coldspots'],
        'optimized_accessibility_variance': optimized_accessibility_variance,

        # Change metrics
        'delta_balance_gap': optimized_balance['balance_gap'] - naive_balance['balance_gap'],
        'delta_morans_i': optimized_spatial['resource_clustering_morans_i'] - naive_spatial['resource_clustering_morans_i'],
        'delta_jains_index': optimized_spatial['jains_fairness_index'] - naive_spatial['jains_fairness_index'],
        'delta_gini': optimized_spatial['gini_coefficient'] - naive_spatial['gini_coefficient'],

        # Convergence info
        'num_iterations': len(history),
        'converged': final_gap < 1.0,  # Define convergence as gap < 1.0
    }

    if verbose:
        print(f"[Map {map_id}] Complete! Gap: {naive_balance['balance_gap']:.2f} -> {optimized_balance['balance_gap']:.2f}")

    return result


def run_batch_experiment(
    num_maps: int,
    evaluator: Optional[Evaluator] = None,
    player_count: int = 6,
    template_name: str = "normal",
    include_pok: bool = True,
    optimization_iterations: int = 200,
    base_seed: Optional[int] = None,
    output_dir: Optional[Path] = None,
    experiment_name: Optional[str] = None,
    verbose: bool = True,
    save_intermediate: bool = True
) -> pd.DataFrame:
    """
    Run a batch of experiments comparing naive vs optimized maps.

    Args:
        num_maps: Number of random maps to generate and optimize
        evaluator: Balance evaluator (uses Joebrew if None)
        player_count: Number of players per map
        template_name: Board template variant
        include_pok: Include PoK expansion
        optimization_iterations: Number of optimization iterations per map
        base_seed: Base random seed (each map gets base_seed + i)
        output_dir: Directory to save results (uses default if None)
        experiment_name: Name for this experiment (auto-generated if None)
        verbose: Print progress messages
        save_intermediate: Save results after each map (for crash recovery)

    Returns:
        DataFrame with all experimental results
    """
    if evaluator is None:
        evaluator = create_joebrew_evaluator()

    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"experiment_{timestamp}_n{num_maps:04d}"

    if output_dir is None:
        # Default to ti4-analysis/results/
        output_dir = Path(__file__).parents[3] / "results" / experiment_name
    else:
        output_dir = Path(output_dir) / experiment_name

    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 80)
        print(f"BATCH EXPERIMENT: {experiment_name}")
        print("=" * 80)
        print(f"Maps to generate: {num_maps}")
        print(f"Player count: {player_count}")
        print(f"Template: {template_name}")
        print(f"Optimization iterations: {optimization_iterations}")
        print(f"Base seed: {base_seed}")
        print(f"Output directory: {output_dir}")
        print("=" * 80)

    results = []
    start_time = time.time()

    for i in range(num_maps):
        if verbose:
            elapsed = time.time() - start_time
            if i > 0:
                avg_time_per_map = elapsed / i
                eta = avg_time_per_map * (num_maps - i)
                print(f"\nProgress: {i}/{num_maps} ({i/num_maps*100:.1f}%) | "
                      f"Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m")

        seed = (base_seed + i) if base_seed is not None else None

        try:
            result = run_single_experiment(
                map_id=i,
                evaluator=evaluator,
                player_count=player_count,
                template_name=template_name,
                include_pok=include_pok,
                optimization_iterations=optimization_iterations,
                random_seed=seed,
                verbose=verbose
            )
            results.append(result)

            # Save intermediate results
            if save_intermediate:
                df_temp = pd.DataFrame(results)
                intermediate_file = output_dir / "raw_data" / "results_intermediate.csv"
                intermediate_file.parent.mkdir(parents=True, exist_ok=True)
                df_temp.to_csv(intermediate_file, index=False)

        except Exception as e:
            print(f"ERROR on map {i}: {e}")
            if verbose:
                import traceback
                traceback.print_exc()

    # Final results
    df = pd.DataFrame(results)

    # Save final results
    final_file = output_dir / "raw_data" / f"results_n{num_maps:04d}.csv"
    final_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(final_file, index=False)

    total_time = time.time() - start_time

    if verbose:
        print("\n" + "=" * 80)
        print(f"EXPERIMENT COMPLETE")
        print("=" * 80)
        print(f"Total maps processed: {len(results)}/{num_maps}")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Average time per map: {total_time/len(results):.1f} seconds")
        print(f"Results saved to: {final_file}")
        print("=" * 80)

    return df


def run_multi_scale_experiment(
    sample_sizes: List[int] = [10, 50, 200],
    base_seed: int = 42,
    output_dir: Optional[Path] = None,
    experiment_name: Optional[str] = None,
    **kwargs
) -> Dict[int, pd.DataFrame]:
    """
    Run experiments at multiple sample sizes (N values).

    This allows comparison of statistical power and effect stability
    across different experiment scales.

    Args:
        sample_sizes: List of N values to test (e.g., [10, 50, 200, 1000])
        base_seed: Base random seed
        output_dir: Output directory for all results
        experiment_name: Name for this multi-scale experiment
        **kwargs: Additional arguments for run_batch_experiment

    Returns:
        Dictionary mapping N -> DataFrame of results
    """
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"multi_scale_{timestamp}"

    if output_dir is None:
        output_dir = Path(__file__).parents[3] / "results" / experiment_name
    else:
        output_dir = Path(output_dir) / experiment_name

    print("=" * 80)
    print(f"MULTI-SCALE EXPERIMENT: {experiment_name}")
    print("=" * 80)
    print(f"Sample sizes to test: {sample_sizes}")
    print("=" * 80)

    all_results = {}

    for n in sample_sizes:
        print(f"\n{'='*80}")
        print(f"Running experiment with N={n}")
        print(f"{'='*80}")

        df = run_batch_experiment(
            num_maps=n,
            base_seed=base_seed,
            output_dir=output_dir,
            experiment_name=f"n{n:04d}",
            **kwargs
        )

        all_results[n] = df

    print("\n" + "=" * 80)
    print("MULTI-SCALE EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"Experiments completed for N = {sample_sizes}")
    print(f"All results saved to: {output_dir}")
    print("=" * 80)

    return all_results

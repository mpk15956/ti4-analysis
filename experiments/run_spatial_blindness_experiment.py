#!/usr/bin/env python3
"""
Spatial Blindness Experiment - Main Orchestration Script

This script runs the complete experimental pipeline to test whether
the "basic" balance optimizer is spatially blind.

Usage:
    python run_spatial_blindness_experiment.py --sample-sizes 10 50 200
    python run_spatial_blindness_experiment.py --quick  # Run N=10 only
    python run_spatial_blindness_experiment.py --full   # Run N=10,50,200,1000
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

from ti4_analysis.evaluation.batch_experiment import run_multi_scale_experiment, create_joebrew_evaluator
from ti4_analysis.evaluation.analysis import (
    analyze_experiment_results,
    test_spatial_blindness,
    find_smoking_gun_cases,
    compute_summary_statistics,
    print_analysis_summary
)
from ti4_analysis.visualization.experiment_viz import create_all_experiment_visualizations
from ti4_analysis.evaluation.report_generator import generate_markdown_report


def main():
    parser = argparse.ArgumentParser(
        description='Run the Spatial Blindness Experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick validation run (N=10, ~5 minutes)
  python run_spatial_blindness_experiment.py --quick

  # Medium scale (N=50, ~25 minutes)
  python run_spatial_blindness_experiment.py --sample-sizes 50

  # Multi-scale experiment (N=10,50,200, ~2 hours)
  python run_spatial_blindness_experiment.py --sample-sizes 10 50 200

  # Full publication-quality (N=10,50,200,1000, ~8 hours)
  python run_spatial_blindness_experiment.py --full

  # Custom configuration
  python run_spatial_blindness_experiment.py --sample-sizes 100 --iterations 300 --seed 12345
        """
    )

    parser.add_argument(
        '--sample-sizes', '-n',
        type=int,
        nargs='+',
        help='List of sample sizes to test (e.g., 10 50 200)'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick validation run (N=10 only)'
    )

    parser.add_argument(
        '--full',
        action='store_true',
        help='Full publication-quality experiment (N=10,50,200,1000)'
    )

    parser.add_argument(
        '--iterations', '-i',
        type=int,
        default=200,
        help='Number of optimization iterations per map (default: 200)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Base random seed (default: 42)'
    )

    parser.add_argument(
        '--player-count', '-p',
        type=int,
        default=6,
        choices=[2, 3, 4, 5, 6, 7, 8],
        help='Number of players (default: 6)'
    )

    parser.add_argument(
        '--template', '-t',
        type=str,
        default='normal',
        help='Board template name (default: normal)'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        help='Output directory for results (default: ti4-analysis/results/)'
    )

    parser.add_argument(
        '--no-pok',
        action='store_true',
        help='Exclude Prophecy of Kings expansion tiles'
    )

    args = parser.parse_args()

    # Determine sample sizes
    if args.quick:
        sample_sizes = [10]
    elif args.full:
        sample_sizes = [10, 50, 200, 1000]
    elif args.sample_sizes:
        sample_sizes = sorted(args.sample_sizes)
    else:
        # Default: medium scale
        sample_sizes = [50]

    # Print experiment configuration
    print("=" * 80)
    print("SPATIAL BLINDNESS EXPERIMENT")
    print("=" * 80)
    print(f"Sample sizes: {sample_sizes}")
    print(f"Player count: {args.player_count}")
    print(f"Template: {args.template}")
    print(f"Optimization iterations: {args.iterations}")
    print(f"Random seed: {args.seed}")
    print(f"Include PoK: {not args.no_pok}")
    print("=" * 80)

    # Estimate time
    total_maps = sum(sample_sizes)
    est_time_min = total_maps * 0.5  # ~30 seconds per map
    print(f"\nEstimated time: ~{est_time_min:.0f} minutes ({est_time_min/60:.1f} hours)")
    print("\nPress Ctrl+C to cancel, or Enter to continue...")

    try:
        input()
    except KeyboardInterrupt:
        print("\n\nExperiment cancelled.")
        return

    # Create evaluator
    evaluator = create_joebrew_evaluator()

    # Run multi-scale experiment
    print("\n" + "=" * 80)
    print("PHASE 1: BATCH EXPERIMENTS")
    print("=" * 80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"spatial_blindness_{timestamp}"

    all_results = run_multi_scale_experiment(
        sample_sizes=sample_sizes,
        base_seed=args.seed,
        output_dir=args.output_dir,
        experiment_name=experiment_name,
        player_count=args.player_count,
        template_name=args.template,
        include_pok=not args.no_pok,
        optimization_iterations=args.iterations,
        verbose=True
    )

    # Analyze each experiment
    print("\n" + "=" * 80)
    print("PHASE 2: STATISTICAL ANALYSIS")
    print("=" * 80)

    for n, df in all_results.items():
        print(f"\n{'='*80}")
        print(f"Analyzing N={n} experiment...")
        print(f"{'='*80}")

        # Statistical tests
        paired_results = analyze_experiment_results(df)
        correlation_results = test_spatial_blindness(df)
        smoking_guns = find_smoking_gun_cases(df)
        summary_stats = compute_summary_statistics(df)

        # Print summary
        print_analysis_summary(paired_results, correlation_results, smoking_guns)

        # Generate visualizations
        print("\n" + "=" * 80)
        print(f"PHASE 3: VISUALIZATIONS (N={n})")
        print("=" * 80)

        output_dir = args.output_dir / experiment_name / f"n{n:04d}" if args.output_dir else Path(__file__).parent.parent / "ti4-analysis" / "results" / experiment_name / f"n{n:04d}"

        figure_paths = create_all_experiment_visualizations(
            df=df,
            paired_results=paired_results,
            correlation_results=correlation_results,
            output_dir=output_dir
        )

        # Generate report
        print("\n" + "=" * 80)
        print(f"PHASE 4: REPORT GENERATION (N={n})")
        print("=" * 80)

        metadata = {
            'Sample Size': n,
            'Player Count': args.player_count,
            'Template': args.template,
            'Optimization Iterations': args.iterations,
            'Random Seed': args.seed,
            'Include PoK': not args.no_pok,
            'Timestamp': timestamp,
        }

        report_path = generate_markdown_report(
            df=df,
            paired_results=paired_results,
            correlation_results=correlation_results,
            summary_stats=summary_stats,
            smoking_guns=smoking_guns,
            figure_paths=figure_paths,
            output_dir=output_dir,
            experiment_name=f"{experiment_name} (N={n})",
            metadata=metadata
        )

        print(f"\n✓ Report generated: {report_path}")

    # Final summary
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to:")
    base_output = args.output_dir / experiment_name if args.output_dir else Path(__file__).parent.parent / "ti4-analysis" / "results" / experiment_name
    print(f"  {base_output}")
    print(f"\nGenerated reports:")
    for n in sample_sizes:
        print(f"  - N={n:4d}: {base_output / f'n{n:04d}' / 'SPATIAL_BLINDNESS_REPORT.md'}")

    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("1. Review the markdown reports for detailed findings")
    print("2. Examine the visualizations in the figures/ directories")
    print("3. Consider implementing multi-objective optimization based on results")
    print("4. Share findings with the TI4 community!")
    print("=" * 80)


if __name__ == '__main__':
    main()

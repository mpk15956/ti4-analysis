"""
Quick pipeline validation test.

This script performs a minimal N=10 experiment to verify all components work together.
Run this before executing large-scale experiments.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        # Data structures
        from ti4_analysis.data.map_structures import (
            Planet, System, MapSpace, Evaluator, PlanetEvalStrategy
        )
        print("  ✓ map_structures")

        # Algorithms
        from ti4_analysis.algorithms.hex_grid import HexCoord
        print("  ✓ hex_grid")

        from ti4_analysis.algorithms.balance_engine import TI4Map, improve_balance
        print("  ✓ balance_engine")

        # Spatial metrics
        from ti4_analysis.spatial_stats.spatial_metrics import comprehensive_spatial_analysis
        print("  ✓ spatial_metrics")

        # Visualization
        from ti4_analysis.visualization.map_viz import plot_hex_map
        print("  ✓ map_viz")

        # Experiments
        from ti4_analysis.evaluation.batch_experiment import run_batch_experiment
        from ti4_analysis.evaluation.analysis import analyze_experiment_results
        print("  ✓ experiments")

        print("\n✓ All imports successful!")
        return True

    except Exception as e:
        print(f"\n✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tile_loading():
    """Test tile database loading."""
    print("\nTesting tile loading...")

    try:
        from ti4_analysis.data.tile_loader import load_tile_database, load_board_template

        # Try to load tile database
        print("  Loading tile database...")
        db = load_tile_database(use_cache=False)

        print(f"  ✓ Loaded {len(db.tiles)} tiles")
        print(f"    - Base: {len(db.base_tiles)}")
        print(f"    - PoK: {len(db.pok_tiles)}")
        print(f"    - Blue: {len(db.blue_tiles)}")
        print(f"    - Red: {len(db.red_tiles)}")

        # Load board template
        print("\n  Loading board template...")
        template = load_board_template(player_count=6, template_name="normal")
        print(f"  ✓ Loaded 6-player normal template")
        print(f"    - Home positions: {len(template['home_worlds'])}")
        print(f"    - Swappable positions: {len(template['primary_tiles']) + len(template['secondary_tiles']) + len(template['tertiary_tiles'])}")

        return True

    except Exception as e:
        print(f"\n✗ Tile loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_map_generation():
    """Test random map generation."""
    print("\nTesting map generation...")

    try:
        from ti4_analysis.algorithms.map_generator import generate_random_map

        print("  Generating random map...")
        ti4_map = generate_random_map(
            player_count=6,
            template_name="normal",
            include_pok=True,
            random_seed=42
        )

        print(f"  ✓ Generated map with {len(ti4_map.spaces)} spaces")
        print(f"    - Home spaces: {len(ti4_map.get_home_spaces())}")
        print(f"    - System spaces: {len(ti4_map.get_system_spaces())}")

        return True

    except Exception as e:
        print(f"\n✗ Map generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_experiment():
    """Test a single experiment run."""
    print("\nTesting single experiment...")

    try:
        from ti4_analysis.evaluation.batch_experiment import (
            run_single_experiment,
            create_joebrew_evaluator
        )

        evaluator = create_joebrew_evaluator()

        print("  Running single experiment (this may take ~30 seconds)...")
        result = run_single_experiment(
            map_id=0,
            evaluator=evaluator,
            player_count=6,
            template_name="normal",
            include_pok=True,
            optimization_iterations=50,  # Reduced for speed
            random_seed=42,
            verbose=False
        )

        print("  ✓ Experiment completed")
        print(f"    - Balance gap: {result['naive_balance_gap']:.2f} → {result['optimized_balance_gap']:.2f}")
        print(f"    - Moran's I: {result['naive_morans_i']:.3f} → {result['optimized_morans_i']:.3f}")

        return True

    except Exception as e:
        print(f"\n✗ Single experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mini_batch():
    """Test a mini batch experiment (N=3)."""
    print("\nTesting mini batch experiment (N=3)...")

    try:
        from ti4_analysis.evaluation.batch_experiment import run_batch_experiment
        from ti4_analysis.evaluation.analysis import analyze_experiment_results
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"  Running batch experiment...")
            df = run_batch_experiment(
                num_maps=3,
                optimization_iterations=50,
                base_seed=42,
                output_dir=Path(tmpdir),
                verbose=False
            )

            print(f"  ✓ Generated {len(df)} results")

            # Test analysis
            print("  Running statistical analysis...")
            paired_results = analyze_experiment_results(df)

            print(f"  ✓ Analyzed {len(paired_results)} metrics")

        return True

    except Exception as e:
        print(f"\n✗ Mini batch failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("="*80)
    print("PIPELINE VALIDATION TEST")
    print("="*80)

    tests = [
        ("Imports", test_imports),
        ("Tile Loading", test_tile_loading),
        ("Map Generation", test_map_generation),
        ("Single Experiment", test_single_experiment),
        ("Mini Batch", test_mini_batch),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(results.values())

    print("="*80)

    if all_passed:
        print("\n🎉 All tests passed! Pipeline is ready for full experiments.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix errors before running experiments.")
        return 1


if __name__ == '__main__':
    sys.exit(main())

from pathlib import Path

from ti4_analysis.algorithms.map_generator import generate_random_map
from ti4_analysis.algorithms.balance_engine import create_joebrew_evaluator
from ti4_analysis.benchmarking.run_experiment import ExperimentConfig, run_full_experiment

BUDGETS = [500, 2000, 8000]
TRIALS = 5
ALGORITHMS = ['G1', 'G1-MO', 'G3-D', 'G5']

for budget in BUDGETS:
    output_dir = Path(f"ti4-analysis/results/final_mo_experiment/budget_{budget}")
    output_dir.mkdir(parents=True, exist_ok=True)

    starting_map = generate_random_map(player_count=6, random_seed=42)
    evaluator = create_joebrew_evaluator()

    config = ExperimentConfig(
        algorithms=ALGORITHMS,
        evaluation_budgets=[budget],
        n_trials=TRIALS,
        base_seed=700 + budget,
        output_dir=str(output_dir),
        save_maps=False,
        save_checkpoints=False,
        n_jobs=1,
    )

    print(f"=== Running final multi-objective comparison at budget {budget} ===")
    run_full_experiment(starting_map=starting_map, evaluator=evaluator, config=config, verbose=True)
    print()

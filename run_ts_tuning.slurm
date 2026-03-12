#!/bin/bash
# TS tuning uses exhaustive grid search over tenure coefficient k (no Optuna).
# Runtime is predictable: 18 k-values x 100 seeds x 50k evals per seed (single node).
# Reduce --time to ~24h once you have a rough runtime; tune per cluster.
#SBATCH --job-name=ti4_ts_tune
#SBATCH --output=logs/ts_tune_%j.out
#SBATCH --error=logs/ts_tune_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16        # --workers 16; multiprocessing avoids GIL for fitness evals
#SBATCH --mem=32G                 # Enough RAM for 16 parallel FastMapState instances
#SBATCH --partition=compute       # Update to your cluster's partition name

# Load modules or activate conda/venv here
# source activate ti4_env

echo "Starting Tabu Search Hyperparameter Tuning (grid over k)"
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"

mkdir -p logs output/ts_retune_highbudget

python scripts/optimize_hyperparameters.py \
  --algo ts \
  --iter-budget 50000 \
  --eval-seeds 100 \
  --workers 16 \
  --output-dir output/ts_retune_highbudget

echo "Job completed at: $(date)"

#!/bin/bash
#SBATCH --job-name=ti4_ts_tune
#SBATCH --output=logs/ts_tune_%j.out
#SBATCH --error=logs/ts_tune_%j.err
#SBATCH --time=48:00:00           # 48 hours to be safe for 50k evals
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16        # Matches your --workers 16
#SBATCH --mem=32G                 # Ensure enough RAM for 16 parallel FastMapStates
#SBATCH --partition=compute       # Update to your cluster's partition name

# Load modules or activate conda/venv here
# source activate ti4_env

echo "Starting Tabu Search Hyperparameter Tuning"
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"

# Create log directory if it doesn't exist
mkdir -p logs output/ts_retune_highbudget

# Run the tuning script
python scripts/optimize_hyperparameters.py \
  --algo ts \
  --iter-budget 50000 \
  --trials 50 \
  --workers 16 \
  --output-dir output/ts_retune_highbudget

echo "Job completed at: $(date)"

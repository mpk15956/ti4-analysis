#!/bin/bash
#SBATCH --job-name=ti4_pintest
#SBATCH --account=geog4592
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/pintest_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
SIF="$PWD/ti4-analysis.sif"
APPTAINER="apptainer exec \
  --bind $PWD/output:/app/output --bind $PWD/src:/app/src \
  --bind $PWD/scripts:/app/scripts --bind $PWD/tests:/app/tests --pwd /app $SIF"
$APPTAINER python -m pytest -v -o addopts="" -p no:cacheprovider tests/test_phase6_canonical_values.py
echo "PINTEST DONE"

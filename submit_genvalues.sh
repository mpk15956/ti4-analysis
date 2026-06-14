#!/bin/bash
#SBATCH --job-name=ti4_genvalues
#SBATCH --account=geog4592
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/genvalues_%j.out

# Derive Phase 6/7 manuscript claim-values from the canonical finalize dir, in
# the SIF with host src/+scripts/ bound over /app (same overlay as Phase 6/7).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
SIF="$PWD/ti4-analysis.sif"
FD=output/paper1_500k_finalize_phase67_20260606_101032
APPTAINER="apptainer exec \
    --bind $PWD/output:/app/output --bind $PWD/src:/app/src --bind $PWD/scripts:/app/scripts \
    --pwd /app $SIF"
$APPTAINER python /app/scripts/generate_manuscript_values.py \
    --finalize-dir "/app/$FD" --budget 500000 \
    --out "/app/$FD/manuscript_values_phase6.json"
echo "GENVALUES DONE"

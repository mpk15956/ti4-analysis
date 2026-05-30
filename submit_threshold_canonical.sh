#!/bin/bash
#SBATCH --job-name=ti4_threshold
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mpk15956@uga.edu

# Canonical LSAP threshold-sensitivity test (Goodhart Test 3 under
# corrected landscape). Closes the "canonical-form re-run pending" footnote
# in docs/limitations/lsap-proxy-goodhart.md and limitations.md §5.3.
#
# Configuration: 50 seeds × budget 1000 (matches the legacy run scale that
# produced τ = 0.949 on the raw-I_i form). Uses canonical SA hyperparameters
# from Phase 0 of paper1_canonical (rate=0.5826, min_temp=0.001195) and the
# --corrected-landscape flag (Gen-0 σ + smooth + √k-LSAP), so the baseline
# LSAP under test is the canonical √k-stabilized form (§3.4.3).
#
# Output lands in the same paper1_canonical_* output tree alongside the
# Phase 4 LISA validation, so all canonical Goodhart artifacts share a
# common run directory.

set -euo pipefail

cleanup() {
    local exit_code=$?
    [ $exit_code -ne 0 ] && echo "ERROR: exit_code=$exit_code at line $1" >&2
    echo "Finished at: $(date -Iseconds)  exit_code=$exit_code"
}
trap 'cleanup $LINENO' EXIT

cd "$SLURM_SUBMIT_DIR"

SIF="$PWD/ti4-analysis.sif"
[ -f "$SIF" ] || { echo "FATAL: $SIF not found." >&2; exit 1; }

# Append into the existing canonical run dir so the threshold artifact lives
# next to lisa_validation_*, optuna_*, benchmark_*, dist_sensitivity_*.
CANONICAL_RUN_DIR="output/paper1_canonical_20260509_134024"

APPTAINER="apptainer exec --bind $PWD/output:/app/output --pwd /app $SIF"

# Canonical SA hyperparameters from Phase 0 best_params.json
SA_RATE=0.582644
SA_MIN_TEMP=0.001195

echo "============================================================"
echo "TI4 Canonical LSAP Threshold-Sensitivity (Goodhart Test 3)"
echo "============================================================"
echo "Host          : $(hostname)"
echo "SLURM Job ID  : ${SLURM_JOB_ID:-local}"
echo "SIF           : $SIF"
echo "Canonical run : $CANONICAL_RUN_DIR"
echo "SA params     : rate=$SA_RATE  min_temp=$SA_MIN_TEMP (canonical Phase 0)"
echo "Started at    : $(date -Iseconds)"
echo "------------------------------------------------------------"

$APPTAINER python /app/scripts/lsap_threshold_sensitivity.py \
    --seeds 50 \
    --budget 1000 \
    --tau 0.05 \
    --base-seed 0 \
    --players 6 \
    --sa-rate "$SA_RATE" \
    --sa-min-temp "$SA_MIN_TEMP" \
    --corrected-landscape \
    --output-dir "/app/$CANONICAL_RUN_DIR"

echo ""
echo "============================================================"
echo "Canonical threshold-sensitivity test complete."
echo "Artifact under: $CANONICAL_RUN_DIR/lsap_threshold_<timestamp>/"
echo "============================================================"

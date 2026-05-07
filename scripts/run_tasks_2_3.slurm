#!/bin/bash
#
# Run only Tasks 2 & 3 of the migration verification, since Task 1 (Phase 1
# C0+C4 reproducibility) already produced clean data in
#   output/verify_pixi_apptainer_20260506_205045/benchmark_*/results.csv
# and was reviewed offline (verdict: env consistent at seed=0; the SLURM
# script's previous gate compared single-seed to 100-seed *population*
# median, which can't be hit from one seed).
#
#   Task 2: scripts/analyze_benchmark.py + scripts/plot_statistical_results.py
#           on the existing Phase 2b results.csv at --budget 500000
#   Task 3: scripts/variance_equalization_diagnostic.py
#
# Both run inside ti4-analysis.sif via apptainer exec.
#
#SBATCH --job-name=ti4_tasks_2_3
#SBATCH --partition=batch
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

SIF="$PWD/ti4-analysis.sif"
[ -f "$SIF" ] || { echo "FATAL: $SIF missing" >&2; exit 1; }

RUN_TAG="verify_pixi_apptainer_20260506_205045"   # reuse Task 1's run dir
OUT="$PWD/output/${RUN_TAG}"
mkdir -p "$OUT"

APPTAINER="apptainer exec --bind $PWD/output:/app/output --pwd /app $SIF"

echo "============================================================"
echo "Tasks 2 & 3 — pixi+apptainer migration verification (continued)"
echo "============================================================"
echo "Host        : $(hostname)"
echo "SLURM JobID : ${SLURM_JOB_ID:-local}"
echo "SIF         : $SIF"
echo "Run tag     : $RUN_TAG"
echo "Started at  : $(date -Iseconds)"
echo "------------------------------------------------------------"

# ── Task 2: Phase 3 stats at budget=500k ─────────────────────────────────────
echo ""
echo "=== Task 2: Phase 3 stats at budget=500k ==="
P2B_CSV="/app/output/saturation_20260314_205919/benchmark_20260315_074617/results.csv"

$APPTAINER python /app/scripts/analyze_benchmark.py \
    "$P2B_CSV" --sensitivity --ablation --budget 500000

$APPTAINER python /app/scripts/plot_statistical_results.py \
    "$P2B_CSV" --budget 500000

echo ""
echo "Phase 3 stats refreshed at: output/saturation_20260314_205919/benchmark_20260315_074617/stats/"
ls -lt output/saturation_20260314_205919/benchmark_20260315_074617/stats/ | head -10

# ── Task 3: variance_equalization_diagnostic.py ──────────────────────────────
echo ""
echo "=== Task 3: variance equalization diagnostic ==="
$APPTAINER python /app/scripts/variance_equalization_diagnostic.py \
    --output-dir "/app/output/${RUN_TAG}"

echo ""
echo "Variance equalization output at: $OUT/variance_equalization_*.csv"
ls -lt "$OUT"/variance_equalization_*.csv 2>/dev/null | head

echo ""
echo "============================================================"
echo "Tasks 2 & 3 complete."
echo "Finished at: $(date -Iseconds)"
echo "============================================================"

#!/bin/bash
#SBATCH --job-name=ti4_500k_canary
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/500k_canary_%j.out

# Fast pre-flight for submit_500k_tail_array.sh. Runs the SAME apptainer + engine
# invocation but at budget=1000 (minutes, not 38h) on a throwaway seed/dir, to
# prove on a real compute node that: apptainer is on PATH, all CLI flags parse,
# and results.csv is produced under --output-dir. Validates everything except
# the (purely longer) 500k runtime. Delete output/canary_500k when done.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
SIF="$PWD/ti4-analysis.sif"
[ -f "$SIF" ] || { echo "FATAL: $SIF missing" >&2; exit 1; }

OUT_HOST="$PWD/output/canary_500k"
OUT_CONTAINER="/app/output/canary_500k"
rm -rf "$OUT_HOST"; mkdir -p "$OUT_HOST"

APPTAINER="apptainer exec \
    --env TI4_GIT_HASH=canary --env TI4_GIT_DIRTY=0 \
    --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 \
    --bind $PWD/output:/app/output --pwd /app $SIF"

echo "Canary on $(hostname) at $(date -Iseconds)"
$APPTAINER python /app/scripts/benchmark_engine.py \
    --algorithms rs,hc,sa,sga,nsga2,ts \
    --base-seed 999 --seeds 1 \
    --budgets 1000 \
    --players 6 --chains 3 --workers 1 \
    --sa-rate 0.550264 --sa-min-temp 0.001365 \
    --sga-blob 0.74242 --sga-mut 0.071364 --sga-warm 0.100439 \
    --nsga-blob 0.74948 --nsga-mut 0.032985 --nsga-warm 0.0 \
    --ts-k 0.5 \
    --corrected-landscape \
    --output-dir "$OUT_CONTAINER"

CSV="$(ls "$OUT_HOST"/benchmark_*/results.csv 2>/dev/null | head -1)"
if [ -f "$CSV" ]; then
    echo "CANARY OK — produced $CSV  ($(wc -l < "$CSV") lines, expect 19 = header + 6 algos x 3 chains)"
else
    echo "CANARY FAILED — no results.csv under $OUT_HOST" >&2
    exit 1
fi

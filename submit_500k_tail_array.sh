#!/bin/bash
#SBATCH --job-name=ti4_500k_tail
#SBATCH --account=geog4592
#SBATCH --partition=batch
#SBATCH --array=16-99%32
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --output=logs/500k_tail_%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mpk15956@uga.edu

# ── Recovery of the budget=500,000 tail for the Paper-1 multi-algo run ──────────
#
# Job 28399 (submit_paper1_multialgo_stage2.sh) hit the 120h walltime ceiling
# partway through the highest budget. State at failure:
#   budgets 1k–200k : 100/100 seeds complete  (12,600 banked rows — DO NOT redo)
#   budget 500k     : seeds 0–15 complete, seeds 16–99 remaining (this job)
#
# Design: one seed per array task (embarrassingly parallel). A walltime/preempt
# hit costs ONE seed, not the batch. Array index == seed, so a failed seed is
# re-run with `sbatch --array=<n> submit_500k_tail_array.sh`. Each task writes
# to its OWN --output-dir, so concurrent tasks cannot collide (the engine only
# ever writes under --output-dir; verified in benchmark_engine.py:788-790).
#
# Empirical per-seed wall at 500k (from banked seeds 0–15): median 27.2h,
# max 37.7h → 48h walltime gives headroom. Memory: banked peak 3G → 8G request.
# Each seed is single-threaded across its 6 algos × 3 chains, so 1 core/task.
#
# After all 84 finish: merge with merge_500k_tail.sh, then re-run Phase 6 + 7.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

SEED="${SLURM_ARRAY_TASK_ID}"

# Pin BLAS/OpenMP so a single-core task never oversubscribes its neighbours.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

SIF="$PWD/ti4-analysis.sif"
[ -f "$SIF" ] || { echo "FATAL: $SIF missing" >&2; exit 1; }

# Provenance: label rows with the git hash the BANKED stage-2 data carried
# (46719be). src/ti4_analysis is byte-identical between 46719be and HEAD
# (2dcf4ff) — the delta is docs/submit-scripts only — and the executed code is
# the SIF either way, so this keeps the 500k cell internally consistent.
GIT_HASH="46719be"
GIT_DIRTY=0

OUT_HOST="$PWD/output/recovery_500k_tail/seed_${SEED}"
OUT_CONTAINER="/app/output/recovery_500k_tail/seed_${SEED}"
mkdir -p "$OUT_HOST"

APPTAINER="apptainer exec \
    --env TI4_GIT_HASH=$GIT_HASH \
    --env TI4_GIT_DIRTY=$GIT_DIRTY \
    --env OMP_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    --env OPENBLAS_NUM_THREADS=1 \
    --bind $PWD/output:/app/output --pwd /app $SIF"

echo "=========================================================="
echo "500k tail recovery — seed ${SEED}"
echo "Host        : $(hostname)"
echo "Array job   : ${SLURM_ARRAY_JOB_ID}  task ${SLURM_ARRAY_TASK_ID}"
echo "Output dir  : $OUT_HOST"
echo "Started     : $(date -Iseconds)"
echo "=========================================================="

# Stage-1 tuned hyperparameters (slurped from the cached, deterministic
# best_params.json under paper1_multialgo_canonical_20260510_110952/ — the same
# values the original stage-2 job used; nsga warm is cold-start 0.0 by design).
$APPTAINER python /app/scripts/benchmark_engine.py \
    --algorithms rs,hc,sa,sga,nsga2,ts \
    --base-seed "$SEED" --seeds 1 \
    --budgets 500000 \
    --players 6 --chains 3 --workers 1 \
    --sa-rate 0.550264 --sa-min-temp 0.001365 \
    --sga-blob 0.74242 --sga-mut 0.071364 --sga-warm 0.100439 \
    --nsga-blob 0.74948 --nsga-mut 0.032985 --nsga-warm 0.0 \
    --ts-k 0.5 \
    --corrected-landscape \
    --output-dir "$OUT_CONTAINER"

echo "Finished seed ${SEED} at: $(date -Iseconds)"

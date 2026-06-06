#!/bin/bash
#SBATCH --job-name=ti4_500k_phase67
#SBATCH --account=geog4592
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/500k_phase67_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mpk15956@uga.edu

# Phase 6 + Phase 7 on the MERGED Paper-1 multi-algo dataset (after the 500k
# tail recovery, array 28494, was merged by merge_500k_tail.sh). Mirrors the
# Phase 6/7 blocks of submit_paper1_multialgo_stage2.sh but points at the
# already-complete merged benchmark dir instead of producing a new Phase 1.
#
# Run only after merge_500k_tail.sh verified budget 500k = 100/100 seeds.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

SIF="$PWD/ti4-analysis.sif"
[ -f "$SIF" ] || { echo "FATAL: $SIF missing" >&2; exit 1; }

# Bind host src/ + scripts/ over the SIF's baked /app copy so the Phase 6/7 fix
# (per-budget non-dominated IGD+ reference + vectorized igd_plus, incl. the new
# moo_indicators.py module) takes effect without rebuilding the image. The SIF
# still supplies the pinned dependency env; only the analysis code is overlaid
# (git-tracked). Editable install is setuptools-lenient, so the new module on
# the bound src/ path is importable (validated by submit_500k_validate.sh).
HOST_GIT_HASH="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
HOST_GIT_DIRTY="$([ -n "$(git status --porcelain 2>/dev/null)" ] && echo 1 || echo 0)"
APPTAINER="apptainer exec \
    --env TI4_GIT_HASH=$HOST_GIT_HASH \
    --env TI4_GIT_DIRTY=$HOST_GIT_DIRTY \
    --bind $PWD/output:/app/output \
    --bind $PWD/src:/app/src \
    --bind $PWD/scripts:/app/scripts \
    --pwd /app $SIF"

MASTER_HOST="$PWD/output/paper1_multialgo_canonical_stage2_20260526_105143/benchmark_20260526_105155"
[ -d "$MASTER_HOST" ] || { echo "FATAL: merged benchmark dir missing: $MASTER_HOST" >&2; exit 1; }
[ -f "$MASTER_HOST/results.csv" ] || { echo "FATAL: merged results.csv missing" >&2; exit 1; }

# Guard: refuse to run unless the merge marker confirms 100/100 seeds at 500k.
if [ ! -f "$PWD/output/recovery_500k_tail/.merged_done" ]; then
    echo "FATAL: merge marker output/recovery_500k_tail/.merged_done absent — run merge_500k_tail.sh first." >&2
    exit 1
fi

MASTER_CONTAINER="$(echo "$MASTER_HOST" | sed "s|^$PWD/output|/app/output|")"
PARETO_ARCHIVES_DIR="$MASTER_CONTAINER/pareto_archives"
UNIFIED_ARCHIVES_DIR="$MASTER_CONTAINER/unified_archives"

OUT_HOST="$PWD/output/paper1_500k_finalize_phase67_$(date +%Y%m%d_%H%M%S)"
OUT_CONTAINER="$(echo "$OUT_HOST" | sed "s|^$PWD/output|/app/output|")"
mkdir -p "$OUT_HOST"

BUDGETS=(1000 5000 10000 25000 50000 100000 200000 500000)

echo "=== Phase 6a: Track B quality indicators (NSGA-II Pareto archives) ==="
$APPTAINER python /app/scripts/quality_indicators.py \
    --archive-dir "$PARETO_ARCHIVES_DIR" --output-dir "$OUT_CONTAINER" --plot

echo "=== Phase 6b: Cross-method IGD ==="
$APPTAINER python /app/scripts/cross_method_igd.py \
    --run-dir "$MASTER_CONTAINER" --output-dir "$OUT_CONTAINER" \
    --algorithms sa,ts,hc,sga,rs --report

echo "=== Phase 6c: Unified HV across all budgets (single pass) ==="
# One call loads all 14,400 unified archives and computes every (algo,seed,
# budget) HV once, writing unified_hv.csv (all budgets) + a per-budget stats
# file each. The old per-budget loop reloaded everything 8x — ~8x redundant at
# 100 seeds x 3 chains. analyze_rq2 derives budgets from the CSV, so one dir is
# sufficient.
$APPTAINER python /app/scripts/unified_hv_analysis.py \
    --archive-dir "$UNIFIED_ARCHIVES_DIR" \
    --output-dir "${OUT_CONTAINER}/unified_hv_all"

echo "=== Phase 7: Statistical analysis (RQ1 + RQ2 + RQ4) ==="
for B in "${BUDGETS[@]}"; do
    echo "--- Phase 7 [budget=$B] ---"
    PER_B_DIR_HOST="${OUT_HOST}/stats_b${B}"
    mkdir -p "$PER_B_DIR_HOST"
    cp "${MASTER_HOST}/results.csv" "${PER_B_DIR_HOST}/results.csv"
    PER_B_CSV_CONTAINER="$(echo "${PER_B_DIR_HOST}/results.csv" | sed "s|^$PWD/output|/app/output|")"
    $APPTAINER python /app/scripts/analyze_benchmark.py \
        "$PER_B_CSV_CONTAINER" --budget "$B" --n-spatial 31
done

echo "--- Phase 7 [RQ2 NSGA-II HV vs scalars] ---"
$APPTAINER python /app/scripts/analyze_rq2_unified_hv.py \
    --hv-dirs "${OUT_CONTAINER}/unified_hv_all" \
    --output "${OUT_CONTAINER}/rq2_unified_hv_report.txt"

echo "--- Phase 7 [RQ3 Spearman] ---"
$APPTAINER python /app/scripts/analyze_rq3_spearman.py \
    "${MASTER_CONTAINER}/results.csv" --output "${OUT_CONTAINER}/rq3_spearman.csv"

echo "=== Phase 6/7 complete. Output: $OUT_HOST ==="
echo "PHASE67_DONE $(date -Iseconds)" > "${OUT_HOST}/.phase67_done"

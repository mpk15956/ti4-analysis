#!/bin/bash
#SBATCH --job-name=ti4_rq4_finalize
#SBATCH --account=geog4592
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/rq4_finalize_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mpk15956@uga.edu

# RQ4 close, fully in-SIF on GACRC (the canonical reproduction of what was first
# assembled on elis). From the banked six-algo run + the NSGA-II evals_to_best
# targeted fill (job 28684), this:
#   1. verifies the cross-container seam (fresh .sif reproduces banked NSGA-II
#      maps bit-identically) — fail-loud gate; refuses to splice on a drift,
#   2. splices the fresh evals_to_best onto the banked dataset (minimal edit),
#   3. assembles a finalize dir: copies the unchanged Phase 6 artifacts
#      (quality_indicators / unified_hv / rq3_spearman) from the 500k finalize,
#      runs analyze_benchmark per budget <=200k on the merged CSV,
#   4. regenerates manuscript_values_phase6.json (RQ4 at 200k; RQ1/RQ2 at 500k),
#   5. runs the Phase 6 canonical pins.
# Run after the close-out commits are pulled onto this cluster (the overlay binds
# host scripts/ + tests/, so the --rq4-budget generator arg + RQ4_BUDGET pins
# must be present in the checked-out tree).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs

export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

SIF="$PWD/ti4-analysis.sif"
[ -f "$SIF" ] || { echo "FATAL: $SIF missing" >&2; exit 1; }

# Inputs (host paths). Adjust if the run dirs differ.
BANKED="output/paper1_multialgo_canonical_stage2_20260526_105143/benchmark_20260526_105155/results.csv"
FRESH="output/rq4_nsga2_fill_20260615_001023/benchmark_20260615_001027/results.csv"
PHASE6_FD="output/paper1_500k_finalize_phase67_20260606_101032"
for f in "$BANKED" "$FRESH" "$PHASE6_FD/quality_indicators.csv" \
         "$PHASE6_FD/unified_hv_all/unified_hv.csv" "$PHASE6_FD/rq3_spearman.csv"; do
    [ -e "$f" ] || { echo "FATAL: missing input $f" >&2; exit 1; }
done

OUT="output/paper1_500k_finalize_phase67_$(date +%Y%m%d_%H%M%S)_rq4splice"
MERGED="$OUT/merged_six_way_results.csv"
mkdir -p "$OUT/unified_hv_all"

APPTAINER="apptainer exec \
    --bind $PWD/output:/app/output --bind $PWD/src:/app/src \
    --bind $PWD/scripts:/app/scripts --bind $PWD/tests:/app/tests --pwd /app $SIF"
c() { echo "/app/$1"; }  # host rel path -> container path

echo "=== 1. cross-container seam check (fail-loud gate) ==="
$APPTAINER python /app/scripts/rq4_verify_cross_container.py "$(c "$FRESH")" "$(c "$BANKED")"

echo "=== 2. splice -> merged six-way results.csv ==="
$APPTAINER python /app/scripts/rq4_build_merged_results.py "$(c "$BANKED")" "$(c "$FRESH")" "$(c "$MERGED")"

echo "=== 3. assemble finalize dir (copy unchanged Phase 6 artifacts) ==="
cp "$PHASE6_FD/quality_indicators.csv" "$OUT/quality_indicators.csv"
cp "$PHASE6_FD/rq3_spearman.csv" "$OUT/rq3_spearman.csv"
cp "$PHASE6_FD/unified_hv_all/unified_hv.csv" "$OUT/unified_hv_all/unified_hv.csv"

echo "=== 3b. per-budget stats from the merged CSV (six-way RQ4) ==="
for B in 1000 5000 10000 25000 50000 100000 200000; do
    PB="$OUT/stats_b${B}"; mkdir -p "$PB"; cp "$MERGED" "$PB/results.csv"
    $APPTAINER python /app/scripts/analyze_benchmark.py "$(c "$PB/results.csv")" \
        --budget "$B" --n-spatial 31
done

echo "=== 4. regenerate manuscript_values_phase6.json (RQ4 @200k, RQ1/RQ2 @500k) ==="
$APPTAINER python /app/scripts/generate_manuscript_values.py \
    --finalize-dir "$(c "$OUT")" --budget 500000 --rq4-budget 200000 \
    --out "$(c "$OUT/manuscript_values_phase6.json")"

echo "=== 5. Phase 6 canonical pins ==="
$APPTAINER python -m pytest -v -o addopts="" -p no:cacheprovider \
    /app/tests/test_phase6_canonical_values.py

echo "RQ4_FINALIZE_DONE $(date -Iseconds) -> $OUT"

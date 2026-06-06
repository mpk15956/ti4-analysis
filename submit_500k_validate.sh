#!/bin/bash
#SBATCH --job-name=ti4_500k_validate
#SBATCH --account=geog4592
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/500k_validate_%j.out

# Cheap gate before the real Phase 6/7 re-run. Confirms, inside the SIF with the
# host src/+scripts/ bound over the baked /app copy:
#   (1) the new moo_indicators module imports (editable-install lenient mode),
#   (2) the vectorized igd_plus + nondominated_filter equal the original loops,
#   (3) the per-budget non-dominated reference is small + the filter is fast
#       (so Phase 6/7 is tractable and walltime can be set with real numbers).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

SIF="$PWD/ti4-analysis.sif"
[ -f "$SIF" ] || { echo "FATAL: $SIF missing" >&2; exit 1; }
APPTAINER="apptainer exec \
    --bind $PWD/output:/app/output \
    --bind $PWD/src:/app/src \
    --bind $PWD/scripts:/app/scripts \
    --bind $PWD/tests:/app/tests \
    --pwd /app $SIF"

echo "=== (1) import check ==="
$APPTAINER python -c "from ti4_analysis.algorithms.moo_indicators import igd_plus, nondominated_filter; print('IMPORT OK')"

echo "=== (2) equivalence tests (vectorized == reference loops) ==="
# Clear pyproject addopts (drops --cov, which tries to write /app/.coverage on
# the read-only SIF fs) and disable the cache provider (also read-only /app).
$APPTAINER python -m pytest -q -o addopts="" -p no:cacheprovider tests/test_moo_indicators.py

echo "=== (3) Phase 6c smoke: unified_hv_analysis on b100000 (the budget that crashed) ==="
RUN=/app/output/paper1_multialgo_canonical_stage2_20260526_105143/benchmark_20260526_105155
rm -rf output/validate_uhv_b100000
$APPTAINER python /app/scripts/unified_hv_analysis.py \
    --archive-dir "$RUN/unified_archives" \
    --output-dir /app/output/validate_uhv_b100000 \
    --budget 100000
echo "--- one-row-per-(algo,seed) check at b100000 (chains must aggregate, not duplicate) ---"
$APPTAINER python -c "
import csv
rows=[r for r in csv.DictReader(open('/app/output/validate_uhv_b100000/unified_hv.csv')) if int(r['budget'])==100000]
keys=set((r['algorithm'],r['seed']) for r in rows)
print(f'  rows@b100000={len(rows)}  distinct (algo,seed)={len(keys)}  one-per-seed={len(rows)==len(keys)}')
assert len(rows)==len(keys), 'DUPLICATE (algo,seed) rows -> chains not aggregated!'
print('  PHASE 6c CHAIN AGGREGATION OK')
"
rm -rf output/validate_uhv_b100000
echo "VALIDATE DONE"

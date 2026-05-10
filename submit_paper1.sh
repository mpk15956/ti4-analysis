#!/bin/bash
#SBATCH --job-name=ti4_paper1
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mpk15956@uga.edu

# Paper 1 (spatial-stats methodology) canonical re-run, under apptainer.
#
# SCOPE. This orchestrator runs ONLY the phases whose artifacts feed the
# geography paper, all under --corrected-landscape (canonical formulation):
#
#     Phase 0  SA hyperparameter tuning, corrected landscape (~6 hr)
#     Phase 1  5-condition ablation, corrected landscape (~12-24 hr)
#     Phase 4  LISA proxy validation, formulation-independent (~3 hr)
#     Phase 5  distance-weight sensitivity, corrected landscape (~6 hr)
#
# PHASES OMITTED (vs submit_all.sh): Phase 2a/2b/3/6/6b — Paper 2 territory.
#
# REPRODUCIBILITY. All phases run inside ti4-analysis.sif (the pixi+apptainer
# container from commit 916df74 / cc7318b). Mirrors the established pattern
# in scripts/verify_pixi_apptainer.slurm: bind host's output/ over /app/output
# so writes persist; the container's /app/{scripts,src,data,.pixi} is the
# canonical code+env.

set -euo pipefail

cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "ERROR: Script failed with exit code $exit_code at line $1" >&2
    fi
    echo "Finished at: $(date -Iseconds)  exit_code=$exit_code"
}
trap 'cleanup $LINENO' EXIT

cd "$SLURM_SUBMIT_DIR"

# ── Container ────────────────────────────────────────────────────────────────
SIF="$PWD/ti4-analysis.sif"
if [ ! -f "$SIF" ]; then
    echo "FATAL: $SIF not found. Run scripts/build_sif.slurm first." >&2
    exit 1
fi

# Forward git state from host into the container so run_config.json records
# the host's git_hash (the SIF has no git on PATH and no .git bind-mounted).
# See run_config._git_state and feedback_canonical_objective_single_source.md.
HOST_GIT_HASH="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
HOST_GIT_DIRTY="$([ -n "$(git status --porcelain 2>/dev/null)" ] && echo 1 || echo 0)"
APPTAINER="apptainer exec \
    --env TI4_GIT_HASH=$HOST_GIT_HASH \
    --env TI4_GIT_DIRTY=$HOST_GIT_DIRTY \
    --bind $PWD/output:/app/output --pwd /app $SIF"

# ── Configuration ─────────────────────────────────────────────────────────────
RUN_TAG="paper1_canonical_$(date +%Y%m%d_%H%M%S)"
HOST_OUTPUT_DIR="$PWD/output/${RUN_TAG}"
CONTAINER_OUTPUT_DIR="/app/output/${RUN_TAG}"
WORKERS=16
mkdir -p "$HOST_OUTPUT_DIR"

SEEDS=100
BUDGETS="1000,5000,10000,25000,50000,100000,200000,500000"
PLAYERS=6

TUNING_TRIALS=50
TUNING_SEEDS=100
TUNING_BASE_SEED=9000

LISA_VALIDATION_SEEDS=30
LISA_VALIDATION_ITER=10000
LISA_PERMS=9999

DIST_SENSITIVITY_SEEDS=50
DIST_SENSITIVITY_BUDGET=10000

echo "============================================================"
echo "TI4 Paper 1 Canonical Re-run — Geography Methodology Paper"
echo "============================================================"
echo "Host          : $(hostname)"
echo "SLURM Job ID  : ${SLURM_JOB_ID:-local}"
echo "SIF           : $SIF ($(stat -c '%s bytes, mtime=%y' $SIF))"
echo "Run tag       : $RUN_TAG"
echo "Host output   : $HOST_OUTPUT_DIR"
echo "Container out : $CONTAINER_OUTPUT_DIR"
echo "Seeds         : $SEEDS"
echo "Budgets       : $BUDGETS"
echo "Tuning seeds  : $TUNING_SEEDS"
echo "Workers       : $WORKERS"
echo "Git hash      : $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
echo "Started at    : $(date -Iseconds)"
echo "Formulation   : --corrected-landscape (canonical)"
echo "------------------------------------------------------------"
$APPTAINER python --version
$APPTAINER python -c "
import numpy, scipy
print(f'  numpy      {numpy.__version__}')
print(f'  scipy      {scipy.__version__}')
try:
    import optuna; print(f'  optuna     {optuna.__version__}')
except ImportError:
    print('  optuna     NOT INSTALLED (Phase 0 will fail)')
"
echo "============================================================"

# ── Phase 0: SA hyperparameter tuning under canonical landscape ──────────────
echo ""
echo "--- Phase 0: SA hyperparameter tuning (${TUNING_TRIALS} trials, seeds ${TUNING_BASE_SEED}+, corrected) ---"
$APPTAINER python /app/scripts/optimize_hyperparameters.py \
    --algo sa \
    --trials "$TUNING_TRIALS" \
    --eval-seeds "$TUNING_SEEDS" \
    --base-seed "$TUNING_BASE_SEED" \
    --players "$PLAYERS" \
    --corrected-landscape \
    --output-dir "$CONTAINER_OUTPUT_DIR"

echo ""
echo "--- Extracting SA tuned hyperparameters (for Phase 1) ---"
read SA_RATE SA_MIN_TEMP < <(
$APPTAINER python -c "
import json, glob, sys
params = {}
for path in sorted(glob.glob('${CONTAINER_OUTPUT_DIR}/optuna_*/best_params.json')):
    with open(path) as f:
        d = json.load(f)
    params[d['algorithm']] = d['best_params']
if 'sa' not in params:
    print('FATAL: No SA tuning results found', file=sys.stderr)
    sys.exit(1)
sa = params['sa']
print(sa['initial_acceptance_rate'], sa['min_temp'])
"
)
echo "  SA: rate=$SA_RATE  min_temp=$SA_MIN_TEMP"
echo "  (under canonical landscape; expect 5-30% shifts vs legacy uncorrected best_params)"

# Health check: cv_std/cv_mean comparable to legacy regime — guards against
# tuning failure under the smooth landscape.
$APPTAINER python -c "
import json, glob
for path in sorted(glob.glob('${CONTAINER_OUTPUT_DIR}/optuna_*/best_params.json')):
    with open(path) as f:
        d = json.load(f)
    cv_mean, cv_std = d.get('cv_mean'), d.get('cv_std')
    if cv_mean and cv_std:
        cv_pct = (cv_std / cv_mean) * 100
        print(f'  cv coefficient of variation = {cv_pct:.1f}%  (sub-30% indicates healthy convergence)')
"

# ── Phase 1: Main experiment (5-condition ablation, SA, canonical) ───────────
echo ""
echo "--- Phase 1: Main experiment — 5-condition ablation (${SEEDS} seeds, SA, canonical) ---"
$APPTAINER python /app/scripts/benchmark_engine.py \
    --conditions jfi_only,moran_only,lsap_only,jfi_moran,full_composite \
    --seeds "$SEEDS" \
    --budgets "$BUDGETS" \
    --players "$PLAYERS" \
    --workers "$WORKERS" \
    --chains 3 \
    --sa-rate "$SA_RATE" \
    --sa-min-temp "$SA_MIN_TEMP" \
    --corrected-landscape \
    --output-dir "$CONTAINER_OUTPUT_DIR"

PRIMARY_RESULTS_CSV=$(find "$HOST_OUTPUT_DIR" -name "results.csv" -path "*/benchmark_*" | sort | tail -1)
echo "Primary results CSV (Phase 1): $PRIMARY_RESULTS_CSV"

# ── Phase 4: Post-hoc LISA proxy validation (formulation-independent) ────────
echo ""
echo "--- Phase 4: LISA proxy validation (${LISA_VALIDATION_SEEDS} seeds, ${LISA_PERMS} perms) ---"
$APPTAINER python /app/scripts/validate_lisa_proxy.py \
    --seeds "$LISA_VALIDATION_SEEDS" \
    --algorithms "sa" \
    --sa-iter "$LISA_VALIDATION_ITER" \
    --hc-iter "$LISA_VALIDATION_ITER" \
    --ts-iter "$LISA_VALIDATION_ITER" \
    --sa-rate "$SA_RATE" \
    --sa-min-temp "$SA_MIN_TEMP" \
    --n-perms "$LISA_PERMS" \
    --workers "$WORKERS" \
    --output-dir "$CONTAINER_OUTPUT_DIR"

# ── Phase 5: Distance-weight sensitivity (canonical) ─────────────────────────
echo ""
echo "--- Phase 5: Distance-weight sensitivity (${DIST_SENSITIVITY_SEEDS} seeds, canonical) ---"
$APPTAINER python /app/scripts/distance_weight_sensitivity.py \
    --seeds "$DIST_SENSITIVITY_SEEDS" \
    --budget "$DIST_SENSITIVITY_BUDGET" \
    --algorithms "sa" \
    --workers "$WORKERS" \
    --sa-rate "$SA_RATE" \
    --sa-min-temp "$SA_MIN_TEMP" \
    --corrected-landscape \
    --output-dir "$CONTAINER_OUTPUT_DIR"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Paper 1 canonical re-run complete."
echo "Host output dir         : $HOST_OUTPUT_DIR"
echo "Primary results (P1)   : $PRIMARY_RESULTS_CSV"
echo "------------------------------------------------------------"
echo "Tuned SA (canonical):    rate=$SA_RATE  min_temp=$SA_MIN_TEMP"
echo "------------------------------------------------------------"
echo "Next steps:"
echo "  1. Verify run_config.json sidecars: every artifact should record"
echo "     corrected_landscape=True with matching git_hash and metric file hashes."
echo "  2. Author MANIFEST entries against the new canonical artifacts;"
echo "     fill in the §3.7 (B) PENDING_CANONICAL_PHASE_0 placeholders."
echo "  3. Run scripts/pre_submission_check.sh — should now go green."
echo "============================================================"

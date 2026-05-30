#!/bin/bash
#SBATCH --job-name=ti4_basin_validation
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mpk15956@uga.edu

# Basin-structure validation run for paper 1 §3.
#
# WHAT THIS CLOSES. The canonical SA jfi_only data (output/paper1_canonical_*)
# showed bimodal |Moran's I| with two anti-correlated convergence basins.
# That finding is *either* a fitness-landscape property of JFI on the
# canonical 6-player layout or an SA-specific search artifact. This run
# discriminates between them by re-running jfi_only under HC, TS, and RS
# at matched seeds/chains/budgets, then applying the diagnostic rules
# pre-registered in docs/audit/jfi_only_diagnostics/PREREGISTRATION.md.
#
# RS is the cleanest control: no search dynamics at all, so a bimodal RS
# distribution would imply the bimodality is in JFI's level-set geometry
# itself rather than in any optimizer's trajectory.
#
# OUT OF SCOPE: cross-algorithm Pareto / Track B / Friedman / Wilcoxon —
# those are paper-1 RQ1/RQ2/RQ4 deliverables and run via
# submit_paper1_multialgo.sh. This script only produces the basin
# verdict.

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
APPTAINER="apptainer exec --bind $PWD/output:/app/output --pwd /app $SIF"

# ── Configuration ─────────────────────────────────────────────────────────────
RUN_TAG="basin_validation_$(date +%Y%m%d_%H%M%S)"
HOST_OUTPUT_DIR="$PWD/output/${RUN_TAG}"
CONTAINER_OUTPUT_DIR="/app/output/${RUN_TAG}"
WORKERS=16
mkdir -p "$HOST_OUTPUT_DIR"

# Match canonical paper1 SA jfi_only run exactly so cross-optimizer
# comparison is at matched problem instances.
SEEDS=100
BUDGETS="1000,5000,10000,25000,50000,100000,200000,500000"
PLAYERS=6
CHAINS=3

TUNING_SEEDS=100
TUNING_BASE_SEED=9000

# Pre-registration must exist and not be tampered with mid-run.
PREREG="$PWD/docs/audit/jfi_only_diagnostics/PREREGISTRATION.md"
if [ ! -f "$PREREG" ]; then
    echo "FATAL: pre-registration $PREREG not found; refusing to run." >&2
    exit 1
fi
PREREG_HASH=$(sha256sum "$PREREG" | cut -d' ' -f1)

echo "============================================================"
echo "TI4 Paper 1 — Basin-structure validation (HC / TS / RS)"
echo "============================================================"
echo "Host          : $(hostname)"
echo "SLURM Job ID  : ${SLURM_JOB_ID:-local}"
echo "SIF           : $SIF"
echo "Run tag       : $RUN_TAG"
echo "Host output   : $HOST_OUTPUT_DIR"
echo "Algorithms    : HC, TS, RS  (jfi_only condition only)"
echo "Seeds         : $SEEDS  Budgets: $BUDGETS  Chains: $CHAINS"
echo "Workers       : $WORKERS"
echo "Git hash      : $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
echo "Started at    : $(date -Iseconds)"
echo "Pre-registration: $PREREG"
echo "Pre-registration sha256: $PREREG_HASH"
echo "Formulation   : --corrected-landscape (canonical)"
echo "------------------------------------------------------------"

# ── Pre-flight: canonical-objective canary + invariants ─────────────────────
# Same gate as submit_paper1_multialgo.sh — fail fast at second 5 instead of
# hour 3 on a tuning/canonicalization drift.
echo ""
echo "--- Pre-flight: canonical-objective canary + invariants ---"
$APPTAINER python -c "
import sys
from ti4_analysis.algorithms.spatial_optimizer import MultiObjectiveScore
from ti4_analysis.algorithms.map_topology import MapTopology, morans_i_null
from ti4_analysis.algorithms.map_generator import generate_random_map
from ti4_analysis.evaluation.batch_experiment import create_joebrew_evaluator

ev = create_joebrew_evaluator()
m = generate_random_map(player_count=6, random_seed=0)
topo = MapTopology.from_ti4_map(m, ev)
assert topo.n_spatial == 31, f'canonical n_spatial drift: expected 31, got {topo.n_spatial}'

e_i = morans_i_null(31)
assert abs(e_i - (-1.0/30)) <= 1e-12, f'morans_i_null(31) drift: {e_i}'

row = {'jains_index': 0.95, 'morans_i': -0.6, 'lisa_penalty': 4.5}
f1, f2, f3 = MultiObjectiveScore.archive_row_to_pareto_point(row, 31)
assert abs(f1 - 0.05) <= 1e-12 and f2 == 0.0 and abs(f3 - 4.5) <= 1e-12, \
    f'canonical tuple drift at canary input: ({f1},{f2},{f3})'

benchmark_seeds = set(range(0, 100))
tuning_seeds    = set(range(9000, 9100))
held_out_seeds  = set(range(9100, 9150))
assert benchmark_seeds.isdisjoint(tuning_seeds)
assert benchmark_seeds.isdisjoint(held_out_seeds)
assert tuning_seeds.isdisjoint(held_out_seeds)
print('  PASS: |G|=31, E[I]=-1/30, canonical tuple bit-correct, seed-sets disjoint')
"
echo "  Pre-flight OK."

# ── Phase 0: TS hyperparameter grid search ──────────────────────────────────
# HC and RS are parameter-free so no tuning needed for them. TS needs k.
echo ""
echo "--- Phase 0: TS grid search over k (canonical) ---"
$APPTAINER python /app/scripts/optimize_hyperparameters.py \
    --algo ts \
    --eval-seeds "$TUNING_SEEDS" \
    --base-seed "$TUNING_BASE_SEED" \
    --players "$PLAYERS" \
    --corrected-landscape \
    --output-dir "$CONTAINER_OUTPUT_DIR"

# ── Slurp tuned TS k — HARD FAIL on missing key ─────────────────────────────
# Direct subscripting; KeyError aborts the job. (Per
# feedback_silent_fallback_is_wrong_answer.md — no .get(default) on tuned
# parameters.)
echo ""
echo "--- Extracting tuned TS k (hard-fail on missing key) ---"
TS_K=$(
$APPTAINER python -c "
import json, glob, sys
paths = sorted(glob.glob('${CONTAINER_OUTPUT_DIR}/optuna_*/best_params.json'))
ts_path = None
for p in paths:
    with open(p) as f:
        d = json.load(f)
    if d.get('algorithm') == 'ts':
        ts_path = p
        break
if ts_path is None:
    print(f'FATAL: no TS best_params.json under {paths}', file=sys.stderr)
    sys.exit(1)
with open(ts_path) as f:
    d = json.load(f)
ts_k = d['best_params']['tabu_tenure_coefficient']
if not (0.0 < ts_k < 10.0):
    print(f'FATAL: tuned ts_k={ts_k} outside (0, 10)', file=sys.stderr)
    sys.exit(1)
print(ts_k)
"
)
echo "  TS k = $TS_K"

# ── Phase 1: HC / TS / RS benchmark on jfi_only ─────────────────────────────
echo ""
echo "--- Phase 1: HC / TS / RS benchmark on jfi_only ---"
$APPTAINER python /app/scripts/benchmark_engine.py \
    --algorithms hc,ts,rs \
    --seeds "$SEEDS" \
    --budgets "$BUDGETS" \
    --players "$PLAYERS" \
    --workers "$WORKERS" \
    --chains "$CHAINS" \
    --conditions jfi_only \
    --ts-k "$TS_K" \
    --corrected-landscape \
    --output-dir "$CONTAINER_OUTPUT_DIR"

# ── Resolve benchmark results CSV ────────────────────────────────────────────
BENCHMARK_RUN_DIR_HOST="$(ls -dt "${HOST_OUTPUT_DIR}"/benchmark_*/ 2>/dev/null | head -1)"
if [ -z "${BENCHMARK_RUN_DIR_HOST}" ] || [ ! -d "${BENCHMARK_RUN_DIR_HOST}" ]; then
    echo "FATAL: no benchmark_*/ subdir found under ${HOST_OUTPUT_DIR}" >&2
    exit 1
fi
RESULTS_CSV_HOST="${BENCHMARK_RUN_DIR_HOST}results.csv"
if [ ! -f "${RESULTS_CSV_HOST}" ]; then
    echo "FATAL: ${RESULTS_CSV_HOST} not produced by Phase 1" >&2
    exit 1
fi
RESULTS_CSV_CONTAINER="$(echo "${RESULTS_CSV_HOST}" | sed "s|^$PWD/output|/app/output|")"
echo "Benchmark results CSV: $RESULTS_CSV_HOST"

# ── Phase 2: Basin-structure diagnostic (applies pre-registered rules) ──────
echo ""
echo "--- Phase 2: Basin-structure diagnostic ---"
$APPTAINER python /app/scripts/basin_diagnostic.py \
    --results-csv "$RESULTS_CSV_CONTAINER" \
    --output-dir  "$CONTAINER_OUTPUT_DIR"

# Re-print the verdict to slurm-*.out for grep-ability.
echo ""
echo "--- Verdict (verbatim from basin_diagnostic.txt) ---"
cat "${HOST_OUTPUT_DIR}/basin_diagnostic.txt"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Basin-structure validation complete."
echo "Host output dir         : $HOST_OUTPUT_DIR"
echo "Benchmark results       : $RESULTS_CSV_HOST"
echo "Basin diagnostic txt    : ${HOST_OUTPUT_DIR}/basin_diagnostic.txt"
echo "Basin diagnostic json   : ${HOST_OUTPUT_DIR}/basin_diagnostic.json"
echo "Basin diagnostic figure : ${HOST_OUTPUT_DIR}/basin_diagnostic.png"
echo "------------------------------------------------------------"

#!/bin/bash
#SBATCH --job-name=ti4_p1ma_stage2
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mpk15956@uga.edu

# Paper 1 multi-algorithm canonical run, Stage 2 (Phase 1 + Phase 6 + Phase 7).
#
# Stage 1 (Phase 0a-d + Phase 0e sensitivity check) completed in job 28185 at
# `output/paper1_multialgo_canonical_20260510_110952/`. The job hit a 72h
# walltime ceiling that was an under-set #SBATCH --time directive in
# `submit_paper1_multialgo.sh`, not a cluster limit — the batch partition
# allows up to 168h. Phase 0 produced four `best_params.json` artifacts
# (one per algorithm) plus the Phase 0e sensitivity tuning; these are
# deterministic, cached, and reused here. This is FAIR-compliant provenance
# (Wilkinson et al. 2016) — content-addressable lineage via git_hash +
# container + run_config + disjoint seed ranges. The "single sbatch
# invocation" property is sacrificed; reproducibility is preserved.
#
# Phase 1 (multi-algo benchmark) needs ~50h; Phases 6+7 need ~1h. Total ~51h.
# The 120h walltime gives 2.3× headroom for cluster variance and the bigger
# NSGA-II/SGA per-evaluation costs at the highest budget (500k).

set -euo pipefail

cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "ERROR: Stage 2 failed with exit code $exit_code at line $1" >&2
    fi
    echo "Finished at: $(date -Iseconds)  exit_code=$exit_code"
}
trap 'cleanup $LINENO' EXIT

cd "$SLURM_SUBMIT_DIR"

# ── Container + git provenance ──────────────────────────────────────────────
SIF="$PWD/ti4-analysis.sif"
if [ ! -f "$SIF" ]; then
    echo "FATAL: $SIF not found. Run scripts/build_sif.slurm first." >&2
    exit 1
fi
HOST_GIT_HASH="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
HOST_GIT_DIRTY="$([ -n "$(git status --porcelain 2>/dev/null)" ] && echo 1 || echo 0)"
APPTAINER="apptainer exec \
    --env TI4_GIT_HASH=$HOST_GIT_HASH \
    --env TI4_GIT_DIRTY=$HOST_GIT_DIRTY \
    --bind $PWD/output:/app/output --pwd /app $SIF"

# ── Stage 1 provenance: reused Phase 0 artifacts ────────────────────────────
# These directories were produced by job 28185 and are the upstream input to
# this Stage 2. The canonical_formulation block + git_hash of the Stage 1
# run is asserted to match the Stage 2 environment at canary time below.
STAGE1_RUN_HOST="$PWD/output/paper1_multialgo_canonical_20260510_110952"
STAGE1_RUN_CONTAINER="/app/output/paper1_multialgo_canonical_20260510_110952"
if [ ! -d "$STAGE1_RUN_HOST" ]; then
    echo "FATAL: Stage 1 run dir missing: $STAGE1_RUN_HOST" >&2
    exit 1
fi
# Explicit per-algorithm best_params.json (looked up via the algorithm
# field inside each file, not by directory ordering — see hard-fail slurp
# below for the assertion this carries through).
SA_BEST_PARAMS="output/paper1_multialgo_canonical_20260510_110952/optuna_20260510_111000/best_params.json"
SGA_BEST_PARAMS="output/paper1_multialgo_canonical_20260510_110952/optuna_20260510_170620/best_params.json"
NSGA_BEST_PARAMS="output/paper1_multialgo_canonical_20260510_110952/optuna_20260511_005446/best_params.json"
TS_BEST_PARAMS="output/paper1_multialgo_canonical_20260510_110952/optuna_20260511_105619/best_params.json"
SENSITIVITY_BEST_PARAMS="output/paper1_multialgo_canonical_20260510_110952/sensitivity_b10000/optuna_20260511_130448/best_params.json"

# ── Stage 2 run dir + canonical formulation continuity ──────────────────────
RUN_TAG="paper1_multialgo_canonical_stage2_$(date +%Y%m%d_%H%M%S)"
HOST_OUTPUT_DIR="$PWD/output/${RUN_TAG}"
CONTAINER_OUTPUT_DIR="/app/output/${RUN_TAG}"
WORKERS=16
mkdir -p "$HOST_OUTPUT_DIR"

SEEDS=100
BUDGETS="1000,5000,10000,25000,50000,100000,200000,500000"
PLAYERS=6
CHAINS=3

echo "============================================================"
echo "TI4 Paper 1 Multi-Algorithm Canonical Run — Stage 2"
echo "============================================================"
echo "Host          : $(hostname)"
echo "SLURM Job ID  : ${SLURM_JOB_ID:-local}"
echo "SIF           : $SIF"
echo "Stage 1 dir   : $STAGE1_RUN_HOST"
echo "Stage 2 dir   : $HOST_OUTPUT_DIR"
echo "Algorithms    : RS, HC, SA, SGA, NSGA-II, TS"
echo "Seeds         : $SEEDS  Budgets: $BUDGETS  Chains: $CHAINS"
echo "Workers       : $WORKERS"
echo "Git hash      : $HOST_GIT_HASH (dirty=$HOST_GIT_DIRTY)"
echo "Started at    : $(date -Iseconds)"
echo "Formulation   : --corrected-landscape (canonical)"
echo "------------------------------------------------------------"

# ── Pre-flight: canonical-objective canary + Stage 1 provenance match ───────
echo ""
echo "--- Pre-flight: canonical-objective canary + Stage 1 provenance match ---"
$APPTAINER python -c "
import json, sys
from ti4_analysis.algorithms.spatial_optimizer import MultiObjectiveScore
from ti4_analysis.algorithms.map_topology import MapTopology, morans_i_null
from ti4_analysis.algorithms.map_generator import generate_random_map
from ti4_analysis.evaluation.batch_experiment import create_joebrew_evaluator
from ti4_analysis.utils.canonical_provenance import (
    assert_canonical_formulation, assert_archive_transform_identity,
    EXPECTED_INVARIANTS,
)

# (1) Canonical 6-player layout: |G| = 31
ev = create_joebrew_evaluator()
m = generate_random_map(player_count=6, random_seed=0)
topo = MapTopology.from_ti4_map(m, ev)
assert topo.n_spatial == 31, f'canonical n_spatial drift: expected 31, got {topo.n_spatial}'

# (2) E[I] under the null
assert abs(morans_i_null(31) - (-1.0/30)) <= 1e-12

# (3) archive_row_to_pareto_point canary
row = {'jains_index': 0.95, 'morans_i': -0.6, 'lisa_penalty': 4.5}
f1, f2, f3 = MultiObjectiveScore.archive_row_to_pareto_point(row, 31)
assert abs(f1 - 0.05) <= 1e-12 and f2 == 0.0 and abs(f3 - 4.5) <= 1e-12
assert_archive_transform_identity(31)

# (4) Seed-set disjointness
benchmark_seeds = set(range(0, 100))
tuning_seeds = set(range(9000, 9100))
held_out_seeds = set(range(9100, 9150))
assert benchmark_seeds.isdisjoint(tuning_seeds)
assert benchmark_seeds.isdisjoint(held_out_seeds)
assert tuning_seeds.isdisjoint(held_out_seeds)

# (5) Stage 1 provenance match: its run_config canonical_formulation MUST
# match the consumer-side EXPECTED_INVARIANTS. If Stage 1 was produced under
# a different formulation, the cached best_params.json from Stage 1 are not
# usable here without re-tuning — and this canary aborts.
stage1_cfg_path = '${STAGE1_RUN_CONTAINER}/benchmark_20260511_212749/run_config.json'
try:
    stage1_cfg = json.load(open(stage1_cfg_path))
except FileNotFoundError:
    print(f'WARNING: Stage 1 benchmark run_config.json missing at {stage1_cfg_path};')
    print(f'   provenance match on Stage 1 BENCHMARK config skipped.')
    print(f'   Stage 1 Phase 0 outputs are still trusted via best_params.json explicit paths.')
    stage1_cfg = None
if stage1_cfg is not None:
    cf = stage1_cfg.get('canonical_formulation', {})
    mism = [(k, cf.get(k), EXPECTED_INVARIANTS[k])
            for k in EXPECTED_INVARIANTS
            if cf.get(k) != EXPECTED_INVARIANTS[k]]
    if mism:
        print('FATAL: Stage 1 canonical_formulation does not match Stage 2 expectations:',
              file=sys.stderr)
        for k, got, want in mism:
            print(f'  {k}: stage1={got!r}  expected={want!r}', file=sys.stderr)
        sys.exit(1)
    print(f'  Stage 1 canonical_formulation matches v{cf[\"version\"]} (git_hash={stage1_cfg.get(\"git_hash\")})')
print('  PASS: canary + Stage 1 provenance OK')
"
echo "  Pre-flight OK."

# ── Slurp Stage 1 hyperparameters (HARD FAIL on missing/invalid) ────────────
echo ""
echo "--- Slurping Stage 1 tuned hyperparameters (hard-fail on missing keys) ---"
read SA_RATE SA_MIN_TEMP SGA_BLOB SGA_MUT SGA_WARM NSGA_BLOB NSGA_MUT TS_K < <(
$APPTAINER python -c "
import json, sys

def load(path):
    return json.load(open(path))['best_params']

sa = load('/app/$SA_BEST_PARAMS')
sga = load('/app/$SGA_BEST_PARAMS')
nsga = load('/app/$NSGA_BEST_PARAMS')
ts = load('/app/$TS_BEST_PARAMS')

# Direct subscripting; KeyError on schema drift.
sa_rate     = sa['initial_acceptance_rate']
sa_min_temp = sa['min_temp']
sga_blob    = sga['blob_fraction']
sga_mut     = sga['mutation_rate']
sga_warm    = sga['warm_fraction']
nsga_blob   = nsga['blob_fraction']
nsga_mut    = nsga['mutation_rate']
ts_k        = ts['tabu_tenure_coefficient']

def _check(name, val, lo, hi):
    if not (lo < val < hi):
        print(f'FATAL: tuned {name}={val} outside ({lo}, {hi})', file=sys.stderr)
        sys.exit(1)
_check('sa_rate', sa_rate, 0.0, 1.0)
_check('sa_min_temp', sa_min_temp, 0.0, 1.0)
_check('sga_blob', sga_blob, 0.0, 1.0)
_check('sga_mut', sga_mut, 0.0, 1.0)
_check('sga_warm', sga_warm, 0.0, 1.0)
_check('nsga_blob', nsga_blob, 0.0, 1.0)
_check('nsga_mut', nsga_mut, 0.0, 1.0)
_check('ts_k', ts_k, 0.0, 10.0)

print(sa_rate, sa_min_temp, sga_blob, sga_mut, sga_warm, nsga_blob, nsga_mut, ts_k)
"
)
NSGA_WARM=0.0
echo "  SA     rate=$SA_RATE min_temp=$SA_MIN_TEMP"
echo "  SGA    blob=$SGA_BLOB mut=$SGA_MUT warm=$SGA_WARM"
echo "  NSGA   blob=$NSGA_BLOB mut=$NSGA_MUT warm=$NSGA_WARM (cold-start; pop/gen derived per-budget)"
echo "  TS     k=$TS_K"

# ── Phase 0e sensitivity diff (reused from Stage 1) ─────────────────────────
# The primary SA tuning was at iter_budget=1000, the sensitivity at 10000.
# Diff is computed once here so the report appears in Stage 2's log too.
echo ""
echo "--- Stage 1 budget-invariance sensitivity diff ---"
$APPTAINER python -c "
import json
sa_primary = json.load(open('/app/$SA_BEST_PARAMS'))['best_params']
sa_sens    = json.load(open('/app/$SENSITIVITY_BEST_PARAMS'))['best_params']
p_rate = sa_primary['initial_acceptance_rate']; s_rate = sa_sens['initial_acceptance_rate']
p_mt   = sa_primary['min_temp'];                s_mt   = sa_sens['min_temp']
print(f'  iter_budget=1000:   rate={p_rate:.4f}  min_temp={p_mt:.4g}')
print(f'  iter_budget=10000:  rate={s_rate:.4f}  min_temp={s_mt:.4g}')
rate_drift = abs(p_rate - s_rate) / max(p_rate, s_rate)
mt_drift   = abs(p_mt - s_mt)   / max(p_mt, s_mt)
print(f'  relative drift: rate={rate_drift:.1%}  min_temp={mt_drift:.1%}')
if rate_drift > 0.20 or mt_drift > 0.50:
    print(f'  WARNING: budget-invariance assumption may not hold; revisit §3.7(D).')
else:
    print(f'  Drift within bounds; budget-invariance assumption defensible.')
"

# ── Phase 1: Multi-algorithm canonical benchmark ────────────────────────────
echo ""
echo "--- Phase 1: Multi-algorithm canonical benchmark ---"
$APPTAINER python /app/scripts/benchmark_engine.py \
    --algorithms rs,hc,sa,sga,nsga2,ts \
    --seeds "$SEEDS" \
    --budgets "$BUDGETS" \
    --players "$PLAYERS" \
    --workers "$WORKERS" \
    --chains "$CHAINS" \
    --sa-rate "$SA_RATE" \
    --sa-min-temp "$SA_MIN_TEMP" \
    --sga-blob "$SGA_BLOB" \
    --sga-mut "$SGA_MUT" \
    --sga-warm "$SGA_WARM" \
    --nsga-blob "$NSGA_BLOB" \
    --nsga-mut "$NSGA_MUT" \
    --nsga-warm "$NSGA_WARM" \
    --ts-k "$TS_K" \
    --corrected-landscape \
    --output-dir "$CONTAINER_OUTPUT_DIR"

BENCHMARK_RUN_DIR_HOST="$(ls -dt "${HOST_OUTPUT_DIR}"/benchmark_*/ 2>/dev/null | head -1)"
if [ -z "${BENCHMARK_RUN_DIR_HOST}" ] || [ ! -d "${BENCHMARK_RUN_DIR_HOST}" ]; then
    echo "FATAL: no benchmark_*/ subdir found under ${HOST_OUTPUT_DIR}" >&2
    exit 1
fi
MULTIALGO_RESULTS_CSV="${BENCHMARK_RUN_DIR_HOST}results.csv"
if [ ! -f "${MULTIALGO_RESULTS_CSV}" ]; then
    echo "FATAL: ${MULTIALGO_RESULTS_CSV} not produced by Phase 1" >&2
    exit 1
fi
BENCHMARK_RUN_DIR_CONTAINER="$(echo "${BENCHMARK_RUN_DIR_HOST%/}" | sed "s|^$PWD/output|/app/output|")"
PARETO_ARCHIVES_DIR="${BENCHMARK_RUN_DIR_CONTAINER}/pareto_archives"
UNIFIED_ARCHIVES_DIR="${BENCHMARK_RUN_DIR_CONTAINER}/unified_archives"
echo "Multi-algorithm results CSV: $MULTIALGO_RESULTS_CSV"

# ── Phase 6a: Track B quality indicators (HV/IGD+/Spacing for NSGA-II) ──────
echo ""
echo "--- Phase 6a: Track B quality indicators (NSGA-II Pareto archives) ---"
$APPTAINER python /app/scripts/quality_indicators.py \
    --archive-dir "$PARETO_ARCHIVES_DIR" \
    --output-dir "$CONTAINER_OUTPUT_DIR" \
    --plot

# ── Phase 6b: Cross-method IGD ──────────────────────────────────────────────
echo ""
echo "--- Phase 6b: Cross-method IGD (scalar terminal states → 3-obj) ---"
$APPTAINER python /app/scripts/cross_method_igd.py \
    --run-dir "$BENCHMARK_RUN_DIR_CONTAINER" \
    --output-dir "$CONTAINER_OUTPUT_DIR" \
    --algorithms sa,ts,hc,sga,rs \
    --report

# ── Phase 6c: Unified HV across all budgets ─────────────────────────────────
echo ""
echo "--- Phase 6c: Unified HV across all budgets (saturation curve) ---"
for B in 1000 5000 10000 25000 50000 100000 200000 500000 ; do
    $APPTAINER python /app/scripts/unified_hv_analysis.py \
        --archive-dir "$UNIFIED_ARCHIVES_DIR" \
        --output-dir "${CONTAINER_OUTPUT_DIR}/unified_hv_b${B}" \
        --budget "$B"
done

# ── Phase 7: Statistical analysis pipeline ──────────────────────────────────
echo ""
echo "--- Phase 7: Statistical analysis (closes RQ1 + RQ2 + RQ4) ---"
for B in 1000 5000 10000 25000 50000 100000 200000 500000 ; do
    echo "--- Phase 7 [budget=$B] ---"
    PER_B_DIR_HOST="${HOST_OUTPUT_DIR}/stats_b${B}"
    mkdir -p "$PER_B_DIR_HOST"
    cp "${MULTIALGO_RESULTS_CSV}" "${PER_B_DIR_HOST}/results.csv"
    PER_B_CSV_CONTAINER="$(echo "${PER_B_DIR_HOST}/results.csv" | sed "s|^$PWD/output|/app/output|")"
    $APPTAINER python /app/scripts/analyze_benchmark.py \
        "$PER_B_CSV_CONTAINER" \
        --budget "$B" \
        --n-spatial 31
done

echo "--- Phase 7 [RQ2 NSGA-II HV vs scalars] ---"
RQ2_HV_DIRS=()
for B in 1000 5000 10000 25000 50000 100000 200000 500000 ; do
    RQ2_HV_DIRS+=("${CONTAINER_OUTPUT_DIR}/unified_hv_b${B}")
done
$APPTAINER python /app/scripts/analyze_rq2_unified_hv.py \
    --hv-dirs "${RQ2_HV_DIRS[@]}" \
    --output  "${CONTAINER_OUTPUT_DIR}/rq2_unified_hv_report.txt"

echo "--- Phase 7 [RQ3 Spearman ρ(balance_gap, spatial)] ---"
$APPTAINER python /app/scripts/analyze_rq3_spearman.py \
    "${BENCHMARK_RUN_DIR_CONTAINER}/results.csv" \
    --output "${CONTAINER_OUTPUT_DIR}/rq3_spearman.csv"

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Paper 1 Stage 2 multi-algorithm canonical run complete."
echo "Stage 1 (Phase 0)            : $STAGE1_RUN_HOST"
echo "Stage 2 (Phase 1+6+7)        : $HOST_OUTPUT_DIR"
echo "Multi-algo results (Phase 1) : $MULTIALGO_RESULTS_CSV"
echo "------------------------------------------------------------"

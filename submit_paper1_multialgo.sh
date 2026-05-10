#!/bin/bash
#SBATCH --job-name=ti4_paper1_multialgo
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mpk15956@uga.edu

# Paper 1 multi-algorithm canonical run.
#
# WHAT THIS SCRIPT CLOSES (vs. data-only). Phase 7 runs the analyzers and emits
# the actual proofs (p-values, effect sizes, Friedman/Wilcoxon panels). Without
# Phase 7, this script would only feed the analysis layer; with Phase 7, it
# closes:
#   RQ1 (cross-algorithm): Friedman + pairwise Wilcoxon + Holm + VDA + d_z on
#                          composite_score, paired by seed at every budget.
#   RQ2 (NSGA-II HV vs scalar): unified-HV Friedman + one-tailed Wilcoxon vs
#                               scalars + Holm.
#   RQ4 (evals-to-best): Friedman + pairwise Wilcoxon + Holm + VDA + d_z on
#                        evals_to_best (per §3.10 — the canonical
#                        anytime-performance metric in the metaheuristics
#                        literature, Hutter et al. 2019).
# Track B (HV/IGD+/Spacing for NSGA-II), Cross-method IGD (scalar terminal
# states → NSGA-II reference front), and Unified HV (all algorithms via
# empirical archives) all consume the canonical objective tuple via
# `MultiObjectiveScore.archive_row_to_pareto_point` — single-source, no
# inline `[1-jfi, abs(mi), lp]` re-implementation.
#
# WHY SELF-CONTAINED. SA hyperparameters are re-tuned in this job rather than
# reused from a sibling run. The 6h cost is paid for self-contained
# reproducibility: every numeric result in the manuscript traces to a single
# sbatch invocation, with one git_hash, one container, one Gen-0 σ regime,
# one canonical formulation flag setting. Reuse-across-jobs imports a
# provenance dependency on a separate run that the orchestrator cannot fully
# assert against.

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
# Compute git state on the HOST (where git + .git are available) and forward
# both into the container as env vars. Without this, run_config.py inside the
# container records git_hash: "unknown" because the SIF has neither git on
# PATH nor .git bind-mounted; that loses the provenance the audit framework
# depends on. Both env vars are read by run_config._git_state() with
# precedence over the in-container `git rev-parse` fallback.
HOST_GIT_HASH="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
HOST_GIT_DIRTY="$([ -n "$(git status --porcelain 2>/dev/null)" ] && echo 1 || echo 0)"
APPTAINER="apptainer exec \
    --env TI4_GIT_HASH=$HOST_GIT_HASH \
    --env TI4_GIT_DIRTY=$HOST_GIT_DIRTY \
    --bind $PWD/output:/app/output --pwd /app $SIF"

# ── Configuration ─────────────────────────────────────────────────────────────
RUN_TAG="paper1_multialgo_canonical_$(date +%Y%m%d_%H%M%S)"
HOST_OUTPUT_DIR="$PWD/output/${RUN_TAG}"
CONTAINER_OUTPUT_DIR="/app/output/${RUN_TAG}"
WORKERS=16
mkdir -p "$HOST_OUTPUT_DIR"

SEEDS=100
BUDGETS="1000,5000,10000,25000,50000,100000,200000,500000"
PLAYERS=6
CHAINS=3

TUNING_TRIALS=50
TUNING_SEEDS=100
TUNING_BASE_SEED=9000

# Tuning iter_budget: 1000 — one order of magnitude above the legacy
# default (200), still affordable in Phase 0, and matches the smallest
# benchmark budget so the in-tuning regime is a strict subset of the
# benchmark regime. The methodology (§3.6/§3.7) ASSUMES hyperparameter
# values are approximately budget-invariant — the same SA cooling rate
# / NSGA-II crossover blob_fraction works at b=1k and b=500k. Phase 0e
# (below) tests this assumption by re-tuning SA at a 10× larger budget
# and checking the parameters don't drift materially.
TUNING_ITER_BUDGET=1000
TUNING_ITER_BUDGET_SENSITIVITY=10000

echo "============================================================"
echo "TI4 Paper 1 Multi-Algorithm Canonical Run"
echo "============================================================"
echo "Host          : $(hostname)"
echo "SLURM Job ID  : ${SLURM_JOB_ID:-local}"
echo "SIF           : $SIF"
echo "Run tag       : $RUN_TAG"
echo "Host output   : $HOST_OUTPUT_DIR"
echo "Algorithms    : RS, HC, SA, SGA, NSGA-II, TS"
echo "Seeds         : $SEEDS  Budgets: $BUDGETS  Chains: $CHAINS"
echo "Workers       : $WORKERS"
echo "Git hash      : $HOST_GIT_HASH (dirty=$HOST_GIT_DIRTY)"
echo "Started at    : $(date -Iseconds)"
echo "Formulation   : --corrected-landscape (canonical)"
echo "------------------------------------------------------------"

# ── Pre-flight: canonical-objective canary + invariants ─────────────────────
# Asserts the canonical objective transform is single-source (no drift in
# the helper since the May 2026 audit), n_spatial=31 on the canonical
# 6-player layout, and seed-set disjointness. Failure here aborts the job
# at second 5 instead of hour 30.
echo ""
echo "--- Pre-flight: canonical-objective canary + invariants ---"
$APPTAINER python -c "
import sys
from ti4_analysis.algorithms.spatial_optimizer import MultiObjectiveScore
from ti4_analysis.algorithms.map_topology import MapTopology, morans_i_null
from ti4_analysis.algorithms.map_generator import generate_random_map
from ti4_analysis.evaluation.batch_experiment import create_joebrew_evaluator

# (1) Canonical 6-player layout: |G| = 31
ev = create_joebrew_evaluator()
m = generate_random_map(player_count=6, random_seed=0)
topo = MapTopology.from_ti4_map(m, ev)
assert topo.n_spatial == 31, f'canonical n_spatial drift: expected 31, got {topo.n_spatial}'

# (2) E[I] under the null on the canonical layout
e_i = morans_i_null(31)
assert abs(e_i - (-1.0/30)) <= 1e-12, f'morans_i_null(31) drift: {e_i}'

# (3) archive_row_to_pareto_point produces the canonical tuple at a known input
row = {'jains_index': 0.95, 'morans_i': -0.6, 'lisa_penalty': 4.5}
f1, f2, f3 = MultiObjectiveScore.archive_row_to_pareto_point(row, 31)
assert abs(f1 - 0.05) <= 1e-12 and f2 == 0.0 and abs(f3 - 4.5) <= 1e-12, \
    f'canonical tuple drift at canary input: ({f1},{f2},{f3})'

# (4) Seed-set disjointness invariant (§3.6)
benchmark_seeds = set(range(0, 100))
tuning_seeds = set(range(9000, 9100))
held_out_seeds = set(range(9100, 9150))
assert benchmark_seeds.isdisjoint(tuning_seeds)
assert benchmark_seeds.isdisjoint(held_out_seeds)
assert tuning_seeds.isdisjoint(held_out_seeds)
print('  PASS: |G|=31, E[I]=-1/30, canonical tuple bit-correct, seed-sets disjoint')
"
echo "  Pre-flight OK."

# ── Phase 0a: SA tuning under canonical landscape (Optuna + SQLite) ──────────
echo ""
echo "--- Phase 0a: SA hyperparameter tuning (canonical, Optuna + SQLite resume) ---"
$APPTAINER python /app/scripts/optimize_hyperparameters.py \
    --algo sa \
    --trials "$TUNING_TRIALS" \
    --eval-seeds "$TUNING_SEEDS" \
    --base-seed "$TUNING_BASE_SEED" \
    --iter-budget "$TUNING_ITER_BUDGET" \
    --players "$PLAYERS" \
    --corrected-landscape \
    --output-dir "$CONTAINER_OUTPUT_DIR" \
    --storage "sqlite:///${CONTAINER_OUTPUT_DIR}/optuna_sa_${RUN_TAG}.db" \
    --study-name "paper1_sa_${RUN_TAG}"

# ── Phase 0b: SGA tuning ─────────────────────────────────────────────────────
echo ""
echo "--- Phase 0b: SGA hyperparameter tuning (canonical, Optuna + SQLite) ---"
$APPTAINER python /app/scripts/optimize_hyperparameters.py \
    --algo sga \
    --trials "$TUNING_TRIALS" \
    --eval-seeds "$TUNING_SEEDS" \
    --base-seed "$TUNING_BASE_SEED" \
    --iter-budget "$TUNING_ITER_BUDGET" \
    --players "$PLAYERS" \
    --corrected-landscape \
    --output-dir "$CONTAINER_OUTPUT_DIR" \
    --storage "sqlite:///${CONTAINER_OUTPUT_DIR}/optuna_sga_${RUN_TAG}.db" \
    --study-name "paper1_sga_${RUN_TAG}"

# ── Phase 0c: NSGA-II tuning ─────────────────────────────────────────────────
echo ""
echo "--- Phase 0c: NSGA-II hyperparameter tuning (canonical, Optuna + SQLite) ---"
$APPTAINER python /app/scripts/optimize_hyperparameters.py \
    --algo nsga2 \
    --trials "$TUNING_TRIALS" \
    --eval-seeds "$TUNING_SEEDS" \
    --base-seed "$TUNING_BASE_SEED" \
    --iter-budget "$TUNING_ITER_BUDGET" \
    --players "$PLAYERS" \
    --corrected-landscape \
    --output-dir "$CONTAINER_OUTPUT_DIR" \
    --storage "sqlite:///${CONTAINER_OUTPUT_DIR}/optuna_nsga2_${RUN_TAG}.db" \
    --study-name "paper1_nsga2_${RUN_TAG}"

# ── Phase 0d: TS grid search ────────────────────────────────────────────────
echo ""
echo "--- Phase 0d: TS grid search over k (canonical) ---"
$APPTAINER python /app/scripts/optimize_hyperparameters.py \
    --algo ts \
    --eval-seeds "$TUNING_SEEDS" \
    --base-seed "$TUNING_BASE_SEED" \
    --iter-budget "$TUNING_ITER_BUDGET" \
    --players "$PLAYERS" \
    --corrected-landscape \
    --output-dir "$CONTAINER_OUTPUT_DIR"

# ── Phase 0e: SA budget-invariance sensitivity check ────────────────────────
# The methodology assumes hyperparameters chosen at TUNING_ITER_BUDGET work
# at all benchmark budgets (1k–500k). This phase re-tunes SA at a 10× larger
# budget and writes the result to a separate study; downstream comparison
# of (rate, min_temp) between the two studies tells us whether the
# assumption holds materially. SA only — re-tuning all algorithms doubles
# Phase 0 cost; SA's cooling parameters are the most regime-sensitive.
echo ""
echo "--- Phase 0e: SA tuning at iter_budget=${TUNING_ITER_BUDGET_SENSITIVITY} (sensitivity check) ---"
$APPTAINER python /app/scripts/optimize_hyperparameters.py \
    --algo sa \
    --trials "$TUNING_TRIALS" \
    --eval-seeds "$TUNING_SEEDS" \
    --base-seed "$TUNING_BASE_SEED" \
    --iter-budget "$TUNING_ITER_BUDGET_SENSITIVITY" \
    --players "$PLAYERS" \
    --corrected-landscape \
    --output-dir "${CONTAINER_OUTPUT_DIR}/sensitivity_b${TUNING_ITER_BUDGET_SENSITIVITY}" \
    --storage "sqlite:///${CONTAINER_OUTPUT_DIR}/optuna_sa_sensitivity_${RUN_TAG}.db" \
    --study-name "paper1_sa_sensitivity_${RUN_TAG}"

echo ""
echo "--- Sensitivity check: SA hyperparameter drift across iter_budget ---"
$APPTAINER python -c "
import json, glob
def _read_sa(path):
    with open(path) as f:
        d = json.load(f)
    return d['best_params']
primary = _read_sa(sorted(glob.glob('${CONTAINER_OUTPUT_DIR}/optuna_*/best_params.json'))[0])
sensitivity_paths = sorted(glob.glob('${CONTAINER_OUTPUT_DIR}/sensitivity_b${TUNING_ITER_BUDGET_SENSITIVITY}/optuna_*/best_params.json'))
if not sensitivity_paths:
    print('Sensitivity SA tuning artifact missing; cannot compare.')
else:
    sens = _read_sa(sensitivity_paths[0])
    p_rate = primary.get('initial_acceptance_rate'); s_rate = sens.get('initial_acceptance_rate')
    p_mt   = primary.get('min_temp');               s_mt   = sens.get('min_temp')
    print(f'  iter_budget={${TUNING_ITER_BUDGET}}:    rate={p_rate:.4f}  min_temp={p_mt:.4g}')
    print(f'  iter_budget={${TUNING_ITER_BUDGET_SENSITIVITY}}:   rate={s_rate:.4f}  min_temp={s_mt:.4g}')
    if p_rate is not None and s_rate is not None:
        rate_drift = abs(p_rate - s_rate) / max(p_rate, s_rate)
        mt_drift   = abs(p_mt - s_mt)   / max(p_mt, s_mt)
        print(f'  relative drift: rate={rate_drift:.1%}  min_temp={mt_drift:.1%}')
        if rate_drift > 0.20 or mt_drift > 0.50:
            print(f'  WARNING: SA hyperparameters drift materially across iter_budget. The')
            print(f'  budget-invariance assumption (§3.7) may not hold; consider re-tuning')
            print(f'  per-budget or revising the methodology section.')
        else:
            print(f'  Drift within acceptable bounds; budget-invariance assumption defensible.')
"

# ── Slurp tuned hyperparameters — HARD FAIL on missing keys ─────────────────
# No `.get(default)`. Direct subscripting; KeyError aborts the job.
# NSGA-II's (pop, gen) are NOT slurped: they are derived per-budget at
# benchmark time via the single-source `nsga2_budget` helper that the tuner
# also uses. See `feedback_canonical_objective_single_source.md` and
# src/ti4_analysis/algorithms/budget_factorization.py — the drift between
# the tuning regime and the benchmark regime is structurally impossible
# when both sides go through the same helper.
echo ""
echo "--- Extracting tuned hyperparameters (hard-fail on missing keys) ---"
read SA_RATE SA_MIN_TEMP SGA_BLOB SGA_MUT SGA_WARM NSGA_BLOB NSGA_MUT TS_K < <(
$APPTAINER python -c "
import json, glob, sys
out = {}
for path in sorted(glob.glob('${CONTAINER_OUTPUT_DIR}/optuna_*/best_params.json')):
    with open(path) as f:
        d = json.load(f)
    out[d['algorithm']] = d['best_params']
required = ['sa', 'sga', 'nsga2', 'ts']
missing = [a for a in required if a not in out]
if missing:
    print(f'FATAL: tuning missing for {missing}; have {sorted(out.keys())}', file=sys.stderr)
    sys.exit(1)

sa = out['sa']
sga = out['sga']
nsga = out['nsga2']
ts = out['ts']

# Direct subscripting — KeyError if any required key is renamed/missing.
sa_rate     = sa['initial_acceptance_rate']
sa_min_temp = sa['min_temp']
sga_blob    = sga['blob_fraction']
sga_mut     = sga['mutation_rate']
sga_warm    = sga['warm_fraction']
nsga_blob   = nsga['blob_fraction']
nsga_mut    = nsga['mutation_rate']
ts_k        = ts['tabu_tenure_coefficient']

# Range checks: catch tuning failures that produced technically-valid JSON
# but nonsensical values. Each parameter has a defensible domain; outside
# it, the benchmark would produce silent garbage.
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
NSGA_WARM=0.0   # tuner-fixed; cold-start NSGA-II per Pareto-diversity rationale
echo "  SA     rate=$SA_RATE min_temp=$SA_MIN_TEMP"
echo "  SGA    blob=$SGA_BLOB mut=$SGA_MUT warm=$SGA_WARM"
echo "  NSGA   blob=$NSGA_BLOB mut=$NSGA_MUT warm=$NSGA_WARM (tuner-fixed cold-start; pop/gen derived per-budget)"
echo "  TS     k=$TS_K"

# ── Phase 1: Multi-algorithm canonical benchmark ─────────────────────────────
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
# Note: --nsga-pop / --nsga-gen are intentionally NOT passed; benchmark_engine
# derives both from the per-budget call to nsga2_budget(), the same helper
# the tuner used. See comment in the slurp block above.

# ── Resolve archive paths from the run directory rather than `find | tail` ───
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
echo "Pareto archives dir        : $PARETO_ARCHIVES_DIR"
echo "Unified archives dir       : $UNIFIED_ARCHIVES_DIR"

# ── Phase 6a: Track B quality indicators (HV/IGD+/Spacing for NSGA-II) ───────
echo ""
echo "--- Phase 6a: Track B quality indicators (NSGA-II Pareto archives) ---"
$APPTAINER python /app/scripts/quality_indicators.py \
    --archive-dir "$PARETO_ARCHIVES_DIR" \
    --output-dir "$CONTAINER_OUTPUT_DIR" \
    --plot

# ── Phase 6b: Cross-method IGD (scalar terminal states → NSGA-II ref front) ──
echo ""
echo "--- Phase 6b: Cross-method IGD (scalar terminal states → 3-obj) ---"
$APPTAINER python /app/scripts/cross_method_igd.py \
    --run-dir "$BENCHMARK_RUN_DIR_CONTAINER" \
    --output-dir "$CONTAINER_OUTPUT_DIR" \
    --algorithms sa,ts,hc,sga,rs \
    --report

# ── Phase 6c: Unified HV across budgets (saturation curve, not single-budget) ─
# RQ2 ("under equal total evaluation budget") is reported at every budget
# level so the saturation curve is visible — single-budget HV would only
# anchor the asymptotic claim.
echo ""
echo "--- Phase 6c: Unified HV across all budgets (saturation curve) ---"
for B in 1000 5000 10000 25000 50000 100000 200000 500000 ; do
    $APPTAINER python /app/scripts/unified_hv_analysis.py \
        --archive-dir "$UNIFIED_ARCHIVES_DIR" \
        --output-dir "${CONTAINER_OUTPUT_DIR}/unified_hv_b${B}" \
        --budget "$B"
done

# ── Phase 7: Statistical analysis pipeline (closes RQ1, RQ2, RQ4) ────────────
# - RQ1 (cross-algorithm Friedman + pairwise Wilcoxon + Holm + VDA + d_z +
#   bootstrap CIs on `composite_score`): analyze_benchmark.py per budget.
# - RQ4 (same panel on `evals_to_best`, the canonical anytime-performance
#   metric per Hutter et al. 2019): analyze_benchmark.py emits both metrics
#   alongside composite_score in its descriptive panel; the formal Friedman
#   on evals_to_best runs as part of the same per-budget invocation.
# - RQ2 (NSGA-II HV does NOT exceed scalar HV; one-tailed Wilcoxon vs each
#   scalar with Holm correction at every budget): analyze_rq2_unified_hv.py
#   consumes the per-budget unified_hv.csv files emitted by Phase 6c.
echo ""
echo "--- Phase 7: Statistical analysis (closes RQ1 + RQ2 + RQ4) ---"
RESULTS_CSV_CONTAINER="${BENCHMARK_RUN_DIR_CONTAINER}/results.csv"
for B in 1000 5000 10000 25000 50000 100000 200000 500000 ; do
    echo "--- Phase 7 [budget=$B] ---"
    # analyze_benchmark.py writes to ${BENCHMARK_RUN_DIR}/stats/, but
    # successive calls would overwrite. Stage each budget into its own
    # subdir by running from a per-budget symlink directory that points
    # at the same results.csv.
    PER_B_DIR_HOST="${HOST_OUTPUT_DIR}/stats_b${B}"
    mkdir -p "$PER_B_DIR_HOST"
    cp "${MULTIALGO_RESULTS_CSV}" "${PER_B_DIR_HOST}/results.csv"
    PER_B_CSV_CONTAINER="$(echo "${PER_B_DIR_HOST}/results.csv" | sed "s|^$PWD/output|/app/output|")"
    $APPTAINER python /app/scripts/analyze_benchmark.py \
        "$PER_B_CSV_CONTAINER" \
        --budget "$B" \
        --n-spatial 31
done

# RQ2: one-tailed Wilcoxon NSGA-II vs each scalar at every budget, Holm
# corrected across the 5 tests at that budget. Reads the per-budget
# unified_hv.csv files from Phase 6c.
echo "--- Phase 7 [RQ2 NSGA-II HV vs scalars] ---"
RQ2_HV_DIRS=()
for B in 1000 5000 10000 25000 50000 100000 200000 500000 ; do
    RQ2_HV_DIRS+=("${CONTAINER_OUTPUT_DIR}/unified_hv_b${B}")
done
$APPTAINER python /app/scripts/analyze_rq2_unified_hv.py \
    --hv-dirs "${RQ2_HV_DIRS[@]}" \
    --output  "${CONTAINER_OUTPUT_DIR}/rq2_unified_hv_report.txt"

# RQ3: per-(algorithm, budget) and pooled Spearman correlations between
# balance_gap and the three spatial metrics across optimized solutions.
# Exploratory per §3.10 RQ3; produces rq3_spearman.csv for the manuscript.
echo "--- Phase 7 [RQ3 Spearman ρ(balance_gap, spatial)] ---"
$APPTAINER python /app/scripts/analyze_rq3_spearman.py \
    "${BENCHMARK_RUN_DIR_CONTAINER}/results.csv" \
    --output "${CONTAINER_OUTPUT_DIR}/rq3_spearman.csv"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Paper 1 multi-algorithm canonical run complete."
echo "Host output dir              : $HOST_OUTPUT_DIR"
echo "Multi-algo results (Phase 1) : $MULTIALGO_RESULTS_CSV"
echo "Stats panel (Phase 7)        : ${HOST_OUTPUT_DIR}/stats_multialgo/"
echo "------------------------------------------------------------"

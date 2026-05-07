#!/bin/bash
#
# Verification run for the pixi+apptainer migration (commit 916df74).
#
# Three tasks, all under ti4-analysis.sif:
#   1) Phase 1 verification: 1 seed of C0 (jfi_only) and C4 (full_composite)
#      at budget=500k. Compares Moran's I to the prior 100-seed medians:
#         C0 ≈ −0.078 (target ≈ −0.079)
#         C4 ≈ −0.657 (target ≈ −0.669)
#   2) Phase 3 stats re-run at budget=500k on the existing Phase 2b results.csv.
#   3) variance_equalization_diagnostic.py on production data (fills §3.7).
#
# Submit AFTER ti4-analysis.sif is built:
#   sbatch --dependency=afterok:<build-jobid> scripts/verify_pixi_apptainer.slurm
#
#SBATCH --job-name=ti4_verify
#SBATCH --partition=batch
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

SIF="$PWD/ti4-analysis.sif"
if [ ! -f "$SIF" ]; then
    echo "FATAL: $SIF not found. Run scripts/build_sif.slurm first." >&2
    exit 1
fi

RUN_TAG="verify_pixi_apptainer_$(date +%Y%m%d_%H%M%S)"
OUT="$PWD/output/${RUN_TAG}"
mkdir -p "$OUT"

# Tuned SA hyperparameters (from optuna_20260314_205923/best_params.json,
# matching the prior Phase 1 100-seed run).
SA_RATE=0.788681
SA_MIN_TEMP=0.001286

# Bind host's output/ over the container's so the verification writes to
# our project tree and reads existing Phase 2b CSVs. Container code, data,
# and pixi env at /app/{scripts,src,data,.pixi} are used as-built.
APPTAINER="apptainer exec --bind $PWD/output:/app/output --pwd /app $SIF"

echo "============================================================"
echo "TI4 pixi+apptainer migration verification"
echo "============================================================"
echo "Host        : $(hostname)"
echo "SLURM JobID : ${SLURM_JOB_ID:-local}"
echo "SIF         : $SIF ($(stat -c '%s bytes, mtime=%y' $SIF))"
echo "Run tag     : $RUN_TAG"
echo "Output dir  : $OUT"
echo "Started at  : $(date -Iseconds)"
echo "Git hash    : $(git rev-parse --short HEAD)"
echo "------------------------------------------------------------"

$APPTAINER python --version
$APPTAINER python -c "
import numpy, scipy, esda, libpysal, pandas
print(f'  numpy {numpy.__version__}, scipy {scipy.__version__}, '
      f'pandas {pandas.__version__}, esda {esda.__version__}, libpysal {libpysal.__version__}')
"
echo "------------------------------------------------------------"

# ── Task 1: Phase 1 verification — C0 + C4, 1 seed, budget=500k ─────────────
echo ""
echo "=== Task 1: Phase 1 verification (C0+C4, 1 seed, 3 chains, budget=500k) ==="
$APPTAINER python /app/scripts/benchmark_engine.py \
    --conditions jfi_only,full_composite \
    --seeds 1 \
    --base-seed 0 \
    --budgets 500000 \
    --players 6 \
    --workers 8 \
    --chains 3 \
    --sa-rate "$SA_RATE" \
    --sa-min-temp "$SA_MIN_TEMP" \
    --corrected-landscape \
    --output-dir "/app/output/${RUN_TAG}"

VERIFY_CSV=$(find "$OUT" -name "results.csv" -path "*/benchmark_*" | sort | tail -1)
echo "Verification CSV: $VERIFY_CSV"

echo ""
echo "--- Verification check: same-seed reproducibility ---"
# Same-seed comparison: with 1 seed × 3 chains we cannot hit the 100-seed
# *population* median (population σ ≈ 0.19 on C0). Comparing seed=0's prior
# 3-chain median to the new container's seed=0 3-chain median is the only
# meaningful single-seed reproducibility check.
# Also tolerate exit non-zero so Tasks 2 & 3 still run.
set +e
# Translate host-side VERIFY_CSV path to its in-container path under /app/output/.
VERIFY_CSV_INC=$(echo "$VERIFY_CSV" | sed "s|^$PWD/output|/app/output|")
$APPTAINER python - <<PY
import sys, pandas as pd
verify = pd.read_csv("$VERIFY_CSV_INC")
prior  = pd.read_csv("/app/output/saturation_20260314_205919/benchmark_20260314_233002/results.csv")

# Tolerance bands sized to natural seed-to-seed variance (≈ 1σ population):
TOL = {"jfi_only": 0.20, "full_composite": 0.10}

print(f"{'condition':<16s} {'prior seed=0':>14s} {'new seed=0':>12s} {'delta':>10s} {'tol (1σ)':>10s} {'within?':>10s}")
print("-" * 80)
ok = True
for cond in ("jfi_only", "full_composite"):
    p = prior.query("condition == @cond and budget == 500000 and seed == 0")["morans_i"]
    v = verify.query("condition == @cond and seed == 0")["morans_i"]
    if not len(p) or not len(v):
        print(f"{cond:<16s} MISSING")
        ok = False
        continue
    pm, vm = p.median(), v.median()
    delta, tol = vm - pm, TOL[cond]
    within = abs(delta) <= tol
    print(f"{cond:<16s} {pm:14.4f} {vm:12.4f} {delta:+10.4f} {tol:10.4f} {('YES' if within else 'NO'):>10s}")
    if not within:
        ok = False

# Population-level context (informational; not a gate)
pop = {c: prior.query("condition == @c and budget == 500000")["morans_i"]
       for c in ("jfi_only", "full_composite")}
print()
print("Population context (prior 100 seeds × 3 chains, budget=500k):")
for c, s in pop.items():
    print(f"  {c:<16s} median={s.median():+.4f}  σ={s.std():.4f}  range=[{s.min():+.4f}, {s.max():+.4f}]")

print()
print("VERDICT:", "consistent — env reproducible at seed=0" if ok else
                  "drift exceeds 1σ at seed=0 — investigate")
sys.exit(0 if ok else 2)
PY
VERIFY_RC=$?
set -e
echo "Verification exit code: $VERIFY_RC  (0 = consistent, 2 = drift)"

# ── Task 2: Phase 3 stats at budget=500k ─────────────────────────────────────
echo ""
echo "=== Task 2: Phase 3 stats at budget=500k ==="
P2B_CSV="/app/output/saturation_20260314_205919/benchmark_20260315_074617/results.csv"

$APPTAINER python /app/scripts/analyze_benchmark.py \
    "$P2B_CSV" --sensitivity --ablation --budget 500000

$APPTAINER python /app/scripts/plot_statistical_results.py \
    "$P2B_CSV" --budget 500000

echo "Phase 3 stats updated at: output/saturation_20260314_205919/benchmark_20260315_074617/stats/"

# ── Task 3: variance_equalization_diagnostic.py on production data ───────────
echo ""
echo "=== Task 3: variance equalization diagnostic ==="
$APPTAINER python /app/scripts/variance_equalization_diagnostic.py \
    --output-dir "/app/output/${RUN_TAG}"

echo ""
echo "============================================================"
echo "Verification complete."
echo "Output dir : $OUT"
echo "Phase 1 CSV: $VERIFY_CSV"
echo "Phase 3 stats overwritten at: output/saturation_20260314_205919/benchmark_20260315_074617/stats/"
echo "Variance eq csv at: $OUT/variance_equalization_*.csv"
echo "Verification verdict: $([ $VERIFY_RC -eq 0 ] && echo PASS || echo FAIL)"
echo "Finished at: $(date -Iseconds)"
echo "============================================================"
exit $VERIFY_RC

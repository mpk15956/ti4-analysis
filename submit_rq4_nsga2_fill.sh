#!/bin/bash
#SBATCH --job-name=ti4_rq4_nsga2_fill
#SBATCH --account=geog4592
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=140:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mpk15956@uga.edu
#
# RQ4 targeted fill: re-run ONLY NSGA-II to record evals_to_best (the -1 sentinel
# in the banked canonical run), at budgets <=200k (the 7 levels where the six-way
# RQ4 omnibus is reported; 500k is the saturated tail, reported 5-way descriptive
# from banked scalars). Reuses the banked NSGA-II tuned params from run_config of
# paper1_multialgo_canonical_stage2_20260526 (blob=0.74948, mut=0.032985,
# warm=0.0), identical seeds/chains/landscape, and the rebuilt .sif (bakes the
# pure evals_to_best instrumentation). NSGA-II pop/gen are derived per-budget
# (NOT passed), matching the banked run. Expect composite/JFI/Moran/LISA to
# reproduce the banked NSGA-II columns; only evals_to_best is newly recorded.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
SIF="$PWD/ti4-analysis.sif"
[ -f "$SIF" ] || { echo "FATAL: $SIF missing" >&2; exit 1; }
HOST_GIT_HASH="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
HOST_GIT_DIRTY="$([ -n "$(git status --porcelain 2>/dev/null)" ] && echo 1 || echo 0)"
APPTAINER="apptainer exec --env TI4_GIT_HASH=$HOST_GIT_HASH --env TI4_GIT_DIRTY=$HOST_GIT_DIRTY --bind $PWD/output:/app/output --pwd /app $SIF"
RUN_TAG="rq4_nsga2_fill_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$PWD/output/${RUN_TAG}"
echo "=== RQ4 NSGA-II fill === git=$HOST_GIT_HASH dirty=$HOST_GIT_DIRTY tag=$RUN_TAG start=$(date -Iseconds)"
$APPTAINER python -c "from ti4_analysis.algorithms.map_topology import morans_i_null; assert abs(morans_i_null(31)+1.0/30)<=1e-12; print(\"canary ok: baked code loads, E[I]=-1/30\")"
$APPTAINER python /app/scripts/benchmark_engine.py \
    --algorithms nsga2 --seeds 100 \
    --budgets 1000,5000,10000,25000,50000,100000,200000 \
    --players 6 --workers 16 --chains 3 \
    --nsga-blob 0.74948 --nsga-mut 0.032985 --nsga-warm 0.0 \
    --corrected-landscape \
    --output-dir "/app/output/${RUN_TAG}"
echo "=== fill complete: $(date -Iseconds) ==="
ls -d "$PWD/output/${RUN_TAG}"/benchmark_*/ 2>/dev/null

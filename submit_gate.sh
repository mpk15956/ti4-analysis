#!/bin/bash
#SBATCH --job-name=ti4_gate
#SBATCH --account=geog4592
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/gate_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mpk15956@uga.edu

# Pre-submission gate checks 2 and 3 (the pytest ones), in-SIF on GACRC where
# the Phase 0/1/4/5 artifacts live. Checks 1 (placeholder grep) and 4 (audit
# SUPERSEDED markers) are pure grep and are verified separately on the login
# node, so this job runs only:
#   Check 2: tests/test_methodology_cross_refs.py  (every section ref resolves)
#   Check 3: tests/test_manuscript_values.py       (manifest <-> artifact <-> prose)
# The host docs/, tests/, output/, src/, scripts/ are bound over the image's
# baked (commit 6a939ca, pre-3.10) copies so the tests read the CURRENT tree.
# Launcher is `python -m pytest` (the SIF's baked pixi default env), mirroring
# submit_rq4_finalize.sh, rather than `pixi run pytest`.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs

SIF="$PWD/ti4-analysis.sif"
[ -f "$SIF" ] || { echo "FATAL: $SIF missing" >&2; exit 1; }

APPTAINER="apptainer exec \
    --bind $PWD/output:/app/output --bind $PWD/docs:/app/docs \
    --bind $PWD/tests:/app/tests --bind $PWD/src:/app/src \
    --bind $PWD/scripts:/app/scripts --pwd /app $SIF"

RC=0

echo "=== Gate check 2: methodology cross-references ==="
$APPTAINER python -m pytest -o addopts="" -p no:cacheprovider \
    /app/tests/test_methodology_cross_refs.py -q || RC=1

echo
echo "=== Gate check 3: manuscript values vs artifacts + prose ==="
$APPTAINER python -m pytest -o addopts="" -p no:cacheprovider \
    /app/tests/test_manuscript_values.py -q || RC=1

echo
if [ "$RC" -eq 0 ]; then
    echo "GATE_PYTEST_CHECKS_PASSED $(date -Iseconds)"
else
    echo "GATE_PYTEST_CHECKS_FAILED $(date -Iseconds)"
fi
exit "$RC"

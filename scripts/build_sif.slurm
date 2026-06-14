#!/bin/bash
#
# Build ti4-analysis.sif on a compute node.
#
# The login node has neither podman nor apptainer; both are system-installed
# on compute nodes. This batch job runs the full multi-stage pipeline that
# `pixi run -e container release` would run on a developer machine:
#   1) podman build  → OCI image
#   2) podman save   → OCI archive
#   3) apptainer build → ti4-analysis.sif
#
# Submit with: sbatch scripts/build_sif.slurm
#
#SBATCH --job-name=ti4_sif_build
#SBATCH --account=geog4592
#SBATCH --partition=batch
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mpk15956@uga.edu

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "=== ti4-analysis .sif build ==="
echo "Host          : $(hostname)"
echo "SLURM Job ID  : ${SLURM_JOB_ID:-local}"
echo "Started at    : $(date -Iseconds)"
echo "podman        : $(command -v podman || echo MISSING)"
echo "apptainer     : $(command -v apptainer || echo MISSING)"
echo "Containerfile : $(stat -c '%s bytes, mtime=%y' Containerfile)"
echo "pixi.lock     : $(stat -c '%s bytes, mtime=%y' pixi.lock)"
echo "------------------------------------------------------------"

# Podman's overlay storage driver requires xattr support on the backing
# filesystem; the project home is NFS-backed and rejects lsetxattr, so we
# stage all build state on the compute node's local /tmp (40 GB on rb1-*).
# Only the final .sif (a flat file) lands back on the project dir.
SCRATCH="/tmp/ti4-build-${SLURM_JOB_ID:-local}"
mkdir -p "$SCRATCH"
# Rootless podman creates layer files owned by mapped subuids, so a plain
# `rm -rf` from our UID hits EPERM on every layer dir. Use `podman unshare`
# to enter the user namespace where those subuids look like root, then unlink.
# Falls back to plain rm in case podman state is already gone.
cleanup() {
    local rc=$?
    if [ -d "$SCRATCH" ]; then
        echo ""
        echo "--- Cleanup (scratch=$SCRATCH) ---"
        podman --root "$PODMAN_GRAPH" --runroot "$PODMAN_RUN" unshare rm -rf "$SCRATCH" 2>/dev/null \
            || rm -rf "$SCRATCH" 2>/dev/null \
            || echo "Cleanup left files at $SCRATCH (will be reaped by /tmp purge)"
    fi
    exit $rc
}
trap cleanup EXIT

# Rootless podman in a batch job has no logind session, so /run/user/$UID
# isn't created. Point XDG_RUNTIME_DIR at scratch and redirect podman's
# graph/runroot there too.
export XDG_RUNTIME_DIR="$SCRATCH/xdg"
mkdir -p "$XDG_RUNTIME_DIR" && chmod 700 "$XDG_RUNTIME_DIR"
PODMAN_GRAPH="$SCRATCH/storage"
PODMAN_RUN="$XDG_RUNTIME_DIR/podman"
mkdir -p "$PODMAN_GRAPH" "$PODMAN_RUN"
PODMAN="podman --root $PODMAN_GRAPH --runroot $PODMAN_RUN"

# Apptainer also extracts the OCI rootfs during conversion; keep it on /tmp.
export TMPDIR="$SCRATCH/tmp"
export APPTAINER_TMPDIR="$SCRATCH/tmp"
mkdir -p "$TMPDIR"

echo "Scratch       : $SCRATCH"
df -h "$SCRATCH"

echo ""
echo "--- Stage 1: podman build (OCI image) ---"
$PODMAN info --format '{{.Host.OS}} {{.Host.Arch}} runtime={{.Host.OCIRuntime.Name}} graphDriver={{.Store.GraphDriverName}}' || true
# TWO things are REQUIRED for this to build on teach (both verified June 2026):
#  (1) --network=host: rootless podman in a batch job (no systemd user session)
#      has no working default RUN-step network on this cluster, so the
#      `pixi install` layer cannot reach conda/pypi.
#  (2) podman build output MUST be redirected to a LOCAL file, not written
#      straight to the NFS-backed slurm-%j.out. With the build stream going
#      directly to NFS the RUN step dies at ~18 s and the error never even
#      flushes (silent failure); the IDENTICAL build with output -> /tmp local
#      succeeds. Proven: diag 28680 (output->NFS) FAILED, 28681 (output->/tmp)
#      PASSED, same node, same flags. We tee the local log back out afterward.
# (The residual "systemd sandbox cgroup" warning is benign.)
BUILD_LOG="$SCRATCH/podman-build.log"
$PODMAN build --network=host -t ti4-analysis:latest -f Containerfile . > "$BUILD_LOG" 2>&1 \
    || { echo "podman build FAILED -- tail of local build log:"; tail -60 "$BUILD_LOG"; exit 1; }
tail -25 "$BUILD_LOG"

echo ""
echo "--- Stage 2: podman save (OCI archive) ---"
$PODMAN save --format oci-archive -o "$SCRATCH/ti4-analysis.oci.tar" ti4-analysis:latest
ls -lh "$SCRATCH/ti4-analysis.oci.tar"

echo ""
echo "--- Stage 3: apptainer build (.sif on scratch) ---"
apptainer build --force "$SCRATCH/ti4-analysis.sif" "oci-archive:$SCRATCH/ti4-analysis.oci.tar"
ls -lh "$SCRATCH/ti4-analysis.sif"

# ── Verify BEFORE publishing ────────────────────────────────────────────────
# Smoke test + canary run against the scratch .sif; only a verified image is
# moved into the project dir. With set -e, any failure aborts here and the
# trap reaps scratch, so a bad rebuild can never clobber the working .sif.
echo ""
echo "--- Smoke test (scratch .sif, before publish) ---"
apptainer exec "$SCRATCH/ti4-analysis.sif" python --version
apptainer exec "$SCRATCH/ti4-analysis.sif" python -c "
import sys, numpy, scipy, pandas, esda, libpysal, matplotlib, networkx
print(f'  python     {sys.version.split()[0]}')
print(f'  numpy      {numpy.__version__}')
print(f'  scipy      {scipy.__version__}')
print(f'  pandas     {pandas.__version__}')
print(f'  esda       {esda.__version__}')
print(f'  libpysal   {libpysal.__version__}')
print(f'  matplotlib {matplotlib.__version__}')
print(f'  networkx   {networkx.__version__}')
print('imports OK')
"

echo ""
echo "--- Canonical-objective canary (baked code, before publish) ---"
# Exercise the FRESHLY BAKED /app/src (no host bind) so a dependency-version
# skew that silently moves a numeric result is caught at build time (~second 60)
# rather than at hour 30 of the re-run. Same invariants as the multi-algo
# pre-flight: |G|=31, E[I]=-1/30, and the canonical objective tuple bit-correct.
apptainer exec "$SCRATCH/ti4-analysis.sif" python -c "
from ti4_analysis.algorithms.spatial_optimizer import MultiObjectiveScore
from ti4_analysis.algorithms.map_topology import MapTopology, morans_i_null
from ti4_analysis.algorithms.map_generator import generate_random_map
from ti4_analysis.evaluation.batch_experiment import create_joebrew_evaluator
ev = create_joebrew_evaluator()
m = generate_random_map(player_count=6, random_seed=0)
topo = MapTopology.from_ti4_map(m, ev)
assert topo.n_spatial == 31, f'n_spatial drift: {topo.n_spatial}'
assert abs(morans_i_null(31) - (-1.0/30)) <= 1e-12, 'E[I] drift'
f1, f2, f3 = MultiObjectiveScore.archive_row_to_pareto_point(
    {'jains_index': 0.95, 'morans_i': -0.6, 'lisa_penalty': 4.5}, 31)
assert abs(f1-0.05) <= 1e-12 and f2 == 0.0 and abs(f3-4.5) <= 1e-12, \
    f'canonical tuple drift: ({f1},{f2},{f3})'
print('  PASS: baked image -> |G|=31, E[I]=-1/30, canonical tuple bit-correct')
"
echo "  Canary OK."

echo ""
echo "--- Publish: move verified .sif into project dir ---"
mv -f "$SCRATCH/ti4-analysis.sif" ./ti4-analysis.sif
ls -lh ti4-analysis.sif

echo ""
echo "--- Provenance record ---"
# Content-address the published .sif to the exact committed tree it was built
# from (FAIR provenance, Wilkinson et al. 2016). Git metadata is taken from the
# submit-time env export when present (the login node always has git), else from
# compute-node git, else 'unknown'. Sidecar lands next to the .sif.
PROV="ti4-analysis.sif.provenance.json"
GIT_HASH="${TI4_GIT_HASH:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}"
GIT_DIRTY="${TI4_GIT_DIRTY:-$([ -n "$(git status --porcelain 2>/dev/null)" ] && echo true || echo false)}"
SIF_SHA="$(sha256sum ti4-analysis.sif | awk '{print $1}')"
CF_SHA="$(sha256sum Containerfile | awk '{print $1}')"
LOCK_SHA="$(sha256sum pixi.lock | awk '{print $1}')"
cat > "$PROV" <<JSON
{
  "sif": "ti4-analysis.sif",
  "sif_sha256": "$SIF_SHA",
  "git_hash": "$GIT_HASH",
  "git_dirty": $GIT_DIRTY,
  "containerfile_sha256": "$CF_SHA",
  "pixi_lock_sha256": "$LOCK_SHA",
  "slurm_job_id": "${SLURM_JOB_ID:-local}",
  "host": "$(hostname)",
  "built_at": "$(date -Iseconds)"
}
JSON
cat "$PROV"

echo ""
echo "============================================================"
echo "Built + verified ti4-analysis.sif at: $PWD/ti4-analysis.sif"
echo "Provenance: $PWD/$PROV (git_hash=$GIT_HASH dirty=$GIT_DIRTY)"
echo "Finished at: $(date -Iseconds)"
echo "============================================================"

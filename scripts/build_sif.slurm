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
#SBATCH --partition=batch
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=slurm-%j.out

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
$PODMAN build -t ti4-analysis:latest -f Containerfile .

echo ""
echo "--- Stage 2: podman save (OCI archive) ---"
$PODMAN save --format oci-archive -o "$SCRATCH/ti4-analysis.oci.tar" ti4-analysis:latest
ls -lh "$SCRATCH/ti4-analysis.oci.tar"

echo ""
echo "--- Stage 3: apptainer build (.sif) ---"
# Build the .sif on scratch, then move (atomic rename within fs) into project dir.
apptainer build --force "$SCRATCH/ti4-analysis.sif" "oci-archive:$SCRATCH/ti4-analysis.oci.tar"
mv -f "$SCRATCH/ti4-analysis.sif" ./ti4-analysis.sif
ls -lh ti4-analysis.sif

echo ""
echo "--- Smoke test ---"
apptainer exec ti4-analysis.sif python --version
apptainer exec ti4-analysis.sif python -c "
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
echo "============================================================"
echo "Built ti4-analysis.sif at: $PWD/ti4-analysis.sif"
echo "Finished at: $(date -Iseconds)"
echo "============================================================"

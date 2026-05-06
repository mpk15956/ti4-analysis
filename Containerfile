# syntax=docker/dockerfile:1.6
#
# One recipe, two artifacts.
#
#   Laptops / CI (OCI):
#     podman build -t ti4-analysis .
#     podman run --rm -it -v "$PWD/output:/app/output" ti4-analysis \
#         python scripts/benchmark_engine.py --seeds 4 --algorithms ts
#
#   HPC / archival (.sif):
#     podman push ti4-analysis ghcr.io/<user>/ti4-analysis:<tag>
#     apptainer build ti4-analysis.sif docker://ghcr.io/<user>/ti4-analysis:<tag>
#
# The build requires pixi.lock to exist in the build context. Generate it
# once with `pixi install` before the first build, and commit it.

ARG PIXI_VERSION=0.67.0

# ── Stage 1: solve and materialize the pixi environment ────────────────────
FROM ghcr.io/prefix-dev/pixi:${PIXI_VERSION} AS build

WORKDIR /app

# Manifests + package source: anything pixi needs to resolve the editable
# install of ti4-analysis. Edits below this layer invalidate the install
# cache; edits to scripts/tests/notebooks (copied in stage 2) do not.
COPY pyproject.toml pixi.lock README.md ./
COPY src/ src/

# --locked refuses to build if pyproject.toml drifted from pixi.lock, which
# is the contract you want for a reproducible image.
RUN pixi install --locked --environment default

# Bake an activation script so the runtime stage doesn't need pixi installed.
# `pixi shell-hook` emits the env activation; appending `exec "$@"` lets it
# be used directly as an entrypoint.
RUN pixi shell-hook --environment default --shell bash > /shell-hook.sh \
 && echo 'exec "$@"' >> /shell-hook.sh

# ── Stage 2: slim runtime ───────────────────────────────────────────────────
FROM debian:12-slim AS runtime

# Pixi env vars are baked as ENV (not just sourced via the shell-hook entrypoint
# below) so `apptainer exec` and `podman exec`, which skip the entrypoint, still
# find python, geopandas/PROJ data dirs, etc. The shell-hook entrypoint stays
# for `apptainer run` / `podman run` paths, which need the full activation
# (LD_LIBRARY_PATH and conda etc/conda/activate.d/ hooks).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PIXI_PROJECT_MANIFEST=/app/pyproject.toml \
    PATH=/app/.pixi/envs/default/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/app/.pixi/envs/default/lib \
    CONDA_PREFIX=/app/.pixi/envs/default \
    PROJ_DATA=/app/.pixi/envs/default/share/proj \
    GDAL_DATA=/app/.pixi/envs/default/share/gdal

WORKDIR /app

# Pull in the materialized environment and activation script first so they
# survive the bulk source copy below.
COPY --from=build /app/.pixi/envs/default /app/.pixi/envs/default
COPY --from=build /shell-hook.sh /shell-hook.sh

# Bring in the rest of the project. .containerignore keeps .pixi/, output/,
# __pycache__/, .git/, etc. out of this copy, so stage 1's env is preserved.
COPY . /app

ENTRYPOINT ["/bin/bash", "/shell-hook.sh"]
CMD ["bash"]

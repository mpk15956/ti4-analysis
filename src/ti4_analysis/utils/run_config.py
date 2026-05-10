"""
Single-source helper for emitting `run_config.json` sidecars next to phase artifacts.

PURPOSE (May 2026 audit follow-up). The audit found that artifact metadata was
inconsistent across phases: `benchmark_engine.py` and `optimize_hyperparameters.py`
each implemented their own run_config writer (with duplicated `git rev-parse`
fallback logic), while `validate_lisa_proxy.py` (Phase 4) and
`distance_weight_sensitivity.py` (Phase 5) wrote no run_config at all — leaving
their artifacts unbound to any code state. This helper consolidates the writer
to one canonical shape and adds the audit's two cheap additions:

  • Resolved weights: capture the actual weight dict the optimizer used, not
    the CLI argparse value (which is None when defaults are taken — exactly
    the case where defaults change silently and break future re-derivations).

  • Per-file content hashes of the metric-defining files
    (spatial_metrics.py, fast_map_state.py, spatial_optimizer.py). The git_hash
    binds the repo state; the file hashes give a tighter contract — if any
    code that affects floating-point output changed between two runs at the
    same git_hash (uncommitted edits, env differences), the hash differs.

USAGE:
    from ti4_analysis.utils.run_config import write_run_config
    write_run_config(
        out_dir,
        args=args,                 # argparse Namespace; vars(args) gets recorded
        resolved_weights={...},    # the actual dict the score object uses
        extra={"phase": "lisa"},   # any additional structured fields
    )

The helper is deliberately tolerant of None args / missing fields so phases
that don't have an argparse Namespace (e.g. ad-hoc diagnostic scripts) can
still call it with just out_dir.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional


# Files whose content directly affects metric outputs. A change here without
# a git commit (uncommitted edit, branch switch, env shadowing) is exactly
# the failure mode the per-file hash catches that git_hash alone misses.
_METRIC_DEFINING_FILES = (
    "src/ti4_analysis/algorithms/fast_map_state.py",
    "src/ti4_analysis/algorithms/map_topology.py",
    "src/ti4_analysis/algorithms/spatial_optimizer.py",
    "src/ti4_analysis/spatial_stats/spatial_metrics.py",
)


def write_run_config(
    out_dir: Path | str,
    *,
    args: Any = None,
    resolved_weights: Optional[Mapping[str, float]] = None,
    extra: Optional[Mapping[str, Any]] = None,
    repo_root: Optional[Path] = None,
) -> Path:
    """
    Write `{out_dir}/run_config.json` recording everything needed to bind
    this run to the code state that produced it.

    The resulting JSON has the shape:
        {
          # caller-supplied: argparse fields
          "seeds": 100, "budgets": [...], "conditions": [...], ...,
          # caller-supplied: resolved post-construction values
          "resolved_weights": {"morans_i": 0.333, "jains_index": 0.333, "lisa_penalty": 0.333},
          # caller-supplied extras
          "phase": "lisa", ...,
          # auto-recorded
          "started_at": "2026-05-07T11:23:45.123456",
          "git_hash": "abc1234",
          "git_dirty": false,
          "env": {"python": "3.11.15", "numpy": "1.26.4", "scipy": "1.13.1"},
          "metric_file_hashes": {
            "src/ti4_analysis/algorithms/fast_map_state.py": "sha256:..."
          }
        }

    Args:
        out_dir: Directory to write run_config.json into. Created if missing.
        args: argparse.Namespace (or any object with __dict__) whose JSON-able
            fields get recorded. Pass None for ad-hoc scripts without argparse.
        resolved_weights: The actual weight dict the optimizer used, captured
            after construction (e.g. via `getattr(score, 'weights', None)`).
            Records None as the literal value so consumers can detect the
            "not captured" case explicitly.
        extra: Additional structured fields to merge in.
        repo_root: Override for the repo root used to compute file hashes.
            Defaults to walking up from this file until a `.git` directory
            is found, with `Path.cwd()` as the fallback.

    Returns:
        The written file path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config: dict[str, Any] = {}

    if args is not None:
        for key, value in vars(args).items():
            if _is_jsonable(value):
                config[key] = value
            else:
                config[key] = repr(value)

    if extra:
        config.update(dict(extra))

    config["resolved_weights"] = (
        dict(resolved_weights) if resolved_weights is not None else None
    )
    config["started_at"] = datetime.now().isoformat()

    git_hash, git_dirty = _git_state()
    config["git_hash"] = git_hash
    config["git_dirty"] = git_dirty

    config["env"] = _env_fingerprint()
    config["metric_file_hashes"] = _metric_file_hashes(
        repo_root or _find_repo_root()
    )

    out_path = out_dir / "run_config.json"
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    return out_path


# ── Internals ─────────────────────────────────────────────────────────────


def _is_jsonable(value: Any) -> bool:
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def _git_state() -> tuple[str, bool]:
    """Return (short_hash, is_dirty). Both fields default safely on failure.

    Resolution order:

      1. Environment variables `TI4_GIT_HASH` and `TI4_GIT_DIRTY` if set —
         the orchestrator (host shell) passes these in via apptainer's
         `--env`, so the container's run_config records the host's git
         state at job submission. Without this, runs inside the container
         silently record `git_hash: "unknown"` because the container has
         neither `git` on PATH nor `.git` bind-mounted, which loses the
         provenance the audit framework depends on.
      2. `git rev-parse --short HEAD` from the cwd, as the host-side
         fallback when the env vars aren't set (i.e., when the script is
         run directly on the host instead of through apptainer).
      3. `("unknown", False)` if both above fail.

    `TI4_GIT_DIRTY` is interpreted as truthy if it equals `"1"`, `"true"`,
    or `"True"` (case-insensitive); any other value (including missing) is
    treated as clean. The orchestrator computes dirtiness via
    `git status --porcelain` on the host and converts to "1"/"0".
    """
    env_hash = os.environ.get("TI4_GIT_HASH", "").strip()
    env_dirty_raw = os.environ.get("TI4_GIT_DIRTY", "").strip().lower()
    env_dirty = env_dirty_raw in ("1", "true", "yes")

    if env_hash:
        return env_hash, env_dirty

    try:
        short = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown", False
    try:
        # `git status --porcelain` emits non-empty output if the working tree
        # has uncommitted changes. `git_dirty=True` is the audit-relevant
        # signal — it means git_hash alone isn't sufficient to reproduce.
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return short, bool(status)
    except Exception:
        return short, False


def _env_fingerprint() -> dict[str, str]:
    fp = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for pkg in ("numpy", "scipy"):
        try:
            mod = __import__(pkg)
            fp[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            fp[pkg] = "missing"
    return fp


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / ".git").exists():
            return parent
    return Path.cwd()


def _metric_file_hashes(repo_root: Path) -> dict[str, str]:
    """SHA256 of each metric-defining file (or 'missing' if absent)."""
    hashes: dict[str, str] = {}
    for rel in _METRIC_DEFINING_FILES:
        path = repo_root / rel
        if path.exists():
            h = hashlib.sha256(path.read_bytes()).hexdigest()
            hashes[rel] = f"sha256:{h[:16]}"  # 16 hex chars = 64 bits, plenty
        else:
            hashes[rel] = "missing"
    return hashes

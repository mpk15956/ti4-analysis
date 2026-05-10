"""
Canonical-formulation provenance assertions for archive consumers.

The single-source helper `MultiObjectiveScore.archive_row_to_pareto_point`
guarantees identity of the *transform* applied to archive rows, but it
cannot guarantee that the archive itself was produced under the canonical
formulation the consumer expects. Producer-side drift (n_spatial change,
smooth_p change, --corrected-landscape flipped, future formulation v2)
would be applied silently by the helper to data from a different regime.

This module reads the producer's `run_config.json` sidecar and asserts the
canonical-formulation invariants the consumer requires. Call
`assert_canonical_formulation(run_dir)` at the entry of every CLI script
that consumes pareto_archives/ or unified_archives/ produced by
benchmark_engine.py.

See:
- `feedback_canonical_objective_single_source.md` for the helper-side rule
- `feedback_silent_fallback_is_wrong_answer.md` for why direct subscripting
  (not .get with defaults) is required when reading required provenance
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

# The canonical formulation contract this code depends on. Bumping version
# here forces every consumer to update to the new regime; mismatched runs
# fail the assertion and the operator is forced to either re-run under the
# new formulation or pin the consumer to the old version.
EXPECTED_CANONICAL_VERSION = "v1.0-corrected-landscape-2026"

EXPECTED_INVARIANTS: dict[str, Any] = {
    "version": EXPECTED_CANONICAL_VERSION,
    "corrected_landscape": True,
    "n_spatial": 31,
    "smooth_p": 8.0,
    "smooth_k": 10.0,
    "gen0_sigma_n_samples": 1000,
    "use_local_variance_lisa": True,
    "smooth_min_form": "power_mean_M_neg_p",
}


def load_run_config(run_dir: Path) -> Mapping[str, Any]:
    """Read run_config.json from a benchmark run directory. Hard-fails if
    missing — provenance is required, not optional."""
    cfg_path = Path(run_dir) / "run_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Provenance check requires {cfg_path} (written by benchmark_engine.py "
            f"via write_run_config). Re-run the producer to populate it, or pin "
            f"the consumer to a benchmark version that emits run_config.json."
        )
    with open(cfg_path) as f:
        return json.load(f)


def assert_canonical_formulation(run_dir: Path,
                                 expected: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    """
    Assert the canonical-formulation invariants for a benchmark run.

    Reads `run_dir/run_config.json` and asserts every key in `expected`
    matches the value recorded by the producer. Default `expected` is
    `EXPECTED_INVARIANTS` (the canonical formulation this code targets).

    Returns the loaded run_config dict on success so callers can read
    additional fields (e.g. `git_hash`, `started_at`) without re-loading.

    Raises a single `AssertionError` enumerating ALL mismatches at once
    so the operator sees the full delta in one shot.
    """
    cfg = load_run_config(run_dir)
    expected = dict(expected) if expected is not None else dict(EXPECTED_INVARIANTS)

    cf = cfg.get("canonical_formulation")
    if cf is None:
        # Backward-compat: some pre-canonical-formulation runs only have
        # the legacy `corrected_landscape` field at the top level. Surface
        # this as a clear error rather than auto-falling-back, since the
        # whole point of this helper is to catch silent regime drift.
        raise AssertionError(
            f"run_config.json at {run_dir} has no 'canonical_formulation' block. "
            f"This run pre-dates the canonical-formulation contract; consumers "
            f"that expect canonical artifacts must reject it. Re-run under the "
            f"current benchmark_engine.py to produce a compliant run_config.json."
        )

    mismatches: list[str] = []
    for key, want in expected.items():
        got = cf.get(key, "<missing>")
        if got != want:
            mismatches.append(f"  canonical_formulation.{key}: expected {want!r}, got {got!r}")
    if mismatches:
        raise AssertionError(
            f"Canonical-formulation provenance mismatch for run at {run_dir}:\n"
            + "\n".join(mismatches)
            + f"\nExpected version: {expected['version']}. The archive's transform "
              f"is applied by archive_row_to_pareto_point under the consumer's "
              f"canonical assumptions; mixing formulations would produce silently "
              f"miscalibrated HV/IGD numbers."
        )
    return cfg


def assert_archive_transform_identity(n_spatial: int) -> None:
    """
    Runtime canary: assert
    `MultiObjectiveScore.archive_row_to_pareto_point(row, n_spatial)`
    matches `MultiObjectiveScore(...).objective_values_for_pareto()` at
    a known fixture input. Catches a future refactor that detaches the
    helper from the underlying canonical method.
    """
    from ti4_analysis.algorithms.spatial_optimizer import MultiObjectiveScore

    row = {"jains_index": 0.95, "morans_i": -0.033, "lisa_penalty": 4.5}
    via_helper = MultiObjectiveScore.archive_row_to_pareto_point(row, n_spatial)
    via_object = MultiObjectiveScore(
        balance_gap=0.0,
        morans_i=row["morans_i"],
        jains_index=row["jains_index"],
        lisa_penalty=row["lisa_penalty"],
        n_spatial=n_spatial,
        use_smooth_objectives=False,
    ).objective_values_for_pareto()
    if via_helper != via_object:
        raise AssertionError(
            f"archive_row_to_pareto_point drifted from objective_values_for_pareto: "
            f"helper={via_helper}, object={via_object}. The consumer-side single-source "
            f"contract is broken; refusing to proceed."
        )

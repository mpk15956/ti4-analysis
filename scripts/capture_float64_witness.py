"""
Re-capture the post-float64-migration witness values.

PURPOSE. The witness file at `tests/_pre_float64_witness.json` carries two
frozen records:

  • `pre_float64_migration` — the metric values under the original float32
    optimizer hot path. Captured 2026-05-07 at git_hash 6ec5031. Frozen;
    this script does not touch it.

  • `post_float64_migration` — the metric values under the migrated (uniformly
    float64) implementation. This script REGENERATES this block. If the
    regenerated values differ from the committed `post_float64_migration`
    block by more than ~1e-12, that's an unexpected drift and the diff
    should be investigated before committing.

The pre/post pair documents that the float32→float64 migration preserved
metric semantics: post values agree with pre to ~1e-7 (within float32
precision contract) AND post-migration the two impls (FastMapState vs
spatial_metrics) agree with each other to bit-equality. See the file's
`_metadata.purpose` for the full contract.

USAGE.
    pixi run python scripts/capture_float64_witness.py

If you intentionally changed the metric formulas (not just refactored), you
must:
  1. Run this script to overwrite `post_float64_migration` with the new values.
  2. Update `delta_vs_pre_float64_migration` to reflect the actual diff.
  3. Document why the change is intentional in the methodology section.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tests.test_metric_parity import _canonical_6p_state  # noqa: E402

from ti4_analysis.algorithms.fast_map_state import FastMapState  # noqa: E402
from ti4_analysis.algorithms.hex_grid import HexCoord  # noqa: E402
from ti4_analysis.spatial_stats.spatial_metrics import (  # noqa: E402
    SpatialWeightMatrix,
    jains_fairness_index,
    morans_i as standalone_morans_i,
)


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        return bool(out)
    except Exception:
        return False


def capture_post_block() -> dict:
    """Build the `post_float64_migration` block by running every metric on the canonical fixture."""
    topo, state = _canonical_6p_state()

    z = state.spatial_values()
    W_dense = topo.spatial_W.toarray().astype(np.float64)
    weights_obj = SpatialWeightMatrix(
        weights=W_dense,
        coords=[HexCoord(0, 0, 0)] * topo.n_spatial,
    )
    standalone_I, standalone_E = standalone_morans_i(
        z.astype(np.float64), weights_obj, row_standardized=False
    )

    return {
        "captured_at": datetime.now().isoformat(),
        "git_hash": _git_hash(),
        "git_dirty": _git_dirty(),
        "dtype_state": {
            "spatial_values": str(z.dtype),
            "spatial_W": str(topo.spatial_W.dtype),
            "system_value": str(state.system_value.dtype),
            "home_resources": str(state.home_resources().dtype),
        },
        "canonical_6p_layout": {
            "n_spatial": int(topo.n_spatial),
            "morans_i_null_expectation": float(topo.morans_i_null_expectation),
            "morans_i_fastmapstate": float(state.morans_i()),
            "morans_i_spatial_metrics": float(standalone_I),
            "morans_i_expected_from_standalone": float(standalone_E),
            "lisa_penalty_fastmapstate": float(state.lisa_penalty()),
            "jfi_resources": float(state.jfi_resources()),
            "jfi_influence": float(state.jfi_influence()),
            "jfi_bottleneck": float(state.jains_index()),
            "jfi_via_spatial_metrics": float(
                jains_fairness_index(state.home_resources())
            ),
        },
    }


def main() -> None:
    witness_path = ROOT / "tests" / "_pre_float64_witness.json"
    if not witness_path.exists():
        sys.exit(
            f"ERROR: {witness_path} does not exist. The pre-migration block "
            "must already be present in this file before re-capturing post values."
        )

    witness = json.loads(witness_path.read_text())
    if "pre_float64_migration" not in witness:
        sys.exit(
            "ERROR: witness file is missing the `pre_float64_migration` block. "
            "Refusing to overwrite — the pre block is the historical record and "
            "must remain intact for the migration to remain auditable."
        )

    new_post = capture_post_block()

    # Preserve any existing delta_vs_pre block under a fresh key so the user can
    # diff old-vs-new manually if values shifted unexpectedly.
    if "post_float64_migration" in witness:
        previous_post = witness["post_float64_migration"]
        if "delta_vs_pre_float64_migration" in previous_post:
            new_post["delta_vs_pre_float64_migration"] = previous_post[
                "delta_vs_pre_float64_migration"
            ]

    witness["post_float64_migration"] = new_post
    witness_path.write_text(json.dumps(witness, indent=2) + "\n")

    print(f"Updated post_float64_migration block: {witness_path}")
    print()
    print("Current post-migration values:")
    for k, v in new_post["canonical_6p_layout"].items():
        print(f"  {k:40s} = {v}")
    print()
    print("dtype state:")
    for k, v in new_post["dtype_state"].items():
        print(f"  {k:30s} = {v}")
    print()
    pre = witness["pre_float64_migration"]["canonical_6p_layout"]
    cur = new_post["canonical_6p_layout"]
    print("Drift vs committed pre-migration block:")
    for k in cur:
        if isinstance(cur[k], (int, float)) and isinstance(pre.get(k), (int, float)):
            d = abs(cur[k] - pre[k])
            flag = "  OK" if d <= 1e-6 else "  ⚠ DRIFT > 1e-6"
            print(f"  {k:40s}  diff = {d:.2e}{flag}")


if __name__ == "__main__":
    main()

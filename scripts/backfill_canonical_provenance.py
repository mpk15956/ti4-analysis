#!/usr/bin/env python3
"""
Backfill the `canonical_formulation` block into a benchmark run's
`run_config.json` for runs predating the canonical-provenance contract.

The canonical-formulation block was added to `benchmark_engine.py`'s run-config
writer in May 2026. Older canonical runs (e.g.
`output/paper1_canonical_20260509_134024/benchmark_20260509_191848/`) were
executed under the canonical formulation but the run_config does not record
the formulation invariants — so consumers that assert
`canonical_formulation.version == "v1.0-corrected-landscape-2026"` reject
those runs even though the data is canonical.

This script writes a `canonical_formulation` block plus an audit-traceable
`backfill_metadata` record. The audit pattern preserves three pieces of
provenance:

  1. The original `git_hash` (from the existing run_config) — the commit
     that produced the data. NOT overwritten by this script.
  2. The canonical invariants known to be true at write time (n_spatial=31,
     smooth_p=8, etc., because the args/code at the original git_hash
     imply them).
  3. `backfill_metadata.{backfilled_at, backfill_git_hash, backfill_script,
     reason}` — when this script ran, against which commit, and why.

Usage:
    python scripts/backfill_canonical_provenance.py \\
        --run-dir output/paper1_canonical_20260509_134024/benchmark_20260509_191848

The script refuses to clobber an existing `canonical_formulation` block
without --force; backfilling a run that already has one is suspicious.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


def _git_hash_short() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True, capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _expected_canonical_block(run_config: Mapping[str, Any]) -> dict:
    """The canonical formulation invariants that hold for any run flagged
    `corrected_landscape: true`. n_spatial / smooth_p / smooth_k / Gen-0 σ
    sample-size are constants in the canonical implementation; the fact
    that the run was executed under the canonical regime is what they
    record. A run that has `corrected_landscape: false` is NOT canonical
    and should not be backfilled."""
    if not run_config.get("corrected_landscape", False):
        raise ValueError(
            f"Run config has corrected_landscape=False (or missing); this "
            f"run is not canonical and must not be backfilled with a "
            f"canonical_formulation block. Re-run under --corrected-landscape "
            f"if you want canonical artifacts."
        )
    return {
        "version": "v1.0-corrected-landscape-2026",
        "corrected_landscape": True,
        "n_spatial": 31,
        "smooth_p": 8.0,
        "smooth_k": 10.0,
        "gen0_sigma_n_samples": 1000,
        "gen0_sigma_seed_offset": 99999,
        "use_local_variance_lisa": True,
        "smooth_min_form": "power_mean_M_neg_p",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Benchmark run directory containing run_config.json")
    p.add_argument("--force", action="store_true",
                   help="Overwrite an existing canonical_formulation block "
                        "(default: refuse and exit non-zero)")
    p.add_argument("--reason", type=str,
                   default="run pre-dates canonical-formulation contract; "
                           "backfilled to allow consumers to assert canonical "
                           "provenance against this run",
                   help="Free-text rationale recorded under backfill_metadata.reason")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir
    cfg_path = run_dir / "run_config.json"

    if not cfg_path.exists():
        print(f"ERROR: {cfg_path} does not exist", file=sys.stderr)
        return 1

    cfg = json.loads(cfg_path.read_text())
    if "canonical_formulation" in cfg and not args.force:
        print(
            f"ERROR: {cfg_path} already has a canonical_formulation block. "
            f"Pass --force to overwrite (and explain why in --reason). "
            f"Refusing silently overwrite is the right default — see "
            f"feedback_idempotent_witness_pattern.md.",
            file=sys.stderr,
        )
        return 1

    try:
        block = _expected_canonical_block(cfg)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    backfill_metadata = {
        "backfilled_at": datetime.now().isoformat(),
        "backfill_git_hash": _git_hash_short(),
        "backfill_script": "scripts/backfill_canonical_provenance.py",
        "original_git_hash": cfg.get("git_hash", "unknown"),
        "reason": args.reason,
    }

    cfg["canonical_formulation"] = block
    cfg["backfill_metadata"] = backfill_metadata

    cfg_path.write_text(json.dumps(cfg, indent=2, default=str))
    print(f"Backfilled canonical_formulation block in {cfg_path}")
    print(f"  version              : {block['version']}")
    print(f"  original git_hash    : {backfill_metadata['original_git_hash']}")
    print(f"  backfill git_hash    : {backfill_metadata['backfill_git_hash']}")
    print(f"  backfilled_at        : {backfill_metadata['backfilled_at']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

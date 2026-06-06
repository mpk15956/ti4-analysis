#!/usr/bin/env python3
"""
Track B multi-objective quality indicators for NSGA-II Pareto front evaluation.

Computes Hypervolume (HV), Inverted Generational Distance Plus (IGD+), and
Spacing for NSGA-II Pareto archives. These indicators evaluate NSGA-II on
its own terms without the scalarization bias of Track A.

Usage:
    python scripts/quality_indicators.py --archive-dir output/pareto_archives/ --seeds 100
    python scripts/quality_indicators.py --archive-dir output/pareto_archives/ --reference auto

Reads:  Pareto archive CSV files (one per seed, produced by benchmark_engine.py)
Writes: quality_indicators.csv, quality_indicators_report.txt
        With --plot: fig_trackb_hypervolume.pdf, fig_trackb_igd_plus.pdf, fig_trackb_spacing.pdf
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

from ti4_analysis.algorithms.spatial_optimizer import MultiObjectiveScore
from ti4_analysis.algorithms.moo_indicators import igd_plus, nondominated_filter

# Canonical 6-player layout has |G| = 31 by the §3.3 derivation. This must
# match the n_spatial the benchmark engine used to produce the archive; if
# the archive comes from a different layout, the per-row pareto-objective
# tuple's hinge term shifts. Pre-flight assertion lives in main().
CANONICAL_N_SPATIAL = 31


# ── Hypervolume (HV) ─────────────────────────────────────────────────────────

def hypervolume_2d(points: np.ndarray, ref: np.ndarray) -> float:
    """
    Exact 2D hypervolume via the inclusion-exclusion sweep-line algorithm.

    For 3+ objectives, uses the Monte Carlo estimator below.
    """
    if len(points) == 0:
        return 0.0
    dominated = np.all(points <= ref, axis=1)
    pts = points[dominated]
    if len(pts) == 0:
        return 0.0
    sorted_pts = pts[pts[:, 0].argsort()]
    hv = 0.0
    prev_y = ref[1]
    for p in sorted_pts:
        hv += (ref[0] - p[0]) * (prev_y - p[1])
        prev_y = min(prev_y, p[1])
    return float(hv)


def hypervolume_mc(points: np.ndarray, ref: np.ndarray,
                   n_samples: int = 100_000, rng=None) -> float:
    """
    Monte Carlo hypervolume estimator for arbitrary dimensions.

    Samples uniformly in the hyperbox defined by ideal point and reference point,
    then counts the fraction dominated by at least one point in the front.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if len(points) == 0:
        return 0.0

    d = points.shape[1]
    ideal = points.min(axis=0)
    box_vol = float(np.prod(ref - ideal))
    if box_vol <= 0:
        return 0.0

    samples = rng.uniform(ideal, ref, size=(n_samples, d))
    dominated_count = 0
    for s in samples:
        if np.any(np.all(points <= s, axis=1)):
            dominated_count += 1

    return box_vol * dominated_count / n_samples


def hypervolume(points: np.ndarray, ref: np.ndarray,
                n_samples: int = 100_000) -> float:
    """Compute hypervolume: exact for 2D, Monte Carlo for 3D+."""
    if points.shape[1] == 2:
        return hypervolume_2d(points, ref)
    return hypervolume_mc(points, ref, n_samples)


# ── IGD+ (Inverted Generational Distance Plus) ──────────────────────────────

# igd_plus now lives in ti4_analysis.algorithms.moo_indicators (vectorized,
# equivalence-tested against the original loop) — single source shared with
# cross_method_igd.py. Imported at module top.


# ── Spacing ───────────────────────────────────────────────────────────────────

def spacing(front: np.ndarray) -> float:
    """
    Spacing metric (Schott 1995): measures uniformity of Pareto front.

    Lower = more uniform. Returns 0 for fronts with <= 2 members.
    """
    n = len(front)
    if n <= 2:
        return 0.0

    d_i = np.empty(n)
    for i in range(n):
        dists = []
        for j in range(n):
            if i != j:
                dists.append(float(np.sum(np.abs(front[i] - front[j]))))
        d_i[i] = min(dists)

    d_bar = d_i.mean()
    return float(np.sqrt(np.sum((d_i - d_bar) ** 2) / (n - 1)))


# ── CLI & Main ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Track B quality indicators for NSGA-II")
    p.add_argument("--archive-dir", type=str, required=True,
                   help="Directory containing Pareto archive CSV files")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output directory (default: same as archive-dir)")
    p.add_argument("--reference", type=str, default="auto",
                   help="Reference point: 'auto' (worst observed + 10%%) or "
                        "comma-separated values")
    p.add_argument("--mc-samples", type=int, default=100_000,
                   help="Monte Carlo samples for HV estimation (default: 100k)")
    p.add_argument("--plot", action="store_true",
                   help="Generate Track B figures (HV, IGD+, Spacing) as PDFs")
    return p.parse_args()


def load_pareto_archive(csv_path: Path, n_spatial: int = CANONICAL_N_SPATIAL) -> np.ndarray:
    """
    Load a Pareto archive CSV into an (N, 3) array of canonical Pareto-objective
    values. Expected columns: jains_index, morans_i, lisa_penalty.
    Each row is transformed via `MultiObjectiveScore.archive_row_to_pareto_point`
    — the SINGLE-SOURCE canonical transformation that returns
    `(1 - jains_index, max(0, morans_i - E[I]), lisa_penalty)`.  # noqa: canonical-transform
    Do NOT
    re-implement the transformation inline; the helper is the contract
    (see `feedback_canonical_objective_single_source.md`).
    """
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(MultiObjectiveScore.archive_row_to_pareto_point(row, n_spatial))
    return np.array(rows) if rows else np.empty((0, 3))


def main() -> int:
    args = parse_args()
    archive_dir = Path(args.archive_dir)
    output_dir = Path(args.output_dir) if args.output_dir else archive_dir

    if not archive_dir.exists():
        print(f"ERROR: {archive_dir} does not exist", file=sys.stderr)
        return 1

    # Provenance assertions: canonical-formulation invariants on the producer
    # side + transform-identity canary on the consumer side. Both fire BEFORE
    # any HV/IGD math runs; failure surfaces as a clean error instead of a
    # silently miscalibrated number.
    from ti4_analysis.utils.canonical_provenance import (
        assert_canonical_formulation, assert_archive_transform_identity,
    )
    benchmark_run_dir = archive_dir.parent
    cfg = assert_canonical_formulation(benchmark_run_dir)
    assert_archive_transform_identity(CANONICAL_N_SPATIAL)
    print(f"Provenance OK: canonical_formulation v{cfg['canonical_formulation']['version']}, "
          f"git_hash={cfg.get('git_hash', 'unknown')}")

    archive_files = sorted(archive_dir.glob("pareto_archive_seed*.csv"))
    if not archive_files:
        print(f"No pareto_archive_seed*.csv files found in {archive_dir}",
              file=sys.stderr)
        return 1

    print(f"Loading {len(archive_files)} Pareto archives from {archive_dir}")

    import re

    all_fronts = {}
    front_budget = {}
    points_by_budget = {}
    all_points = []
    for f in archive_files:
        key = f.stem.replace("pareto_archive_seed", "")  # "{seed}_b{budget}"
        m = re.search(r"_b(\d+)$", key)
        if not m:
            print(f"WARNING: cannot parse budget from {f.name}; skipping",
                  file=sys.stderr)
            continue
        budget = int(m.group(1))
        front = load_pareto_archive(f)
        if len(front) > 0:
            all_fronts[key] = front
            front_budget[key] = budget
            points_by_budget.setdefault(budget, []).append(front)
            all_points.append(front)

    if not all_points:
        print("No valid Pareto archives found", file=sys.stderr)
        return 1

    # HV reference POINT: global nadir (worst observed x 1.1) for cross-row
    # comparability. HV is computed per-front against this point — unchanged.
    combined = np.vstack(all_points)
    if args.reference == "auto":
        worst = combined.max(axis=0)
        ref = worst * 1.1
    else:
        ref = np.array([float(x) for x in args.reference.split(",")])
    print(f"Reference point (HV nadir): {ref}")

    # IGD+ reference SET: per-budget non-dominated front. Ishibuchi et al. (2015)
    # require the reference to approximate the true Pareto front; Methodology
    # 3.8 specifies "merging all observed Pareto points across seeds" (i.e. per
    # budget, non-dominated). The non-dominated filter also collapses the raw
    # per-budget union to its frontier, which is what makes IGD+ tractable.
    ref_by_budget = {}
    for budget, fronts in sorted(points_by_budget.items()):
        raw = np.vstack(fronts)
        nd = nondominated_filter(raw)
        ref_by_budget[budget] = nd
        print(f"  budget {budget:>7}: IGD+ reference {len(raw)} -> {len(nd)} non-dominated")

    results = []
    for key, front in sorted(all_fronts.items()):
        budget = front_budget[key]
        hv = hypervolume(front, ref, args.mc_samples)
        igdp = igd_plus(front, ref_by_budget[budget])
        sp = spacing(front)
        results.append({
            "seed": key,
            "front_size": len(front),
            "hypervolume": round(hv, 6),
            "igd_plus": round(igdp, 6),
            "spacing": round(sp, 6),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "quality_indicators.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "front_size",
                                               "hypervolume", "igd_plus", "spacing"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    hvs = [r["hypervolume"] for r in results]
    igds = [r["igd_plus"] for r in results]
    sps = [r["spacing"] for r in results]
    front_sizes = [r["front_size"] for r in results]

    import statistics
    report = [
        "TRACK B: MULTI-OBJECTIVE QUALITY INDICATORS",
        "=" * 60,
        f"Seeds evaluated  : {len(results)}",
        f"Reference point  : {ref.tolist()}",
        "",
        "Hypervolume (higher = better):",
        f"  mean = {statistics.mean(hvs):.4f}  "
        f"std = {statistics.stdev(hvs) if len(hvs) > 1 else 0:.4f}",
        "",
        "IGD+ (lower = better):",
        f"  mean = {statistics.mean(igds):.4f}  "
        f"std = {statistics.stdev(igds) if len(igds) > 1 else 0:.4f}",
        "",
        "Spacing (lower = more uniform):",
        f"  mean = {statistics.mean(sps):.4f}  "
        f"std = {statistics.stdev(sps) if len(sps) > 1 else 0:.4f}",
        "",
        "Pareto front size:",
        f"  mean = {statistics.mean(front_sizes):.1f}  "
        f"range = [{min(front_sizes)}, {max(front_sizes)}]",
    ]

    report_text = "\n".join(report)
    print(f"\n{report_text}")

    report_path = output_dir / "quality_indicators_report.txt"
    report_path.write_text(report_text)
    print(f"\nResults → {csv_path}")
    print(f"Report  → {report_path}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("Warning: matplotlib not available, skipping --plot", file=sys.stderr)
        else:
            _write_trackb_figures(output_dir, hvs, igds, sps)

    return 0


def _write_trackb_figures(output_dir: Path,
                         hvs: List[float], igds: List[float], sps: List[float]) -> None:
    """Write Track B indicator figures (boxplots) to output_dir."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.boxplot([hvs], labels=["NSGA-II"])
    ax.set_ylabel("Hypervolume")
    ax.set_title("Track B: Hypervolume (higher = better)")
    fig.tight_layout()
    path = output_dir / "fig_trackb_hypervolume.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure → {path}")

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.boxplot([igds], labels=["NSGA-II"])
    ax.set_ylabel("IGD+")
    ax.set_title("Track B: IGD+ (lower = better)")
    fig.tight_layout()
    path = output_dir / "fig_trackb_igd_plus.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure → {path}")

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.boxplot([sps], labels=["NSGA-II"])
    ax.set_ylabel("Spacing")
    ax.set_title("Track B: Spacing (lower = more uniform)")
    fig.tight_layout()
    path = output_dir / "fig_trackb_spacing.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure → {path}")


if __name__ == "__main__":
    sys.exit(main())

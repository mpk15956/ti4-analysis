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

def igd_plus(front: np.ndarray, reference_front: np.ndarray) -> float:
    """
    IGD+ (Ishibuchi et al. 2015): modified IGD that is Pareto-compliant.

    For each reference point, find the minimum modified distance to any
    point in the front, where the modified distance only penalises dimensions
    where the front point is worse.
    """
    if len(front) == 0 or len(reference_front) == 0:
        return float('inf')

    total = 0.0
    for r in reference_front:
        min_dist = float('inf')
        for p in front:
            diff = np.maximum(p - r, 0.0)
            d = float(np.sqrt(np.sum(diff ** 2)))
            if d < min_dist:
                min_dist = d
        total += min_dist

    return total / len(reference_front)


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


def load_pareto_archive(csv_path: Path) -> np.ndarray:
    """
    Load a Pareto archive CSV into an (N, 3) array of objective values.

    Expected columns: jains_index, morans_i, lisa_penalty
    Converts to minimisation objectives: (1-JFI, |morans_i|, lisa_penalty)
    """
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            jfi = float(row["jains_index"])
            mi = float(row["morans_i"])
            lp = float(row["lisa_penalty"])
            rows.append([1.0 - jfi, abs(mi), lp])
    return np.array(rows) if rows else np.empty((0, 3))


def main() -> int:
    args = parse_args()
    archive_dir = Path(args.archive_dir)
    output_dir = Path(args.output_dir) if args.output_dir else archive_dir

    if not archive_dir.exists():
        print(f"ERROR: {archive_dir} does not exist", file=sys.stderr)
        return 1

    archive_files = sorted(archive_dir.glob("pareto_archive_seed*.csv"))
    if not archive_files:
        print(f"No pareto_archive_seed*.csv files found in {archive_dir}",
              file=sys.stderr)
        return 1

    print(f"Loading {len(archive_files)} Pareto archives from {archive_dir}")

    all_fronts = {}
    all_points = []
    for f in archive_files:
        seed = f.stem.replace("pareto_archive_seed", "")
        front = load_pareto_archive(f)
        if len(front) > 0:
            all_fronts[seed] = front
            all_points.append(front)

    if not all_points:
        print("No valid Pareto archives found", file=sys.stderr)
        return 1

    combined = np.vstack(all_points)

    if args.reference == "auto":
        worst = combined.max(axis=0)
        ref = worst * 1.1
    else:
        ref = np.array([float(x) for x in args.reference.split(",")])

    print(f"Reference point: {ref}")
    print(f"Combined reference front: {len(combined)} points")

    results = []
    for seed, front in sorted(all_fronts.items()):
        hv = hypervolume(front, ref, args.mc_samples)
        igdp = igd_plus(front, combined)
        sp = spacing(front)
        results.append({
            "seed": seed,
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

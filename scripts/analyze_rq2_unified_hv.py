#!/usr/bin/env python3
"""
RQ2 formal closer: at every evaluation budget, one-tailed Wilcoxon
signed-rank test of NSGA-II hypervolume vs each scalar algorithm
(RS, HC, SA, SGA, TS) with Holm-Bonferroni correction across the 5
pairwise tests at that budget.

§3.10 RQ2:
    H_0: NSGA-II's hypervolume does not exceed scalar algorithms'
         hypervolume under the canonical formulation at equal total
         evaluation budget.

The descriptive panel (`unified_hv_analysis.py`) runs a two-tailed
Wilcoxon over all algorithm pairs. RQ2 specifically requires a
one-tailed comparison against NSGA-II, so this script consumes the
per-budget `unified_hv.csv` files emitted by `unified_hv_analysis.py`
and runs the formal RQ2 panel.

Usage:
    python scripts/analyze_rq2_unified_hv.py \\
        --hv-dirs output/run_tag/unified_hv_b*/  \\
        --output  output/run_tag/rq2_unified_hv.txt
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests


SCALAR_ALGOS = ("rs", "hc", "sa", "sga", "ts")
NSGA_KEY = "nsga2"


def load_unified_hv(csv_path: Path) -> List[Dict]:
    """Read a unified_hv.csv as a list of {algorithm, seed, budget, hv} dicts."""
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "algorithm": r["algorithm"].lower(),
                "seed":      int(r["seed"]),
                "budget":    int(r["budget"]),
                "hv":        float(r["hv"]),
            })
    return rows


def rq2_one_budget(rows: List[Dict], budget: int) -> Tuple[List[Dict], float, float]:
    """
    For one budget, run one-tailed Wilcoxon NSGA-II vs each scalar (alternative
    'greater' — H_1: NSGA-II HV > scalar HV) on paired-by-seed samples, then
    Holm-correct across the 5 tests. Returns (per-pair test rows,
    nsga_median_hv, scalars_median_hv).
    """
    rows_b = [r for r in rows if r["budget"] == budget]
    if not rows_b:
        return [], float("nan"), float("nan")

    by_algo_seed: Dict[Tuple[str, int], float] = {}
    for r in rows_b:
        # In multi-chain runs, unified archives can have one row per chain;
        # take the per-seed mean to align with the §3.9 chain aggregation.
        key = (r["algorithm"], r["seed"])
        by_algo_seed.setdefault(key, [])
        by_algo_seed[key].append(r["hv"])
    by_algo_seed_mean = {k: float(np.mean(v)) for k, v in by_algo_seed.items()}

    nsga_seeds = {s for (a, s) in by_algo_seed_mean if a == NSGA_KEY}
    if not nsga_seeds:
        return [], float("nan"), float("nan")

    panel = []
    raw_p = []
    for scalar in SCALAR_ALGOS:
        scalar_seeds = {s for (a, s) in by_algo_seed_mean if a == scalar}
        common = sorted(nsga_seeds & scalar_seeds)
        if len(common) < 5:
            continue
        x = np.array([by_algo_seed_mean[(NSGA_KEY, s)] for s in common])  # NSGA-II
        y = np.array([by_algo_seed_mean[(scalar, s)] for s in common])    # scalar
        try:
            W, p = stats.wilcoxon(x, y, alternative="greater")
        except ValueError:
            W, p = float("nan"), 1.0
        # Vargha-Delaney A12: P(NSGA-II > scalar) + 0.5 * P(tie)
        diff = x[:, None] - y[None, :]
        a12 = float((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(common) ** 2)
        panel.append({
            "scalar":     scalar,
            "n":          len(common),
            "W":          W,
            "p_raw":      p,
            "vda_A":      a12,
            "median_nsga": float(np.median(x)),
            "median_scalar": float(np.median(y)),
        })
        raw_p.append(p)

    if raw_p:
        reject, p_corr, _, _ = multipletests(raw_p, method="holm")
        for row, pc, sig in zip(panel, p_corr, reject):
            row["p_holm"] = float(pc)
            row["significant"] = bool(sig)

    nsga_vals = [v for (a, _), v in by_algo_seed_mean.items() if a == NSGA_KEY]
    scalar_vals = [v for (a, _), v in by_algo_seed_mean.items() if a in SCALAR_ALGOS]
    return panel, float(np.median(nsga_vals)), float(np.median(scalar_vals))


def write_report(per_budget: Dict[int, Tuple[List[Dict], float, float]],
                 out_path: Path) -> None:
    lines: List[str] = []
    rule = "=" * 92
    lines.append(rule)
    lines.append("RQ2: NSGA-II HYPERVOLUME vs SCALAR ALGORITHMS — ONE-TAILED WILCOXON + HOLM")
    lines.append(rule)
    lines.append("")
    lines.append("H_0: NSGA-II HV does not exceed scalar HV under the canonical formulation.")
    lines.append("Test: paired Wilcoxon signed-rank, alternative='greater', Holm-Bonferroni")
    lines.append("      across the 5 (NSGA-II vs scalar) tests at each budget.")
    lines.append("")
    for budget in sorted(per_budget):
        panel, m_nsga, m_scalar = per_budget[budget]
        lines.append("-" * 92)
        lines.append(f"Budget = {budget}")
        lines.append(f"  median(NSGA-II HV) = {m_nsga:.6f}   median(all scalars HV) = {m_scalar:.6f}")
        if not panel:
            lines.append("  (no comparable seeds at this budget)")
            continue
        for row in panel:
            sig = "***" if row.get("significant", False) else "   "
            lines.append(
                f"  NSGA-II vs {row['scalar']:<4s}  n={row['n']:3d}  "
                f"W={row['W']:.3f}  p_raw={row['p_raw']:.4g}  "
                f"p_holm={row.get('p_holm', float('nan')):.4g} {sig}  "
                f"VDA(A_NSGA>scalar)={row['vda_A']:.3f}  "
                f"medians={row['median_nsga']:.5f}/{row['median_scalar']:.5f}"
            )
    lines.append("")
    lines.append(rule)
    lines.append("RQ2 verdict per budget: NSGA-II's HV exceeds the scalar's at p_holm < 0.05.")
    lines.append("Significant pairs are flagged ***. Direction also confirmed by VDA > 0.5.")
    lines.append(rule)
    out_path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hv-dirs", nargs="+", required=True, type=Path,
                   help="One or more directories each containing a unified_hv.csv")
    p.add_argument("--output", type=Path, required=True,
                   help="Path for the RQ2 report.txt")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rows: List[Dict] = []
    for d in args.hv_dirs:
        csv_path = d / "unified_hv.csv"
        if not csv_path.exists():
            print(f"WARNING: {csv_path} missing — skipping", file=sys.stderr)
            continue
        rows.extend(load_unified_hv(csv_path))
    if not rows:
        print("ERROR: no unified_hv.csv data loaded", file=sys.stderr)
        return 1

    budgets = sorted({r["budget"] for r in rows})
    per_budget = {b: rq2_one_budget(rows, b) for b in budgets}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_report(per_budget, args.output)
    print(f"RQ2 report written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

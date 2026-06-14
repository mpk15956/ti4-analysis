#!/usr/bin/env python3
"""
Derive the Phase 6/7 manuscript claim-values FROM canonical artifacts.

Single source: this reads the finalized Phase 6/7 output dir (Track B
quality_indicators.csv, unified HV, cross-method IGD, RQ3 Spearman) and reuses
`analyze_rq2_unified_hv`'s own functions for the RQ2 panel — it does NOT
re-implement any statistic or hand-transcribe any number. Output is a JSON of
the claim-defining quantities (Track B magnitudes; RQ2 VDA/Holm-p/medians at the
canonical budget; RQ3 pooled rho+sign), which `tests/test_phase6_canonical_values.py`
pins and which the §3.10 prose cites once written.

Usage:
    python scripts/generate_manuscript_values.py \
        --finalize-dir output/paper1_500k_finalize_phase67_YYYYMMDD_HHMMSS \
        --budget 500000 \
        --out output/paper1_500k_finalize_phase67_YYYYMMDD_HHMMSS/manuscript_values_phase6.json
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

NSGA = "nsga2"
SCALARS = ["rs", "hc", "sa", "sga", "ts"]


def track_b(finalize_dir: Path) -> dict:
    """Mean HV / IGD+ / Spacing over all Track B rows (quality_indicators.csv)."""
    path = finalize_dir / "quality_indicators.csv"
    hv, igd, sp = [], [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            hv.append(float(r["hypervolume"]))
            igd.append(float(r["igd_plus"]))
            sp.append(float(r["spacing"]))
    return {
        "n_rows": len(hv),
        "hv_mean": statistics.mean(hv),
        "igd_plus_mean": statistics.mean(igd),
        "spacing_mean": statistics.mean(sp),
    }


def rq2_all_budgets(finalize_dir: Path, scripts_dir: Path) -> dict:
    """RQ2 panels at EVERY budget, via analyze_rq2's own functions (single
    source). All budgets are needed to pin the NSGA-II-vs-SA crossover (b1000
    endpoint, 500k endpoint, and the trend between)."""
    sys.path.insert(0, str(scripts_dir))
    import analyze_rq2_unified_hv as rq2mod
    rows = rq2mod.load_unified_hv(finalize_dir / "unified_hv_all" / "unified_hv.csv")
    budgets = sorted({r["budget"] for r in rows})
    out = {}
    for b in budgets:
        panel, m_nsga, _ = rq2mod.rq2_one_budget(rows, b)
        by_scalar = {row["scalar"]: row for row in panel}
        pairs = {}
        for s in SCALARS:
            row = by_scalar.get(s)
            if row is None:
                continue
            pairs[s] = {
                "n": row["n"],
                "vda_A": row["vda_A"],            # P(NSGA-II HV > scalar HV)
                "p_holm": row.get("p_holm"),
                "significant": bool(row.get("significant", False)),
                "median_nsga": row["median_nsga"],
                "median_scalar": row["median_scalar"],
            }
        out[str(b)] = {"median_nsga_hv": m_nsga, "pairs": pairs}
    return out


def rq3_pooled(finalize_dir: Path) -> dict:
    """Pooled Spearman rho (+sign) per metric from rq3_spearman.csv."""
    path = finalize_dir / "rq3_spearman.csv"
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            # Pooled rows are tagged algorithm=POOLED (budget -1). Column names
            # are discovered defensively (rho vs spearman_rho).
            algo = (r.get("algorithm") or r.get("algo") or "").upper()
            if algo != "POOLED":
                continue
            metric = r.get("metric")
            rho_key = next((k for k in r if "rho" in k.lower() or "spearman" in k.lower()), None)
            if metric and rho_key:
                out[metric] = float(r[rho_key])
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--finalize-dir", type=Path, required=True)
    p.add_argument("--budget", type=int, default=500000)
    p.add_argument("--scripts-dir", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    fd = args.finalize_dir
    values = {
        "provenance": {
            "finalize_dir": str(fd),
            "canonical_budget": args.budget,
            "note": "Derived from canonical Phase 6/7 artifacts; do not hand-edit.",
        },
        "track_b": track_b(fd),
        "rq2_by_budget": rq2_all_budgets(fd, args.scripts_dir),
        "rq3_pooled_spearman": rq3_pooled(fd),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.out}")
    print(json.dumps(values, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

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


def rq4_evals_to_best(finalize_dir: Path, budget: int = 500000) -> dict:
    """RQ4 anytime-composite: the evals_to_best Friedman omnibus across all six
    algorithms, read single-source from analyze_benchmark's structured CSV
    (rq4_friedman_evals_to_best.csv).

    Active verification, two regimes:
      - CSV absent  → RQ4 not yet finalized (the pre-re-run state). Return
        {"available": False, ...} so RQ2/RQ3 still regenerate; the pin stays red.
      - CSV present but REFUSED / NaN chi2 / missing an algorithm → a corrupt or
        partial omnibus (the NSGA-II sentinel defect). RAISE — RQ4 is pre-
        registered over all six, so a broken result is a hard stop, never a
        silently-emitted number.
    """
    # The canonical-budget evals_to_best Friedman is written by analyze_benchmark
    # to <finalize>/stats_b<budget>/stats/. Prefer a top-level copy if a finalize
    # script surfaced one; otherwise read the per-budget stats the Phase 7 loop
    # already produces (works for both submit_500k_phase67.sh and stage2 without
    # editing their assembly).
    candidates = [
        finalize_dir / "rq4_friedman_evals_to_best.csv",
        finalize_dir / f"stats_b{budget}" / "stats" / "rq4_friedman_evals_to_best.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return {"available": False,
                "reason": (f"rq4_friedman_evals_to_best.csv not found (looked in finalize "
                           f"root and stats_b{budget}/stats) — RQ4 re-run not yet finalized")}
    with open(path) as f:
        row = next(csv.DictReader(f))
    note = (row.get("note") or "").strip()
    chi2 = (row.get("chi2") or "").strip()
    if note or chi2.lower() in ("", "nan"):
        raise ValueError(
            f"RQ4 Friedman present but unusable (note={note!r}, chi2={chi2!r}); "
            f"refusing to emit a manuscript value. Re-run the multi-algo benchmark "
            f"with NSGA-II evals_to_best instrumented."
        )
    algos = [a for a in (row.get("algorithms") or "").split(",") if a]
    expected = ["rs", "hc", "sa", "sga", "nsga2", "ts"]
    missing = [a for a in expected if a not in algos]
    if missing:
        raise ValueError(
            f"RQ4 omnibus is missing algorithms {missing} (got {algos}); the "
            f"pre-registered test is paired across all six."
        )
    return {
        "available": True,
        "chi2": float(chi2),
        "p_friedman": float(row["p_friedman"]),
        "df": int(float(row["df"])),
        "n": int(float(row["n"])),
        "significant": str(row.get("significant", "")).strip().lower() in ("true", "1"),
        "algorithms": algos,
    }


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
        "rq4_evals_to_best": rq4_evals_to_best(fd, args.budget),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.out}")
    print(json.dumps(values, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

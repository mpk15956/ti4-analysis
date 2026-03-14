#!/usr/bin/env python3
"""Run preflight checks: conditions, row counts, and run_config (corrected_landscape + use_local_variance_lisa)."""
import csv
import glob
import json
import sys

def main():
    f = sorted(glob.glob("output/preflight_final/benchmark_*/results.csv"))[0]
    rows = list(csv.DictReader(open(f)))
    conditions = sorted(set(r["condition"] for r in rows))
    bad = [r for r in rows if r.get("composite_score", "") in ("nan", "inf", "")]

    print("Check 1 - Conditions:", conditions)
    assert conditions == [
        "full_composite",
        "jfi_moran",
        "jfi_only",
        "lsap_only",
        "moran_only",
    ], f"FAIL: {conditions}"

    print("Check 2 - Rows:", len(rows), "Bad:", len(bad))
    assert len(rows) == 15, f"Expected 15, got {len(rows)}"
    assert len(bad) == 0, f"Bad rows: {bad}"

    cfg_path = sorted(glob.glob("output/preflight_final/*/run_config.json"))[0]
    cfg = json.load(open(cfg_path))
    print("Check 3 - corrected_landscape:", cfg.get("corrected_landscape"))
    print("Check 3 - use_local_variance_lisa:", cfg.get("use_local_variance_lisa"))
    use_lisa = cfg.get("use_local_variance_lisa")
    if use_lisa is None:
        use_lisa = cfg.get("corrected_landscape", False)
    assert cfg.get("corrected_landscape") is True, "corrected_landscape must be True"
    assert use_lisa is True, "use_local_variance_lisa must be True (or inferred from corrected_landscape)"
    print("ALL CHECKS PASS")
    return 0

if __name__ == "__main__":
    sys.exit(main())

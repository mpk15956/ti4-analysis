"""
Canonical-pins regression test for the Phase 6/7 manuscript claims (json<->canonical
edge). Re-derives Track B / RQ2 / RQ3 quantities FROM the canonical finalize dir
via the single-source generator (scripts/generate_manuscript_values.py, which in
turn reuses analyze_rq2's own functions), then asserts each claim at the
granularity the claim is made — effect-size bands for the RQ2 SA crossover, a
VDA floor + significance for dominance, magnitude-to-precision for Track B, and
sign+magnitude for RQ3. It guards the CLAIM, not a brittle point estimate.

The prose<->json edge (matching these against §3.10 text) is added to
manuscript_values.yaml once §3.10's RQ2 results are written (line 212 is the
placeholder slot today).

Skips cleanly if the canonical finalize dir is not present (e.g. a fresh
checkout without output/), matching the existing manuscript-values convention.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "src"))

CANONICAL_BUDGET = 500000
SCALARS_DOMINATED = ["rs", "hc", "sga", "ts"]  # SA is the strong baseline, handled separately


def _finalize_dir() -> Path:
    cands = sorted((ROOT / "output").glob("paper1_500k_finalize_phase67_*"))
    if not cands:
        pytest.skip("no canonical Phase 6/7 finalize dir present under output/")
    return cands[-1]


@pytest.fixture(scope="module")
def values():
    import generate_manuscript_values as gen
    fd = _finalize_dir()
    if not (fd / "unified_hv_all" / "unified_hv.csv").exists():
        pytest.skip(f"finalize dir {fd} missing unified_hv_all/unified_hv.csv")
    return {
        "track_b": gen.track_b(fd),
        "rq2": gen.rq2_all_budgets(fd, SCRIPTS),
        "rq3": gen.rq3_pooled(fd),
    }


# ── Track B: magnitudes to reported precision (guards the IGD+ reference fix) ──

def test_track_b_igd_plus_small(values):
    # ~0.0011 with the per-budget non-dominated reference; a reversion to the
    # raw-union reference would inflate this by orders of magnitude.
    assert 0.0008 < values["track_b"]["igd_plus_mean"] < 0.0014


def test_track_b_hv_and_spacing(values):
    assert 0.235 < values["track_b"]["hv_mean"] < 0.245
    assert 0.020 < values["track_b"]["spacing_mean"] < 0.033


# ── RQ2: NSGA-II vs SA budget-dependent crossover (the load-bearing finding) ──

def test_rq2_sa_low_budget_tie(values):
    # b1000: SA matches/leads — NSGA-II does NOT significantly exceed it.
    sa = values["rq2"]["1000"]["pairs"]["sa"]
    assert sa["vda_A"] < 0.50
    assert sa["significant"] is False


def test_rq2_sa_canonical_significant_but_negligible(values):
    # 500k: significant (paired) but negligible effect; near-identical medians.
    sa = values["rq2"][str(CANONICAL_BUDGET)]["pairs"]["sa"]
    assert sa["significant"] is True
    assert 0.50 < sa["vda_A"] < 0.60
    assert abs(sa["median_nsga"] - sa["median_scalar"]) < 0.0005


def test_rq2_sa_crossover_and_negligible_everywhere(values):
    # The operational claim: once past the crossover (>= b5000) NSGA-II is
    # consistently significant, and at EVERY budget the effect stays negligible
    # (VDA never escapes the small band) — SA never separates like the others.
    for b, panel in values["rq2"].items():
        sa = panel["pairs"]["sa"]
        assert 0.42 < sa["vda_A"] < 0.65, (b, sa["vda_A"])
        if int(b) >= 5000:
            assert sa["significant"] is True, (b, sa["p_holm"])


# ── RQ2: NSGA-II dominance over the OTHER scalars at 500k (different class) ──

def test_rq2_dominance_over_other_scalars(values):
    pairs = values["rq2"][str(CANONICAL_BUDGET)]["pairs"]
    for s in SCALARS_DOMINATED:
        assert pairs[s]["vda_A"] > 0.80, (s, pairs[s]["vda_A"])
        assert pairs[s]["significant"] is True
        assert pairs[s]["p_holm"] < 0.01, (s, pairs[s]["p_holm"])


# ── RQ3: pooled Spearman rho — signs are load-bearing for the 3-test defense ──

def test_rq3_pooled_signs_and_magnitudes(values):
    rho = values["rq3"]
    assert 0.15 < rho["morans_i"] < 0.28
    assert 0.17 < rho["lisa_penalty"] < 0.30
    assert -0.60 < rho["jains_index"] < -0.45

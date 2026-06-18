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
# RQ4's six-way omnibus is read at 200k, NOT the 500k RQ1/RQ2 canonical: the
# targeted NSGA-II evals_to_best fill (Option 1) covers budgets <=200k, so 200k
# is the LARGEST budget where all six algorithms carry a real (non-sentinel)
# evals_to_best. At 500k NSGA-II was not re-run, so RQ4 there is five-way
# descriptive only. Decoupled from CANONICAL_BUDGET so RQ1/RQ2 stay 500k-canonical.
RQ4_BUDGET = 200000
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
        "rq4": gen.rq4_evals_to_best(fd, RQ4_BUDGET),
        "rq4_breadth_tax": gen.rq4_breadth_tax(fd, RQ4_BUDGET),
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


def test_rq2_sa_crossover_and_below_threshold_everywhere(values):
    # The operational claim: once past the crossover (>= b5000) NSGA-II is
    # consistently significant, and at EVERY budget the SA effect stays below the
    # pre-registered practical-significance threshold (A >= 0.64; Design_Rationale
    # section 3), so SA never separates from NSGA-II the way the other four
    # scalars do. NB: 0.61 max is "small" by the textbook VDA bands, not
    # "negligible"; the load-bearing line is the pre-committed 0.64, not 0.56.
    for b, panel in values["rq2"].items():
        sa = panel["pairs"]["sa"]
        assert 0.42 < sa["vda_A"] < 0.64, (b, sa["vda_A"])
        if int(b) >= 5000:
            assert sa["significant"] is True, (b, sa["p_holm"])


# ── RQ2: NSGA-II dominance over the OTHER scalars at 500k (different class) ──

def test_rq2_dominance_over_other_scalars(values):
    pairs = values["rq2"][str(CANONICAL_BUDGET)]["pairs"]
    for s in SCALARS_DOMINATED:
        assert pairs[s]["vda_A"] > 0.80, (s, pairs[s]["vda_A"])
        assert pairs[s]["significant"] is True
        assert pairs[s]["p_holm"] < 0.01, (s, pairs[s]["p_holm"])


# ── RQ4: evals_to_best six-way Friedman (RED until the multi-algo re-run) ──
# Structural pins only: they guard that the pre-registered six-way omnibus ran on
# real, non-sentinel data for all six algorithms. The specific p/significance is
# folded in alongside the §3.10 RQ4 prose after the re-run (step G).

def test_rq4_available_after_rerun(values):
    # Red until F: the canonical multi-algo re-run must finalize the RQ4 omnibus
    # CSV. Absent → NSGA-II evals_to_best not yet instrumented + re-run.
    assert values["rq4"].get("available") is True, values["rq4"].get("reason")


def test_rq4_omnibus_over_all_six_algorithms(values):
    # The anti-sentinel guarantee: all six algorithms (including NSGA-II) entered
    # the omnibus on real data — no silent five-algorithm subset, no constant -1
    # column corrupting the rank-ANOVA.
    rq4 = values["rq4"]
    assert rq4.get("available") is True, rq4.get("reason")
    assert set(rq4["algorithms"]) == {"rs", "hc", "sa", "sga", "nsga2", "ts"}
    assert rq4["df"] == 5


def test_rq4_omnibus_statistic_is_real(values):
    rq4 = values["rq4"]
    assert rq4.get("available") is True, rq4.get("reason")
    assert rq4["chi2"] == rq4["chi2"] and rq4["chi2"] > 0  # finite (not NaN), positive
    assert 0.0 <= rq4["p_friedman"] <= 1.0
    assert rq4["n"] > 0


def test_rq4_breadth_tax_canonical(values):
    # The depth-vs-breadth anchor for §3.10: at the 200k RQ4 budget NSGA-II
    # reaches its best 1:1:1 composite only after a large but sub-budget median
    # number of evaluations (the "breadth tax"). Guards the json<-csv edge for
    # the 134,100 / 67% figure the §3.10 prose cites; the prose<-json edge is in
    # manuscript_values.yaml.
    bt = values["rq4_breadth_tax"]
    assert bt.get("available") is True, bt.get("reason")
    assert bt["budget"] == RQ4_BUDGET
    assert 100_000 < bt["nsga2_median_evals_to_best"] < RQ4_BUDGET
    assert 0.60 < bt["budget_fraction"] < 0.72


# ── RQ3: pooled Spearman rho — signs are load-bearing for the 3-test defense ──

def test_rq3_pooled_signs_and_magnitudes(values):
    rho = values["rq3"]
    assert 0.15 < rho["morans_i"] < 0.28
    assert 0.17 < rho["lisa_penalty"] < 0.30
    assert -0.60 < rho["jains_index"] < -0.45

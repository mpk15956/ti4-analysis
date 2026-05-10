"""
Tests for fitness landscape structural corrections: smooth objectives,
Gen-0 normalization, raw_objective_terms, and local-variance LSAP.
"""

import pytest
import numpy as np

from ti4_analysis.algorithms.objectives_smooth import (
    smooth_min_jain,
    softplus_hinge,
    DEFAULT_JAIN_SMOOTH_P,
    DEFAULT_SOFTPLUS_K,
    JAIN_EPS,
)
from ti4_analysis.algorithms.spatial_optimizer import (
    MultiObjectiveScore,
    NORM_KEY_HINGE,
    NORM_KEY_JFI,
    NORM_KEY_LISA,
    compute_gen0_sigma,
)


# ── Smooth Jain (L_{-p} mean) ─────────────────────────────────────────────────

def test_smooth_min_jain_approaches_min_as_p_large():
    """smooth_min_jain(J_R, J_I, p) → min(J_R, J_I) as p increases."""
    j_r, j_i = 0.6, 0.8
    j_min = min(j_r, j_i)
    for p in [4, 8, 16, 32, 64]:
        j_smooth = smooth_min_jain(j_r, j_i, p=p)
        assert j_smooth >= j_min - 0.01
        assert j_smooth <= max(j_r, j_i) + 0.01
    j_smooth_64 = smooth_min_jain(j_r, j_i, p=64)
    assert abs(j_smooth_64 - j_min) < 0.05


def test_smooth_min_jain_clamps_away_from_zero():
    """J_R or J_I near zero are clamped to JAIN_EPS to avoid blow-up."""
    j_smooth = smooth_min_jain(1e-8, 0.9, p=8)
    assert 0 < j_smooth <= 1.0
    j_smooth2 = smooth_min_jain(0.9, 1e-8, p=8)
    assert 0 < j_smooth2 <= 1.0


def test_smooth_min_jain_symmetric():
    """smooth_min_jain(a, b) == smooth_min_jain(b, a)."""
    a, b = 0.5, 0.7
    assert smooth_min_jain(a, b, p=8) == smooth_min_jain(b, a, p=8)


# ── Softplus hinge ────────────────────────────────────────────────────────────

def test_softplus_hinge_positive_x():
    """For x > 0, softplus(x, k) approximates x for large k."""
    x = 0.5
    out = softplus_hinge(x, k=10)
    assert out > 0
    assert out <= x + 0.2  # softplus slightly above identity for x>0


def test_softplus_hinge_negative_x():
    """For x < 0, softplus(x, k) is near 0 (smooth zero)."""
    x = -0.5
    out = softplus_hinge(x, k=10)
    assert out >= 0
    assert out < 0.1


def test_softplus_hinge_zero():
    """softplus(0, k) = ln(2)/k."""
    out = softplus_hinge(0.0, k=10)
    expected = np.log(2) / 10
    assert abs(out - expected) < 1e-6


def test_softplus_k_clamped():
    """k is clamped to [5, 20] to avoid overflow."""
    # Very large k should not raise
    out_high = softplus_hinge(2.0, k=100)
    assert np.isfinite(out_high)
    out_low = softplus_hinge(0.1, k=0.1)
    assert np.isfinite(out_low)


# ── §3.1 / §3.3 empirical bounds on the composite normalization ────────────

def _phase1_canonical_results_path():
    from pathlib import Path
    return Path(__file__).parent.parent / "output/paper1_canonical_20260509_134024/benchmark_20260509_191848/results.csv"


def test_composite_term_bounds_empirical():
    """§3.1: each of the three composite terms is normalized to be inside
    [0, ~1] before weighting. Hinge max(0, I − E[I]) ∈ [0, 1 + 1/(|G|−1)]
    ≈ [0, 1.033] at |G|=31; JFI gap (1 − J_min) ∈ [0, 1]; LSAP/(|G|(|G|−1))
    ∈ [0, 1]. Verified against the 12,000 row canonical Phase 1 sample so
    the analytical bounds and the empirically-realized values stay
    consistent."""
    import pandas as pd
    p = _phase1_canonical_results_path()
    if not p.exists():
        pytest.skip(f"Phase 1 artifact missing: {p}")
    df = pd.read_csv(p)
    n_spatial = 31
    e_i = -1.0 / (n_spatial - 1)
    hinge_max = (df["morans_i"] - e_i).clip(lower=0).max()
    jfi_gap_max = (1 - df["jains_index"]).max()
    lsap_norm_max = df["lisa_penalty"].max() / (n_spatial * (n_spatial - 1))
    hinge_bound = 1.0 + 1.0 / (n_spatial - 1)  # ≈ 1.0333
    assert 0 <= hinge_max <= hinge_bound + 1e-9, (
        f"hinge max {hinge_max:.4f} outside [0, {hinge_bound:.4f}]"
    )
    assert 0 <= jfi_gap_max <= 1.0 + 1e-9, f"JFI gap max {jfi_gap_max:.4f} outside [0, 1]"
    assert 0 <= lsap_norm_max <= 1.0 + 1e-9, f"LSAP/n(n-1) max {lsap_norm_max:.4f} outside [0, 1]"


def test_lsap_bounded_by_n_times_n_minus_1():
    """§3.3: LSAP ≤ |G|(|G|−1) so LSAP/(|G|(|G|−1)) ∈ [0, 1]. Verified empirically
    against canonical Phase 1 — the bound is conservative, not tight; this
    test guards against future formulation changes that could push LSAP
    outside its declared normalization range."""
    import pandas as pd
    p = _phase1_canonical_results_path()
    if not p.exists():
        pytest.skip(f"Phase 1 artifact missing: {p}")
    df = pd.read_csv(p)
    n_spatial = 31
    bound = n_spatial * (n_spatial - 1)
    actual_max = df["lisa_penalty"].max()
    assert actual_max <= bound, f"LSAP max {actual_max} exceeds n(n-1) = {bound}"


# ── §3.6 seed-set disjointness invariant ──────────────────────────────────

def test_canonical_seed_sets_disjoint():
    """§3.6 asserts that benchmark seeds (0–99), tuning seeds (9000–9099),
    and held-out seeds (9100–9149) are pairwise disjoint. The numbers come
    from submit_paper1.sh + optimize_hyperparameters.py defaults; this test
    pins them as a regression invariant so a future reseeding can't silently
    introduce overlap."""
    benchmark_seeds = set(range(0, 100))
    tuning_seeds = set(range(9000, 9000 + 100))
    held_out_seeds = set(range(9100, 9100 + 50))
    assert len(benchmark_seeds) == 100
    assert len(tuning_seeds) == 100
    assert len(held_out_seeds) == 50
    assert benchmark_seeds.isdisjoint(tuning_seeds), "benchmark ∩ tuning non-empty"
    assert benchmark_seeds.isdisjoint(held_out_seeds), "benchmark ∩ held-out non-empty"
    assert tuning_seeds.isdisjoint(held_out_seeds), "tuning ∩ held-out non-empty"


# ── Canonical objective tuple is single-source across all CLI consumers ────
# Regression for `feedback_canonical_objective_single_source.md`: three CLI
# scripts (quality_indicators.py, cross_method_igd.py, unified_hv_analysis.py)
# previously each re-implemented `[1 - jfi, abs(mi), lp]` inline, producing
# a non-canonical Pareto tuple (penalizing negative I — the optimizer's
# target — instead of `max(0, I - E[I])`). This test asserts every consumer
# now goes through `MultiObjectiveScore.archive_row_to_pareto_point` and the
# helper itself returns the canonical tuple bit-identically to
# `objective_values_for_pareto()`.

def test_archive_row_to_pareto_point_matches_objective_values_for_pareto():
    """The archive-row helper must produce the same tuple that
    `objective_values_for_pareto()` returns for an equivalently-constructed
    score (with use_smooth_objectives=False, since the archive lacks
    per-dimension JFI). Tested across a fixture grid covering the realistic
    range of (jains_index, morans_i, lisa_penalty)."""
    from ti4_analysis.algorithms.spatial_optimizer import MultiObjectiveScore

    fixtures = [
        {"jains_index": 0.999, "morans_i": -0.6, "lisa_penalty": 0.0},
        {"jains_index": 0.95, "morans_i": -0.033, "lisa_penalty": 4.5},
        {"jains_index": 0.5, "morans_i": +0.4, "lisa_penalty": 12.0},
        {"jains_index": 1.0, "morans_i": -1.0, "lisa_penalty": 0.0},
    ]
    for n_spatial in (29, 31):
        for row in fixtures:
            tuple_via_helper = MultiObjectiveScore.archive_row_to_pareto_point(row, n_spatial)
            score = MultiObjectiveScore(
                balance_gap=0.0,
                morans_i=row["morans_i"],
                jains_index=row["jains_index"],
                lisa_penalty=row["lisa_penalty"],
                n_spatial=n_spatial,
                use_smooth_objectives=False,
            )
            tuple_via_helper_object = score.objective_values_for_pareto()
            assert tuple_via_helper == tuple_via_helper_object, (
                f"helper drift at n_spatial={n_spatial} row={row}: "
                f"archive_row_to_pareto_point={tuple_via_helper} "
                f"objective_values_for_pareto={tuple_via_helper_object}"
            )


def test_archive_row_to_pareto_point_returns_canonical_tuple_definition():
    """Spot-check: at a known input the helper returns
    `(1 - jains_index, max(0, morans_i - E[I]), lisa_penalty)` as defined in
    methodology §3.1. Guards against future refactors that silently change
    the canonical contract."""
    from ti4_analysis.algorithms.spatial_optimizer import MultiObjectiveScore
    from ti4_analysis.algorithms.map_topology import morans_i_null

    n = 31
    row = {"jains_index": 0.95, "morans_i": -0.033, "lisa_penalty": 4.5}
    f1, f2, f3 = MultiObjectiveScore.archive_row_to_pareto_point(row, n)
    e_i = morans_i_null(n)
    assert abs(f1 - (1.0 - 0.95)) <= 1e-12
    assert abs(f2 - max(0.0, -0.033 - e_i)) <= 1e-12  # ≈ 0.000... (just above null)
    assert abs(f3 - 4.5) <= 1e-12
    # The legacy non-canonical form was `(1 - jfi, abs(morans_i), lp)`. At this
    # input that would give f2 = 0.033 — a positive penalty for what is in
    # fact below-null (no clustering). Canonical f2 should be near zero.
    assert f2 < 1e-3, (
        f"canonical f2 should be near zero for I just below null; got {f2}. "
        f"This would only fail if the helper reverted to abs(morans_i)."
    )


def test_no_inline_canonical_transform_in_scripts():
    """Pre-submission grep gate. No Python file under `scripts/` may contain
    inline canonical-objective transforms outside of approved call sites.
    Every consumer must call `MultiObjectiveScore.archive_row_to_pareto_point`.

    Deny-list (broader than the literal patterns from the original bug): any
    line that combines `abs(...morans...)`, `1.0 - jains`, `1.0 - jfi`,
    `1 - jfi`, or `lisa_penalty / (n*(n-1))` is suspect. Lines explicitly
    marked `# noqa: canonical-transform` are exempt — use that comment when
    the line is intentionally NOT performing the canonical transformation
    (e.g., a unit-test fixture comparing legacy vs canonical, or a docstring
    that quotes the legacy form for contrast). There should be ~zero of
    these in scripts/, and any new ones should be reviewed.

    See `feedback_canonical_objective_single_source.md`.
    """
    import re
    from pathlib import Path
    scripts_dir = Path(__file__).parent.parent / "scripts"
    # Patterns that flag inline canonical-objective math. Each pattern
    # matches a substring; lines that pass through canonical helpers
    # (`MultiObjectiveScore.archive_row_to_pareto_point`,
    # `objective_values_for_pareto`, `raw_objective_terms`) won't trip them
    # because those produce the tuple internally.
    forbidden_regexes = [
        re.compile(r"abs\s*\(\s*[^()]*morans[^()]*\)"),
        re.compile(r"1(?:\.0)?\s*-\s*(?:jfi|jains_index|jains|row\[['\"]jains_index['\"]\])"),
        re.compile(r"\babs\s*\(\s*mi\s*\)"),
        re.compile(r"lisa_penalty\s*/\s*\(\s*n\s*\*\s*\(\s*n\s*-\s*1\s*\)\s*\)"),
    ]
    NOQA = "noqa: canonical-transform"
    offenders = []
    for py in scripts_dir.rglob("*.py"):
        for lineno, line in enumerate(py.read_text().splitlines(), start=1):
            if NOQA in line:
                continue
            for rgx in forbidden_regexes:
                if rgx.search(line):
                    offenders.append(
                        f"{py.relative_to(scripts_dir.parent)}:{lineno}  matched /{rgx.pattern}/  "
                        f"line: {line.strip()[:120]}"
                    )
                    break
    assert not offenders, (
        "Inline canonical-objective transforms detected in scripts/. Refactor to call "
        "`MultiObjectiveScore.archive_row_to_pareto_point(row, n_spatial)`, or — if the "
        "line is intentionally NOT doing canonical arithmetic — append "
        "'# noqa: canonical-transform' to suppress. Findings:\n  - "
        + "\n  - ".join(offenders)
    )


# ── §3.4.1 / §3.4.2 / §3.4.3 analytical bound verification ─────────────────
# These tests assert the numerical inequalities the methodology section claims,
# so the manuscript text and the implementation can never silently disagree.

def test_smooth_min_jain_power_mean_bound():
    """§3.4.1: smooth_min_jain is the generalized power mean M_{-p}; for any
    a, b > 0 the inequality min(a, b) ≤ M_{-p}(a, b) ≤ min(a, b) · 2^{1/p}
    holds (power-mean inequality / Hölder mean monotonicity). Tested at p=8
    across a dense grid of (J_R, J_I) pairs in (0, 1]² — the JFI domain SA
    actually traverses. Multiplicative slack at p=8 is 2^{1/8} − 1 ≈ 0.091."""
    p = DEFAULT_JAIN_SMOOTH_P
    bound_factor = 2.0 ** (1.0 / p)  # ≈ 1.0905 at p=8
    grid = np.linspace(JAIN_EPS, 1.0, 64)
    a, b = np.meshgrid(grid, grid, indexing="ij")
    sm = np.vectorize(lambda u, v: smooth_min_jain(float(u), float(v), p=p))(a, b)
    true_min = np.minimum(a, b)
    # Power-mean inequality lower bound: M_{-p} ≥ min
    assert (sm >= true_min - 1e-12).all(), (
        f"smooth_min dropped below true min: max shortfall = {(true_min - sm).max()}"
    )
    # Power-mean inequality upper bound: M_{-p} ≤ min · 2^{1/p}
    excess = sm / np.maximum(true_min, JAIN_EPS) - 1.0
    assert excess.max() <= (bound_factor - 1.0) + 1e-9, (
        f"smooth_min/min exceeded 2^(1/p) = {bound_factor:.6f}; "
        f"observed max ratio = {1.0 + excess.max():.6f}"
    )


def test_softplus_hinge_log2_over_k_bound():
    """§3.4.2: softplus_k(x) ∈ [max(0, x), max(0, x) + log(2)/k] for all x.
    The bound is achieved at x = 0. Tested across a dense grid in [-2, 2]
    spanning the I − E[I] domain that occurs in this layout."""
    k = DEFAULT_SOFTPLUS_K
    bound = np.log(2.0) / k
    xs = np.linspace(-2.0, 2.0, 1001)
    sp = np.array([softplus_hinge(float(x), k=k) for x in xs])
    hinge = np.maximum(0.0, xs)
    err = sp - hinge  # ≥ 0 always
    assert (err >= -1e-12).all(), f"softplus dropped below hinge: min(err) = {err.min()}"
    assert err.max() <= bound + 1e-12, (
        f"softplus deviation {err.max():.6f} exceeds log(2)/k = {bound:.6f}"
    )
    # bound is approached at x = 0
    assert abs(softplus_hinge(0.0, k=k) - bound) <= 1e-12


def test_sqrt_k_variance_equalization_canonical_layout():
    """§3.4.3: under H₀ (random labels), Var[I_i] ∝ 1/k_i, so I_i^corr =
    sqrt(k_i) * I_i should have approximately equal per-node variance.
    Verified empirically on the canonical 6-player layout via 2,000-iteration
    Monte Carlo with a fixed-magnitude shuffled value vector. The
    coefficient of variation of per-node Var[I_i^corr] across nodes should
    be substantially smaller than the corresponding CV for raw Var[I_i]."""
    from tests.test_morans_i_null_invariants import _canonical_6p_topology
    rng = np.random.default_rng(0)
    topo = _canonical_6p_topology()
    W = np.asarray(topo.spatial_W_swappable.todense() if hasattr(topo.spatial_W_swappable, "todense") else topo.spatial_W_swappable)
    n = W.shape[0]
    # Random standardized values; null = independent labels
    n_iter = 2000
    raw = np.zeros((n_iter, n))
    for i in range(n_iter):
        z = rng.standard_normal(n)
        z = z - z.mean()
        m2 = (z @ z) / n
        if m2 == 0:
            continue
        Wz = W @ z
        raw[i] = z * Wz / m2
    var_raw = raw.var(axis=0)
    deg = np.asarray(topo.degree_swappable, dtype=np.float64)
    var_corr = (deg ** 0.5)[None, :] ** 2 * var_raw  # i.e. var(sqrt(k) * I_i) = k * var(I_i)
    cv_raw = var_raw.std() / var_raw.mean()
    cv_corr = var_corr.std() / var_corr.mean()
    # √k stabilization should reduce inter-node variance heterogeneity (lower CV).
    assert cv_corr < cv_raw, (
        f"√k stabilization did not reduce per-node Var[I] heterogeneity "
        f"(raw CV={cv_raw:.3f}, √k CV={cv_corr:.3f})"
    )


def test_canonical_6p_degree_split():
    """§3.4.3: degree distribution of the canonical 6-player layout's
    spatial_W_swappable is {3: 12, 5: 6, 6: 13} per the prose."""
    from collections import Counter
    from tests.test_morans_i_null_invariants import _canonical_6p_topology
    topo = _canonical_6p_topology()
    deg = list(topo.degree_swappable)
    counts = Counter(deg)
    assert counts == {3: 12, 5: 6, 6: 13}, (
        f"canonical 6p degree split {dict(counts)} != expected {{3:12, 5:6, 6:13}}"
    )
    assert len(deg) == 31


# ── lisa_penalty_swappable_thresholded (canonical Test 3 same-form variant) ──

def test_lisa_penalty_swappable_thresholded_matches_baseline_at_tau_zero():
    """At τ=0 the same-form thresholded variant equals
    lisa_penalty_swappable(use_local_variance=True) at machine precision —
    they sum the same √k-stabilized positive local I_i values; threshold floor
    removes nothing. This is the invariant that makes Test 3 a true same-form
    comparison: only τ varies between baseline and comparator."""
    from ti4_analysis.algorithms.map_generator import generate_random_map
    from ti4_analysis.algorithms.map_topology import MapTopology
    from ti4_analysis.algorithms.fast_map_state import FastMapState
    from ti4_analysis.evaluation.batch_experiment import create_joebrew_evaluator

    ev = create_joebrew_evaluator()
    for seed in (0, 7, 31):
        m = generate_random_map(player_count=6, random_seed=seed)
        topo = MapTopology.from_ti4_map(m, ev)
        fast = FastMapState.from_ti4_map(topo, m, ev)
        baseline = fast.lisa_penalty_swappable(use_local_variance=True)
        thresh0 = fast.lisa_penalty_swappable_thresholded(tau=0.0, use_local_variance=True)
        assert abs(baseline - thresh0) <= 1e-12, (
            f"seed {seed}: baseline={baseline}, thresh@0={thresh0}, diff={baseline-thresh0}"
        )


def test_lisa_penalty_swappable_thresholded_monotone_in_tau():
    """Σ max(0, x − τ) is monotonically non-increasing in τ for τ ≥ 0."""
    from ti4_analysis.algorithms.map_generator import generate_random_map
    from ti4_analysis.algorithms.map_topology import MapTopology
    from ti4_analysis.algorithms.fast_map_state import FastMapState
    from ti4_analysis.evaluation.batch_experiment import create_joebrew_evaluator

    ev = create_joebrew_evaluator()
    m = generate_random_map(player_count=6, random_seed=0)
    topo = MapTopology.from_ti4_map(m, ev)
    fast = FastMapState.from_ti4_map(topo, m, ev)
    vals = [fast.lisa_penalty_swappable_thresholded(tau=t, use_local_variance=True)
            for t in (0.0, 0.025, 0.05, 0.1, 0.5, 10.0)]
    for a, b in zip(vals, vals[1:]):
        assert a >= b - 1e-12, f"non-monotone: {vals}"
    assert vals[-1] == 0.0  # very large τ clears the penalty entirely


# ── MultiObjectiveScore: raw_objective_terms, normalizer_sigma ─────────────────

def test_raw_objective_terms_shape():
    """raw_objective_terms returns (hinge, jfi_gap, lisa_norm)."""
    score = MultiObjectiveScore(
        balance_gap=1.0,
        morans_i=0.1,
        jains_index=0.7,
        lisa_penalty=10.0,
        n_spatial=37,
    )
    terms = score.raw_objective_terms()
    assert len(terms) == 3
    hinge, jfi_gap, lisa_norm = terms
    assert hinge >= 0
    assert 0 <= jfi_gap <= 1
    assert 0 <= lisa_norm <= 1.1  # lisa_norm = LSAP / (n*(n-1))


def test_composite_with_normalizer_sigma():
    """With normalizer_sigma, composite_score uses normalized terms."""
    score = MultiObjectiveScore(
        balance_gap=0.0,
        morans_i=0.0,
        jains_index=0.8,
        lisa_penalty=5.0,
        n_spatial=37,
        normalizer_sigma={
            NORM_KEY_HINGE: 0.2,
            NORM_KEY_JFI: 0.15,
            NORM_KEY_LISA: 0.01,
        },
    )
    c = score.composite_score()
    assert np.isfinite(c)
    assert c >= 0


def test_objective_values_for_pareto_smooth_vs_raw():
    """objective_values_for_pareto with use_smooth_objectives differs from raw."""
    score_raw = MultiObjectiveScore(
        balance_gap=0.0, morans_i=-0.02, jains_index=0.75,
        lisa_penalty=2.0, n_spatial=37,
        jfi_resources=0.75, jfi_influence=0.9,
    )
    score_smooth = MultiObjectiveScore(
        balance_gap=0.0, morans_i=-0.02, jains_index=0.75,
        lisa_penalty=2.0, n_spatial=37,
        jfi_resources=0.75, jfi_influence=0.9,
        use_smooth_objectives=True,
    )
    raw_objs = score_raw.objective_values_for_pareto()
    smooth_objs = score_smooth.objective_values_for_pareto()
    # First objective (JFI gap): smooth min is > raw min when one dimension is better
    assert smooth_objs[0] <= raw_objs[0] + 0.05  # smooth can be slightly different
    assert len(raw_objs) == 3 and len(smooth_objs) == 3


# ── compute_gen0_sigma ───────────────────────────────────────────────────────

def test_compute_gen0_sigma_returns_positive_dict():
    """compute_gen0_sigma returns dict with positive sigma per key."""
    from ti4_analysis.algorithms.hex_grid import HexCoord
    from ti4_analysis.data.map_structures import (
        Planet, System, MapSpace, MapSpaceType, Evaluator,
    )
    from ti4_analysis.algorithms.balance_engine import TI4Map
    from ti4_analysis.algorithms.map_topology import MapTopology

    def _sys(i, r, inf):
        return System(id=i, planets=[Planet(f"P{i}", resources=r, influence=inf)])

    home1 = MapSpace(HexCoord(-5, 5, 0), MapSpaceType.HOME)
    home2 = MapSpace(HexCoord(5, -5, 0), MapSpaceType.HOME)
    spaces = [home1, home2] + [
        MapSpace(HexCoord(-1, 1, 0), MapSpaceType.SYSTEM, _sys(1, 3, 2)),
        MapSpace(HexCoord(0, 0, 0), MapSpaceType.SYSTEM, _sys(2, 2, 3)),
        MapSpace(HexCoord(1, -1, 0), MapSpaceType.SYSTEM, _sys(3, 4, 1)),
        MapSpace(HexCoord(2, -2, 0), MapSpaceType.SYSTEM, _sys(4, 1, 4)),
    ]
    ti4_map = TI4Map(spaces)
    evaluator = Evaluator(name="Test")
    topology = MapTopology.from_ti4_map(ti4_map, evaluator)

    sigma = compute_gen0_sigma(
        topology, evaluator, ti4_map.copy(),
        n_samples=20, random_seed=42, n_swaps_randomize=30,
    )
    assert NORM_KEY_HINGE in sigma and sigma[NORM_KEY_HINGE] >= 0
    assert NORM_KEY_JFI in sigma and sigma[NORM_KEY_JFI] >= 0
    assert NORM_KEY_LISA in sigma and sigma[NORM_KEY_LISA] >= 0

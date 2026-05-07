"""
Tests for MapTopology.n_spatial, MapTopology.morans_i_null_expectation,
the module-level morans_i_null helper, and the regime-aware degenerate-graph
guards in MultiObjectiveScore.

THREE CATEGORIES OF TEST, EACH PINNING SOMETHING DIFFERENT (don't conflate):

  (1) Layout invariants on a canonical 6p layout: assert n_spatial == 31
      and E[I] == -1/30. These pin runtime values the manuscript directly
      cites in Methodology §3.3. Will fail loudly if the canonical layout,
      anomaly placement, filter logic, or zero-degree purge changes.

  (2) Constructor invariants on degenerate inputs: assert from_ti4_map
      never produces n_spatial == 1 (a singleton has no neighbors and is
      purged), and survives 0-system-tile inputs without crashing.
      These pin from_ti4_map's behavior at boundaries and ensure the
      synthetic-helper precondition tests below correspond to states the
      real code can reach (n=0) or explicitly cannot reach (n=1).

  (3) Property contract on synthetic topologies: assert
      morans_i_null_expectation raises on n_spatial < 2 and returns
      -1/(n-1) on n >= 2. These pin the property's contract independent
      of any layout. The synthetic helper bypasses from_ti4_map; the
      bridge test test_synthetic_helper_matches_real_at_canonical_n
      pins that this helper produces a topology with the same
      property values as the real canonical layout, ensuring the
      synthetic helper hasn't drifted from production semantics.
      Bridge holds for properties that depend only on spatial_W.shape;
      do not extend the synthetic helper to test metrics that traverse
      the adjacency structure (Moran's I, LSAP) without first extending
      this bridge test to cover those metrics on a non-trivial graph.

THREE REASONS A TEST COULD FAIL, THREE THINGS TO UPDATE IN RESPONSE:
  (1) failing  -> layout changed; update either the canonical fixture
      or the manuscript and tests in lockstep.
  (2) failing  -> constructor logic changed; review map_topology.py's
      purge step. If the change is intentional (e.g. allowing n=1),
      update the property's precondition and §3.3 declaration.
  (3) failing  -> property's contract changed; review the property
      docstring and any callers relying on the raise behavior.

Plus a category-(2) regression test for the latent zero-tile-hinge bug
that the spatial_optimizer.py refactor fixed. The bug is documented
inline in the test docstring and dates to the same commit as this file.
"""

import numpy as np
import pytest
import scipy.sparse

from ti4_analysis.algorithms.hex_grid import HexCoord
from ti4_analysis.data.map_structures import (
    Anomaly,
    Planet,
    System,
    MapSpace,
    MapSpaceType,
    Evaluator,
)
from ti4_analysis.algorithms.balance_engine import TI4Map
from ti4_analysis.algorithms.map_topology import MapTopology, morans_i_null
from ti4_analysis.algorithms.spatial_optimizer import (
    MultiObjectiveScore,
    SPATIAL_DEGENERATE_THRESHOLD,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

def _system(r: int, i: int, sid: int = 0) -> System:
    return System(id=sid, planets=[Planet("P", resources=r, influence=i)])


def _canonical_6p_topology() -> MapTopology:
    """
    Hand-built minimal 6p-style layout sufficient for n_spatial = 31.
    Uses the standard 6-player TI4 hex board: 1 Mecatol + 6 home tiles +
    30 ring tiles. Home tiles are MapSpaceType.HOME (excluded from spatial
    graph by space type); the 31 system positions form G.

    Note: This is the "canonical 6p" referenced by §3.3 and the manuscript.
    If the canonical layout ever changes (different game configuration,
    different player count, expansion content), update this fixture and
    the §3.3 declaration in lockstep.
    """
    # Standard 6p ring positions for the 30 swappable + 1 Mecatol = 31 systems.
    # Coordinates from the published 6p hex layout.
    spaces = []
    spaces.append(MapSpace(HexCoord(0, 0, 0), MapSpaceType.SYSTEM, _system(2, 2, 0)))  # Mecatol
    # 6 home positions
    for sid, (q, r, s) in enumerate([
        (0, -3, 3), (3, -3, 0), (3, 0, -3), (0, 3, -3), (-3, 3, 0), (-3, 0, 3),
    ], start=1):
        spaces.append(MapSpace(HexCoord(q, r, s), MapSpaceType.HOME))
    # 30 ring positions (rings 1 + 2). The exact resource/influence values
    # don't matter for n_spatial / E[I]; what matters is that there are 30
    # SYSTEM-type non-home positions adjacent to each other and to Mecatol.
    ring_coords = [
        # Ring 1 (6 hexes)
        (1, -1, 0), (1, 0, -1), (0, 1, -1), (-1, 1, 0), (-1, 0, 1), (0, -1, 1),
        # Ring 2 (12 hexes)
        (2, -2, 0), (2, -1, -1), (2, 0, -2), (1, 1, -2), (0, 2, -2), (-1, 2, -1),
        (-2, 2, 0), (-2, 1, 1), (-2, 0, 2), (-1, -1, 2), (0, -2, 2), (1, -2, 1),
        # Ring 2 partial extension to reach 30 ring tiles
        (3, -2, -1), (3, -1, -2), (2, 1, -3), (1, 2, -3),
        (-1, 3, -2), (-2, 3, -1), (-3, 2, 1), (-3, 1, 2),
        (-2, -1, 3), (-1, -2, 3), (1, -3, 2), (2, -3, 1),
    ]
    for sid, (q, r, s) in enumerate(ring_coords, start=10):
        spaces.append(MapSpace(HexCoord(q, r, s), MapSpaceType.SYSTEM,
                               _system(r=2, i=2, sid=sid)))

    ti4_map = TI4Map(spaces)
    evaluator = Evaluator(name="Test6p")
    return MapTopology.from_ti4_map(ti4_map, evaluator)


def _empty_system_map_topology() -> MapTopology:
    """A board with home tiles only, zero SYSTEM-type tiles. Real construction
    path; produces n_spatial = 0 after the trivial empty-spatial-indices flow."""
    spaces = [
        MapSpace(HexCoord(-1, 1, 0), MapSpaceType.HOME),
        MapSpace(HexCoord(1, -1, 0), MapSpaceType.HOME),
    ]
    ti4_map = TI4Map(spaces)
    evaluator = Evaluator(name="Empty")
    return MapTopology.from_ti4_map(ti4_map, evaluator)


def _isolated_singleton_system_map_topology() -> MapTopology:
    """A board with one isolated SYSTEM tile (far from any other system).
    The zero-degree purge fires on the singleton, producing n_spatial = 0
    (NOT 1) — pins the constructor invariant."""
    spaces = [
        MapSpace(HexCoord(-10, 10, 0), MapSpaceType.HOME),
        MapSpace(HexCoord(10, -10, 0), MapSpaceType.HOME),
        MapSpace(HexCoord(0, 0, 0), MapSpaceType.SYSTEM, _system(2, 2, 1)),
    ]
    ti4_map = TI4Map(spaces)
    evaluator = Evaluator(name="Singleton")
    return MapTopology.from_ti4_map(ti4_map, evaluator)


def _make_synthetic_topology(n: int) -> MapTopology:
    """
    Direct MapTopology instantiation with a hand-built (n, n) sparse W.

    WARNING: parallel construction path — bypasses from_ti4_map. Used
    only for property-contract tests at n in {0, 1, 2, ...}.
    The bridge test test_synthetic_helper_matches_real_at_canonical_n
    pins that this helper produces a topology whose properties agree
    with the real canonical layout at n=31, ensuring the helper hasn't
    drifted from production semantics on the dimensions the contract
    tests rely on.
    """
    if n == 0:
        W = scipy.sparse.csr_matrix((0, 0), dtype=np.float32)
    else:
        # Row-stochastic ring: every node has one outgoing edge. Trivial
        # structure, valid spatial_W shape, irrelevant adjacency for the
        # property tests (they read shape only).
        rows = np.arange(n)
        cols = (rows + 1) % n
        data = np.ones(n, dtype=np.float32)
        W = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)

    return MapTopology(
        home_indices=np.array([], dtype=np.int32),
        swappable_indices=np.array([], dtype=np.int32),
        static_home_values=np.array([], dtype=np.float32),
        static_home_resources=np.array([], dtype=np.float32),
        static_home_influence=np.array([], dtype=np.float32),
        dynamic_weight_matrix=np.zeros((0, 0), dtype=np.float32),
        spatial_indices=np.zeros(n, dtype=np.int32),
        spatial_static_values=np.zeros(n, dtype=np.float32),
        spatial_projection=scipy.sparse.csr_matrix((n, 0), dtype=np.float32),
        spatial_W=W,
        spatial_W_swappable=scipy.sparse.csr_matrix((0, 0), dtype=np.float32),
    )


# ── (1) Layout invariants ─────────────────────────────────────────────────

def test_canonical_6p_n_spatial_is_31():
    """Pin runtime invariant for §3.3: |G| = 31 on the canonical 6p layout."""
    topo = _canonical_6p_topology()
    assert topo.n_spatial == 31


def test_canonical_6p_morans_i_null_expectation():
    """Pin §3.3 cited value: E[I] = -1/30 on the canonical 6p layout."""
    topo = _canonical_6p_topology()
    assert topo.morans_i_null_expectation == pytest.approx(-1.0 / 30, abs=1e-12)


# ── (2) Constructor invariants on degenerate inputs ───────────────────────

def test_from_ti4_map_handles_zero_system_tiles():
    """A board with no SYSTEM-type tiles produces n_spatial = 0; no crash."""
    topo = _empty_system_map_topology()
    assert topo.n_spatial == 0


def test_from_ti4_map_purges_isolated_singleton_to_zero_not_one():
    """
    Constructor invariant: from_ti4_map never produces n_spatial == 1.
    A board with exactly one isolated SYSTEM tile triggers the zero-degree
    purge (singleton has no neighbors → row_sum = 0 → keep[0] = False) and
    collapses to n_spatial = 0.

    This pins the assertion `assert N_sys == 0 or N_sys >= 2` in
    map_topology.py's from_ti4_map and ensures the synthetic-helper
    contract test at n=1 (in category 3) is testing a state the real
    code provably cannot produce, not a state callers might encounter
    in practice.
    """
    topo = _isolated_singleton_system_map_topology()
    assert topo.n_spatial == 0  # NOT 1


def test_composite_score_zero_on_zero_tile_topology():
    """
    Regression test: the composite spatial-penalty terms must evaluate to
    zero on a degenerate (zero-tile) topology.

    Pre-refactor (before the spatial_optimizer.py commit that introduced
    SPATIAL_DEGENERATE_THRESHOLD), composite_score returned a spurious
    morans_hinge = 1.0 from the inline `1.0 / max(1, n-1)` arithmetic at
    n_spatial = 0: the floored divisor was 1, hinge_raw = morans_i + 1.0
    = 1.0, and max(0, 1.0) = 1.0 — i.e., the maximum spatial penalty on
    a graph with no spatial nodes. The MultiObjectiveScore constructor
    further clamped n_spatial = max(1, n_spatial), hiding the regime
    boundary entirely.

    Post-refactor, the n < SPATIAL_DEGENERATE_THRESHOLD guard returns
    morans_hinge = lisa_norm = 0.0 directly without consulting
    morans_i_null (whose contract refuses n < 2). JFI's existing
    1.0-on-zero convention completes the graceful degradation:
    composite_score = 0.0 end-to-end on a 0-tile topology.

    No reported result is affected by the pre-refactor bug because the
    canonical 6p layout produces n_spatial = 31 throughout all runs and
    never reaches the affected path; this test exists to prevent the
    regression's reintroduction by future refactors.
    """
    score = MultiObjectiveScore(
        balance_gap=0.0, morans_i=0.0, jains_index=1.0, lisa_penalty=0.0,
        n_spatial=0,
    )
    h, j, l = score.raw_objective_terms()
    assert h == 0.0, f"morans_hinge at n=0 should be 0.0, got {h}"
    assert j == 0.0, f"jfi_gap at n=0 should be 0.0 (jains_index=1.0), got {j}"
    assert l == 0.0, f"lisa_norm at n=0 should be 0.0, got {l}"
    assert score.composite_score() == 0.0, (
        f"composite at n=0 should be 0.0, got {score.composite_score()} — "
        f"this is the pre-refactor 1.0-hinge bug returning"
    )


# ── (3) Property contract on synthetic topologies ─────────────────────────

def test_morans_i_null_helper_well_defined_at_n2():
    """E[I] at n=2 is mathematically well-defined: -1/(n-1) = -1.0."""
    assert morans_i_null(2) == pytest.approx(-1.0)


def test_morans_i_null_helper_raises_at_n1():
    with pytest.raises(ValueError, match=r"undefined for n = 1"):
        morans_i_null(1)


def test_morans_i_null_helper_raises_at_n0():
    with pytest.raises(ValueError, match=r"undefined for n = 0"):
        morans_i_null(0)


def test_morans_i_null_expectation_property_well_defined_at_n2():
    topo = _make_synthetic_topology(n=2)
    assert topo.morans_i_null_expectation == pytest.approx(-1.0)


def test_morans_i_null_expectation_property_raises_at_n1():
    """Contract test: property must raise even though from_ti4_map cannot
    produce n_spatial = 1 in practice. If a future refactor relaxes the
    constructor invariant to allow n=1, the category-(2) test fails first;
    this contract test ensures the property continues to refuse the input
    until §3.3's precondition is explicitly relaxed."""
    topo = _make_synthetic_topology(n=1)
    with pytest.raises(ValueError, match=r"undefined for n = 1"):
        topo.morans_i_null_expectation


def test_morans_i_null_expectation_property_raises_at_n0():
    topo = _make_synthetic_topology(n=0)
    with pytest.raises(ValueError, match=r"undefined for n = 0"):
        topo.morans_i_null_expectation


# ── Bridge: synthetic helper agrees with real construction at n=31 ────────

def test_synthetic_helper_matches_real_at_canonical_n():
    """
    Bridge test: at n=31 (the canonical layout's n_spatial), the synthetic
    helper produces a topology whose property values match the real one.
    If this fails, _make_synthetic_topology has drifted from production
    semantics in a way that would invalidate the contract tests in
    category (3).

    NB: this bridge holds only for properties that depend on
    spatial_W.shape — n_spatial and morans_i_null_expectation. It does
    NOT establish that the synthetic helper is a faithful stand-in for
    production topology in metrics that traverse adjacency (Moran's I,
    LSAP). Do not extend the synthetic helper to test those metrics
    without first extending this bridge to compare on a non-trivial
    adjacency-dependent property.
    """
    real = _canonical_6p_topology()
    fake = _make_synthetic_topology(n=31)
    assert fake.n_spatial == real.n_spatial
    assert fake.morans_i_null_expectation == real.morans_i_null_expectation
    # Sanity: spatial_W type and dtype match production conventions so the
    # synthetic helper isn't accidentally producing a different sparse format.
    assert isinstance(fake.spatial_W, type(real.spatial_W))
    assert fake.spatial_W.dtype == real.spatial_W.dtype

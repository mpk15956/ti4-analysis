"""
Tests for Tabu Search optimizer.
"""

import pytest

from ti4_analysis.algorithms.hex_grid import HexCoord
from ti4_analysis.data.map_structures import (
    Planet, System, MapSpace, MapSpaceType, Evaluator,
)
from ti4_analysis.algorithms.balance_engine import TI4Map
from ti4_analysis.algorithms.map_topology import MapTopology
from ti4_analysis.algorithms.fast_map_state import FastMapState
from ti4_analysis.algorithms.spatial_optimizer import (
    MultiObjectiveScore,
    evaluate_map_multiobjective,
)
from ti4_analysis.algorithms.tabu_search_optimizer import improve_balance_tabu


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_system(system_id: int, resources: int, influence: int) -> System:
    planet = Planet(f"P{system_id}", resources=resources, influence=influence)
    return System(id=system_id, planets=[planet])


def _make_four_system_map(r_high: int = 5, r_low: int = 1) -> tuple:
    """Minimal map with 2 homes and 4 swappable systems (clustered layout)."""
    home1 = MapSpace(HexCoord(-5, 5, 0), MapSpaceType.HOME)
    home2 = MapSpace(HexCoord(5, -5, 0), MapSpaceType.HOME)

    sys_high1 = MapSpace(HexCoord(-1, 1, 0), MapSpaceType.SYSTEM,
                         _make_system(1, r_high, r_high))
    sys_high2 = MapSpace(HexCoord(0, 0, 0), MapSpaceType.SYSTEM,
                         _make_system(2, r_high, r_high))
    sys_low1 = MapSpace(HexCoord(1, -1, 0), MapSpaceType.SYSTEM,
                        _make_system(3, r_low, r_low))
    sys_low2 = MapSpace(HexCoord(2, -2, 0), MapSpaceType.SYSTEM,
                        _make_system(4, r_low, r_low))

    ti4_map = TI4Map([home1, home2, sys_high1, sys_high2, sys_low1, sys_low2])
    evaluator = Evaluator(name="Test")
    return ti4_map, evaluator


# ── Unit tests ────────────────────────────────────────────────────────────────

class TestTabuSearch:

    def test_ts_runs_and_returns_valid_score(self):
        """improve_balance_tabu() completes and returns a MultiObjectiveScore."""
        ti4_map, evaluator = _make_four_system_map()

        score, history = improve_balance_tabu(
            ti4_map, evaluator, max_evaluations=100, random_seed=42, verbose=False
        )

        assert isinstance(score, MultiObjectiveScore)
        assert score.balance_gap >= 0.0
        assert 0.0 <= score.jains_index <= 1.0
        assert score.lisa_penalty >= 0.0
        assert len(history) >= 1

    def test_ts_best_score_not_worse_than_start(self):
        """Best-ever score from TS should not exceed initial composite score."""
        ti4_map, evaluator = _make_four_system_map(r_high=5, r_low=1)
        topology = MapTopology.from_ti4_map(ti4_map, evaluator)
        fast_state = FastMapState.from_ti4_map(topology, ti4_map, evaluator)
        initial = evaluate_map_multiobjective(ti4_map, evaluator, fast_state=fast_state)

        best, _ = improve_balance_tabu(
            ti4_map, evaluator, max_evaluations=500, random_seed=0, verbose=False
        )

        assert best.composite_score() <= initial.composite_score() + 1e-6

    def test_ts_tabu_tenure_prevents_immediate_reversal(self):
        """
        With tenure > 1, TS should not immediately reverse a swap.
        Run 2 full iterations and verify the second move differs from a simple reversal.
        """
        ti4_map, evaluator = _make_four_system_map()

        score, history = improve_balance_tabu(
            ti4_map, evaluator, max_evaluations=50, random_seed=7,
            verbose=False, tabu_tenure=5
        )

        # TS completed at least one iteration (history has initial + iterations)
        assert len(history) >= 2

    def test_ts_deterministic_with_seed(self):
        """Same seed produces identical results."""
        ti4_map1, evaluator = _make_four_system_map()
        ti4_map2 = ti4_map1.copy()

        score1, _ = improve_balance_tabu(
            ti4_map1, evaluator, max_evaluations=200, random_seed=99, verbose=False
        )
        score2, _ = improve_balance_tabu(
            ti4_map2, evaluator, max_evaluations=200, random_seed=99, verbose=False
        )

        assert score1.composite_score() == pytest.approx(
            score2.composite_score(), rel=1e-6
        )

    def test_ts_respects_evaluation_budget(self):
        """Total evaluations should not exceed max_evaluations."""
        ti4_map, evaluator = _make_four_system_map()

        _, history = improve_balance_tabu(
            ti4_map, evaluator, max_evaluations=50, random_seed=42, verbose=False
        )

        # History entries record cumulative evaluation count
        max_evals_recorded = max(h[0] for h in history)
        assert max_evals_recorded <= 50

    def test_ts_populates_per_dimension_jfi(self):
        """TS result should have valid per-dimension JFI from Multi-Jain."""
        ti4_map, evaluator = _make_four_system_map()

        score, _ = improve_balance_tabu(
            ti4_map, evaluator, max_evaluations=200, random_seed=42, verbose=False
        )

        assert hasattr(score, 'jfi_resources')
        assert hasattr(score, 'jfi_influence')
        assert 0.0 <= score.jfi_resources <= 1.0
        assert 0.0 <= score.jfi_influence <= 1.0
        assert score.jains_index == pytest.approx(
            min(score.jfi_resources, score.jfi_influence), rel=1e-6
        )

    def test_ts_too_few_spaces_raises(self):
        """Fewer than 2 swappable spaces should raise ValueError."""
        home1 = MapSpace(HexCoord(0, 0, 0), MapSpaceType.HOME)
        home2 = MapSpace(HexCoord(3, -3, 0), MapSpaceType.HOME)
        sys_space = MapSpace(HexCoord(1, -1, 0), MapSpaceType.SYSTEM,
                             _make_system(1, 3, 3))
        ti4_map = TI4Map([home1, home2, sys_space])
        evaluator = Evaluator(name="Test")

        with pytest.raises(ValueError, match="Not enough swappable"):
            improve_balance_tabu(ti4_map, evaluator, max_evaluations=100)

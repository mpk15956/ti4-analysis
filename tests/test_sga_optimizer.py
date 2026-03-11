"""
Tests for Single-Objective GA (SGA) optimizer.
"""

import random
import pytest
import numpy as np

from ti4_analysis.algorithms.hex_grid import HexCoord
from ti4_analysis.data.map_structures import (
    Planet, System, MapSpace, MapSpaceType, Evaluator,
)
from ti4_analysis.algorithms.balance_engine import TI4Map
from ti4_analysis.algorithms.map_topology import MapTopology
from ti4_analysis.algorithms.fast_map_state import FastMapState
from ti4_analysis.algorithms.spatial_optimizer import (
    evaluate_map_multiobjective,
    MultiObjectiveScore,
)
from ti4_analysis.algorithms.sga_optimizer import sga_optimize, _scalar_tournament
from ti4_analysis.algorithms.nsga2_optimizer import Individual


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_system(system_id: int, resources: int, influence: int) -> System:
    return System(id=system_id, planets=[Planet(f"P{system_id}", resources, influence)])


def _make_six_system_map() -> tuple:
    """Map with 2 HOME spaces and 6 swappable SYSTEM spaces."""
    home1 = MapSpace(HexCoord(-5, 5, 0), MapSpaceType.HOME)
    home2 = MapSpace(HexCoord(5, -5, 0), MapSpaceType.HOME)
    systems = [
        MapSpace(HexCoord(-2,  2, 0), MapSpaceType.SYSTEM, _make_system(1, 3, 1)),
        MapSpace(HexCoord(-1,  1, 0), MapSpaceType.SYSTEM, _make_system(2, 1, 3)),
        MapSpace(HexCoord( 0,  0, 0), MapSpaceType.SYSTEM, _make_system(3, 4, 2)),
        MapSpace(HexCoord( 1, -1, 0), MapSpaceType.SYSTEM, _make_system(4, 2, 4)),
        MapSpace(HexCoord( 2, -2, 0), MapSpaceType.SYSTEM, _make_system(5, 5, 1)),
        MapSpace(HexCoord( 3, -3, 0), MapSpaceType.SYSTEM, _make_system(6, 1, 5)),
    ]
    ti4_map = TI4Map([home1, home2] + systems)
    evaluator = Evaluator(name="Test")
    return ti4_map, evaluator


# ── Scalar tournament tests ───────────────────────────────────────────────────

def _make_individual_with_composite(composite_val):
    """Construct a minimal Individual with a specific composite_score."""
    score = MultiObjectiveScore(
        balance_gap=0.0, morans_i=0.0, jains_index=1.0,
        lisa_penalty=0.0, n_spatial=37,
    )
    score._test_composite = composite_val
    original_composite = score.composite_score
    score.composite_score = lambda: composite_val
    ind = object.__new__(Individual)
    ind.score = score
    ind.map = None
    ind.fast_state = None
    ind.rank = 0
    ind.crowding_distance = 0.0
    return ind


class TestScalarTournament:

    def test_lower_composite_wins(self):
        """Tournament should return the individual with lower composite score."""
        a = _make_individual_with_composite(0.1)
        b = _make_individual_with_composite(0.9)
        pool = [a, b]
        rng = random.Random(42)
        winner = _scalar_tournament(pool, rng)
        assert winner.score.composite_score() <= 0.1


# ── Integration tests ─────────────────────────────────────────────────────────

class TestSgaOptimize:

    def test_sga_returns_score_and_history(self):
        """sga_optimize completes and returns (score, history) tuple."""
        ti4_map, evaluator = _make_six_system_map()
        score, history = sga_optimize(
            ti4_map, evaluator,
            generations=3,
            population_size=10,
            random_seed=42,
            verbose=False,
        )
        assert isinstance(score, MultiObjectiveScore)
        assert score.composite_score() >= 0.0
        assert len(history) > 0

    def test_sga_improves_over_generations(self):
        """Best composite score at the end should be <= initial."""
        ti4_map, evaluator = _make_six_system_map()
        score, history = sga_optimize(
            ti4_map, evaluator,
            generations=10,
            population_size=10,
            random_seed=42,
            verbose=False,
        )
        initial_composite = history[0][1].composite_score()
        final_composite = score.composite_score()
        assert final_composite <= initial_composite

    def test_sga_deterministic_with_seed(self):
        """Two runs with the same seed produce identical results."""
        ti4_map, evaluator = _make_six_system_map()
        score1, _ = sga_optimize(
            ti4_map, evaluator,
            generations=5, population_size=8,
            random_seed=99, verbose=False,
        )
        ti4_map2, evaluator2 = _make_six_system_map()
        score2, _ = sga_optimize(
            ti4_map2, evaluator2,
            generations=5, population_size=8,
            random_seed=99, verbose=False,
        )
        assert abs(score1.composite_score() - score2.composite_score()) < 1e-6

    def test_sga_history_is_monotonically_nonincreasing(self):
        """Best composite in history should never increase."""
        ti4_map, evaluator = _make_six_system_map()
        _, history = sga_optimize(
            ti4_map, evaluator,
            generations=10, population_size=10,
            random_seed=42, verbose=False,
        )
        composites = [s.composite_score() for _, s in history]
        for i in range(1, len(composites)):
            assert composites[i] <= composites[i - 1] + 1e-9, (
                f"History not monotonic at index {i}: "
                f"{composites[i]} > {composites[i-1]}"
            )

    def test_sga_raises_on_insufficient_spaces(self):
        """SGA should raise ValueError if fewer than 2 swappable spaces."""
        home1 = MapSpace(HexCoord(-5, 5, 0), MapSpaceType.HOME)
        home2 = MapSpace(HexCoord(5, -5, 0), MapSpaceType.HOME)
        sys1 = MapSpace(HexCoord(0, 0, 0), MapSpaceType.SYSTEM,
                        _make_system(1, 3, 1))
        ti4_map = TI4Map([home1, home2, sys1])
        evaluator = Evaluator(name="Test")
        with pytest.raises(ValueError, match="Not enough swappable spaces"):
            sga_optimize(ti4_map, evaluator, generations=2, population_size=4,
                         verbose=False)

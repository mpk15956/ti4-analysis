"""Smoke tests for composite-scoring hill-climber (hc_optimizer)."""

import pytest

from ti4_analysis.algorithms.hex_grid import HexCoord
from ti4_analysis.data.map_structures import (
    Planet, System, MapSpace, MapSpaceType, Evaluator,
)
from ti4_analysis.algorithms.balance_engine import TI4Map
from ti4_analysis.algorithms.map_topology import MapTopology
from ti4_analysis.algorithms.fast_map_state import FastMapState
from ti4_analysis.algorithms.hc_optimizer import hc_optimize
from ti4_analysis.algorithms.spatial_optimizer import (
    MultiObjectiveScore,
    evaluate_map_multiobjective,
)


def _make_system(sid: int, resources: int, influence: int) -> System:
    return System(id=sid, planets=[Planet(f"P{sid}", resources=resources, influence=influence)])


def _make_map():
    """Minimal map: 2 homes, 4 swappable systems."""
    spaces = [
        MapSpace(HexCoord(-5, 5, 0), MapSpaceType.HOME),
        MapSpace(HexCoord(5, -5, 0), MapSpaceType.HOME),
        MapSpace(HexCoord(-1, 1, 0), MapSpaceType.SYSTEM, _make_system(1, 5, 5)),
        MapSpace(HexCoord(0, 0, 0), MapSpaceType.SYSTEM, _make_system(2, 5, 5)),
        MapSpace(HexCoord(1, -1, 0), MapSpaceType.SYSTEM, _make_system(3, 1, 1)),
        MapSpace(HexCoord(2, -2, 0), MapSpaceType.SYSTEM, _make_system(4, 1, 1)),
    ]
    return TI4Map(spaces), Evaluator(name="Test")


class TestHcOptimizer:

    def test_return_type(self):
        """hc_optimize returns (MultiObjectiveScore, history, evals_to_best)."""
        ti4_map, evaluator = _make_map()
        score, history, evals_to_best = hc_optimize(
            ti4_map, evaluator, iterations=20, random_seed=42, verbose=False
        )
        assert isinstance(score, MultiObjectiveScore)
        assert isinstance(history, list)
        assert all(isinstance(h[0], int) and isinstance(h[1], MultiObjectiveScore) for h in history)
        assert isinstance(evals_to_best, int)
        assert evals_to_best >= 0

    def test_improvement_over_iterations(self):
        """Best composite score should not worsen vs initial (strict improvement only)."""
        ti4_map, evaluator = _make_map()
        topo = MapTopology.from_ti4_map(ti4_map, evaluator)
        fs = FastMapState.from_ti4_map(topo, ti4_map, evaluator)
        initial = evaluate_map_multiobjective(ti4_map, evaluator, fast_state=fs)

        best, _, _ = hc_optimize(
            ti4_map, evaluator, iterations=50, random_seed=0, verbose=False
        )
        assert best.composite_score() <= initial.composite_score() + 1e-6

    def test_determinism(self):
        """Same seed yields same best composite score."""
        m1, evaluator = _make_map()
        m2, _ = _make_map()
        best1, _, _ = hc_optimize(m1, evaluator, iterations=30, random_seed=123, verbose=False)
        best2, _, _ = hc_optimize(m2, evaluator, iterations=30, random_seed=123, verbose=False)
        assert best1.composite_score() == pytest.approx(best2.composite_score())

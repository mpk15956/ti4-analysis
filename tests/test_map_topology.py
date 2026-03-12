"""Tests for MapTopology, including anomaly-aware spatial W."""

import numpy as np

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
from ti4_analysis.algorithms.map_topology import MapTopology


def _system(r: int, i: int, sid: int = 0) -> System:
    return System(id=sid, planets=[Planet("P", resources=r, influence=i)])


def test_supernova_purged_from_spatial_W():
    """
    Zero-degree nodes (e.g. Supernova with no navigable neighbors) are purged
    from spatial_indices and spatial_W so variance and spatial lag share the
    same domain (no island variance leak).
    """
    # Map: 2 homes, 2 normal systems, 1 supernova (impassable).
    supernova = System(id=0, planets=[], anomalies=[Anomaly.SUPERNOVA])
    spaces = [
        MapSpace(HexCoord(-5, 5, 0), MapSpaceType.HOME),
        MapSpace(HexCoord(5, -5, 0), MapSpaceType.HOME),
        MapSpace(HexCoord(-1, 1, 0), MapSpaceType.SYSTEM, _system(5, 5, 1)),
        MapSpace(HexCoord(0, 0, 0), MapSpaceType.SYSTEM, _system(3, 3, 2)),
        MapSpace(HexCoord(1, -1, 0), MapSpaceType.SYSTEM, supernova),
    ]
    ti4_map = TI4Map(spaces)
    evaluator = Evaluator(name="Test")

    topo = MapTopology.from_ti4_map(ti4_map, evaluator)
    spatial_indices = topo.spatial_indices
    supernova_space_idx = 4

    # Supernova has no navigable neighbors → zero degree → purged from spatial domain.
    assert supernova_space_idx not in spatial_indices, (
        "Zero-degree (island) nodes must be purged from spatial_indices"
    )
    # All remaining rows in spatial_W are connected (no zero rows).
    row_sums = np.array(topo.spatial_W.sum(axis=1)).flatten()
    assert (row_sums > 0).all(), "After purge, spatial_W has no zero-degree rows"

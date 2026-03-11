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


def test_supernova_has_zero_row_and_column_sums_in_spatial_W():
    """
    Impassable tiles (e.g. Supernova) must have zero row and column sums in
    spatial_W so Moran's I and LSAP use navigable topology only.
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
    W = topo.spatial_W
    if hasattr(W, "toarray"):
        W_dense = W.toarray()
    else:
        W_dense = np.asarray(W.todense())

    # Find the row index for the Supernova space (space index 4).
    # spatial_indices are system spaces with non-None system; order = [2,3,4] for indices 2,3,4.
    # So supernova is at the last row (index 2 in 0-based spatial_indices).
    spatial_indices = topo.spatial_indices
    supernova_space_idx = 4
    supernova_row = np.flatnonzero(spatial_indices == supernova_space_idx)[0]

    row_sum = W_dense[supernova_row, :].sum()
    col_sum = W_dense[:, supernova_row].sum()
    assert row_sum == 0.0, "Supernova row in spatial_W should sum to 0 (no outgoing edges)"
    assert col_sum == 0.0, "Supernova column in spatial_W should sum to 0 (no incoming edges)"

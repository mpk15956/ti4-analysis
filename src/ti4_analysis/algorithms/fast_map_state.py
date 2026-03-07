"""
Mutable numpy representation of system values at swappable map positions.

Designed for the inner optimization loop. A swap is two scalar assignments;
fitness evaluation is a single matrix-vector product executed at C level.

Usage:
    topology = MapTopology.from_ti4_map(ti4_map, evaluator)
    state = FastMapState.from_ti4_map(topology, ti4_map, evaluator)

    # Inside optimizer loop:
    state.swap(s_i, s_j)           # O(1)
    gap = state.balance_gap()      # O(H * S) via numpy matmul, cached
    state.swap(s_i, s_j)           # revert if not accepted (same operation)
"""

import numpy as np
from typing import TYPE_CHECKING

from .map_topology import MapTopology
from ..data.map_structures import Evaluator

if TYPE_CHECKING:
    from .balance_engine import TI4Map


class FastMapState:
    """
    Vectorized map state for high-throughput fitness evaluation.

    system_value[s] holds the evaluator score of whichever system tile
    currently occupies swappable position s (matching topology.swappable_indices[s]).

    Home values are lazily recomputed from:
        home_values = topology.weight_matrix @ system_value

    The weight_matrix is static (topology never changes during optimization),
    so only system_value needs to be updated on each swap.
    """

    __slots__ = ("topology", "system_value", "_home_values", "_dirty")

    def __init__(self, topology: MapTopology, system_value: np.ndarray) -> None:
        self.topology = topology
        self.system_value: np.ndarray = system_value  # shape (S,) float32
        self._home_values: np.ndarray = (
            topology.static_home_values + topology.dynamic_weight_matrix @ system_value
        )
        self._dirty: bool = False

    @classmethod
    def from_ti4_map(
        cls, topology: MapTopology, ti4_map: 'TI4Map', evaluator: Evaluator
    ) -> 'FastMapState':
        """
        Build initial state by evaluating each swappable system.

        The ordering of system_value matches topology.swappable_indices exactly,
        so swappable_spaces[s] in the optimizer corresponds to system_value[s].
        """
        spaces = ti4_map.spaces
        system_value = np.array(
            [
                spaces[i].system.evaluate(evaluator) if spaces[i].system else 0.0
                for i in topology.swappable_indices
            ],
            dtype=np.float32,
        )
        return cls(topology, system_value)

    def swap(self, s_i: int, s_j: int) -> None:
        """
        Swap the system values at two swappable-position indices.

        Call again with the same indices to revert (swap is its own inverse).
        O(1) — two scalar assignments.
        """
        self.system_value[s_i], self.system_value[s_j] = (
            self.system_value[s_j],
            self.system_value[s_i],
        )
        self._dirty = True

    def home_values(self) -> np.ndarray:
        """
        Return home values for all H home positions.

        Lazily recomputed via matmul when dirty; cached otherwise.
        Shape: (H,) float32.

        Formula: static_home_values + dynamic_weight_matrix @ system_value
        The static term (non-swappable tiles) is pre-computed in MapTopology.
        """
        if self._dirty:
            self._home_values = (
                self.topology.static_home_values
                + self.topology.dynamic_weight_matrix @ self.system_value
            )
            self._dirty = False
        return self._home_values

    def balance_gap(self) -> float:
        """max(home_values) - min(home_values). Lower is better."""
        hv = self.home_values()
        return float(hv.max() - hv.min())

    def clone(self) -> 'FastMapState':
        """
        Return an independent copy sharing the same immutable topology.

        Used by population-based optimizers (Pareto, NSGA-II) to create
        per-individual state without duplicating the weight matrix.
        """
        new = object.__new__(FastMapState)
        new.topology = self.topology          # shared — topology is frozen
        new.system_value = self.system_value.copy()
        new._home_values = self._home_values.copy()
        new._dirty = self._dirty
        return new

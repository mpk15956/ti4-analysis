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

    Per-dimension resource/influence arrays support Multi-Jain (bottleneck
    JFI across resource dimensions) without changing the spatial metrics,
    which continue to operate on the combined evaluator scalar.

    The weight_matrix is static (topology never changes during optimization),
    so only system_value needs to be updated on each swap.
    """

    __slots__ = ("topology", "system_value",
                 "system_resources", "system_influence",
                 "_home_values", "_dirty",
                 "_home_resources", "_home_influence", "_home_ri_dirty",
                 "_spatial_values", "_spatial_dirty")

    def __init__(
        self,
        topology: MapTopology,
        system_value: np.ndarray,
        system_resources: np.ndarray,
        system_influence: np.ndarray,
    ) -> None:
        self.topology = topology
        self.system_value: np.ndarray = system_value        # shape (S,) float32
        self.system_resources: np.ndarray = system_resources  # shape (S,) float32
        self.system_influence: np.ndarray = system_influence  # shape (S,) float32
        self._home_values: np.ndarray = (
            topology.static_home_values + topology.dynamic_weight_matrix @ system_value
        )
        self._dirty: bool = False
        self._home_resources: np.ndarray = (
            topology.static_home_resources + topology.dynamic_weight_matrix @ system_resources
        )
        self._home_influence: np.ndarray = (
            topology.static_home_influence + topology.dynamic_weight_matrix @ system_influence
        )
        self._home_ri_dirty: bool = False
        self._spatial_values: np.ndarray = np.empty(0, dtype=np.float32)
        self._spatial_dirty: bool = True

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
        system_resources = np.array(
            [
                sum(p.resources for p in spaces[i].system.planets)
                if spaces[i].system else 0.0
                for i in topology.swappable_indices
            ],
            dtype=np.float32,
        )
        system_influence = np.array(
            [
                sum(p.influence for p in spaces[i].system.planets)
                if spaces[i].system else 0.0
                for i in topology.swappable_indices
            ],
            dtype=np.float32,
        )
        return cls(topology, system_value, system_resources, system_influence)

    def swap(self, s_i: int, s_j: int) -> None:
        """
        Swap the system values at two swappable-position indices.

        Call again with the same indices to revert (swap is its own inverse).
        O(1) — six scalar assignments (value + resources + influence).
        """
        self.system_value[s_i], self.system_value[s_j] = (
            self.system_value[s_j],
            self.system_value[s_i],
        )
        self.system_resources[s_i], self.system_resources[s_j] = (
            self.system_resources[s_j],
            self.system_resources[s_i],
        )
        self.system_influence[s_i], self.system_influence[s_j] = (
            self.system_influence[s_j],
            self.system_influence[s_i],
        )
        self._dirty = True
        self._home_ri_dirty = True
        self._spatial_dirty = True

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

    def spatial_values(self) -> np.ndarray:
        """
        Full N_sys resource values vector for Moran's I computation.

        Lazily recomputed when dirty; cached otherwise.
        Formula: spatial_static_values + spatial_projection @ system_value
        Shape: (N_sys,) float32.
        """
        if self._spatial_dirty:
            topo = self.topology
            proj = np.asarray(
                topo.spatial_projection @ self.system_value, dtype=np.float32
            ).ravel()
            self._spatial_values = topo.spatial_static_values + proj
            self._spatial_dirty = False
        return self._spatial_values

    def morans_i(self) -> float:
        """
        Global Moran's I spatial autocorrelation via sparse matmul.

        O(nnz) ≈ O(6·N_sys) — uses pre-computed row-standardized adjacency W.
        I > 0: clustering; I ≈ 0: random; I < 0: dispersion.
        """
        z = self.spatial_values()
        n = len(z)
        if n < 3:
            return 0.0
        z_dev = z - z.mean()
        denom = float(z_dev @ z_dev)
        if denom == 0.0:
            return 0.0
        W = self.topology.spatial_W
        W_sum = float(W.sum())
        if W_sum == 0.0:
            return 0.0
        numer = float(z_dev @ (W @ z_dev))
        return (n / W_sum) * (numer / denom)

    def lisa_penalty(self) -> float:
        """
        LISA penalty: sum of positive variance-normalised local Moran's I values.

        local_I[i] = z_dev[i] * (W @ z_dev)[i] / m2
        where m2 = Σ(z_dev²) / n  (the spatial variance).

        Dividing by m2 makes local_I dimensionless and ensures Σ local_I ≈ n × I_global.
        Without this normalisation the values scale with the square of system values,
        causing LISA to dominate the composite score at any fixed weight.

        Positive local_I indicates H-H or L-L spatial clusters (resource
        hoarding or resource deserts). Negative local_I indicates spatial
        outliers (H-L or L-H borders), which we do not penalize.

        O(nnz) via sparse matmul — same cost as morans_i().
        """
        z = self.spatial_values()
        n = len(z)
        if n < 3:
            return 0.0
        z_dev = z - z.mean()
        m2 = float(z_dev @ z_dev) / n
        if m2 == 0.0:
            return 0.0
        Wz = self.topology.spatial_W @ z_dev
        local_I = z_dev * np.asarray(Wz).ravel() / m2
        return float(local_I[local_I > 0].sum())

    def home_resources(self) -> np.ndarray:
        """Distance-weighted raw resource totals per home, shape (H,)."""
        if self._home_ri_dirty:
            self._home_resources = (
                self.topology.static_home_resources
                + self.topology.dynamic_weight_matrix @ self.system_resources
            )
            self._home_influence = (
                self.topology.static_home_influence
                + self.topology.dynamic_weight_matrix @ self.system_influence
            )
            self._home_ri_dirty = False
        return self._home_resources

    def home_influence(self) -> np.ndarray:
        """Distance-weighted raw influence totals per home, shape (H,)."""
        if self._home_ri_dirty:
            self.home_resources()  # recomputes both
        return self._home_influence

    @staticmethod
    def _jfi(v: np.ndarray) -> float:
        """Jain's Fairness Index on a 1-D value vector."""
        n = len(v)
        if n == 0:
            return 1.0
        sum_x = float(v.sum())
        sum_x2 = float((v ** 2).sum())
        if sum_x2 == 0.0:
            return 1.0
        return (sum_x ** 2) / (n * sum_x2)

    def jfi_resources(self) -> float:
        """JFI on distance-weighted raw resources per player."""
        return self._jfi(self.home_resources())

    def jfi_influence(self) -> float:
        """JFI on distance-weighted raw influence per player."""
        return self._jfi(self.home_influence())

    def jains_index(self) -> float:
        """
        Multi-Jain bottleneck: min(JFI_resources, JFI_influence).

        DRF-inspired: map fairness is limited by the least-fair resource
        dimension. Range [1/H, 1]; 1 = perfect fairness on both axes.
        """
        return min(self.jfi_resources(), self.jfi_influence())

    def clone(self) -> 'FastMapState':
        """
        Return an independent copy sharing the same immutable topology.

        Used by population-based optimizers (Pareto, NSGA-II) to create
        per-individual state without duplicating the weight matrix.
        """
        new = object.__new__(FastMapState)
        new.topology = self.topology          # shared — topology is frozen
        new.system_value = self.system_value.copy()
        new.system_resources = self.system_resources.copy()
        new.system_influence = self.system_influence.copy()
        new._home_values = self._home_values.copy()
        new._dirty = self._dirty
        new._home_resources = self._home_resources.copy()
        new._home_influence = self._home_influence.copy()
        new._home_ri_dirty = self._home_ri_dirty
        new._spatial_values = (
            self._spatial_values.copy() if not self._spatial_dirty
            else np.empty(0, dtype=np.float32)
        )
        new._spatial_dirty = self._spatial_dirty
        return new

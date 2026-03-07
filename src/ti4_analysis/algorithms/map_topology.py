"""
Static pre-computed topology for a fixed TI4 map template.

The hex grid topology (which positions are adjacent, which are blocked by
Supernovas, which are connected via wormholes) is immutable during an
optimization run because anomaly and wormhole tiles are excluded from
swapping. This allows the distance-weighted evaluation to be baked into
a static (H x S) weight matrix computed once at optimizer initialization.

After construction, fitness evaluation reduces to:
    home_values = weight_matrix @ system_value   # single C-level matmul
"""

import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

from ..data.map_structures import Evaluator, MapSpaceType
from .hex_grid import hex_distance

if TYPE_CHECKING:
    from .balance_engine import TI4Map, can_swap_system


@dataclass(frozen=True)
class MapTopology:
    """
    Immutable topology for a specific map template and evaluator.

    Build once via MapTopology.from_ti4_map(); reuse for all swap iterations.

    Home value decomposition:
        home_values = static_home_values + dynamic_weight_matrix @ system_value

    where:
        static_home_values  — contribution from non-swappable systems (Mecatol,
                              wormhole tiles, anomaly tiles, etc.). Constant for
                              the entire optimization run.
        dynamic_weight_matrix — per-swappable-position weights, shape (H, S).
        system_value        — current evaluator score at each swappable position,
                              maintained by FastMapState and updated O(1) per swap.

    Attributes:
        home_indices: Indices into TI4Map.spaces for home positions, shape (H,).
        swappable_indices: Indices into TI4Map.spaces for swappable system
            positions, shape (S,). Ordering matches FastMapState.system_value.
        static_home_values: Pre-summed contribution of non-swappable systems,
            shape (H,). Added back in FastMapState.home_values().
        dynamic_weight_matrix: Pre-computed fitness weights for swappable
            positions, shape (H, S).
            dynamic_weight_matrix[h, s] = evaluator.get_distance_multiplier(
                modded_routing_distance(home_h, swappable_s))
            0.0 where Supernova-blocked or geometric distance >= 5.
    """

    home_indices: np.ndarray          # shape (H,) int32
    swappable_indices: np.ndarray     # shape (S,) int32
    static_home_values: np.ndarray   # shape (H,) float32
    dynamic_weight_matrix: np.ndarray # shape (H, S) float32

    @classmethod
    def from_ti4_map(cls, ti4_map: 'TI4Map', evaluator: Evaluator) -> 'MapTopology':
        """
        Build topology by running get_shortest_modded_distance once per
        (home, system) pair. O(H * N_systems * BFS) — runs once at optimizer init.

        Non-swappable systems (Mecatol, wormhole tiles, anomaly tiles, empty
        tiles with planets) contribute a fixed amount to each home value and are
        baked into static_home_values. Only swappable system positions are
        included in dynamic_weight_matrix.

        Args:
            ti4_map: Map whose topology to pre-compute.
            evaluator: Evaluator controlling anomaly traversal costs.

        Returns:
            MapTopology instance ready for use with FastMapState.
        """
        from .balance_engine import can_swap_system

        spaces = ti4_map.spaces

        home_indices = np.array(
            [i for i, s in enumerate(spaces) if s.space_type == MapSpaceType.HOME],
            dtype=np.int32,
        )
        swappable_indices = np.array(
            [i for i, s in enumerate(spaces) if can_swap_system(s)],
            dtype=np.int32,
        )
        swappable_set = set(swappable_indices.tolist())

        H = len(home_indices)
        S = len(swappable_indices)
        static_home_values = np.zeros(H, dtype=np.float32)
        dynamic_weight_matrix = np.zeros((H, S), dtype=np.float32)

        # s_pos lookup: space index → column in dynamic_weight_matrix
        s_pos_of = {int(idx): pos for pos, idx in enumerate(swappable_indices)}

        for h_pos, h_idx in enumerate(home_indices):
            home_space = spaces[h_idx]

            for space_idx, space in enumerate(spaces):
                if space.space_type != MapSpaceType.SYSTEM or space.system is None:
                    continue

                # Geometric range gate — matches get_home_value's cutoff
                if hex_distance(home_space.coord, space.coord) >= 5:
                    continue

                system_value = space.system.evaluate(evaluator)
                if system_value <= 0:
                    continue

                # Topological routing distance (None if blocked by Supernova)
                modded_dist = ti4_map.get_shortest_modded_distance(
                    home_space, space, evaluator
                )
                if modded_dist is None:
                    continue

                weight = evaluator.get_distance_multiplier(modded_dist)

                if space_idx in swappable_set:
                    # Dynamic: tile value changes during optimization
                    dynamic_weight_matrix[h_pos, s_pos_of[space_idx]] = weight
                else:
                    # Static: tile never moves; bake value × weight into constant
                    static_home_values[h_pos] += weight * system_value

        return cls(
            home_indices=home_indices,
            swappable_indices=swappable_indices,
            static_home_values=static_home_values,
            dynamic_weight_matrix=dynamic_weight_matrix,
        )

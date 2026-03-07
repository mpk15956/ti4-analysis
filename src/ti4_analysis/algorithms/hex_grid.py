"""
Hexagonal grid mathematics for TI4 map analysis.

This module implements cube coordinate system for hexagonal grids,
including distance calculations, adjacency, and pathfinding operations.

References:
    - Red Blob Games: https://www.redblobgames.com/grids/hexagons/
    - Cube coordinates: x + y + z = 0
"""

from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class HexCoord:
    """
    Hexagonal coordinate in cube coordinate system.

    Invariant: x + y + z == 0
    """
    x: int
    y: int
    z: int

    def __post_init__(self):
        """Validate cube coordinate invariant."""
        if self.x + self.y + self.z != 0:
            raise ValueError(
                f"Invalid cube coordinates: ({self.x}, {self.y}, {self.z}). "
                f"Must satisfy x + y + z = 0"
            )

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def __eq__(self, other):
        if not isinstance(other, HexCoord):
            return False
        return self.x == other.x and self.y == other.y and self.z == other.z

    def to_tuple(self) -> Tuple[int, int, int]:
        """Convert to tuple representation."""
        return (self.x, self.y, self.z)


def hex_distance(a: HexCoord, b: HexCoord) -> int:
    """
    Calculate Manhattan distance between two hexes in cube coordinates.

    This is the minimum number of steps to move from hex a to hex b.

    Formula: max(|Δx|, |Δy|, |Δz|)

    Args:
        a: First hex coordinate
        b: Second hex coordinate

    Returns:
        Integer distance (number of hex steps)

    Examples:
        >>> origin = HexCoord(0, 0, 0)
        >>> neighbor = HexCoord(1, -1, 0)
        >>> hex_distance(origin, neighbor)
        1
        >>> far = HexCoord(3, -1, -2)
        >>> hex_distance(origin, far)
        3
    """
    return max(
        abs(a.x - b.x),
        abs(a.y - b.y),
        abs(a.z - b.z)
    )


def get_adjacent_coordinates(coord: HexCoord) -> List[HexCoord]:
    """
    Get all 6 adjacent hexagonal coordinates.

    The six directions in cube coordinates are:
    - (+1, -1, 0), (+1, 0, -1)   [east directions]
    - (-1, +1, 0), (0, +1, -1)   [west directions]
    - (-1, 0, +1), (0, -1, +1)   [north/south directions]

    Args:
        coord: Center hex coordinate

    Returns:
        List of 6 adjacent coordinates

    Examples:
        >>> origin = HexCoord(0, 0, 0)
        >>> neighbors = get_adjacent_coordinates(origin)
        >>> len(neighbors)
        6
        >>> all(hex_distance(origin, n) == 1 for n in neighbors)
        True
    """
    directions = [
        (1, -1, 0),   # East
        (1, 0, -1),   # Southeast
        (0, 1, -1),   # Southwest
        (-1, 1, 0),   # West
        (-1, 0, 1),   # Northwest
        (0, -1, 1),   # Northeast
    ]

    return [
        HexCoord(coord.x + dx, coord.y + dy, coord.z + dz)
        for dx, dy, dz in directions
    ]


def get_ring(center: HexCoord, radius: int) -> List[HexCoord]:
    """
    Get all hexes at exactly 'radius' distance from center.

    Args:
        center: Center hex coordinate
        radius: Distance from center

    Returns:
        List of hexes forming a ring

    Examples:
        >>> origin = HexCoord(0, 0, 0)
        >>> ring1 = get_ring(origin, 1)
        >>> len(ring1)
        6
        >>> ring2 = get_ring(origin, 2)
        >>> len(ring2)
        12
    """
    if radius == 0:
        return [center]

    hexes = []

    # Start at one corner of the ring
    hex_coord = HexCoord(
        center.x + radius,
        center.y - radius,
        center.z
    )

    # Walk around the ring in 6 directions
    directions = [
        (0, 1, -1),   # Southwest
        (-1, 1, 0),   # West
        (-1, 0, 1),   # Northwest
        (0, -1, 1),   # Northeast
        (1, -1, 0),   # East
        (1, 0, -1),   # Southeast
    ]

    for direction in directions:
        for _ in range(radius):
            hexes.append(hex_coord)
            hex_coord = HexCoord(
                hex_coord.x + direction[0],
                hex_coord.y + direction[1],
                hex_coord.z + direction[2]
            )

    return hexes


def get_hexes_in_range(center: HexCoord, max_radius: int) -> List[HexCoord]:
    """
    Get all hexes within max_radius distance from center (inclusive).

    Args:
        center: Center hex coordinate
        max_radius: Maximum distance from center

    Returns:
        List of all hexes within range

    Examples:
        >>> origin = HexCoord(0, 0, 0)
        >>> hexes = get_hexes_in_range(origin, 1)
        >>> len(hexes)
        7  # center + 6 neighbors
        >>> hexes = get_hexes_in_range(origin, 2)
        >>> len(hexes)
        19  # 1 + 6 + 12
    """
    hexes = []
    for radius in range(max_radius + 1):
        hexes.extend(get_ring(center, radius))
    return hexes


def breadth_first_search(
    start: HexCoord,
    valid_coords: Set[HexCoord],
    max_distance: int = 5
) -> dict[HexCoord, int]:
    """
    Perform BFS to find distances from start to all reachable hexes.

    Args:
        start: Starting hex coordinate
        valid_coords: Set of valid/traversable coordinates
        max_distance: Maximum search distance

    Returns:
        Dictionary mapping hex coordinates to their distance from start

    Examples:
        >>> start = HexCoord(0, 0, 0)
        >>> valid = {start, HexCoord(1, -1, 0), HexCoord(2, -2, 0)}
        >>> distances = breadth_first_search(start, valid, max_distance=5)
        >>> distances[HexCoord(2, -2, 0)]
        2
    """
    if start not in valid_coords:
        return {}

    distances = {start: 0}
    frontier = [start]

    for current_distance in range(1, max_distance + 1):
        new_frontier = []

        for coord in frontier:
            for neighbor in get_adjacent_coordinates(coord):
                if neighbor in valid_coords and neighbor not in distances:
                    distances[neighbor] = current_distance
                    new_frontier.append(neighbor)

        if not new_frontier:
            break

        frontier = new_frontier

    return distances


def line_interpolation(a: HexCoord, b: HexCoord, t: float) -> Tuple[float, float, float]:
    """
    Linear interpolation between two hex coordinates.

    Used for line-of-sight and hex line algorithms.

    Args:
        a: Start coordinate
        b: End coordinate
        t: Interpolation parameter (0.0 to 1.0)

    Returns:
        Tuple of interpolated (x, y, z) coordinates
    """
    return (
        a.x + (b.x - a.x) * t,
        a.y + (b.y - a.y) * t,
        a.z + (b.z - a.z) * t
    )


def hex_round(x: float, y: float, z: float) -> HexCoord:
    """
    Round fractional cube coordinates to nearest hex.

    Since rounding can violate the x+y+z=0 constraint, we round
    the component with the largest rounding error and recalculate it.

    Args:
        x, y, z: Fractional cube coordinates

    Returns:
        Nearest valid hex coordinate
    """
    rx, ry, rz = round(x), round(y), round(z)

    x_diff = abs(rx - x)
    y_diff = abs(ry - y)
    z_diff = abs(rz - z)

    if x_diff > y_diff and x_diff > z_diff:
        rx = -ry - rz
    elif y_diff > z_diff:
        ry = -rx - rz
    else:
        rz = -rx - ry

    return HexCoord(rx, ry, rz)


def hex_line(a: HexCoord, b: HexCoord) -> List[HexCoord]:
    """
    Get all hexes along the line from a to b.

    Args:
        a: Start coordinate
        b: End coordinate

    Returns:
        List of hexes forming a line from a to b (inclusive)

    Examples:
        >>> a = HexCoord(0, 0, 0)
        >>> b = HexCoord(3, -3, 0)
        >>> line = hex_line(a, b)
        >>> len(line)
        4
    """
    distance = hex_distance(a, b)
    if distance == 0:
        return [a]

    results = []
    for i in range(distance + 1):
        t = i / distance
        fx, fy, fz = line_interpolation(a, b, t)
        results.append(hex_round(fx, fy, fz))

    return results


def compute_spatial_center(coords: List[HexCoord]) -> HexCoord:
    """
    Compute the approximate center of a collection of hexes.

    Args:
        coords: List of hex coordinates

    Returns:
        Hex coordinate closest to the geometric center

    Raises:
        ValueError: If coords is empty
    """
    if not coords:
        raise ValueError("Cannot compute center of empty coordinate list")

    avg_x = sum(c.x for c in coords) / len(coords)
    avg_y = sum(c.y for c in coords) / len(coords)
    avg_z = sum(c.z for c in coords) / len(coords)

    return hex_round(avg_x, avg_y, avg_z)


def rotate_hex(coord: HexCoord, rotations: int = 1) -> HexCoord:
    """
    Rotate hex coordinate around origin by 60-degree increments.

    Args:
        coord: Hex coordinate to rotate
        rotations: Number of 60-degree clockwise rotations (can be negative)

    Returns:
        Rotated hex coordinate

    Examples:
        >>> coord = HexCoord(1, -1, 0)
        >>> rotate_hex(coord, 1)  # 60 degrees clockwise
        HexCoord(x=1, y=0, z=-1)
        >>> rotate_hex(coord, 6)  # Full rotation
        HexCoord(x=1, y=-1, z=0)
    """
    # Normalize rotations to 0-5
    rotations = rotations % 6

    # Rotation matrices for 60-degree increments (cube coordinates)
    # Each rotation: (x, y, z) -> (-y, -z, -x)
    x, y, z = coord.x, coord.y, coord.z

    for _ in range(rotations):
        x, y, z = -z, -x, -y

    return HexCoord(x, y, z)

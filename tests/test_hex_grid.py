"""
Tests for hexagonal grid mathematics.

Includes both standard unit tests and property-based tests using hypothesis.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings
import numpy as np

from ti4_analysis.algorithms.hex_grid import (
    HexCoord, hex_distance, get_adjacent_coordinates, get_ring,
    get_hexes_in_range, breadth_first_search, hex_round, rotate_hex
)


class TestHexCoord:
    """Tests for HexCoord dataclass."""

    def test_valid_coord_creation(self):
        """Test creating valid hex coordinates."""
        coord = HexCoord(1, -1, 0)
        assert coord.x == 1
        assert coord.y == -1
        assert coord.z == 0

    def test_invalid_coord_raises_error(self):
        """Test that invalid coordinates raise ValueError."""
        with pytest.raises(ValueError, match="Invalid cube coordinates"):
            HexCoord(1, 1, 1)  # Violates x + y + z = 0

    def test_origin(self):
        """Test origin coordinate."""
        origin = HexCoord(0, 0, 0)
        assert origin.x + origin.y + origin.z == 0

    @given(
        x=st.integers(min_value=-10, max_value=10),
        y=st.integers(min_value=-10, max_value=10)
    )
    def test_coordinate_invariant(self, x, y):
        """Property test: all coordinates must satisfy x + y + z = 0."""
        z = -x - y
        coord = HexCoord(x, y, z)
        assert coord.x + coord.y + coord.z == 0


class TestHexDistance:
    """Tests for hex distance calculations."""

    def test_distance_to_self(self):
        """Distance from a hex to itself should be 0."""
        coord = HexCoord(0, 0, 0)
        assert hex_distance(coord, coord) == 0

    def test_distance_to_neighbor(self):
        """Distance to adjacent hex should be 1."""
        origin = HexCoord(0, 0, 0)
        neighbor = HexCoord(1, -1, 0)
        assert hex_distance(origin, neighbor) == 1

    def test_distance_symmetry(self):
        """Distance should be symmetric."""
        a = HexCoord(1, -1, 0)
        b = HexCoord(2, -3, 1)
        assert hex_distance(a, b) == hex_distance(b, a)

    @given(
        x1=st.integers(min_value=-10, max_value=10),
        y1=st.integers(min_value=-10, max_value=10),
        x2=st.integers(min_value=-10, max_value=10),
        y2=st.integers(min_value=-10, max_value=10)
    )
    def test_distance_symmetry_property(self, x1, y1, x2, y2):
        """Property test: distance(a, b) == distance(b, a)."""
        z1, z2 = -x1 - y1, -x2 - y2
        a = HexCoord(x1, y1, z1)
        b = HexCoord(x2, y2, z2)
        assert hex_distance(a, b) == hex_distance(b, a)

    @given(
        x1=st.integers(min_value=-5, max_value=5),
        y1=st.integers(min_value=-5, max_value=5),
        x2=st.integers(min_value=-5, max_value=5),
        y2=st.integers(min_value=-5, max_value=5),
        x3=st.integers(min_value=-5, max_value=5),
        y3=st.integers(min_value=-5, max_value=5)
    )
    @settings(max_examples=50)  # Reduce examples for triangle inequality
    def test_triangle_inequality(self, x1, y1, x2, y2, x3, y3):
        """Property test: triangle inequality holds."""
        a = HexCoord(x1, y1, -x1 - y1)
        b = HexCoord(x2, y2, -x2 - y2)
        c = HexCoord(x3, y3, -x3 - y3)

        d_ab = hex_distance(a, b)
        d_bc = hex_distance(b, c)
        d_ac = hex_distance(a, c)

        assert d_ac <= d_ab + d_bc

    @given(
        x=st.integers(min_value=-10, max_value=10),
        y=st.integers(min_value=-10, max_value=10)
    )
    def test_non_negative_distance(self, x, y):
        """Property test: distances are always non-negative."""
        origin = HexCoord(0, 0, 0)
        coord = HexCoord(x, y, -x - y)
        assert hex_distance(origin, coord) >= 0


class TestAdjacentCoordinates:
    """Tests for adjacent hex coordinate generation."""

    def test_has_six_neighbors(self):
        """Every hex should have exactly 6 neighbors."""
        coord = HexCoord(0, 0, 0)
        neighbors = get_adjacent_coordinates(coord)
        assert len(neighbors) == 6

    def test_all_neighbors_distance_one(self):
        """All neighbors should be distance 1."""
        coord = HexCoord(0, 0, 0)
        neighbors = get_adjacent_coordinates(coord)
        for neighbor in neighbors:
            assert hex_distance(coord, neighbor) == 1

    def test_neighbors_unique(self):
        """All neighbors should be unique."""
        coord = HexCoord(0, 0, 0)
        neighbors = get_adjacent_coordinates(coord)
        assert len(neighbors) == len(set(neighbors))

    @given(
        x=st.integers(min_value=-10, max_value=10),
        y=st.integers(min_value=-10, max_value=10)
    )
    def test_always_six_neighbors(self, x, y):
        """Property test: every hex has exactly 6 neighbors."""
        coord = HexCoord(x, y, -x - y)
        neighbors = get_adjacent_coordinates(coord)
        assert len(neighbors) == 6

    @given(
        x=st.integers(min_value=-10, max_value=10),
        y=st.integers(min_value=-10, max_value=10)
    )
    def test_neighbor_distance_always_one(self, x, y):
        """Property test: all neighbors are distance 1."""
        coord = HexCoord(x, y, -x - y)
        neighbors = get_adjacent_coordinates(coord)
        for neighbor in neighbors:
            assert hex_distance(coord, neighbor) == 1


class TestRing:
    """Tests for ring generation."""

    def test_ring_radius_zero(self):
        """Ring of radius 0 should contain only the center."""
        center = HexCoord(0, 0, 0)
        ring = get_ring(center, 0)
        assert len(ring) == 1
        assert ring[0] == center

    def test_ring_radius_one(self):
        """Ring of radius 1 should contain 6 hexes."""
        center = HexCoord(0, 0, 0)
        ring = get_ring(center, 1)
        assert len(ring) == 6

    def test_ring_radius_two(self):
        """Ring of radius 2 should contain 12 hexes."""
        center = HexCoord(0, 0, 0)
        ring = get_ring(center, 2)
        assert len(ring) == 12

    @given(radius=st.integers(min_value=1, max_value=5))
    def test_ring_size(self, radius):
        """Property test: ring of radius r contains 6r hexes."""
        center = HexCoord(0, 0, 0)
        ring = get_ring(center, radius)
        assert len(ring) == 6 * radius

    @given(radius=st.integers(min_value=0, max_value=5))
    def test_ring_all_same_distance(self, radius):
        """Property test: all hexes in ring are same distance from center."""
        center = HexCoord(0, 0, 0)
        ring = get_ring(center, radius)
        for hex_coord in ring:
            assert hex_distance(center, hex_coord) == radius


class TestHexesInRange:
    """Tests for getting hexes within range."""

    def test_range_zero(self):
        """Range 0 should contain only center."""
        center = HexCoord(0, 0, 0)
        hexes = get_hexes_in_range(center, 0)
        assert len(hexes) == 1

    def test_range_one(self):
        """Range 1 should contain center + 6 neighbors = 7."""
        center = HexCoord(0, 0, 0)
        hexes = get_hexes_in_range(center, 1)
        assert len(hexes) == 7

    def test_range_two(self):
        """Range 2 should contain 1 + 6 + 12 = 19."""
        center = HexCoord(0, 0, 0)
        hexes = get_hexes_in_range(center, 2)
        assert len(hexes) == 19

    @given(max_radius=st.integers(min_value=0, max_value=4))
    def test_range_count(self, max_radius):
        """Property test: range R contains 1 + 6 + 12 + ... + 6R hexes."""
        center = HexCoord(0, 0, 0)
        hexes = get_hexes_in_range(center, max_radius)

        # Formula: 1 + 6(1 + 2 + ... + R) = 1 + 6R(R+1)/2 = 1 + 3R(R+1)
        expected_count = 1 + 3 * max_radius * (max_radius + 1)
        assert len(hexes) == expected_count

    @given(max_radius=st.integers(min_value=0, max_value=5))
    def test_range_all_within_distance(self, max_radius):
        """Property test: all hexes are within max_radius."""
        center = HexCoord(0, 0, 0)
        hexes = get_hexes_in_range(center, max_radius)
        for hex_coord in hexes:
            assert hex_distance(center, hex_coord) <= max_radius


class TestHexRound:
    """Tests for hex coordinate rounding."""

    def test_round_exact_coordinate(self):
        """Rounding exact coordinates should return same value."""
        coord = HexCoord(1, -1, 0)
        rounded = hex_round(1.0, -1.0, 0.0)
        assert rounded == coord

    def test_round_near_coordinate(self):
        """Rounding near a coordinate should return that coordinate."""
        rounded = hex_round(1.1, -1.1, 0.0)
        assert rounded == HexCoord(1, -1, 0)

    def test_round_maintains_invariant(self):
        """Rounded coordinates must satisfy x + y + z = 0."""
        rounded = hex_round(1.3, -0.8, -0.5)
        assert rounded.x + rounded.y + rounded.z == 0


class TestRotateHex:
    """Tests for hex rotation."""

    def test_rotate_six_times_returns_original(self):
        """Rotating 6 times (360 degrees) should return original."""
        coord = HexCoord(1, -1, 0)
        rotated = rotate_hex(coord, 6)
        assert rotated == coord

    def test_rotate_zero(self):
        """Rotating 0 times should return original."""
        coord = HexCoord(1, -1, 0)
        rotated = rotate_hex(coord, 0)
        assert rotated == coord

    @given(
        x=st.integers(min_value=-5, max_value=5),
        y=st.integers(min_value=-5, max_value=5)
    )
    def test_rotate_preserves_distance_from_origin(self, x, y):
        """Property test: rotation preserves distance from origin."""
        coord = HexCoord(x, y, -x - y)
        origin = HexCoord(0, 0, 0)

        original_dist = hex_distance(origin, coord)

        for rotations in range(1, 6):
            rotated = rotate_hex(coord, rotations)
            assert hex_distance(origin, rotated) == original_dist


class TestBreadthFirstSearch:
    """Tests for BFS pathfinding."""

    def test_bfs_single_hex(self):
        """BFS on single hex should return distance 0."""
        start = HexCoord(0, 0, 0)
        valid = {start}
        distances = breadth_first_search(start, valid, max_distance=5)
        assert distances[start] == 0
        assert len(distances) == 1

    def test_bfs_line(self):
        """BFS on a line should compute correct distances."""
        hexes = [HexCoord(i, 0, -i) for i in range(5)]
        valid = set(hexes)
        distances = breadth_first_search(hexes[0], valid, max_distance=10)

        for i, hex_coord in enumerate(hexes):
            assert distances[hex_coord] == i

    def test_bfs_respects_max_distance(self):
        """BFS should not exceed max_distance."""
        hexes = [HexCoord(i, 0, -i) for i in range(10)]
        valid = set(hexes)
        distances = breadth_first_search(hexes[0], valid, max_distance=3)

        assert max(distances.values()) <= 3
        assert len(distances) <= 4  # 0, 1, 2, 3

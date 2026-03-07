# Adjacency Logic Verification

## Question

Does the one-shot script (`generate_game_map.py`) use the same adjacency logic as the comprehensive analysis experiments?

## Answer

✅ **YES** - Both use the exact same adjacency logic from the shared `TI4Map` class.

## Shared Components

### 1. Core Adjacency Methods

Both the one-shot script and comprehensive analysis use the same `TI4Map` class from `balance_engine.py`:

**File:** `ti4-analysis/src/ti4_analysis/algorithms/balance_engine.py`

**Methods:**

- `get_adjacent_spaces(space)` - Handles hex adjacency + hyperlanes
- `get_adjacent_spaces_including_wormholes(space)` - Adds wormhole connections

### 2. Hyperlane Patterns

Both use the same `HYPERLANE_PATTERNS` dictionary:

```python
HYPERLANE_PATTERNS = {
    "83A": [[1, 4]], "83B": [[0, 3], [0, 2], [3, 5]],
    "84A": [[2, 5]], "84B": [[0, 3], [0, 4], [1, 3]],
    "85A": [[1, 5]], "85B": [[0, 3], [0, 2], [3, 5]],
    # ... etc
}
```

### 3. Hyperlane Connection Logic

Both use `_get_hyperlane_connected_spaces()` which:

- Takes hyperlane tile ID and rotation
- Applies rotation to pattern edges
- Returns connected spaces based on entry direction

## Code Flow

### One-Shot Script (`generate_game_map.py`)

1. Creates `TI4Map` via `generate_random_map()` → `TI4Map(spaces)`
2. Uses `TI4Map.get_adjacent_spaces_including_wormholes()` for pathfinding
3. Calls `get_shortest_modded_distance()` which uses adjacency methods

### Comprehensive Analysis (G1, G1-MO, G3-D, G5)

1. All experiments create `TI4Map` objects
2. All use `TI4Map.get_adjacent_spaces_including_wormholes()` for pathfinding
3. All call `get_shortest_modded_distance()` which uses adjacency methods

## Adjacency Features

Both implementations handle:

✅ **Hex Grid Adjacency**

- 6-directional hex neighbors using cube coordinates
- `get_adjacent_coordinates()` from `hex_grid.py`

✅ **Hyperlane Connections**

- Pattern-based connections (e.g., edge 1 → edge 4)
- Rotation-aware (applies rotation to pattern edges)
- Topological "teleportation" across void spaces

✅ **Wormhole Connections**

- Same-type wormholes treated as adjacent
- Alpha ↔ Alpha, Beta ↔ Beta, etc.

✅ **Anomaly Handling**

- Nebula movement rules
- Gravity rift +1 move bonus
- Supernova/asteroid blocking

## Verification

**Shared Code Path:**

```
balance_engine.py
  ├── TI4Map class
  │   ├── get_adjacent_spaces() → uses HYPERLANE_PATTERNS
  │   └── get_adjacent_spaces_including_wormholes() → adds wormholes
  └── get_shortest_modded_distance() → uses adjacency methods
```

**Both Use:**

- Same `TI4Map` class
- Same `HYPERLANE_PATTERNS` dictionary
- Same `_get_hyperlane_connected_spaces()` function
- Same `get_adjacent_coordinates()` from `hex_grid.py`

## Conclusion

✅ **The one-shot script and comprehensive analysis use identical adjacency logic.**

There are no differences in:

- Hyperlane pattern definitions
- Hyperlane connection resolution
- Wormhole adjacency
- Hex grid adjacency
- Anomaly movement rules

The only difference is the **optimization algorithm** (G1 hill-climbing vs. G3-D NSGA-II), but both operate on the same `TI4Map` objects with the same adjacency methods.

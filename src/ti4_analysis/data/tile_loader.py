"""
Tile database loader for TI4 map generation.

Parses the JavaScript tile database and converts it to Python data structures.
Creates a cached JSON version for faster subsequent loads.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass

from .map_structures import (
    System, Planet, Anomaly, Wormhole,
    PlanetTrait, TechSpecialty
)


# Anomaly and wormhole type mappings
ANOMALY_MAP = {
    "nebula": Anomaly.NEBULA,
    "gravity-rift": Anomaly.GRAVITY_RIFT,
    "asteroid-field": Anomaly.ASTEROID_FIELD,
    "supernova": Anomaly.SUPERNOVA,
}

WORMHOLE_MAP = {
    "alpha": Wormhole.ALPHA,
    "beta": Wormhole.BETA,
    "gamma": Wormhole.GAMMA,
    "delta": Wormhole.DELTA,
    "epsilon": Wormhole.EPSILON,
    "zeta": Wormhole.ZETA,
    "eta": Wormhole.ETA,
    "theta": Wormhole.THETA,
    "iota": Wormhole.IOTA,
    "kappa": Wormhole.KAPPA,
}

TRAIT_MAP = {
    "hazardous": PlanetTrait.HAZARDOUS,
    "industrial": PlanetTrait.INDUSTRIAL,
    "cultural": PlanetTrait.CULTURAL,
    None: None,
}

SPECIALTY_MAP = {
    "biotic": TechSpecialty.BIOTIC,
    "warfare": TechSpecialty.WARFARE,
    "propulsion": TechSpecialty.PROPULSION,
    "cybernetic": TechSpecialty.CYBERNETIC,
    None: None,
}


@dataclass
class TileDatabase:
    """Container for all tile data."""
    tiles: Dict[str, System]
    base_tiles: List[str]
    pok_tiles: List[str]
    uncharted_tiles: List[str]
    blue_tiles: List[str]
    red_tiles: List[str]
    home_tiles: List[str]
    hyperlane_tiles: List[str]

    def get_swappable_tiles(
        self,
        include_pok: bool = True,
        include_uncharted: bool = False,
        blue_count: int = 18,
        red_count: int = 12
    ) -> List[System]:
        """
        Get a pool of tiles suitable for random map generation.

        Args:
            include_pok: Include Prophecy of Kings expansion
            include_uncharted: Include Uncharted Space expansion
            blue_count: Number of blue (planet) tiles to include
            red_count: Number of red (empty/anomaly) tiles to include

        Returns:
            List of System objects
        """
        # Build allowed tile pool
        allowed_ids: Set[str] = set(self.base_tiles)
        if include_pok:
            allowed_ids.update(self.pok_tiles)
        if include_uncharted:
            allowed_ids.update(self.uncharted_tiles)

        # Filter to swappable tiles only (no homes, no Mecatol)
        blue_pool = [
            self.tiles[tid] for tid in self.blue_tiles
            if tid in allowed_ids
        ]
        red_pool = [
            self.tiles[tid] for tid in self.red_tiles
            if tid in allowed_ids
        ]

        # Randomly sample (caller should shuffle/seed for reproducibility)
        import random
        selected_blue = random.sample(blue_pool, min(blue_count, len(blue_pool)))
        selected_red = random.sample(red_pool, min(red_count, len(red_pool)))

        return selected_blue + selected_red


def parse_javascript_tile_data(js_file_path: Path) -> Dict:
    """
    Parse the JavaScript tileData.js file into a Python dictionary.

    This uses regex-based parsing to extract the JSON-like structure
    from the JavaScript export.

    Args:
        js_file_path: Path to tileData.js

    Returns:
        Dictionary with tile data
    """
    with open(js_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract the main tileData object (everything between "const tileData = {" and the closing "}")
    # This is a simplified parser - may need refinement for complex cases
    match = re.search(r'const tileData = ({.*?});?\s*tileData\.green', content, re.DOTALL)
    if not match:
        raise ValueError("Could not find tileData object in JavaScript file")

    tile_data_str = match.group(1)

    # Convert JavaScript object notation to JSON
    # Replace single quotes with double quotes
    tile_data_str = re.sub(r"'", '"', tile_data_str)
    # Remove trailing commas before closing braces/brackets
    tile_data_str = re.sub(r',(\s*[}\]])', r'\1', tile_data_str)

    try:
        return json.loads(tile_data_str)
    except json.JSONDecodeError as e:
        # If direct parsing fails, save intermediate result for debugging
        debug_path = js_file_path.parent / "tileData_debug.json"
        with open(debug_path, 'w') as f:
            f.write(tile_data_str)
        raise ValueError(f"Failed to parse JavaScript as JSON. Debug output saved to {debug_path}") from e


def convert_tile_to_system(tile_id: str, tile_data: Dict) -> System:
    """
    Convert a tile data dictionary to a System object.

    Args:
        tile_id: The tile ID (string)
        tile_data: Dictionary with tile properties

    Returns:
        System object
    """
    # Convert planets
    planets = []
    for p_data in tile_data.get("planets", []):
        traits = [TRAIT_MAP[p_data.get("trait")]] if p_data.get("trait") else None
        specialty = [SPECIALTY_MAP[p_data.get("specialty")]] if p_data.get("specialty") else None

        planet = Planet(
            name=p_data["name"],
            resources=p_data["resources"],
            influence=p_data["influence"],
            traits=traits,
            tech_specialties=specialty
        )
        planets.append(planet)

    # Convert anomalies
    anomaly_list = [ANOMALY_MAP[a] for a in tile_data.get("anomaly", []) if a in ANOMALY_MAP]
    anomalies = anomaly_list if anomaly_list else None

    # Convert wormholes
    wormhole_list = tile_data.get("wormhole", [])
    wormhole = WORMHOLE_MAP[wormhole_list[0]] if wormhole_list else None

    # Create system
    return System(
        id=int(tile_id) if tile_id.isdigit() else hash(tile_id) % (10**9),  # Hash for non-numeric IDs
        planets=planets,
        anomalies=anomalies,
        wormhole=wormhole
    )


def load_tile_database(
    project_root: Optional[Path] = None,
    use_cache: bool = True,
    force_reload: bool = False
) -> TileDatabase:
    """
    Load the complete tile database from JavaScript source or cache.

    Args:
        project_root: Root directory of the ti4_map_generator project.
                     If None, attempts to find it automatically.
        use_cache: If True, load from cached JSON if available
        force_reload: If True, ignore cache and reload from JavaScript

    Returns:
        TileDatabase object with all tiles
    """
    # Find project root
    if project_root is None:
        current = Path(__file__).resolve()
        # Navigate up from ti4-analysis/src/ti4_analysis/data/tile_loader.py
        # to find the parent ti4_map_generator directory
        for parent in current.parents:
            if (parent / "src" / "data" / "tileData.js").exists():
                project_root = parent
                break
        if project_root is None:
            raise FileNotFoundError("Could not find project root with src/data/tileData.js")

    js_file = project_root / "src" / "data" / "tileData.js"
    cache_file = Path(__file__).parent / "tiles_cache.json"
    board_data_file = project_root / "src" / "data" / "boardData.json"

    # Check cache
    if use_cache and not force_reload and cache_file.exists():
        print(f"Loading tiles from cache: {cache_file}")
        with open(cache_file, 'r') as f:
            cached = json.load(f)

        # Reconstruct System objects
        tiles = {
            tid: convert_tile_to_system(tid, tdata)
            for tid, tdata in cached["tiles"].items()
        }

        return TileDatabase(
            tiles=tiles,
            base_tiles=cached["base_tiles"],
            pok_tiles=cached["pok_tiles"],
            uncharted_tiles=cached["uncharted_tiles"],
            blue_tiles=cached["blue_tiles"],
            red_tiles=cached["red_tiles"],
            home_tiles=cached["home_tiles"],
            hyperlane_tiles=cached["hyperlane_tiles"]
        )

    # Parse from JavaScript
    print(f"Parsing JavaScript tile database: {js_file}")
    raw_data = parse_javascript_tile_data(js_file)

    # Load expansion lists
    with open(js_file, 'r') as f:
        content = f.read()

    # Extract tile lists using regex
    def extract_list(list_name: str) -> List[str]:
        pattern = f'"{list_name}":\s*\[(.*?)\]'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            items_str = match.group(1)
            # Extract quoted strings
            items = re.findall(r'"([^"]+)"', items_str)
            return items
        return []

    base_tiles = extract_list("base")
    pok_tiles = extract_list("pok")
    uncharted_tiles = extract_list("uncharted")
    hyperlane_tiles = raw_data.get("hyperlanes", [])

    # Convert all tiles to System objects
    all_tiles = raw_data["all"]
    tiles = {}
    blue_tiles = []
    red_tiles = []
    home_tiles = []

    for tile_id, tile_data in all_tiles.items():
        system = convert_tile_to_system(tile_id, tile_data)
        tiles[tile_id] = system

        # Categorize
        tile_type = tile_data.get("type")
        is_special = tile_data.get("special", False)

        if tile_type == "green" and not is_special:
            home_tiles.append(tile_id)
        elif tile_type == "blue" and not is_special:
            blue_tiles.append(tile_id)
        elif tile_type == "red" and not is_special:
            red_tiles.append(tile_id)

    db = TileDatabase(
        tiles=tiles,
        base_tiles=base_tiles,
        pok_tiles=pok_tiles,
        uncharted_tiles=uncharted_tiles,
        blue_tiles=blue_tiles,
        red_tiles=red_tiles,
        home_tiles=home_tiles,
        hyperlane_tiles=hyperlane_tiles
    )

    # Save cache
    if use_cache:
        print(f"Saving tile cache: {cache_file}")
        cache_data = {
            "tiles": {
                tid: {
                    "type": "blue" if tid in blue_tiles else ("red" if tid in red_tiles else "green"),
                    "planets": [
                        {
                            "name": p.name,
                            "resources": p.resources,
                            "influence": p.influence,
                            "trait": p.traits[0].value if p.traits else None,
                            "specialty": p.tech_specialties[0].value if p.tech_specialties else None,
                            "legendary": False  # Simplified for now
                        }
                        for p in system.planets
                    ],
                    "anomaly": [a.value for a in system.anomalies] if system.anomalies else [],
                    "wormhole": [system.wormhole.value] if system.wormhole else []
                }
                for tid, system in tiles.items()
            },
            "base_tiles": base_tiles,
            "pok_tiles": pok_tiles,
            "uncharted_tiles": uncharted_tiles,
            "blue_tiles": blue_tiles,
            "red_tiles": red_tiles,
            "home_tiles": home_tiles,
            "hyperlane_tiles": hyperlane_tiles
        }

        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)

    print(f"Loaded {len(tiles)} tiles:")
    print(f"  - Base: {len(base_tiles)}")
    print(f"  - PoK: {len(pok_tiles)}")
    print(f"  - Blue: {len(blue_tiles)}")
    print(f"  - Red: {len(red_tiles)}")
    print(f"  - Home: {len(home_tiles)}")

    return db


def load_board_template(
    player_count: int = 6,
    template_name: str = "normal",
    project_root: Optional[Path] = None
) -> Dict:
    """
    Load a board template configuration.

    Args:
        player_count: Number of players (2-8)
        template_name: Template variant (e.g., "normal", "spiral", "large")
        project_root: Project root directory

    Returns:
        Dictionary with home_worlds, primary_tiles, secondary_tiles, etc.
    """
    if project_root is None:
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / "src" / "data" / "boardData.json").exists():
                project_root = parent
                break

    board_file = project_root / "src" / "data" / "boardData.json"

    with open(board_file, 'r') as f:
        board_data = json.load(f)

    template = board_data["styles"][str(player_count)][template_name]

    return template

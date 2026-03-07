# TI4 Map Balance Analysis

Scientific analysis and visualization toolkit for Twilight Imperium 4 map balancing algorithms.

This Python package provides rigorous mathematical analysis of the balancing algorithms used in the [TI4 Map Generator](https://github.com/KeeganW/ti4), implementing advanced spatial statistics proposed in the research documentation.

## 🚀 Quick Start: Export to Frontend

Generate balanced maps in Python and visualize them in the 3D TypeScript frontend:

```python
from ti4_analysis.algorithms.map_generator import generate_random_map
from ti4_analysis.data.ti4proj_exporter import export_to_ti4proj

# Generate a balanced 6-player map
map_obj = generate_random_map(player_count=6, random_seed=42)

# Export to .ti4proj format (loads directly in the frontend)
export_to_ti4proj(
    map_obj,
    "output/my_map.ti4proj",
    map_name="Balanced 6-Player Map",
    jaines_index=0.95,  # Balance metric
    morans_i=0.12       # Spatial autocorrelation
)
```

**Next:** Open `my_map.ti4proj` in TI4 Map Analyzer to see your map in 3D!

📖 **See:** [QUICK_START_EXPORT.md](QUICK_START_EXPORT.md) | [Full Integration Guide](PYTHON_TO_TYPESCRIPT_INTEGRATION.md)

---

## Features

### Core Algorithms
- **Hexagonal Grid Mathematics**: Cube coordinate system with distance calculations, pathfinding, and spatial operations
- **Balance Optimization Engine**: Iterative swapping algorithm (ti4-map-lab) with greedy hill climbing
- **Distance-Weighted Evaluation**: Accessibility-based scoring with configurable evaluator strategies

### Advanced Spatial Statistics
Implementation of metrics from `docs/Twilight Imperium Map Balance Research.md`:

- **Moran's I**: Spatial autocorrelation (resource clustering detection)
- **Getis-Ord Gi***: Hot spot analysis (identifies resource-rich clusters)
- **Gravity Model**: Distance-weighted accessibility calculation
- **Jain's Fairness Index**: Equality of resource distribution
- **Local Indicators of Spatial Association (LISA)**: Neighborhood-based clustering

### Visualization
Publication-quality plots using Matplotlib and Seaborn:
- Hexagonal map rendering
- System value heatmaps
- Balance convergence plots
- Before/after comparisons
- Comprehensive balance reports

### Testing
- **Property-based testing** with Hypothesis
- **Mathematical verification** of key algorithms
- **Unit tests** for all core functionality

---

## Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Setup

1. **Clone the repository** (if not already done):
```bash
cd ti4_map_generator/ti4-analysis
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Install in development mode**:
```bash
pip install -e .
```

5. **Verify installation**:
```bash
python -c "import ti4_analysis; print('✓ Installation successful')"
```

---

## Quick Start

### Example 1: Basic Balance Analysis

```python
from ti4_analysis.algorithms.hex_grid import HexCoord
from ti4_analysis.data.map_structures import Planet, System, MapSpace, MapSpaceType, Evaluator
from ti4_analysis.algorithms.balance_engine import TI4Map, improve_balance, analyze_balance

# Create a simple map
home1 = MapSpace(HexCoord(0, 0, 0), MapSpaceType.HOME)
home2 = MapSpace(HexCoord(3, -3, 0), MapSpaceType.HOME)

planet = Planet("Test Planet", resources=3, influence=2)
system = System(id=1, planets=[planet])
sys_space = MapSpace(HexCoord(1, -1, 0), MapSpaceType.SYSTEM, system)

ti4_map = TI4Map([home1, home2, sys_space])

# Create evaluator
evaluator = Evaluator(name="Joebrew")

# Analyze balance
analysis = analyze_balance(ti4_map, evaluator)
print(f"Balance Gap: {analysis['balance_gap']:.2f}")
print(f"Fairness Index: {analysis['fairness_index']:.3f}")

# Optimize balance
final_gap, history = improve_balance(ti4_map, evaluator, iterations=100)
print(f"Optimized Gap: {final_gap:.2f}")
```

### Example 2: Spatial Statistics

```python
from ti4_analysis.spatial_stats.spatial_metrics import (
    comprehensive_spatial_analysis, identify_hotspots
)

# Run comprehensive spatial analysis
results = comprehensive_spatial_analysis(ti4_map, evaluator)

print(f"Moran's I (clustering): {results['resource_clustering_morans_i']:.3f}")
print(f"Gini Coefficient: {results['gini_coefficient']:.3f}")
print(f"Hot spots detected: {results['num_hotspots']}")

# Identify specific hot/cold spots
hotspots = identify_hotspots(ti4_map, evaluator)
for coord, gi_star, spot_type in hotspots:
    print(f"{spot_type}: {coord} (Gi* = {gi_star:.2f})")
```

### Example 3: Visualization

```python
import matplotlib.pyplot as plt
from ti4_analysis.visualization.map_viz import create_balance_report

# Create comprehensive report
fig = create_balance_report(ti4_map, evaluator, history)
plt.savefig('balance_report.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## Jupyter Notebooks

Interactive examples are provided in the [notebooks/](./notebooks/) directory:

- **01_balance_analysis_example.ipynb**: Complete walkthrough of balance optimization and spatial statistics

### Running Notebooks

```bash
jupyter notebook notebooks/
```

---
## Documentation

Detailed technical documentation is available in the [`docs/`](./docs/) directory:

- **[Anomaly Pathfinding](./docs/ANOMALY_PATHFINDING.md)**: Comprehensive guide to how the pathfinding algorithm handles TI4 anomaly systems (asteroid fields, supernovas, gravity rifts, nebulas). Includes mathematical models, edge cases, and implementation details.

For game design research and spatial theory, see the root [docs/](../docs/) directory.

---


## Project Structure

```
ti4-analysis/
├── src/ti4_analysis/
│   ├── algorithms/
│   │   ├── hex_grid.py          # Hexagonal grid mathematics
│   │   └── balance_engine.py    # Balance optimization algorithm
│   ├── data/
│   │   └── map_structures.py    # Planet, System, Map classes
│   ├── spatial_stats/
│   │   └── spatial_metrics.py   # Moran's I, Getis-Ord, etc.
│   └── visualization/
│       └── map_viz.py           # Plotting functions
├── tests/
│   ├── test_hex_grid.py         # Property-based tests
│   └── test_balance_engine.py   # Balance algorithm tests
├── notebooks/
│   └── 01_balance_analysis_example.ipynb
├── docs/                         # Technical documentation
├── results/                      # Output directory for plots
├── pyproject.toml               # Modern Python packaging
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

---

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ti4_analysis --cov-report=html

# Run specific test file
pytest tests/test_hex_grid.py

# Run property-based tests with more examples
pytest tests/test_hex_grid.py --hypothesis-show-statistics

# Run the minimal validation suite (TS-Hybrid vs. SATS-BO experiments)
pytest -m minimal_validation
```

The validation harnesses look for a cascade dataset via the environment variable
`TI4_VALIDATION_DATA`. When the path is absent (the default for CI runs), the tests
fallback to deterministic synthetic samples, keeping the suite lightweight.

The TS-Hybrid calibration assumes a calm regime with importance-weight variance
below `0.05` and temporal-difference drift below `0.2`. The heavy-tail probe pushes
importance weights up to ~3 and drives TD drift far beyond that limit, providing the
decisive pass/fail contrast described in `docs/Defending Complex System Design.md`.

---

## Key Concepts

### Hexagonal Grid (Cube Coordinates)

We use cube coordinates for hexagonal grids where each hex is represented by `(x, y, z)` with the constraint `x + y + z = 0`. This provides elegant distance and pathfinding algorithms.

**Distance Formula**:
```
d(a, b) = max(|ax - bx|, |ay - by|, |az - bz|)
```

### Balance Optimization

The balance engine minimizes the "balance gap" (difference between most and least advantaged players) through iterative system swapping:

1. Calculate home value for each player position
2. Compute gap = max(values) - min(values)
3. Randomly swap two eligible systems
4. Accept swap if gap decreases (greedy)
5. Repeat for N iterations

This is a **local search / hill climbing** algorithm that may find local minima.

### Evaluator Strategies

Three planet evaluation strategies:

- **SUM**: Total value = Resources + Influence + Tech
- **GREATEST**: Value = max(Resources, Influence, Tech)
- **GREATEST_PLUS_TECH**: Value = max(Resources, Influence) + Tech

The "Joebrew" evaluator uses GREATEST_PLUS_TECH with resource multiplier 3×, influence 2×, and tech specialty bonus 5.

### Spatial Statistics

**Moran's I**:
```
I = (N/W) × Σᵢ Σⱼ wᵢⱼ(xᵢ - x̄)(xⱼ - x̄) / Σᵢ(xᵢ - x̄)²
```
- I > 0: Spatial clustering (similar values near each other)
- I ≈ 0: Random spatial pattern
- I < 0: Spatial dispersion (dissimilar values near each other)

**Getis-Ord Gi*** (hot spot detection):
```
Gi* = [Σⱼ wᵢⱼxⱼ - X̄ Σⱼ wᵢⱼ] / [S√((n Σⱼ wᵢⱼ² - (Σⱼ wᵢⱼ)²) / (n-1))]
```
- |Gi*| > 1.96: Statistically significant at 95% confidence
- Gi* > 0: Hot spot (high-value cluster)
- Gi* < 0: Cold spot (low-value cluster)

**Jain's Fairness Index**:
```
J = (Σxᵢ)² / (n × Σxᵢ²)
```
- Range: [1/n, 1]
- J = 1: Perfect fairness
- J = 1/n: Maximum unfairness

---

## Research Implementation Status

### ✅ Implemented
- Hexagonal grid mathematics
- Distance-weighted accessibility (Gravity Model)
- Iterative balance optimization
- Moran's I spatial autocorrelation
- Getis-Ord Gi* hot spot analysis
- Jain's Fairness Index
- Local Moran's I (LISA)
- Gini coefficient
- Comprehensive visualization suite

### 🚧 Proposed (from research docs)
- Multi-objective Pareto optimization
- Betweenness centrality (strategic chokepoints)
- Voronoi tessellation (territorial analysis)
- Forward dock positioning analysis
- Player-to-player distance symmetry
- Empirical validation studies

---

## Dependencies

### Core Scientific Stack
- **NumPy** (≥1.24): Numerical computing
- **SciPy** (≥1.11): Scientific algorithms and spatial analysis
- **Pandas** (≥2.0): Data manipulation

### Spatial Analysis
- **PySAL** (≥23.7): Python Spatial Analysis Library
- **libpysal** (≥4.9): Spatial weights and I/O
- **esda** (≥2.5): Exploratory Spatial Data Analysis
- **GeoPandas** (≥0.14): Geographic data structures

### Visualization
- **Matplotlib** (≥3.7): Publication-quality plots
- **Seaborn** (≥0.12): Statistical visualization
- **Plotly** (≥5.17): Interactive plots

### Graph Theory
- **NetworkX** (≥3.1): Graph algorithms (for future betweenness centrality)

### Testing
- **pytest** (≥7.4): Testing framework
- **pytest-cov** (≥4.1): Coverage reports
- **Hypothesis** (≥6.88): Property-based testing

### Interactive Analysis
- **Jupyter** (≥1.0): Notebook environment
- **IPyKernel** (≥6.25): Jupyter kernel
- **IPyWidgets** (≥8.1): Interactive widgets

---

## Performance Notes

- **Balance Optimization**: O(iterations × systems × players) - typically <1s for 100 iterations
- **Spatial Statistics**: O(n²) for weight matrix construction, then O(n) for metrics
- **Visualization**: Matplotlib rendering scales well up to ~100 hexes

For large maps (>100 systems), consider:
- Reducing max pathfinding distance
- Using sparse weight matrices
- Enabling Numba JIT compilation (future enhancement)

---

## Contributing

This is a research and analysis tool. Contributions welcome for:

- Additional spatial metrics (betweenness centrality, Voronoi analysis)
- Optimization algorithms (simulated annealing, genetic algorithms)
- Integration with JavaScript codebase (JSON import/export)
- Performance improvements (Numba, Cython)
- Empirical validation studies

---

## Citation

If you use this toolkit in research, please cite:

```
TI4 Map Balance Analysis Toolkit
Repository: https://github.com/KeeganW/ti4
Documentation: docs/Twilight Imperium Map Balance Research.md
```

---

## References

### Spatial Statistics
- Anselin, L. (1995). Local indicators of spatial association—LISA. *Geographical Analysis*, 27(2), 93-115.
- Getis, A., & Ord, J. K. (1992). The analysis of spatial association by use of distance statistics. *Geographical Analysis*, 24(3), 189-206.

### Fairness Metrics
- Jain, R., Chiu, D. M., & Hawe, W. R. (1984). A quantitative measure of fairness and discrimination. *DEC Research Report TR-301*.

### Hexagonal Grids
- Red Blob Games: [Hexagonal Grids](https://www.redblobgames.com/grids/hexagons/)

---

## License

Same as parent repository (see root LICENSE file).

---

## Support

For questions or issues:
1. Check the [Jupyter notebooks](./notebooks/) for examples
2. Review the research documentation in [../docs/](../docs/)
3. Open an issue on GitHub

---

**Happy analyzing!** 🎲🔬

# TI4 Map Optimization: Academic Benchmarking Approach

## Overview

This repository implements a rigorous academic framework for comparing multi-objective optimization algorithms on hexagonal game map generation.

**NO GAMEPLAY VALIDATION** - This is a pure computational experiment comparing optimization algorithms.

## Research Questions

1. **RQ1**: Does NSGA-II converge faster than hill-climbing?
2. **RQ2**: Does NSGA-II produce superior Pareto fronts?
3. **RQ3**: Does unconstrained action space (red tile swapping) expand the Pareto front?
4. **RQ4**: What trade-offs exist between balance gap and spatial distribution?
5. **RQ5**: Which algorithm is most computationally efficient?

## Three Algorithms

| Algorithm | Description | Action Space | Objectives |
|-----------|-------------|--------------|------------|
| **G1** | Hill-climbing baseline | Blue tiles only | Minimize balance_gap |
| **G2** | NSGA-II constrained | Blue tiles only | [balance_gap, \|Moran's I\|, 1-Jain's] |
| **G3** | NSGA-II unconstrained | All tiles (blue+red) | [balance_gap, \|Moran's I\|, 1-Jain's] |

## Dependent Variables (Gold-Standard Quality Indicators)

### Primary Metrics
- **Hypervolume (HV)**: Volume dominated by Pareto front (higher = better)
- **IGD+**: Distance to reference Pareto front (lower = better)
- **Spacing**: Uniformity of solutions along front (lower = better)

### Secondary Metrics
- Iterations to 95% final HV (convergence speed)
- Function evaluations to convergence
- Best balance gap, best Jain's Index
- Knee point objectives (best compromise)

### Efficiency Metrics
- Wall-clock time
- Function evaluations
- Swaps accepted

## Quick Start

### 1. Install Dependencies

```bash
cd ti4-analysis
pip install -r requirements.txt
```

### 2. Run Academic Benchmark

Open `notebooks/08_academic_benchmark_study.ipynb` in Jupyter and execute all cells.

**Configuration options:**
- `n_trials=30` - Number of independent runs per algorithm (reduce to 3 for testing)
- `iterations=500` - Optimization iterations/generations
- `population_size=100` - NSGA-II population size

**Estimated runtime:** 1-2 hours for full 30 trials × 3 algorithms

### 3. Analyze Results

The notebook automatically:
- Runs 90 optimization trials (30 per algorithm)
- Calculates all quality indicators
- Performs statistical tests (Wilcoxon with Bonferroni correction)
- Generates publication-ready figures
- Exports LaTeX tables

**Output directory:** `results/academic_benchmark/`

## File Structure

```
ti4-analysis/
├── src/ti4_analysis/
│   ├── algorithms/
│   │   ├── nsga2_optimizer.py     (PyMOO NSGA-II implementation)
│   │   ├── balance_engine.py      (Hill-climbing baseline)
│   │   └── spatial_optimizer.py   (Legacy multi-objective)
│   ├── benchmarking/              (NEW: Academic framework)
│   │   ├── quality_indicators.py  (HV, IGD+, Spacing)
│   │   ├── statistical_tests.py   (Wilcoxon, Friedman, effect sizes)
│   │   └── run_experiment.py      (Automated experiment runner)
│   ├── spatial_stats/
│   │   └── spatial_metrics.py     (Moran's I, Jain's Index)
│   └── visualization/
│       └── map_viz.py             (Rendering utilities)
├── notebooks/
│   └── 08_academic_benchmark_study.ipynb  (PRIMARY WORKFLOW)
└── tests/

```

## Key Python Packages

- **pymoo**: Multi-objective optimization (NSGA-II)
- **scipy**: Statistical tests (Wilcoxon, Friedman)
- **networkx**: Graph algorithms (betweenness centrality)
- **pysal/esda**: Spatial statistics (Moran's I)
- **matplotlib/seaborn**: Visualization

## What This Proves (Academically)

### ✅ You CAN claim:
- "We compare three optimization algorithms using gold-standard quality indicators"
- "30 independent trials provide robust statistical power"
- "Wilcoxon tests with Bonferroni correction control family-wise error"
- "Effect sizes (Cohen's d, A12) quantify practical significance"
- "NSGA-II efficiently optimizes multiple spatial objectives simultaneously"

### ❌ You CANNOT claim:
- ~~"AI agents played full games"~~
- ~~"This measures actual gameplay outcomes"~~
- ~~"Win-rate variance validates experiential balance"~~

## Publication Output

The notebook generates:
- **CSV**: `results.csv` with all trial data
- **Figures**: Box plots, violin plots, CDFs, scatter plots
- **LaTeX Tables**: Statistical comparisons for papers
- **JSON**: Summary statistics and test results

## References

- Deb et al. (2002): NSGA-II algorithm
- Blank & Deb (2020): PyMOO framework
- Ishibuchi et al. (2015): IGD+ metric
- Zitzler & Thiele (1999): Hypervolume indicator
- García et al. (2010): Non-parametric statistical tests for MOO

## License

See main repository LICENSE file.

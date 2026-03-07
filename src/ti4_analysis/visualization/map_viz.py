"""
Visualization tools for TI4 maps and balance analysis.

Provides functions for:
- Hexagonal map rendering
- Balance statistics plots
- Optimization convergence plots
- Spatial heatmaps
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
import numpy as np
import seaborn as sns
from typing import List, Dict, Optional, Tuple

from ..algorithms.hex_grid import HexCoord
from ..data.map_structures import MapSpace, MapSpaceType, Evaluator
from ..algorithms.balance_engine import TI4Map
from ..algorithms.balance_engine import HomeValue, get_home_values


# Hexagon visualization constants
HEX_RADIUS = 1.0
HEX_HEIGHT = np.sqrt(3) * HEX_RADIUS
HEX_WIDTH = 2 * HEX_RADIUS


def cube_to_pixel(coord: HexCoord, size: float = HEX_RADIUS) -> Tuple[float, float]:
    """
    Convert cube coordinates to pixel coordinates for visualization.

    Uses flat-top hexagon orientation.

    Args:
        coord: Hex coordinate
        size: Hexagon size (radius)

    Returns:
        (x, y) pixel coordinates
    """
    x = size * (3/2 * coord.x)
    y = size * (np.sqrt(3)/2 * coord.x + np.sqrt(3) * coord.y)
    return x, y


def create_hexagon_patch(coord: HexCoord, size: float = HEX_RADIUS) -> mpatches.RegularPolygon:
    """
    Create a matplotlib hexagon patch at given coordinate.

    Args:
        coord: Hex coordinate
        size: Hexagon size (radius)

    Returns:
        RegularPolygon patch
    """
    x, y = cube_to_pixel(coord, size)
    return mpatches.RegularPolygon(
        (x, y), 6, radius=size,
        orientation=0,  # Flat-top
        edgecolor='black',
        linewidth=1
    )


def plot_hex_map(
    ti4_map: 'TI4Map',
    ax: Optional[plt.Axes] = None,
    color_by: str = 'type',
    value_map: Optional[Dict[HexCoord, float]] = None,
    show_coords: bool = False,
    show_system_ids: bool = True,
    title: Optional[str] = None,
    size: float = HEX_RADIUS
) -> plt.Axes:
    """
    Plot TI4 hexagonal map.

    Args:
        ti4_map: TI4 map object
        ax: Matplotlib axes (creates new if None)
        color_by: How to color hexes ('type', 'value', 'resources')
        value_map: Optional dict mapping coords to values for coloring
        show_coords: Whether to show hex coordinates
        show_system_ids: Whether to show system IDs
        title: Optional plot title
        size: Hexagon size

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))

    patches = []
    colors = []

    # Color schemes
    type_colors = {
        MapSpaceType.OPEN: '#f0f0f0',
        MapSpaceType.SYSTEM: '#a8d5ff',
        MapSpaceType.HOME: '#ffcccc',
        MapSpaceType.CLOSED: '#333333',
        MapSpaceType.WARP: '#ffcc99'
    }

    for space in ti4_map.spaces:
        patch = create_hexagon_patch(space.coord, size)
        patches.append(patch)

        # Determine color
        if color_by == 'type':
            color = type_colors.get(space.space_type, '#ffffff')
        elif color_by == 'value' and value_map:
            val = value_map.get(space.coord, 0)
            color = val  # Will use colormap
        else:
            color = '#ffffff'

        colors.append(color)

    # Create patch collection
    if color_by == 'value' and value_map:
        collection = PatchCollection(patches, cmap='YlOrRd', edgecolors='black', linewidths=1)
        collection.set_array(np.array(colors))
        ax.add_collection(collection)
        plt.colorbar(collection, ax=ax, label='Value')
    else:
        collection = PatchCollection(patches, facecolors=colors, edgecolors='black', linewidths=1)
        ax.add_collection(collection)

    # Add text annotations
    for space in ti4_map.spaces:
        x, y = cube_to_pixel(space.coord, size)

        if show_coords:
            ax.text(x, y - 0.3, f"({space.coord.x},{space.coord.y})",
                   ha='center', va='center', fontsize=6, color='gray')

        if show_system_ids and space.system:
            ax.text(x, y, f"{space.system.id}",
                   ha='center', va='center', fontsize=10, fontweight='bold')

        # Show home markers
        if space.space_type == MapSpaceType.HOME:
            ax.scatter([x], [y], s=200, c='red', marker='*', zorder=10, alpha=0.7)

    ax.set_aspect('equal')
    ax.autoscale_view()
    ax.axis('off')

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')

    return ax


def plot_balance_convergence(
    history: List[Tuple[int, float]],
    ax: Optional[plt.Axes] = None,
    title: str = "Balance Optimization Convergence"
) -> plt.Axes:
    """
    Plot balance gap convergence over iterations.

    Args:
        history: List of (iteration, balance_gap) tuples
        ax: Matplotlib axes
        title: Plot title

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    iterations, gaps = zip(*history)

    ax.plot(iterations, gaps, linewidth=2, color='#2E86AB')
    ax.fill_between(iterations, gaps, alpha=0.3, color='#2E86AB')

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Balance Gap', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add improvement annotation
    initial_gap = gaps[0]
    final_gap = gaps[-1]
    improvement = ((initial_gap - final_gap) / initial_gap) * 100

    ax.text(0.95, 0.95,
           f'Initial: {initial_gap:.2f}\nFinal: {final_gap:.2f}\nImprovement: {improvement:.1f}%',
           transform=ax.transAxes,
           fontsize=10,
           verticalalignment='top',
           horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    return ax


def plot_balance_distribution(
    home_values: List[HomeValue],
    ax: Optional[plt.Axes] = None,
    title: str = "Player Position Value Distribution"
) -> plt.Axes:
    """
    Plot distribution of home values across players.

    Args:
        home_values: List of HomeValue objects
        ax: Matplotlib axes
        title: Plot title

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    values = [hv.value for hv in home_values]
    positions = [f"P{i+1}" for i in range(len(values))]

    # Bar plot
    bars = ax.bar(positions, values, color='#A23B72', alpha=0.7, edgecolor='black')

    # Add mean line
    mean_val = np.mean(values)
    ax.axhline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.1f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Player Position', fontsize=12)
    ax.set_ylabel('Home Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    return ax


def plot_balance_comparison(
    before_values: List[float],
    after_values: List[float],
    ax: Optional[plt.Axes] = None,
    title: str = "Balance Before vs After Optimization"
) -> plt.Axes:
    """
    Compare balance before and after optimization.

    Args:
        before_values: Home values before optimization
        after_values: Home values after optimization
        ax: Matplotlib axes
        title: Plot title

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(before_values))
    width = 0.35

    bars1 = ax.bar(x - width/2, before_values, width, label='Before',
                   color='#E63946', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, after_values, width, label='After',
                   color='#06A77D', alpha=0.7, edgecolor='black')

    ax.set_xlabel('Player Position', fontsize=12)
    ax.set_ylabel('Home Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'P{i+1}' for i in range(len(before_values))])
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    # Add gap annotations
    before_gap = max(before_values) - min(before_values)
    after_gap = max(after_values) - min(after_values)

    ax.text(0.02, 0.98,
           f'Gap Before: {before_gap:.2f}\nGap After: {after_gap:.2f}\nReduction: {before_gap - after_gap:.2f}',
           transform=ax.transAxes,
           fontsize=10,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    return ax


def plot_value_heatmap(
    ti4_map: 'TI4Map',
    evaluator: Evaluator,
    ax: Optional[plt.Axes] = None,
    title: str = "System Value Heatmap"
) -> plt.Axes:
    """
    Create heatmap of system values across the map.

    Args:
        ti4_map: TI4 map object
        evaluator: Evaluator for system values
        ax: Matplotlib axes
        title: Plot title

    Returns:
        Matplotlib axes
    """
    # Calculate values for each space
    value_map = {}
    for space in ti4_map.spaces:
        if space.system:
            value_map[space.coord] = space.system.evaluate(evaluator)
        else:
            value_map[space.coord] = 0

    return plot_hex_map(
        ti4_map,
        ax=ax,
        color_by='value',
        value_map=value_map,
        show_system_ids=True,
        title=title
    )


def create_balance_report(
    ti4_map: 'TI4Map',
    evaluator: Evaluator,
    history: Optional[List[Tuple[int, float]]] = None,
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """
    Create comprehensive balance analysis report.

    Args:
        ti4_map: TI4 map object
        evaluator: Evaluator parameters
        history: Optional optimization history
        figsize: Figure size

    Returns:
        Matplotlib figure with multiple subplots
    """
    fig = plt.figure(figsize=figsize)

    # Layout: 2x2 grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Plot 1: Hex map
    ax1 = fig.add_subplot(gs[0, 0])
    plot_hex_map(ti4_map, ax=ax1, title="Map Layout")

    # Plot 2: Value heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    plot_value_heatmap(ti4_map, evaluator, ax=ax2)

    # Plot 3: Balance distribution
    ax3 = fig.add_subplot(gs[1, 0])
    home_values = get_home_values(ti4_map, evaluator)
    plot_balance_distribution(home_values, ax=ax3)

    # Plot 4: Convergence (if history provided)
    if history:
        ax4 = fig.add_subplot(gs[1, 1])
        plot_balance_convergence(history, ax=ax4)
    else:
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.text(0.5, 0.5, 'No optimization history',
                ha='center', va='center', fontsize=14)
        ax4.axis('off')

    fig.suptitle('TI4 Map Balance Analysis', fontsize=16, fontweight='bold', y=0.98)

    return fig


def plot_fairness_metrics(
    analysis_results: Dict,
    ax: Optional[plt.Axes] = None,
    title: str = "Balance Fairness Metrics"
) -> plt.Axes:
    """
    Visualize fairness metrics from balance analysis.

    Args:
        analysis_results: Results from analyze_balance()
        ax: Matplotlib axes
        title: Plot title

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    metrics = {
        'Balance Gap': analysis_results['balance_gap'],
        'Std Dev': analysis_results['std'],
        'Mean': analysis_results['mean'],
        'Fairness Index': analysis_results['fairness_index'] * 100  # Scale to 0-100
    }

    y_pos = np.arange(len(metrics))
    values = list(metrics.values())
    labels = list(metrics.keys())

    bars = ax.barh(y_pos, values, color=['#E63946', '#F77F00', '#06A77D', '#118AB2'])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val, i, f' {val:.2f}', va='center', fontweight='bold')

    ax.grid(True, axis='x', alpha=0.3)

    return ax

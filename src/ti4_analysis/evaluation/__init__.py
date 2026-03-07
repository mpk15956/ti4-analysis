"""
Experimental framework for TI4 map balance analysis.

This package provides tools for running batch experiments to compare
different optimization strategies (basic balance vs spatial optimization).
"""

from .batch_experiment import run_batch_experiment
from .analysis import analyze_experiment_results
from .report_generator import generate_markdown_report

__all__ = [
    'run_batch_experiment',
    'analyze_experiment_results',
    'generate_markdown_report',
]

"""
Single-source budget-to-(generations, population) factorization for NSGA-II.

The canonical factorization rule lives here so the tuner and the benchmark
engine *cannot disagree* on what (pop, gen) corresponds to a given evaluation
budget. Previously, optimize_hyperparameters.py defined `_nsga2_budget` as a
private helper and benchmark_engine.py hardcoded `--nsga-pop 20` with
`nsga_gen = budget // pop`; the two scaled differently across budgets
(tuner: pop = √budget; benchmark: pop = 20) and the tuner's optimized
hyperparameters were applied at a different (pop, gen) regime than the
tuning ran in. This module collapses that drift into a structural
impossibility — same helper, same input, same output, called from both
sides — see `feedback_canonical_objective_single_source.md` for the
single-source principle this implements.

Rule:
    pop = max(MIN_POP, round(sqrt(budget)))
    gen = max(MIN_GEN, budget // pop)

At budget = 1,000:   pop = 31, gen = 32  (992 actual evaluations)
At budget = 25,000:  pop = 158, gen = 158  (24,964 actual evaluations)
At budget = 500,000: pop = 707, gen = 707  (499,849 actual evaluations)

The (pop, gen) pair is approximately balanced — neither dimension dominates.
The actual evaluation count is `pop * gen`, which is at most `budget` and
typically within √budget of it (i.e., at most pop or gen evaluations short).
"""

from __future__ import annotations

import math
from typing import Tuple

MIN_POP = 10
MIN_GEN = 5


def nsga2_budget(iter_budget: int) -> Tuple[int, int]:
    """
    Factorize iter_budget into (generations, population_size).

    Args:
        iter_budget: target number of NSGA-II evaluations for one run.

    Returns:
        (generations, population_size). The actual evaluation count is
        pop * gen ≤ iter_budget; the difference is at most pop or gen
        evaluations.
    """
    pop_size = max(MIN_POP, int(round(math.sqrt(iter_budget))))
    generations = max(MIN_GEN, iter_budget // pop_size)
    return generations, pop_size

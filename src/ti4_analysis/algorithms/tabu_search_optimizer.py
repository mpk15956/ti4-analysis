"""
Tabu Search optimizer for TI4 map balance.

Classic full-neighbourhood Tabu Search: each iteration evaluates all C(S,2)
possible 2-swaps among S swappable positions, selects the best non-tabu move
(or the best tabu move when the aspiration criterion is satisfied), and
unconditionally applies it.

The key distinction from Simulated Annealing:
    SA escapes local optima *stochastically* (Metropolis criterion);
    TS escapes them *deterministically* via short-term memory (tabu list).

Aspiration criterion: a tabu move is accepted if it produces a composite score
strictly better than the best-ever observed, overriding the tabu restriction.

Evaluation budget accounting: each candidate swap costs one evaluation. A single
TS iteration costs C(S,2) evaluations, making budget comparisons with SA/HC
evaluation-for-evaluation fair. At low budgets, TS completes fewer sequential
decisions (iterations) than SA — the depth-vs-breadth tradeoff mirrors
the NSGA-II population overhead argument.
"""

import random
from itertools import combinations
from typing import Tuple, List, Optional, Dict

from .balance_engine import TI4Map
from .map_topology import MapTopology
from .fast_map_state import FastMapState
from ..data.map_structures import Evaluator
from .spatial_optimizer import MultiObjectiveScore, evaluate_map_multiobjective


def improve_balance_tabu(
    ti4_map: TI4Map,
    evaluator: Evaluator,
    max_evaluations: int = 1000,
    weights: Optional[Dict[str, float]] = None,
    random_seed: Optional[int] = None,
    verbose: bool = True,
    tabu_tenure: Optional[int] = None,
) -> Tuple[MultiObjectiveScore, List[Tuple[int, MultiObjectiveScore]]]:
    """
    Improve map balance using Tabu Search over the composite fitness.

    Each iteration scans the full 2-swap neighbourhood, applies the best
    non-tabu move (or aspirational tabu move), and adds the reverse swap
    to the tabu list for `tabu_tenure` iterations.

    Args:
        ti4_map: Map to optimize (modified in-place)
        evaluator: Balance evaluator
        max_evaluations: Total evaluation budget (comparable to SA iterations)
        weights: Objective weights (morans_i, jains_index, lisa_penalty; sum to 1.0)
        random_seed: Random seed for reproducibility
        verbose: Print progress messages
        tabu_tenure: Number of iterations a swap remains forbidden.
            Default: ceil(sqrt(S)), a standard heuristic (Glover 1989).

    Returns:
        Tuple of (best_score, history)
        history is a list of (evaluation_count, score) tuples
    """
    if random_seed is not None:
        random.seed(random_seed)

    topology = MapTopology.from_ti4_map(ti4_map, evaluator)
    fast_state = FastMapState.from_ti4_map(topology, ti4_map, evaluator)
    swappable_spaces = [ti4_map.spaces[i] for i in topology.swappable_indices]
    S = len(swappable_spaces)

    if S < 2:
        raise ValueError("Not enough swappable spaces to optimize")

    swap_pairs = list(combinations(range(S), 2))
    neighborhood_size = len(swap_pairs)

    if tabu_tenure is None:
        import math as _math
        tabu_tenure = max(3, int(_math.ceil(S ** 0.5)))

    current_score = evaluate_map_multiobjective(ti4_map, evaluator, weights, fast_state)
    best_score = current_score
    best_composite = best_score.composite_score()
    history: List[Tuple[int, MultiObjectiveScore]] = [(0, current_score)]

    # (min_idx, max_idx) → iteration number when the tabu expires
    tabu_dict: Dict[Tuple[int, int], int] = {}

    total_evals = 0
    iteration = 0

    if verbose:
        print(
            f"TS: S={S}, neighborhood={neighborhood_size}, "
            f"tenure={tabu_tenure}, budget={max_evaluations}"
        )
        print(f"Initial: {current_score}")

    while total_evals + neighborhood_size <= max_evaluations:
        iteration += 1

        best_move: Optional[Tuple[int, int]] = None
        best_move_score: Optional[MultiObjectiveScore] = None
        best_move_composite = float('inf')

        best_tabu_move: Optional[Tuple[int, int]] = None
        best_tabu_score: Optional[MultiObjectiveScore] = None
        best_tabu_composite = float('inf')

        for s_i, s_j in swap_pairs:
            fast_state.swap(s_i, s_j)
            candidate = evaluate_map_multiobjective(
                ti4_map, evaluator, weights, fast_state
            )
            c_composite = candidate.composite_score()
            total_evals += 1

            is_tabu = (s_i, s_j) in tabu_dict and tabu_dict[(s_i, s_j)] > iteration

            if is_tabu:
                if c_composite < best_tabu_composite:
                    best_tabu_composite = c_composite
                    best_tabu_move = (s_i, s_j)
                    best_tabu_score = candidate
            else:
                if c_composite < best_move_composite:
                    best_move_composite = c_composite
                    best_move = (s_i, s_j)
                    best_move_score = candidate

            fast_state.swap(s_i, s_j)  # revert trial

        # Aspiration: allow tabu move if it beats global best
        chosen_move = best_move
        chosen_score = best_move_score

        if best_tabu_move is not None and best_tabu_composite < best_composite:
            chosen_move = best_tabu_move
            chosen_score = best_tabu_score

        if chosen_move is None:
            # All non-tabu moves exhausted and no aspiration — fallback to best tabu
            if best_tabu_move is not None:
                chosen_move = best_tabu_move
                chosen_score = best_tabu_score
            else:
                break

        # Apply chosen move to both fast_state and ti4_map
        s_i, s_j = chosen_move
        fast_state.swap(s_i, s_j)
        swappable_spaces[s_i].system, swappable_spaces[s_j].system = (
            swappable_spaces[s_j].system, swappable_spaces[s_i].system
        )
        current_score = chosen_score

        tabu_dict[(s_i, s_j)] = iteration + tabu_tenure

        if current_score.composite_score() < best_composite:
            best_score = current_score
            best_composite = best_score.composite_score()
            if verbose:
                print(
                    f"  Iter {iteration} (evals={total_evals}): "
                    f"NEW BEST {best_score}"
                )

        history.append((total_evals, current_score))

    # Use remaining budget for a partial neighbourhood scan
    remaining = max_evaluations - total_evals
    if remaining > 0 and len(swap_pairs) > 0:
        iteration += 1
        partial_pairs = swap_pairs.copy()
        random.shuffle(partial_pairs)
        partial_pairs = partial_pairs[:remaining]

        best_move = None
        best_move_score = None
        best_move_composite = float('inf')
        best_tabu_move = None
        best_tabu_score = None
        best_tabu_composite = float('inf')

        for s_i, s_j in partial_pairs:
            fast_state.swap(s_i, s_j)
            candidate = evaluate_map_multiobjective(
                ti4_map, evaluator, weights, fast_state
            )
            c_composite = candidate.composite_score()
            total_evals += 1

            is_tabu = (s_i, s_j) in tabu_dict and tabu_dict[(s_i, s_j)] > iteration

            if is_tabu:
                if c_composite < best_tabu_composite:
                    best_tabu_composite = c_composite
                    best_tabu_move = (s_i, s_j)
                    best_tabu_score = candidate
            else:
                if c_composite < best_move_composite:
                    best_move_composite = c_composite
                    best_move = (s_i, s_j)
                    best_move_score = candidate

            fast_state.swap(s_i, s_j)

        chosen_move = best_move
        chosen_score = best_move_score
        if best_tabu_move is not None and best_tabu_composite < best_composite:
            chosen_move = best_tabu_move
            chosen_score = best_tabu_score
        if chosen_move is None and best_tabu_move is not None:
            chosen_move = best_tabu_move
            chosen_score = best_tabu_score

        if chosen_move is not None:
            s_i, s_j = chosen_move
            fast_state.swap(s_i, s_j)
            swappable_spaces[s_i].system, swappable_spaces[s_j].system = (
                swappable_spaces[s_j].system, swappable_spaces[s_i].system
            )
            current_score = chosen_score
            tabu_dict[(s_i, s_j)] = iteration + tabu_tenure
            if current_score.composite_score() < best_composite:
                best_score = current_score
                best_composite = best_score.composite_score()
            history.append((total_evals, current_score))

    if verbose:
        print(f"TS complete: {iteration} iterations, {total_evals} evaluations")
        print(f"Final best: {best_score}")

    history.append((total_evals, best_score))
    return best_score, history

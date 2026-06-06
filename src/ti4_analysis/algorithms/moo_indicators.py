"""
Multi-objective quality-indicator primitives — single source of truth.

`scripts/quality_indicators.py` (Track B) and `scripts/cross_method_igd.py`
(cross-method IGD) previously each carried their own pure-Python `igd_plus`
double-loop and built reference sets as the *raw* union of archive points (no
non-dominated filtering). On the full 100-seed dataset that produced a
204k-point reference front and an O(|ref|x|front|) interpreted loop that blew
past the Phase 6 walltime — and, more importantly, an IGD+ reference set that
violates the cited construction (Ishibuchi et al. 2015: the reference set must
approximate the true Pareto front, i.e. the *non-dominated* subset of the union
of observed solutions).

These helpers fix both: `nondominated_filter` restores the correct reference
construction, and the vectorized `igd_plus` removes the interpreted loop. Both
operate directly on the transformed minimization-space triples
`(1 - J, moran-hinge, LSAP)` that the archive points are already stored in —
the same space `MultiObjectiveScore.dominates()` compares (all-<=, any-<,
minimization), so the filter is consistent with the canonical dominance
predicate by construction.
"""
from __future__ import annotations

import numpy as np


def nondominated_filter(points: np.ndarray) -> np.ndarray:
    """
    Return the non-dominated (Pareto-minimal) subset of `points`.

    Minimization on every column. Point j dominates point i iff
    all(points[j] <= points[i]) and any(points[j] < points[i]) — identical to
    `MultiObjectiveScore.dominates()` operating in the stored objective space.
    The inner comparison is vectorized over all candidate dominators; duplicate
    rows are mutually non-dominated and are all kept (matching the naive
    O(n^2) predicate).
    """
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n <= 1:
        return pts
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        # j strictly dominates i: every objective <=, at least one <.
        leq = np.all(pts <= pts[i], axis=1)
        lt = np.any(pts < pts[i], axis=1)
        dominators = leq & lt
        dominators[i] = False  # a point never dominates itself
        if dominators.any():
            keep[i] = False
    return pts[keep]


def igd_plus(front: np.ndarray, reference_front: np.ndarray,
             chunk: int = 2048) -> float:
    """
    IGD+ (Ishibuchi et al. 2015), vectorized. Numerically equivalent to the
    reference double loop:

        total = 0
        for r in reference_front:
            total += min_p sqrt(sum( max(p - r, 0)^2 ))
        return total / len(reference_front)

    For each reference point r, the modified distance to a front point p only
    penalizes objectives where p is worse than r (p > r, minimization). We
    chunk over the reference axis to bound the (chunk, |front|, d) broadcast.
    """
    front = np.asarray(front, dtype=float)
    ref = np.asarray(reference_front, dtype=float)
    if len(front) == 0 or len(ref) == 0:
        return float("inf")

    total = 0.0
    for start in range(0, len(ref), chunk):
        rchunk = ref[start:start + chunk]                  # (c, d)
        diff = front[None, :, :] - rchunk[:, None, :]      # (c, |front|, d)
        np.maximum(diff, 0.0, out=diff)
        d = np.sqrt(np.einsum("cpd,cpd->cp", diff, diff))  # (c, |front|)
        total += float(d.min(axis=1).sum())
    return total / len(ref)

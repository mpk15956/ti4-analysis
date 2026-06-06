"""
Equivalence tests for the vectorized MOO indicator helpers.

These pin the vectorized `igd_plus` and `nondominated_filter` to the *original*
reference implementations they replaced, so the Phase 6 refactor is provably
semantics-preserving (the performance fix changes runtime, not numbers).
"""
import numpy as np

from ti4_analysis.algorithms.moo_indicators import igd_plus, nondominated_filter


def _igd_plus_reference(front: np.ndarray, reference_front: np.ndarray) -> float:
    """Verbatim re-statement of the original quality_indicators.py double loop."""
    if len(front) == 0 or len(reference_front) == 0:
        return float("inf")
    total = 0.0
    for r in reference_front:
        min_dist = float("inf")
        for p in front:
            diff = np.maximum(p - r, 0.0)
            d = float(np.sqrt(np.sum(diff ** 2)))
            if d < min_dist:
                min_dist = d
        total += min_dist
    return total / len(reference_front)


def _nondominated_reference(points: np.ndarray) -> np.ndarray:
    """Naive O(n^2) non-dominated set (minimization), same predicate as dominates()."""
    pts = np.asarray(points, dtype=float)
    keep = []
    for i in range(len(pts)):
        dominated = False
        for j in range(len(pts)):
            if j == i:
                continue
            if np.all(pts[j] <= pts[i]) and np.any(pts[j] < pts[i]):
                dominated = True
                break
        if not dominated:
            keep.append(pts[i])
    return np.array(keep) if keep else np.empty((0, pts.shape[1]))


def test_igd_plus_matches_reference_loop():
    rng = np.random.default_rng(0)
    for _ in range(20):
        # Points on BOTH sides of the reference so the max(.,0) clamp is exercised.
        front = rng.normal(0.0, 1.0, size=(rng.integers(1, 40), 3))
        ref = rng.normal(0.0, 1.0, size=(rng.integers(1, 120), 3))
        got = igd_plus(front, ref)
        want = _igd_plus_reference(front, ref)
        assert np.isclose(got, want, rtol=1e-12, atol=1e-12), (got, want)


def test_igd_plus_chunk_invariance():
    rng = np.random.default_rng(1)
    front = rng.normal(size=(25, 3))
    ref = rng.normal(size=(500, 3))
    assert np.isclose(igd_plus(front, ref, chunk=1),
                      igd_plus(front, ref, chunk=10_000), rtol=1e-12, atol=1e-12)


def test_igd_plus_empty():
    assert igd_plus(np.empty((0, 3)), np.ones((5, 3))) == float("inf")
    assert igd_plus(np.ones((5, 3)), np.empty((0, 3))) == float("inf")


def test_nondominated_filter_matches_reference():
    rng = np.random.default_rng(2)
    for _ in range(20):
        pts = rng.integers(0, 5, size=(rng.integers(1, 60), 3)).astype(float)
        got = nondominated_filter(pts)
        want = _nondominated_reference(pts)
        got_set = {tuple(r) for r in got}
        want_set = {tuple(r) for r in want}
        assert got_set == want_set


def test_nondominated_filter_keeps_duplicates_and_shrinks():
    # A clearly dominated cloud collapses to its non-dominated frontier.
    pts = np.array([[0.0, 0.0, 0.0],   # dominates everything below
                    [1.0, 1.0, 1.0],
                    [2.0, 0.5, 3.0],
                    [0.0, 0.0, 0.0]])  # duplicate of the optimum — both kept
    nd = nondominated_filter(pts)
    nd_set = {tuple(r) for r in nd}
    assert nd_set == {(0.0, 0.0, 0.0)}
    assert len(nd) == 2  # duplicates of a non-dominated point are retained

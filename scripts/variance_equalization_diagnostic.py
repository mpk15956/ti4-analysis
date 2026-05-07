#!/usr/bin/env python3
"""
Variance equalization diagnostic for the composite objective weights.

Addresses the reviewer concern (Methodology §3.7) that if the empirical
variance of LSAP across the map space is substantially larger than the
variance of JFI or Moran's I, the largest-σ term will silently dominate
the Markov chain acceptance dynamics regardless of the nominal weight
ratio.

This script computes the standard deviation of the three normalized
objective terms (via MultiObjectiveScore.raw_objective_terms, which
applies the same regime-aware degenerate-graph guard the optimizer uses):
  - hinge_morans_i  = max(0, I − E[I])      (E[I] from morans_i_null)
  - jfi_gap         = 1 − J_min
  - lisa_norm       = LSAP / (|G| × (|G|−1))

at two points:
  1. Generation 0: across N_RANDOM unoptimized maps (random configurations).
  2. Convergence: across N_SA SA-optimized solutions, where SA runs under
     the SAME --weights vector this script reports against. (Single-source
     weight flow: the weight vector parameterizes both the SA composite
     SA optimizes AND the share verdict reported below.)

Reports the empirical σ at each stage, the weighted-share verdict under
the active weight vector, and the σ shift Gen-0 → convergence.

Default weights are 1:1:1 (matches Methodology §3.1's nominal weighting).
The historical 5:5:3 formulation is supported as a sensitivity probe but
requires --sensitivity-probe to acknowledge the manuscript-vs-CLI mismatch.

Usage:
    # Manuscript-aligned default:
    python scripts/variance_equalization_diagnostic.py \\
        [--n-random 1000] [--n-sa 50] [--sa-budget 2000]

    # Historical 5:5:3 sensitivity probe:
    python scripts/variance_equalization_diagnostic.py \\
        --weights 5,5,3 --sensitivity-probe
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Variance equalization diagnostic for composite objective weights"
    )
    p.add_argument("--n-random",   type=int,   default=1000,
                   help="Number of unoptimized maps to sample (Gen-0 proxy; default 1000)")
    p.add_argument("--n-sa",       type=int,   default=50,
                   help="Number of SA runs for convergence sample (default 50)")
    p.add_argument("--sa-budget",  type=int,   default=2000,
                   help="SA evaluation budget per run (default 2000)")
    p.add_argument("--players",    type=int,   default=6,
                   help="Number of players (default 6)")
    p.add_argument("--base-seed",  type=int,   default=0,
                   help="First random seed (default 0)")
    p.add_argument("--output-dir", type=str,   default="output",
                   help="Root output directory (default output/)")
    p.add_argument("--weights",    type=str,   default="1,1,1",
                   help=("Composite weights for the spatial-fairness terms, "
                         "comma-separated as morans_i,jains_index,lisa_penalty. "
                         "Default '1,1,1' matches Methodology §3.1's nominal "
                         "weighting; pass '5,5,3' (with --sensitivity-probe) "
                         "to reproduce the legacy diagnostic. The same weight "
                         "vector parameterizes both the embedded SA composite "
                         "AND the share verdict reported here, eliminating the "
                         "two-source mismatch in earlier versions of this script."))
    p.add_argument("--sensitivity-probe", action="store_true",
                   help=("Required when --weights is not the manuscript default "
                         "'1,1,1'. Without this flag, non-default weights are "
                         "rejected to prevent accidental authoritative-looking "
                         "output that doesn't match the live nominal weighting."))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Weight handling — single source, flows to both SA composite and reporting
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = (1.0, 1.0, 1.0)

# Mapping from the diagnostic's term names to MultiObjectiveScore weight keys.
# raw_objective_terms returns (hinge_morans_i, jfi_gap, lisa_norm); the
# composite SA optimizes against has weights keyed (morans_i, jains_index,
# lisa_penalty). The two name conventions correspond term-for-term.
TERM_NAMES = ("hinge_morans_i", "jfi_gap", "lisa_norm")
SCORE_WEIGHT_KEYS = ("morans_i", "jains_index", "lisa_penalty")


def parse_weights(raw: str, sensitivity_probe: bool) -> Tuple[Dict[str, float], Tuple[float, float, float]]:
    """
    Parse --weights into both the diagnostic-side normalized share dict and
    the MultiObjectiveScore-side weight dict. Both views derive from the
    same input; downstream consumers cannot drift.

    Returns (diag_weights, score_weights_tuple) where:
      diag_weights      : keyed by TERM_NAMES, sums to 1.0
      score_weights_raw : 3-tuple in (morans_i, jains_index, lisa_penalty) order
    """
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 3:
        raise SystemExit(f"--weights must be 3 comma-separated numbers; got: {raw!r}")
    try:
        nums = tuple(float(p) for p in parts)
    except ValueError as e:
        raise SystemExit(f"--weights values must be numeric: {e}")
    if any(n < 0 for n in nums) or sum(nums) <= 0:
        raise SystemExit(f"--weights must be non-negative with positive sum; got: {nums}")

    if nums != DEFAULT_WEIGHTS and not sensitivity_probe:
        raise SystemExit(
            f"--weights {raw} differs from the manuscript default 1,1,1; "
            f"pass --sensitivity-probe to acknowledge that the result is a "
            f"sensitivity probe, not the live diagnostic. (This guard exists "
            f"because variance shares are weight-dependent by construction; "
            f"reporting non-default-weighted shares without explicit "
            f"acknowledgment risks producing authoritative-looking output "
            f"that doesn't match Methodology §3.1.)"
        )

    total = sum(nums)
    normalized = tuple(n / total for n in nums)
    diag_weights = dict(zip(TERM_NAMES, normalized))
    return diag_weights, nums  # raw nums passed to SA so it can normalize itself


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_terms(score) -> dict:
    """
    Extract the three normalized objective terms from a MultiObjectiveScore.

    Delegates to score.raw_objective_terms() which applies the same
    regime-aware degenerate-graph guard as composite_score (returns 0.0
    spatial terms on n_spatial < SPATIAL_DEGENERATE_THRESHOLD). Single
    source: this script does not duplicate the term arithmetic.
    """
    morans_hinge, jfi_gap, lisa_norm = score.raw_objective_terms()
    return {
        "hinge_morans_i": morans_hinge,
        "jfi_gap":        jfi_gap,
        "lisa_norm":      lisa_norm,
    }


def _sigma_table(records: list[dict]) -> dict:
    """Compute mean and std for each term across a list of records."""
    result = {}
    for key in TERM_NAMES:
        vals = np.array([r[key] for r in records])
        result[key] = {
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        }
    return result


def _dominance_verdict(sigma: dict, weights: dict) -> str:
    """
    Verdict on whether one term's weighted σ dominates acceptance dynamics
    under the active weight vector. Threshold: dominant share > 60%.
    """
    weighted = {k: sigma[k]["std"] * weights[k] for k in sigma}
    total = sum(weighted.values()) or 1.0
    shares = {k: weighted[k] / total for k in weighted}

    dominant = max(shares, key=shares.get)
    dominant_share = shares[dominant]
    weight_str = ":".join(f"{weights[k]:.3f}" for k in TERM_NAMES)

    if dominant_share > 0.60:
        return (
            f"WARNING — '{dominant}' accounts for {dominant_share:.1%} of "
            f"weighted variance under active weights ({weight_str}). This "
            f"term may dominate Markov chain acceptance dynamics under the "
            f"current weighting."
        )
    else:
        return (
            f"OK — no single term dominates under active weights "
            f"({weight_str}); largest share: '{dominant}' at "
            f"{dominant_share:.1%}."
        )


def _print_table(label: str, sigma: dict, weights: dict) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  {'Term':<20s}  {'Mean':>8s}  {'Std':>8s}  {'W':>7s}  {'w×σ':>8s}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*8}")
    for key in TERM_NAMES:
        w = weights[key]
        s = sigma[key]["std"]
        print(f"  {key:<20s}  {sigma[key]['mean']:>8.5f}  {s:>8.5f}  "
              f"{w:>7.4f}  {w*s:>8.5f}")
    print(f"\n  Verdict: {_dominance_verdict(sigma, weights)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    diag_weights, raw_weight_tuple = parse_weights(args.weights, args.sensitivity_probe)
    score_weights_dict = dict(zip(SCORE_WEIGHT_KEYS,
                                  tuple(n / sum(raw_weight_tuple) for n in raw_weight_tuple)))

    from ti4_analysis.algorithms.map_generator import generate_random_map
    from ti4_analysis.algorithms.map_topology import MapTopology
    from ti4_analysis.algorithms.fast_map_state import FastMapState
    from ti4_analysis.algorithms.spatial_optimizer import (
        evaluate_map_multiobjective,
        improve_balance_spatial,
    )
    from ti4_analysis.evaluation.batch_experiment import create_joebrew_evaluator

    evaluator = create_joebrew_evaluator()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    weights_label = ":".join(f"{w:g}" for w in raw_weight_tuple)

    print(f"Variance Equalization Diagnostic — {timestamp}")
    print(f"  Gen-0 sample size : {args.n_random} random maps")
    print(f"  Convergence sample: {args.n_sa} SA runs × {args.sa_budget} evals")
    print(f"  Players           : {args.players}")
    print(f"  Base seed         : {args.base_seed}")
    print(f"  Weights (raw)     : {weights_label}  "
          f"({'manuscript default' if raw_weight_tuple == DEFAULT_WEIGHTS else 'sensitivity probe'})")
    print(f"  Weights (norm.)   : "
          f"morans_i={score_weights_dict['morans_i']:.4f}, "
          f"jains_index={score_weights_dict['jains_index']:.4f}, "
          f"lisa_penalty={score_weights_dict['lisa_penalty']:.4f}")

    # ── Generation-0 sample (random unoptimized configurations) ──────────────
    print(f"\nSampling {args.n_random} random configurations (Gen-0)...")
    gen0_records: list[dict] = []
    for i in range(args.n_random):
        seed = args.base_seed + i
        try:
            ti4_map = generate_random_map(player_count=args.players, random_seed=seed)
            topology = MapTopology.from_ti4_map(ti4_map, evaluator)
            fast_state = FastMapState.from_ti4_map(topology, ti4_map, evaluator)
            score = evaluate_map_multiobjective(ti4_map, evaluator, fast_state=fast_state)
            rec = _extract_terms(score)
            rec["seed"] = seed
            gen0_records.append(rec)
        except Exception as exc:
            print(f"  [seed={seed}] ERROR: {exc}", file=sys.stderr)
        if (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{args.n_random} done")

    gen0_sigma = _sigma_table(gen0_records)
    _print_table(f"GEN-0 (random configurations, n={len(gen0_records)})",
                 gen0_sigma, diag_weights)

    # ── Convergence sample (SA-optimized solutions) ────────────────────────
    # SA optimizes against the SAME weight vector this script reports against,
    # eliminating the parallel-source bug in earlier versions of this script
    # where SA used optimizer defaults while reporting used a separate
    # NOMINAL_WEIGHTS constant.
    print(f"\nRunning {args.n_sa} SA optimizations (convergence sample)"
          f" under weights {weights_label}...")
    conv_records: list[dict] = []
    for i in range(args.n_sa):
        seed = args.base_seed + 10_000 + i   # distinct from Gen-0 seeds
        try:
            ti4_map = generate_random_map(player_count=args.players, random_seed=seed)
            t0 = time.time()
            best_score, _, _ = improve_balance_spatial(
                ti4_map,
                evaluator,
                iterations=args.sa_budget,
                weights=score_weights_dict,
            )
            elapsed = time.time() - t0
            rec = _extract_terms(best_score)
            rec["seed"] = seed
            rec["elapsed_sec"] = round(elapsed, 2)
            conv_records.append(rec)
        except Exception as exc:
            print(f"  [seed={seed}] ERROR: {exc}", file=sys.stderr)
        if (i + 1) % 10 == 0:
            print(f"  ... {i + 1}/{args.n_sa} done")

    conv_sigma = _sigma_table(conv_records)
    _print_table(f"CONVERGENCE (SA-optimized solutions, n={len(conv_records)})",
                 conv_sigma, diag_weights)

    # ── σ shift summary ────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  σ SHIFT (Gen-0 → Convergence)")
    print(f"{'=' * 60}")
    print(f"  {'Term':<20s}  {'σ Gen-0':>10s}  {'σ Conv.':>10s}  {'Ratio':>8s}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*8}")
    for key in ("hinge_morans_i", "jfi_gap", "lisa_norm"):
        s0 = gen0_sigma[key]["std"]
        sc = conv_sigma[key]["std"]
        ratio = sc / s0 if s0 > 0 else float("nan")
        flag = " ← shrinks" if ratio < 0.5 else (" ← inflates" if ratio > 2.0 else "")
        print(f"  {key:<20s}  {s0:>10.5f}  {sc:>10.5f}  {ratio:>8.3f}{flag}")

    # ── Write CSV ──────────────────────────────────────────────────────────
    # Each row carries the weight vector in dedicated columns so the CSV is
    # self-describing: if rows are ever concatenated across sensitivity runs
    # (default vs 5:5:3 probe vs other), the weight provenance survives the
    # merge. Same principle as the topology-summary-in-run-log idea — the
    # artifact carries its own context.
    csv_path = output_dir / f"variance_equalization_{timestamp}.csv"
    weight_cols = {
        "weight_morans_i":     score_weights_dict["morans_i"],
        "weight_jains_index":  score_weights_dict["jains_index"],
        "weight_lisa_penalty": score_weights_dict["lisa_penalty"],
        "weight_label":        weights_label,
    }
    fieldnames = (["phase", "seed", "hinge_morans_i", "jfi_gap", "lisa_norm"]
                  + list(weight_cols.keys()))
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for rec in gen0_records:
            w.writerow({**rec, "phase": "gen0", **weight_cols})
        for rec in conv_records:
            w.writerow({**rec, "phase": "convergence", **weight_cols})
    print(f"\nRaw data written to: {csv_path}")


if __name__ == "__main__":
    main()

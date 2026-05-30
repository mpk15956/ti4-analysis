"""Basin-structure diagnostic: applies the pre-registered rules from
`docs/audit/jfi_only_diagnostics/PREREGISTRATION.md` to a basin-validation
benchmark output and emits the formal verdict.

Inputs:  --results-csv PATH (the multi-algorithm jfi_only run)
         --output-dir  DIR  (where to write basin_diagnostic.txt + .png)

Outputs: basin_diagnostic.txt — the verdict, machine-pasteable into the
         manuscript or commit message.
         basin_diagnostic.png — per-algorithm distributions and joint scatter.

Note: this script is intentionally *not* parameterised on thresholds —
they are baked in from the pre-registration. If you change them here you
have un-pre-registered the analysis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BC_THRESHOLD = 0.555
DBIC_ROBUST = 10.0
DBIC_SOFT = 2.0


def sarle_bc(x: np.ndarray) -> float:
    n = len(x)
    g = stats.skew(x)
    k = stats.kurtosis(x, fisher=True)
    return (g * g + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))


def gmm_dbic(x: np.ndarray) -> tuple[float, dict]:
    """Return ΔBIC = BIC(k=1) − BIC(k=2). Positive favors k=2."""
    xv = x.reshape(-1, 1)
    gmm1 = GaussianMixture(n_components=1, random_state=0, n_init=5).fit(xv)
    gmm2 = GaussianMixture(n_components=2, random_state=0, n_init=5).fit(xv)
    means = sorted(gmm2.means_.flatten().tolist())
    sds = sorted(np.sqrt(gmm2.covariances_.flatten()).tolist())
    weights = sorted(gmm2.weights_.flatten().tolist())
    return gmm1.bic(xv) - gmm2.bic(xv), {
        "k1_bic": gmm1.bic(xv),
        "k2_bic": gmm2.bic(xv),
        "k2_means": means,
        "k2_sds": sds,
        "k2_weights": weights,
    }


def per_algo_verdict(bc: float, dbic: float) -> str:
    """Apply the pre-registered table per PREREGISTRATION.md."""
    if bc >= BC_THRESHOLD and dbic > DBIC_ROBUST:
        return "Bimodal (robust)"
    if bc >= BC_THRESHOLD and DBIC_SOFT <= dbic <= DBIC_ROBUST:
        return "Bimodal (soft)"
    if bc < BC_THRESHOLD and dbic > DBIC_ROBUST:
        return "Bimodal (soft) — moments and GMM disagree"
    if bc < BC_THRESHOLD and DBIC_SOFT <= dbic <= DBIC_ROBUST:
        return "Inconclusive"
    return "Unimodal"


def joint_verdict(verdicts: dict[str, str]) -> tuple[str, str]:
    """Apply the joint-verdict table per PREREGISTRATION.md."""
    hc = verdicts.get("hc", "Unimodal")
    ts = verdicts.get("ts", "Unimodal")
    rs = verdicts.get("rs", "Unimodal")
    hc_bm = "Bimodal" in hc
    ts_bm = "Bimodal" in ts
    rs_bm = "Bimodal" in rs

    if hc_bm and ts_bm and rs_bm:
        return ("Reading A — fitness-landscape property",
                "JFI's solution geometry contains anti-correlated basins; "
                "any search method, including random sampling, exhibits "
                "the bimodality. Strongest framing for §3.")
    if hc_bm and ts_bm and not rs_bm:
        return ("Reading A' — local-search property of JFI",
                "Any local-improvement search on JFI exhibits the basin "
                "structure; uniform sampling does not. Publishable as a "
                "fitness-landscape claim conditional on local-improvement "
                "dynamics.")
    if hc_bm != ts_bm:
        return ("Reading B' — algorithm-specific",
                "Bimodality appears in only one of HC/TS. The cross-"
                "algorithm framing fails; soften §3 to 'in our SA "
                "experiments, the optimizer converges into two basins' "
                "with explicit scope-limit.")
    return ("Reading B — SA-specific",
            "Neither HC nor TS shows bimodality. The bimodality is a "
            "property of SA's trajectory, not of JFI's landscape. §3 "
            "must drop the basin framing; load-bearing claim becomes "
            "the magnitude gaps.")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results-csv", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--sa-baseline-csv", type=Path, default=None,
                   help="optional: canonical SA jfi_only results.csv for "
                        "side-by-side comparison")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.results_csv)
    if "condition" not in df.columns:
        print("FATAL: results.csv has no 'condition' column", file=__import__("sys").stderr)
        return 1
    maxb = df["budget"].max()
    final = df[(df["budget"] == maxb) & (df["condition"] == "jfi_only")].copy()
    final["abs_i"] = final["morans_i"].abs()
    if len(final) == 0:
        print("FATAL: no jfi_only rows at final budget", file=__import__("sys").stderr)
        return 1

    algos = sorted(final["algorithm"].unique())
    summary: dict = {"final_budget": int(maxb), "algorithms": algos, "per_algo": {}}
    verdicts_i: dict[str, str] = {}

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("BASIN-STRUCTURE DIAGNOSTIC")
    lines.append("Applied per docs/audit/jfi_only_diagnostics/PREREGISTRATION.md")
    lines.append("=" * 72)
    lines.append(f"Results CSV     : {args.results_csv}")
    lines.append(f"Final budget    : {maxb}")
    lines.append(f"Algorithms found: {algos}")
    lines.append("")

    for algo in algos:
        sub = final[final["algorithm"] == algo]
        n = len(sub)
        ai = sub["abs_i"].to_numpy()
        lp = sub["lisa_penalty"].to_numpy()

        bc_i = sarle_bc(ai)
        dbic_i, gmm_i = gmm_dbic(ai)
        verdict_i = per_algo_verdict(bc_i, dbic_i)
        verdicts_i[algo] = verdict_i

        bc_l = sarle_bc(lp)
        dbic_l, gmm_l = gmm_dbic(lp)
        verdict_l = per_algo_verdict(bc_l, dbic_l)

        rho_s, p_s = stats.spearmanr(lp, ai)

        summary["per_algo"][algo] = {
            "n": int(n),
            "abs_i": {
                "mean": float(ai.mean()), "median": float(np.median(ai)),
                "std": float(ai.std()), "skew": float(stats.skew(ai)),
                "kurt": float(stats.kurtosis(ai)),
                "BC": float(bc_i), "dBIC_k2_over_k1": float(dbic_i),
                "GMM_k2": gmm_i, "verdict": verdict_i,
            },
            "lsap": {
                "mean": float(lp.mean()), "median": float(np.median(lp)),
                "std": float(lp.std()), "skew": float(stats.skew(lp)),
                "kurt": float(stats.kurtosis(lp)),
                "BC": float(bc_l), "dBIC_k2_over_k1": float(dbic_l),
                "GMM_k2": gmm_l, "verdict": verdict_l,
            },
            "spearman_lsap_vs_abs_i": {"rho": float(rho_s), "p": float(p_s)},
        }

        lines.append(f"--- {algo.upper()}  (n={n}) ---")
        lines.append(f"  |I|   mean={ai.mean():.3f}  median={np.median(ai):.3f}  std={ai.std():.3f}")
        lines.append(f"        skew={stats.skew(ai):+.3f}  excess_kurt={stats.kurtosis(ai):+.3f}")
        lines.append(f"        Sarle's BC = {bc_i:.3f}  (threshold {BC_THRESHOLD})")
        lines.append(f"        GMM ΔBIC (k=2 over k=1) = {dbic_i:.1f}")
        lines.append(f"        k=2 fit: means={[f'{m:.3f}' for m in gmm_i['k2_means']]}  "
                     f"sd={[f'{s:.3f}' for s in gmm_i['k2_sds']]}  "
                     f"w={[f'{w:.2f}' for w in gmm_i['k2_weights']]}")
        lines.append(f"        VERDICT (|I|): {verdict_i}")
        lines.append("")
        lines.append(f"  LSAP  mean={lp.mean():.3f}  median={np.median(lp):.3f}  std={lp.std():.3f}")
        lines.append(f"        skew={stats.skew(lp):+.3f}  excess_kurt={stats.kurtosis(lp):+.3f}")
        lines.append(f"        Sarle's BC = {bc_l:.3f}")
        lines.append(f"        GMM ΔBIC (k=2 over k=1) = {dbic_l:.1f}")
        lines.append(f"        VERDICT (LSAP): {verdict_l}")
        lines.append("")
        lines.append(f"  Spearman ρ(LSAP, |I|) = {rho_s:+.3f}  (p = {p_s:.2g})")
        lines.append("")

    joint_label, joint_text = joint_verdict(verdicts_i)
    summary["joint_verdict"] = {"label": joint_label, "text": joint_text,
                                "verdicts_used": verdicts_i}

    lines.append("=" * 72)
    lines.append("JOINT VERDICT")
    lines.append("=" * 72)
    for a in ("hc", "ts", "rs"):
        lines.append(f"  {a.upper():>3}: {verdicts_i.get(a, '(not run)')}")
    lines.append("")
    lines.append(f"  → {joint_label}")
    lines.append(f"    {joint_text}")
    lines.append("")

    txt_path = args.output_dir / "basin_diagnostic.txt"
    txt_path.write_text("\n".join(lines))
    json_path = args.output_dir / "basin_diagnostic.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print("\n".join(lines))
    print(f"\nSaved: {txt_path}")
    print(f"Saved: {json_path}")

    # Figure
    n_algos = len(algos)
    fig, axes = plt.subplots(2, n_algos, figsize=(5 * n_algos, 8), squeeze=False)
    for i, algo in enumerate(algos):
        sub = final[final["algorithm"] == algo]
        ai = sub["abs_i"].to_numpy()
        lp = sub["lisa_penalty"].to_numpy()

        ax = axes[0, i]
        ax.hist(ai, bins=30, density=True, alpha=0.5, color="C2", edgecolor="white")
        xs = np.linspace(0, max(0.001, ai.max()), 400)
        try:
            kde = stats.gaussian_kde(ai)
            ax.plot(xs, kde(xs), color="C2", lw=2)
        except Exception:
            pass
        ax.set_xlabel("|Moran's I|")
        ax.set_ylabel("density")
        ax.set_title(f"{algo.upper()}  |I|  (n={len(sub)})\n"
                     f"BC={sarle_bc(ai):.3f}  ΔBIC={gmm_dbic(ai)[0]:.1f}\n"
                     f"{verdicts_i[algo]}")

        ax = axes[1, i]
        ax.scatter(lp, ai, s=14, alpha=0.55, color="C1", edgecolor="none")
        rho, p = stats.spearmanr(lp, ai)
        ax.set_xlabel("LSAP")
        ax.set_ylabel("|Moran's I|")
        ax.set_title(f"{algo.upper()}  joint\n"
                     f"Spearman ρ = {rho:+.3f}  (p = {p:.2g})")

    fig.suptitle(f"Basin-structure diagnostic (jfi_only, final budget = {maxb})",
                 fontsize=13)
    fig.tight_layout()
    fig_path = args.output_dir / "basin_diagnostic.png"
    fig.savefig(fig_path, dpi=140)
    print(f"Saved: {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

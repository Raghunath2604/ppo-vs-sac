"""Aggregate multi-seed runs: learning curves with 95% CIs and a summary table.

python -m algo_compare.analysis --runs runs/ablation --out results
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def load_study(root: str) -> dict[str, list[dict]]:
    """root/<variant>/seed_<k>/{metrics.csv,final_eval.json} -> {variant: [run, ...]}"""
    study: dict[str, list[dict]] = {}
    for metrics in sorted(glob.glob(os.path.join(root, "*", "seed_*", "metrics.csv"))):
        run_dir = os.path.dirname(metrics)
        variant = os.path.basename(os.path.dirname(run_dir))
        df = pd.read_csv(metrics)
        ev_path = os.path.join(run_dir, "final_eval.json")
        ev = json.load(open(ev_path)) if os.path.exists(ev_path) else None
        study.setdefault(variant, []).append({"seed": os.path.basename(run_dir), "df": df, "eval": ev})
    return study


def ci95_halfwidth(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Half-width of the 95% t-interval of the mean across seeds (0 if only one seed)."""
    n = x.shape[axis]
    if n < 2:
        return np.zeros(np.delete(np.array(x.shape), axis))
    sem = np.nanstd(x, axis=axis, ddof=1) / np.sqrt(n)
    return stats.t.ppf(0.975, n - 1) * sem


def curves(runs: list[dict], grid: np.ndarray) -> np.ndarray:
    """(n_seeds, len(grid)) matrix of mean_return_10 interpolated on a shared step grid."""
    out = []
    for r in runs:
        d = r["df"].dropna(subset=["mean_return_10"])
        out.append(np.interp(grid, d["global_step"], d["mean_return_10"], left=np.nan))
    return np.vstack(out)


def final_performance(run: dict, frac: float = 0.1) -> float:
    """Mean training return over the last `frac` of training for one run."""
    d = run["df"].dropna(subset=["mean_return_10"])
    cutoff = (1 - frac) * d["global_step"].max()
    return float(d.loc[d["global_step"] >= cutoff, "mean_return_10"].mean())


def summarize(study: dict[str, list[dict]], baseline: str = "baseline") -> pd.DataFrame:
    rows = []
    base_vals = None
    if baseline in study:
        base_vals = np.array([final_performance(r) for r in study[baseline]])
    for variant, runs in study.items():
        vals = np.array([final_performance(r) for r in runs])
        evals = np.array([r["eval"]["mean_return"] for r in runs if r["eval"]])
        row = {
            "variant": variant,
            "n_seeds": len(vals),
            "final_return_mean": vals.mean(),
            "ci95_halfwidth": float(ci95_halfwidth(vals[:, None])[0]),
            "std": vals.std(ddof=1) if len(vals) > 1 else 0.0,
            "min": vals.min(),
            "max": vals.max(),
            "eval_return_mean": evals.mean() if len(evals) else np.nan,
        }
        if base_vals is not None and variant != baseline:
            row["delta_vs_baseline"] = vals.mean() - base_vals.mean()
            if len(vals) > 1 and len(base_vals) > 1:
                with warnings.catch_warnings():
                    # identical values within a group (e.g. every seed hit the max return) -> scipy warns
                    warnings.simplefilter("ignore", RuntimeWarning)
                    row["welch_p"] = float(stats.ttest_ind(vals, base_vals, equal_var=False).pvalue)
            else:
                row["welch_p"] = np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    if baseline in set(df["variant"]):
        df = pd.concat([df[df.variant == baseline], df[df.variant != baseline]], ignore_index=True)
    return df


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(
                f"{v:.2f}" if isinstance(v, (float, np.floating)) and not pd.isna(v) else ("" if pd.isna(v) else str(v))
            )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _plot(ax, grid, runs, label, color=None):
    m = curves(runs, grid)
    with warnings.catch_warnings():
        # early grid points have no completed episode yet -> all-NaN columns; that is expected
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(m, axis=0)
        hw = ci95_halfwidth(m)
    ax.plot(grid, mean, label=f"{label} (n={len(runs)})", color=color)
    ax.fill_between(grid, mean - hw, mean + hw, alpha=0.2, color=color)


def plot_all(study: dict[str, list[dict]], out_dir: str, baseline: str = "baseline") -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    max_step = min(r["df"]["global_step"].max() for runs in study.values() for r in runs)
    grid = np.linspace(0, max_step, 200)
    written = []

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for variant, runs in study.items():
        _plot(ax, grid, runs, variant)
    ax.set(
        xlabel="environment steps",
        ylabel="episodic return (mean of last 10 episodes)",
        title="All variants: mean and 95% CI across seeds",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    path = os.path.join(out_dir, "curves_all.png")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(path)

    if baseline in study:
        for variant, runs in study.items():
            if variant == baseline:
                continue
            fig, ax = plt.subplots(figsize=(7, 4.5))
            _plot(ax, grid, study[baseline], baseline, "tab:blue")
            _plot(ax, grid, runs, variant, "tab:red")
            ax.set(xlabel="environment steps", ylabel="episodic return", title=f"{variant} vs {baseline}")
            ax.legend()
            ax.grid(alpha=0.3)
            path = os.path.join(out_dir, f"vs_baseline_{variant}.png")
            fig.tight_layout()
            fig.savefig(path, dpi=140)
            plt.close(fig)
            written.append(path)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarise and plot a multi-seed ablation study.")
    ap.add_argument("--runs", required=True, help="directory containing <variant>/seed_<k>/")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    study = load_study(args.runs)
    if not study:
        raise SystemExit(f"no runs found under {args.runs}")
    os.makedirs(args.out, exist_ok=True)
    summary = summarize(study)
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(to_markdown(summary) + "\n")
    plots = plot_all(study, args.out)
    print(to_markdown(summary))
    print(f"\nwrote summary.csv, summary.md and {len(plots)} plots to {args.out}/")


if __name__ == "__main__":
    main()

"""Compare algorithms: sample efficiency, final performance, stability, wall-clock cost and where the time goes.

python -m algo_compare.compare_report --runs runs/algos_hc --out results/algos_hc --reference ppo
"""

from __future__ import annotations

import argparse
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from .analysis import ci95_halfwidth, load_study, to_markdown  # noqa: E402


def _valid(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=["mean_return_10"])


def run_final_return(df: pd.DataFrame, frac: float = 0.1) -> float:
    d = _valid(df)
    cutoff = (1 - frac) * d["global_step"].max()
    return float(d.loc[d["global_step"] >= cutoff, "mean_return_10"].mean())


def run_auc(df: pd.DataFrame) -> float:
    """Mean return over the whole training run (area under the learning curve / steps): rewards learning *early*."""
    d = _valid(df)
    x, y = d["global_step"].to_numpy(float), d["mean_return_10"].to_numpy(float)
    if len(x) < 2:
        return float(y.mean())
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def run_stability(df: pd.DataFrame, frac: float = 0.2) -> float:
    """Std of the smoothed return over the last `frac` of training (lower = steadier)."""
    d = _valid(df)
    cutoff = (1 - frac) * d["global_step"].max()
    return float(d.loc[d["global_step"] >= cutoff, "mean_return_10"].std(ddof=0))


def first_crossing(df: pd.DataFrame, threshold: float, col: str = "global_step") -> float:
    d = _valid(df)
    hit = d.loc[d["mean_return_10"] >= threshold, col]
    return float(hit.iloc[0]) if len(hit) else np.nan


def pick_threshold(study: dict[str, list[dict]], frac: float = 0.8) -> float:
    """threshold = start + frac * (best final - start): 'frac of the improvement the best algorithm achieved'."""
    starts = [float(_valid(r["df"])["mean_return_10"].iloc[0]) for runs in study.values() for r in runs]
    best = max(np.mean([run_final_return(r["df"]) for r in runs]) for runs in study.values())
    start = float(np.median(starts))
    return start + frac * (best - start)


def per_run_table(study: dict[str, list[dict]], threshold: float) -> pd.DataFrame:
    rows = []
    for variant, runs in study.items():
        for r in runs:
            df = r["df"]
            row = {
                "variant": variant,
                "seed": r["seed"],
                "final_return": run_final_return(df),
                "auc": run_auc(df),
                "stability_std_last20": run_stability(df),
                "steps_to_threshold": first_crossing(df, threshold),
                "total_steps": float(df["global_step"].max()),
                "eval_return": r["eval"]["mean_return"] if r["eval"] else np.nan,
            }
            if "wall_time_s" in df:
                wall = float(df["wall_time_s"].iloc[-1])
                row["wall_time_s"] = wall
                row["sps"] = row["total_steps"] / wall
                row["time_to_threshold_s"] = first_crossing(df, threshold, "wall_time_s")
                roll, upd = float(df["time_rollout_s"].iloc[-1]), float(df["time_update_s"].iloc[-1])
                row["rollout_s_per_1k"] = 1000 * roll / row["total_steps"]
                row["update_s_per_1k"] = 1000 * upd / row["total_steps"]
                row["update_fraction"] = upd / (roll + upd) if roll + upd > 0 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def _mean_ci(x) -> tuple[float, float]:
    v = np.asarray(pd.Series(x).dropna(), dtype=float)
    if len(v) == 0:
        return np.nan, np.nan
    return float(v.mean()), float(ci95_halfwidth(v[:, None])[0])


def _welch(a, b) -> float:
    a, b = np.asarray(pd.Series(a).dropna(), float), np.asarray(pd.Series(b).dropna(), float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def summarize(pr: pd.DataFrame, reference: str | None = None) -> pd.DataFrame:
    variants = list(dict.fromkeys(pr.variant))
    if reference in variants:
        variants = [reference] + [v for v in variants if v != reference]
    rows = []
    for v in variants:
        g = pr[pr.variant == v]
        row = {"variant": v, "n_seeds": len(g)}
        for metric in ("final_return", "auc", "stability_std_last20"):
            m, ci = _mean_ci(g[metric])
            row[metric], row[f"{metric}_ci95"] = m, ci
        reached = g["steps_to_threshold"].notna()
        row["reached_threshold"] = f"{int(reached.sum())}/{len(g)}"
        row["median_steps_to_threshold"] = float(g["steps_to_threshold"].median()) if reached.any() else np.nan
        if "wall_time_s" in g:
            row["wall_minutes"] = float(g["wall_time_s"].mean() / 60)
            row["sps"] = float(g["sps"].mean())
            row["update_fraction"] = float(g["update_fraction"].mean())
            row["median_minutes_to_threshold"] = (
                float(g["time_to_threshold_s"].median() / 60) if g["time_to_threshold_s"].notna().any() else np.nan
            )
        if reference in variants and v != reference:
            ref = pr[pr.variant == reference]
            row["final_vs_ref"] = row["final_return"] - float(ref["final_return"].mean())
            row["final_welch_p"] = _welch(g["final_return"], ref["final_return"])
            row["auc_vs_ref"] = row["auc"] - float(ref["auc"].mean())
            row["auc_welch_p"] = _welch(g["auc"], ref["auc"])
        rows.append(row)
    return pd.DataFrame(rows)


def _curve(runs: list[dict], xcol: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xmax = min(_valid(r["df"])[xcol].max() for r in runs)
    grid = np.linspace(0, xmax, 200)
    m = np.vstack(
        [np.interp(grid, _valid(r["df"])[xcol], _valid(r["df"])["mean_return_10"], left=np.nan) for r in runs]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean, hw = np.nanmean(m, axis=0), ci95_halfwidth(m)
    return grid, mean, hw


def plot_curves(study, out_dir: str, xcol: str, xlabel: str, fname: str, threshold: float | None) -> str | None:
    if not all(xcol in r["df"] for runs in study.values() for r in runs):
        return None
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for variant, runs in study.items():
        grid, mean, hw = _curve(runs, xcol)
        ax.plot(grid, mean, label=f"{variant} (n={len(runs)})")
        ax.fill_between(grid, mean - hw, mean + hw, alpha=0.15)
    if threshold is not None:
        ax.axhline(threshold, color="gray", ls="--", lw=0.8, label="threshold")
    ax.set(xlabel=xlabel, ylabel="episodic return (mean of last 10 episodes)", title=f"Return vs {xlabel}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_profile(pr: pd.DataFrame, out_dir: str) -> str | None:
    if "update_s_per_1k" not in pr:
        return None
    g = pr.groupby("variant")[["rollout_s_per_1k", "update_s_per_1k"]].mean().reindex(list(dict.fromkeys(pr.variant)))
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(g.index, g["rollout_s_per_1k"], label="acting + simulation")
    ax.bar(g.index, g["update_s_per_1k"], bottom=g["rollout_s_per_1k"], label="gradient updates")
    for i, (_, r) in enumerate(g.iterrows()):
        tot = r["rollout_s_per_1k"] + r["update_s_per_1k"]
        ax.text(i, tot, f"{100 * r['update_s_per_1k'] / tot:.0f}% updates", ha="center", va="bottom", fontsize=8)
    ax.set(ylabel="seconds per 1,000 environment steps", title="Where the wall-clock time goes")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "profile.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def build_report(
    runs_dir: str, out_dir: str, reference: str | None, frac: float
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    study = load_study(runs_dir)
    if not study:
        raise SystemExit(f"no runs found under {runs_dir}")
    os.makedirs(out_dir, exist_ok=True)
    thr = pick_threshold(study, frac)
    pr = per_run_table(study, thr)
    summary = summarize(pr, reference)
    pr.to_csv(os.path.join(out_dir, "per_run.csv"), index=False)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(f"Threshold = {thr:.2f} ({frac:.0%} of the best algorithm's improvement over the starting return)\n\n")
        f.write(to_markdown(summary) + "\n")
    plot_curves(study, out_dir, "global_step", "environment steps", "curves_steps.png", thr)
    plot_curves(study, out_dir, "wall_time_s", "wall-clock seconds", "curves_wall.png", thr)
    plot_profile(pr, out_dir)
    return pr, summary, thr


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare algorithms from a runs directory.")
    ap.add_argument("--runs", required=True, help="directory containing <variant>/seed_<k>/")
    ap.add_argument("--out", default="results/compare")
    ap.add_argument("--reference", default="ppo", help="variant to test the others against (Welch)")
    ap.add_argument("--threshold-frac", type=float, default=0.8)
    args = ap.parse_args()
    _, summary, thr = build_report(args.runs, args.out, args.reference, args.threshold_frac)
    print(f"threshold = {thr:.2f}\n")
    print(to_markdown(summary))
    print(f"\nwrote tables and plots to {args.out}/")


if __name__ == "__main__":
    main()

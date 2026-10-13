"""Cross-task evaluation for the multi-task study, and the multi-task-vs-specialist report.

    python -m algo_compare.task_eval --runs runs/mt_hc --targets 1.0,2.5,4.0 --out results/mt_hc/task_eval.csv
    python -m algo_compare.task_eval --report results/mt_hc/task_eval.csv --out results/mt_hc

Every trained policy is evaluated *deterministically* on every target velocity (same episode seeds). A specialist is
not told the target, so its score on other tasks shows how task-specific it is; a multi-task policy is told the target.
"""

from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy import stats  # noqa: E402

from .analysis import ci95_halfwidth, to_markdown  # noqa: E402
from .checkpoint import load_checkpoint  # noqa: E402
from .envs import make_env_for, parse_targets  # noqa: E402
from .evaluate import evaluate_policy  # noqa: E402

EVAL_SEED = 20_000


def _eval_run(job: tuple[str, str, str, list[float], int]) -> list[dict]:
    variant, seed_name, model_path, targets, episodes = job
    torch.set_num_threads(1)
    agent, obs_norm, cfg, lo, hi = load_checkpoint(model_path)
    rows = []
    for t in targets:
        env = make_env_for(cfg, seed=EVAL_SEED, fixed_target=t)
        res = evaluate_policy(agent, obs_norm, cfg.env_id, lo, hi, episodes, seed=EVAL_SEED, env=env)
        env.close()
        rows.append({"variant": variant, "seed": seed_name, "target": float(t), "mean_return": res["mean_return"]})
    return rows


def evaluate_all(runs_dir: str, targets: list[float], episodes: int, jobs: int = 1) -> pd.DataFrame:
    work = []
    for path in sorted(glob.glob(os.path.join(runs_dir, "*", "seed_*", "model.pt"))):
        run = os.path.dirname(path)
        work.append((os.path.basename(os.path.dirname(run)), os.path.basename(run), path, targets, episodes))
    if not work:
        raise SystemExit(f"no models under {runs_dir}")
    if jobs <= 1:
        results = [_eval_run(w) for w in work]
    else:
        with mp.get_context("spawn").Pool(jobs) as pool:
            results = pool.map(_eval_run, work)
    return pd.DataFrame([r for rows in results for r in rows])


def _welch(a, b) -> float:
    if len(a) < 2 or len(b) < 2:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def report(df: pd.DataFrame, out_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (matrix, comparison). matrix: variant x target mean/CI. comparison: multi vs matching specialist."""
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for (variant, target), g in df.groupby(["variant", "target"]):
        v = g["mean_return"].to_numpy(float)
        rows.append(
            {
                "variant": variant,
                "target": target,
                "mean_return": v.mean(),
                "ci95": float(ci95_halfwidth(v[:, None])[0]),
                "n_seeds": len(v),
            }
        )
    matrix = pd.DataFrame(rows)

    comp = []
    for variant in sorted(set(df.variant)):
        if not variant.endswith("_multi"):
            continue
        algo = variant[: -len("_multi")]
        for target in sorted(set(df.target)):
            spec_name = f"{algo}_task_{target:g}"
            m = df[(df.variant == variant) & (df.target == target)]["mean_return"].to_numpy(float)
            s = df[(df.variant == spec_name) & (df.target == target)]["mean_return"].to_numpy(float)
            if len(m) == 0 or len(s) == 0:
                continue
            comp.append(
                {
                    "algo": algo,
                    "target": target,
                    "multi_mean": m.mean(),
                    "specialist_mean": s.mean(),
                    "multi_minus_specialist": m.mean() - s.mean(),
                    "welch_p": _welch(m, s),
                }
            )
    comparison = pd.DataFrame(comp)

    matrix.to_csv(os.path.join(out_dir, "task_matrix.csv"), index=False)
    comparison.to_csv(os.path.join(out_dir, "task_comparison.csv"), index=False)
    with open(os.path.join(out_dir, "task_summary.md"), "w") as f:
        f.write("## Return of each policy on each target velocity (mean +- 95% CI across seeds)\n\n")
        f.write(to_markdown(matrix) + "\n\n## Multi-task policy vs. the specialist trained on that task\n\n")
        f.write(to_markdown(comparison) + "\n")

    piv = matrix.pivot(index="variant", columns="target", values="mean_return")
    fig, ax = plt.subplots(figsize=(1.6 * len(piv.columns) + 3, 0.5 * len(piv) + 2))
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(piv.columns)), [f"{c:g}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), list(piv.index))
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.to_numpy()[i, j]:.0f}", ha="center", va="center", color="w", fontsize=8)
    ax.set(xlabel="target velocity", title="Return on each task (higher = closer to target; <= 0)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "task_heatmap.png"), dpi=140)
    plt.close(fig)
    return matrix, comparison


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-task evaluation and multi-task report.")
    ap.add_argument("--runs", default=None, help="runs directory to evaluate")
    ap.add_argument("--targets", default="1.0,2.5,4.0")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", required=True, help="eval.csv path (evaluate mode) or output directory (--report mode)")
    ap.add_argument("--report", default=None, help="path to a task_eval csv; writes tables/plots to --out")
    args = ap.parse_args()
    if args.report:
        matrix, comparison = report(pd.read_csv(args.report), args.out)
        print(to_markdown(comparison))
        return
    if not args.runs:
        raise SystemExit("--runs is required unless --report is given")
    df = evaluate_all(args.runs, parse_targets(args.targets), args.episodes, args.jobs)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()

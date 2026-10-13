from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from algo_compare.compare_report import (
    build_report,
    first_crossing,
    per_run_table,
    pick_threshold,
    run_auc,
    run_final_return,
    run_stability,
    summarize,
)
from algo_compare.compare_study import ALGOS, apply_overrides, build_jobs
from algo_compare.config import OffPolicyConfig, PPOConfig
from algo_compare.task_eval import evaluate_all, report
from algo_compare.train import train


# ---------------- study construction ----------------
def test_algos_study_jobs_and_overrides(tmp_path):
    jobs = build_jobs(
        "algos",
        "HalfCheetah-v5",
        ["ppo", "sac", "td3"],
        2,
        1000,
        str(tmp_path),
        overrides=["sac.hidden_size=64", "all.batch_size=32", "td3.policy_delay=4"],
    )
    assert len(jobs) == 6
    by = {(a, c["seed"]): c for a, c, _ in jobs}
    assert by[("sac", 1)]["hidden_size"] == 64 and by[("td3", 2)]["hidden_size"] == 256
    assert by[("td3", 1)]["policy_delay"] == 4 and by[("sac", 1)]["batch_size"] == 32
    assert "policy_delay" not in by[("ppo", 1)]  # 'all.' skips fields a config lacks
    assert {Path(j[2]).parent.name for j in jobs} == {"ppo", "sac", "td3"}


def test_override_errors():
    cfg = PPOConfig()
    with pytest.raises(ValueError):
        apply_overrides(cfg, "ppo", ["ppo.not_a_field=1"])
    apply_overrides(cfg, "ppo", ["ppo.norm_obs=false", "sac.hidden_size=7"])
    assert cfg.norm_obs is False and cfg.hidden_size == 64


def test_multitask_jobs(tmp_path):
    jobs = build_jobs(
        "multitask", "HalfCheetah-v5", ["ppo", "sac"], 2, 1000, str(tmp_path), targets=[1.0, 2.5, 4.0], multi_factor=3.0
    )
    names = {Path(j[2]).parent.name for j in jobs}
    assert names == {
        "ppo_multi",
        "ppo_task_1",
        "ppo_task_2.5",
        "ppo_task_4",
        "sac_multi",
        "sac_task_1",
        "sac_task_2.5",
        "sac_task_4",
    }
    cfgs = {Path(j[2]).parent.name: j[1] for j in jobs}
    assert cfgs["ppo_multi"]["task_condition"] is True and cfgs["ppo_multi"]["task_velocities"] == "1,2.5,4"
    assert cfgs["sac_task_2.5"]["task_condition"] is False and cfgs["sac_task_2.5"]["task_velocities"] == "2.5"
    assert cfgs["ppo_multi"]["total_timesteps"] == 3000 and cfgs["ppo_task_1"]["total_timesteps"] == 1000


def test_bad_study_arguments(tmp_path):
    with pytest.raises(ValueError):
        build_jobs("algos", "X", ["dqn"], 1, 10, str(tmp_path))
    with pytest.raises(ValueError):
        build_jobs("multitask", "X", ["ppo"], 1, 10, str(tmp_path), targets=[1.0])
    assert set(ALGOS) == {"ppo", "sac", "td3"} and OffPolicyConfig().algo == "sac"


# ---------------- metrics ----------------
def curve(values, steps=None, wall_per_step=None, upd_frac=0.5):
    steps = steps if steps is not None else np.arange(1, len(values) + 1) * 1000
    df = pd.DataFrame({"global_step": steps, "mean_return_10": values})
    if wall_per_step is not None:
        df["wall_time_s"] = df["global_step"] * wall_per_step
        df["time_update_s"] = df["wall_time_s"] * upd_frac
        df["time_rollout_s"] = df["wall_time_s"] * (1 - upd_frac)
    return df


def test_metric_definitions():
    ramp = curve(np.linspace(0, 100, 11))
    assert abs(run_auc(ramp) - 50.0) < 1e-6  # mean of a linear ramp
    assert abs(run_final_return(ramp, frac=0.1) - 95.0) < 1e-6  # window holds the last two points: 90 and 100
    assert abs(run_final_return(ramp, frac=0.05) - 100.0) < 1e-6
    assert first_crossing(ramp, 55.0) == 7000.0
    assert np.isnan(first_crossing(ramp, 500.0))
    assert run_stability(curve([5.0] * 10)) == 0.0 and run_stability(curve([0, 10] * 5)) > 4


def test_planted_sample_efficiency_and_profile():
    rng = np.random.default_rng(1)
    study = {}
    for name, rate, wall, upd in (("fast_learner", 3.0, 0.01, 0.9), ("slow_learner", 12.0, 0.001, 0.2)):
        steps = np.arange(1, 41) * 1000
        study[name] = [
            {
                "seed": f"seed_{s}",
                "df": curve(
                    1000 * (1 - np.exp(-steps / (rate * 10_000))) + rng.normal(0, 5, size=40), steps, wall, upd
                ),
                "eval": None,
            }
            for s in (1, 2, 3)
        ]
    thr = pick_threshold(study, 0.5)
    pr = per_run_table(study, thr)
    s = summarize(pr).set_index("variant")
    # the slow learner never reaches the threshold within the budget: censored, reported as 0/3 with no median
    assert s.loc["fast_learner", "reached_threshold"] == "3/3" and s.loc["slow_learner", "reached_threshold"] == "0/3"
    assert np.isfinite(s.loc["fast_learner", "median_steps_to_threshold"])
    assert np.isnan(s.loc["slow_learner", "median_steps_to_threshold"])
    assert s.loc["fast_learner", "auc"] > s.loc["slow_learner", "auc"]
    assert s.loc["fast_learner", "update_fraction"] > 0.8
    assert s.loc["slow_learner", "update_fraction"] < 0.3
    ref = summarize(pr, reference="slow_learner").set_index("variant")
    assert ref.loc["fast_learner", "auc_welch_p"] < 0.05 and ref.loc["fast_learner", "auc_vs_ref"] > 0
    # the fast learner costs more wall-clock per step, so it can lose the wall-clock race
    assert s.loc["fast_learner", "wall_minutes"] > s.loc["slow_learner", "wall_minutes"]


def test_build_report_writes_files(tmp_path):
    rng = np.random.default_rng(2)
    for name, rate in (("ppo", 8.0), ("sac", 2.0)):
        for s in (1, 2):
            d = tmp_path / "runs" / name / f"seed_{s}"
            d.mkdir(parents=True)
            steps = np.arange(1, 31) * 1000
            curve(1000 * (1 - np.exp(-steps / (rate * 10_000))) + rng.normal(0, 5, size=30), steps, 0.002, 0.5).to_csv(
                d / "metrics.csv", index=False
            )
    out = tmp_path / "out"
    _, summary, thr = build_report(str(tmp_path / "runs"), str(out), "ppo", 0.8)
    for f in ("summary.csv", "summary.md", "per_run.csv", "curves_steps.png", "curves_wall.png", "profile.png"):
        assert (out / f).exists(), f
    assert np.isfinite(thr) and set(summary.variant) == {"ppo", "sac"}


# ---------------- multi-task evaluation / report ----------------
def test_task_report_recovers_planted_gap(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for seed in (1, 2, 3, 4):
        for target in (1.0, 3.0):
            rows.append(
                dict(variant="ppo_multi", seed=f"seed_{seed}", target=target, mean_return=-200 + rng.normal(0, 5))
            )
            rows.append(
                dict(
                    variant=f"ppo_task_{target:g}",
                    seed=f"seed_{seed}",
                    target=target,
                    mean_return=-100 + rng.normal(0, 5),
                )
            )
            other = 3.0 if target == 1.0 else 1.0
            rows.append(
                dict(
                    variant=f"ppo_task_{other:g}",
                    seed=f"seed_{seed}",
                    target=target,
                    mean_return=-600 + rng.normal(0, 5),
                )
            )
    matrix, comparison = report(pd.DataFrame(rows), str(tmp_path))
    assert len(comparison) == 2 and (comparison["multi_minus_specialist"] < -80).all()
    assert (comparison["welch_p"] < 0.001).all()
    off_diag = matrix[(matrix.variant == "ppo_task_1") & (matrix.target == 3.0)]["mean_return"].iloc[0]
    assert off_diag < -500  # a specialist is bad on the other task
    assert (tmp_path / "task_heatmap.png").exists() and (tmp_path / "task_summary.md").exists()


def test_cross_task_evaluation_on_trained_models(tmp_path):
    for name, kw in (
        ("ppo_multi", {"task_velocities": "1,3", "task_condition": True}),
        ("ppo_task_1", {"task_velocities": "1", "task_condition": False}),
    ):
        train(
            PPOConfig(
                env_id="HalfCheetah-v5",
                total_timesteps=128,
                num_steps=64,
                num_minibatches=2,
                update_epochs=1,
                eval_episodes=1,
                log_dir=str(tmp_path / name / "seed_1"),
                **kw,
            )
        )
    df = evaluate_all(str(tmp_path), [1.0, 3.0], episodes=1)
    assert len(df) == 4 and set(df.variant) == {"ppo_multi", "ppo_task_1"} and np.isfinite(df.mean_return).all()
    assert (df.mean_return <= 0.5).all()  # tracking reward is (almost) non-positive

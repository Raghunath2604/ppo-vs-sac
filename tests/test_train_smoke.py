import json
import os

import numpy as np
import pandas as pd

from algo_compare.checkpoint import load_checkpoint
from algo_compare.config import PPOConfig
from algo_compare.evaluate import evaluate_policy
from algo_compare.train import CSV_COLUMNS, train


def tiny_cfg(tmp_path, **kw):
    base = dict(
        env_id="InvertedPendulum-v5",
        total_timesteps=512,
        num_steps=128,
        num_minibatches=2,
        update_epochs=2,
        eval_episodes=1,
        log_dir=str(tmp_path),
    )
    base.update(kw)
    return PPOConfig(**base)


def test_train_writes_artifacts(tmp_path):
    train(tiny_cfg(tmp_path))
    for f in ("config.json", "metrics.csv", "model.pt", "final_eval.json"):
        assert os.path.exists(tmp_path / f)
    df = pd.read_csv(tmp_path / "metrics.csv")
    assert list(df.columns) == CSV_COLUMNS
    assert len(df) == 4
    assert np.isfinite(df[["policy_loss", "value_loss", "entropy"]].to_numpy()).all()
    assert json.load(open(tmp_path / "final_eval.json"))["episodes"] == 1


def test_same_seed_is_reproducible(tmp_path):
    train(tiny_cfg(tmp_path / "a"))
    train(tiny_cfg(tmp_path / "b"))
    a = pd.read_csv(tmp_path / "a" / "metrics.csv")
    b = pd.read_csv(tmp_path / "b" / "metrics.csv")
    np.testing.assert_allclose(a["policy_loss"], b["policy_loss"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(a["mean_return_10"].fillna(-1), b["mean_return_10"].fillna(-1))


def test_different_seeds_differ(tmp_path):
    train(tiny_cfg(tmp_path / "a", seed=1))
    train(tiny_cfg(tmp_path / "b", seed=2))
    a = pd.read_csv(tmp_path / "a" / "metrics.csv")
    b = pd.read_csv(tmp_path / "b" / "metrics.csv")
    assert not np.allclose(a["policy_loss"], b["policy_loss"])


def test_time_limit_truncation_path_runs(tmp_path):
    # Pendulum-v1 never terminates, it only truncates at 200 steps -> exercises the bootstrap path
    train(tiny_cfg(tmp_path, env_id="Pendulum-v1", total_timesteps=400, num_steps=200, num_minibatches=2))
    df = pd.read_csv(tmp_path / "metrics.csv")
    assert df["episodes_done"].iloc[-1] >= 1
    assert np.isfinite(df["value_loss"]).all()


def test_ablation_switches_run(tmp_path):
    for name, kw in {
        "a": dict(norm_obs=False),
        "b": dict(norm_reward=False),
        "c": dict(norm_adv=False),
        "d": dict(clip_vloss=False),
        "e": dict(num_envs=2, num_steps=64),
    }.items():
        train(tiny_cfg(tmp_path / name, **kw))
        assert os.path.exists(tmp_path / name / "model.pt")


def test_checkpoint_roundtrip_and_eval(tmp_path):
    train(tiny_cfg(tmp_path))
    agent, obs_norm, cfg, lo, hi = load_checkpoint(str(tmp_path / "model.pt"))
    res = evaluate_policy(agent, obs_norm, cfg.env_id, lo, hi, episodes=2)
    assert res["episodes"] == 2 and np.isfinite(res["mean_return"])

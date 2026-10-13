import json

import numpy as np
import pandas as pd
import pytest
import torch

from algo_compare.checkpoint import load_checkpoint
from algo_compare.config import OffPolicyConfig
from algo_compare.export import export_torchscript
from algo_compare.offpolicy import CSV_COLUMNS, SACAgent, TD3Agent, soft_update, train_offpolicy
from algo_compare.offpolicy_nets import SACActor, TD3Actor
from algo_compare.replay import ReplayBuffer

LOW, HIGH = np.array([-2.0, -1.0], dtype=np.float32), np.array([2.0, 3.0], dtype=np.float32)


def small_cfg(**kw):
    base = {"hidden_size": 32, "batch_size": 64, "learning_starts": 100}
    base.update(kw)
    return OffPolicyConfig(**base)


def random_batch(n=64, obs_dim=3, act_dim=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "obs": torch.randn(n, obs_dim, generator=g),
        "act": torch.rand(n, act_dim, generator=g) * 2 - 1,
        "rew": torch.randn(n, generator=g),
        "next_obs": torch.randn(n, obs_dim, generator=g),
        "term": (torch.rand(n, generator=g) < 0.1).float(),
    }


# ---------------- replay buffer ----------------
def test_replay_buffer_wraps_and_samples_valid_rows():
    buf = ReplayBuffer(2, 1, capacity=5)
    for i in range(8):
        buf.add(np.full(2, i), np.full(1, i), float(i), np.full(2, i + 1), terminated=(i == 7))
    assert len(buf) == 5 and buf.ptr == 3
    batch = buf.sample(200)
    assert set(batch["rew"].numpy().astype(int)) <= {3, 4, 5, 6, 7}  # the oldest three were overwritten
    assert batch["obs"].shape == (200, 2) and batch["term"].sum() > 0


# ---------------- networks ----------------
@pytest.mark.parametrize("cls", [SACActor, TD3Actor])
def test_actor_respects_action_bounds(cls):
    actor = cls(3, 2, 32, LOW, HIGH)
    obs = torch.randn(256, 3) * 10
    det = actor.act_deterministic(obs)
    assert (det >= torch.as_tensor(LOW) - 1e-5).all() and (det <= torch.as_tensor(HIGH) + 1e-5).all()
    torch.testing.assert_close(actor.actor_mean(obs), det)  # the exported module is the same function
    if cls is SACActor:
        a, logp = actor(obs)
        assert (a >= torch.as_tensor(LOW) - 1e-5).all() and (a <= torch.as_tensor(HIGH) + 1e-5).all()
        assert torch.isfinite(logp).all() and logp.shape == (256,)


def test_soft_update_interpolates():
    a, b = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    before = a.weight.detach().clone()
    soft_update(a, b, 0.0)
    torch.testing.assert_close(a.weight, before)
    soft_update(a, b, 0.5)
    torch.testing.assert_close(a.weight, 0.5 * before + 0.5 * b.weight)
    soft_update(a, b, 1.0)
    torch.testing.assert_close(a.weight, b.weight)


# ---------------- agents ----------------
@pytest.mark.parametrize("cls", [SACAgent, TD3Agent])
def test_update_is_finite_and_changes_parameters(cls):
    torch.manual_seed(0)
    agent = cls(3, 2, LOW, HIGH, small_cfg(policy_delay=1))
    before = [p.detach().clone() for p in agent.critic.parameters()]
    stats = agent.update(random_batch())
    assert np.isfinite(stats["critic_loss"]) and np.isfinite(stats["q_mean"])
    assert any(not torch.equal(b, p) for b, p in zip(before, agent.critic.parameters(), strict=True))
    assert agent.select_action(np.zeros(3, dtype=np.float32)).shape == (2,)


def test_sac_alpha_is_tuned_only_when_enabled():
    torch.manual_seed(0)
    on = SACAgent(3, 2, LOW, HIGH, small_cfg(autotune_alpha=True))
    off = SACAgent(3, 2, LOW, HIGH, small_cfg(autotune_alpha=False))
    for _ in range(5):
        on.update(random_batch())
        off.update(random_batch())
    assert on.alpha != 1.0 and off.alpha == 1.0


def test_td3_delays_actor_updates():
    torch.manual_seed(0)
    agent = TD3Agent(3, 2, LOW, HIGH, small_cfg(policy_delay=2))
    snap = lambda: [p.detach().clone() for p in agent.actor.parameters()]  # noqa: E731
    p0 = snap()
    agent.update(random_batch())  # update 1: critic only
    assert all(torch.equal(a, b) for a, b in zip(p0, snap(), strict=True))
    agent.update(random_batch())  # update 2: actor moves
    assert any(not torch.equal(a, b) for a, b in zip(p0, snap(), strict=True))


@pytest.mark.parametrize("cls", [SACAgent, TD3Agent])
def test_terminal_transitions_bootstrap_nothing(cls):
    """With term=1 the target is just the reward, so Q must converge to it regardless of the next state."""
    torch.manual_seed(0)
    agent = cls(3, 2, LOW, HIGH, small_cfg(policy_delay=1, learning_rate=3e-3))
    batch = random_batch()
    batch["rew"] = torch.ones(64)
    batch["term"] = torch.ones(64)
    for _ in range(400):
        stats = agent.update(batch)
    assert abs(stats["q_mean"] - 1.0) < 0.15


@pytest.mark.parametrize("cls,kw", [(TD3Agent, {}), (SACAgent, {"autotune_alpha": False, "init_alpha": 0.02})])
def test_agent_learns_a_one_step_task(cls, kw):
    """reward = -|a - 0.5| with a constant observation: the deterministic policy must go to a = 0.5."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    low, high = np.array([-1.0], dtype=np.float32), np.array([1.0], dtype=np.float32)
    agent = cls(1, 1, low, high, small_cfg(learning_rate=3e-3, policy_delay=1, **kw))
    buf = ReplayBuffer(1, 1, 2000)
    for _ in range(2000):
        a = rng.uniform(-1, 1, size=1).astype(np.float32)
        buf.add(np.zeros(1), a, -abs(float(a[0]) - 0.5), np.zeros(1), terminated=True)
    for _ in range(800):
        agent.update(buf.sample(64))
    learned = float(agent.actor.act_deterministic(torch.zeros(1, 1))[0, 0])
    assert abs(learned - 0.5) < 0.2, learned


# ---------------- training loop, checkpoint, deployment ----------------
@pytest.fixture(scope="module", params=["sac", "td3"])
def trained(request, tmp_path_factory):
    d = tmp_path_factory.mktemp(request.param)
    cfg = small_cfg(
        algo=request.param,
        env_id="InvertedPendulum-v5",
        total_timesteps=600,
        log_interval=200,
        eval_episodes=1,
        log_dir=str(d),
    )
    train_offpolicy(cfg)
    return request.param, d


def test_training_writes_artifacts(trained):
    algo, d = trained
    for f in ("config.json", "metrics.csv", "model.pt", "final_eval.json"):
        assert (d / f).exists()
    df = pd.read_csv(d / "metrics.csv")
    assert list(df.columns) == CSV_COLUMNS and len(df) == 3
    assert (df["wall_time_s"].diff().dropna() > 0).all()
    assert (df["time_update_s"].iloc[-1] > 0) and (df["time_rollout_s"].iloc[-1] > 0)
    assert json.load(open(d / "config.json"))["algo"] == algo


def test_checkpoint_roundtrip_and_torchscript_export(trained, tmp_path):
    algo, d = trained
    actor, norm, cfg, lo, hi = load_checkpoint(str(d / "model.pt"))
    assert norm is None and cfg.algo == algo
    ts = torch.jit.load(export_torchscript(str(d / "model.pt"), str(tmp_path)))
    obs = torch.randn(5, actor.obs_dim)
    torch.testing.assert_close(
        ts(obs), torch.max(torch.min(actor.act_deterministic(obs), torch.as_tensor(hi)), torch.as_tensor(lo))
    )


def test_served_off_policy_actor(trained):
    from fastapi.testclient import TestClient

    from algo_compare.serve import create_app

    algo, d = trained
    client = TestClient(create_app(str(d / "model.pt")))
    h = client.get("/health").json()
    assert h["algo"] == algo and h["obs_dim"] == 4
    assert client.post("/act", json={"observation": [0.0, 0.1, 0.0, 0.0]}).status_code == 200


def test_sac_training_is_reproducible(tmp_path):
    kw = {
        "algo": "sac",
        "env_id": "InvertedPendulum-v5",
        "total_timesteps": 400,
        "log_interval": 200,
        "eval_episodes": 1,
    }
    train_offpolicy(small_cfg(log_dir=str(tmp_path / "a"), **kw))
    train_offpolicy(small_cfg(log_dir=str(tmp_path / "b"), **kw))
    a, b = (pd.read_csv(tmp_path / x / "metrics.csv") for x in "ab")
    np.testing.assert_allclose(a["critic_loss"], b["critic_loss"], rtol=1e-5)

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from algo_compare.checkpoint import load_checkpoint
from algo_compare.config import PPOConfig
from algo_compare.export import export_torchscript
from algo_compare.serve import create_app
from algo_compare.train import train


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("model")
    train(
        PPOConfig(
            env_id="InvertedPendulum-v5",
            total_timesteps=512,
            num_steps=128,
            num_minibatches=2,
            update_epochs=2,
            eval_episodes=1,
            log_dir=str(d),
        )
    )
    return d


def reference_action(model_path, obs):
    agent, norm, _, lo, hi = load_checkpoint(model_path)
    x = norm(obs[None, :]) if norm is not None else obs[None, :].astype(np.float32)
    a = agent.act_deterministic(torch.as_tensor(x)).numpy()[0]
    return np.clip(a, lo, hi)


def test_torchscript_matches_checkpoint(model_dir, tmp_path):
    ts_path = export_torchscript(str(model_dir / "model.pt"), str(tmp_path))
    scripted = torch.jit.load(ts_path)
    rng = np.random.default_rng(0)
    for _ in range(5):
        obs = rng.normal(size=4).astype(np.float32)
        out = scripted(torch.as_tensor(obs[None, :]))[0].detach().numpy()
        np.testing.assert_allclose(out, reference_action(str(model_dir / "model.pt"), obs), atol=1e-5)


def test_api_health_and_act(model_dir):
    client = TestClient(create_app(str(model_dir / "model.pt")))
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["obs_dim"] == 4 and h["act_dim"] == 1
    obs = [0.1, -0.2, 0.3, 0.0]
    r = client.post("/act", json={"observation": obs})
    assert r.status_code == 200
    np.testing.assert_allclose(
        r.json()["action"], reference_action(str(model_dir / "model.pt"), np.array(obs, dtype=np.float32)), atol=1e-5
    )
    assert client.get("/health").json()["requests_served"] == 1


def test_api_rejects_wrong_observation_size(model_dir):
    client = TestClient(create_app(str(model_dir / "model.pt")))
    assert client.post("/act", json={"observation": [1.0, 2.0]}).status_code == 422

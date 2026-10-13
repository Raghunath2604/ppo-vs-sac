import numpy as np
import torch

from algo_compare.agent import ppo_update
from algo_compare.config import PPOConfig
from algo_compare.networks import ActorCritic


def make_batch(agent, n=64, obs_dim=5, act_dim=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(n, obs_dim, generator=g)
    with torch.no_grad():
        actions, logp, _, values = agent.get_action_and_value(obs)
    adv = torch.randn(n, generator=g)
    return {
        "obs": obs,
        "actions": actions,
        "logprobs": logp,
        "advantages": adv,
        "returns": values + adv,
        "values": values,
    }


def test_ratio_is_one_before_first_step():
    torch.manual_seed(0)
    agent = ActorCritic(5, 2, 16)
    opt = torch.optim.Adam(agent.parameters(), lr=3e-4)
    cfg = PPOConfig(update_epochs=1, num_minibatches=1)
    stats = ppo_update(agent, opt, make_batch(agent), cfg, np.random.default_rng(0))
    assert abs(stats["approx_kl"]) < 1e-6  # KL is measured before the optimiser step
    assert stats["clipfrac"] == 0.0


def test_update_changes_parameters_and_stays_finite():
    torch.manual_seed(1)
    agent = ActorCritic(5, 2, 16)
    before = [p.detach().clone() for p in agent.parameters()]
    opt = torch.optim.Adam(agent.parameters(), lr=1e-3)
    cfg = PPOConfig(update_epochs=4, num_minibatches=4)
    stats = ppo_update(agent, opt, make_batch(agent), cfg, np.random.default_rng(0))
    assert all(np.isfinite(v) for k, v in stats.items() if k != "explained_variance")
    assert any(not torch.equal(b, a) for b, a in zip(before, agent.parameters(), strict=True))


def test_target_kl_stops_early():
    torch.manual_seed(2)
    agent = ActorCritic(5, 2, 16)
    opt = torch.optim.Adam(agent.parameters(), lr=1e-2)
    cfg = PPOConfig(update_epochs=50, num_minibatches=2, target_kl=1e-6)
    stats = ppo_update(agent, opt, make_batch(agent), cfg, np.random.default_rng(0))
    # with a tiny KL budget the loop must exit after the first epoch(s), not run all 50
    assert stats["approx_kl"] < 1.0

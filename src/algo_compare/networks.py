"""Actor-critic networks for continuous control."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class ActorCritic(nn.Module):
    """Separate tanh MLPs for policy mean and value; state-independent log-std."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, act_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        mean = self.actor_mean(obs)
        std = torch.exp(self.actor_logstd.expand_as(mean))
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, logprob, entropy, self.get_value(obs)

    @torch.no_grad()
    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor_mean(obs)

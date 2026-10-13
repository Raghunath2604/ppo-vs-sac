"""Networks for SAC and TD3."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


def mlp(inp: int, hidden: int, out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(inp, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out)
    )


class _Affine(nn.Module):
    def __init__(self, scale: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self.register_buffer("scale", scale)
        self.register_buffer("bias", bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.bias


class TwinQ(nn.Module):
    """Two independent Q(s, a) networks (clipped double-Q learning)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int):
        super().__init__()
        self.q1 = mlp(obs_dim + act_dim, hidden, 1)
        self.q2 = mlp(obs_dim + act_dim, hidden, 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


def _bounds(act_low, act_high) -> tuple[torch.Tensor, torch.Tensor]:
    lo, hi = np.asarray(act_low, dtype=np.float32), np.asarray(act_high, dtype=np.float32)
    return torch.as_tensor((hi - lo) / 2.0), torch.as_tensor((hi + lo) / 2.0)


class SACActor(nn.Module):
    """Squashed-Gaussian policy: a = tanh(u) * scale + bias, u ~ N(mean(s), std(s))."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int, act_low, act_high):
        super().__init__()
        self.obs_dim = obs_dim
        self.trunk = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.mean_head = nn.Linear(hidden, act_dim)
        self.logstd_head = nn.Linear(hidden, act_dim)
        scale, bias = _bounds(act_low, act_high)
        self.register_buffer("scale", scale)
        self.register_buffer("bias", bias)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reparameterised sample and its log-probability (with the tanh/scale change-of-variables term)."""
        h = self.trunk(obs)
        mean = self.mean_head(h)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (torch.tanh(self.logstd_head(h)) + 1)
        std = log_std.exp()
        u = mean + std * torch.randn_like(mean)
        a = torch.tanh(u)
        logp = torch.distributions.Normal(mean, std).log_prob(u)
        logp = logp - torch.log(self.scale * (1 - a.pow(2)) + 1e-6)
        return a * self.scale + self.bias, logp.sum(-1)

    @torch.no_grad()
    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.mean_head(self.trunk(obs))) * self.scale + self.bias

    @property
    def actor_mean(self) -> nn.Module:
        """Deterministic policy as a single module (used by export.py)."""
        return nn.Sequential(self.trunk, self.mean_head, nn.Tanh(), _Affine(self.scale, self.bias))


class TD3Actor(nn.Module):
    """Deterministic policy: a = tanh(f(s)) * scale + bias."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int, act_low, act_high):
        super().__init__()
        self.obs_dim = obs_dim
        self.net = mlp(obs_dim, hidden, act_dim)
        scale, bias = _bounds(act_low, act_high)
        self.register_buffer("scale", scale)
        self.register_buffer("bias", bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs)) * self.scale + self.bias

    @torch.no_grad()
    def act_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        return self(obs)

    @property
    def actor_mean(self) -> nn.Module:
        return nn.Sequential(self.net, nn.Tanh(), _Affine(self.scale, self.bias))

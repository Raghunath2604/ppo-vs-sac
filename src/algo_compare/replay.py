"""A simple uniform replay buffer (numpy, preallocated)."""

from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int, seed: int = 0):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros(self.capacity, dtype=np.float32)
        self.term = np.zeros(self.capacity, dtype=np.float32)  # TERMINATION only; truncation still bootstraps
        self.ptr = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, obs, act, rew, next_obs, terminated: bool) -> None:
        i = self.ptr
        self.obs[i], self.act[i], self.rew[i] = obs, act, rew
        self.next_obs[i], self.term[i] = next_obs, float(terminated)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        idx = self.rng.integers(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[idx]),
            "act": torch.as_tensor(self.act[idx]),
            "rew": torch.as_tensor(self.rew[idx]),
            "next_obs": torch.as_tensor(self.next_obs[idx]),
            "term": torch.as_tensor(self.term[idx]),
        }

    def __len__(self) -> int:
        return self.size

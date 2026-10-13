"""Running normalisation utilities (Welford / parallel-variance update)."""

from __future__ import annotations

import numpy as np


class RunningMeanStd:
    """Tracks running mean/variance of a stream of batches (Chan et al. parallel update)."""

    def __init__(self, shape: tuple[int, ...] = (), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        self._update_from_moments(batch_mean, batch_var, x.shape[0])

    def _update_from_moments(self, batch_mean, batch_var, batch_count) -> None:
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot
        self.mean = new_mean
        self.var = m2 / tot
        self.count = tot

    def state_dict(self) -> dict:
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": float(self.count)}

    def load_state_dict(self, d: dict) -> None:
        self.mean = np.asarray(d["mean"], dtype=np.float64)
        self.var = np.asarray(d["var"], dtype=np.float64)
        self.count = float(d["count"])


class ObsNormalizer:
    """Normalises observations with running statistics and clips the result."""

    def __init__(self, shape: tuple[int, ...], clip: float = 10.0, eps: float = 1e-8):
        self.rms = RunningMeanStd(shape)
        self.clip = clip
        self.eps = eps

    def update(self, obs: np.ndarray) -> None:
        self.rms.update(obs)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        out = (obs - self.rms.mean) / np.sqrt(self.rms.var + self.eps)
        return np.clip(out, -self.clip, self.clip).astype(np.float32)


class RewardScaler:
    """Scales rewards by the running std of the discounted return (as in the PPO baselines).

    Rewards are *not* mean-shifted, so the sign and relative ordering of rewards is preserved.
    """

    def __init__(self, num_envs: int, gamma: float, clip: float = 10.0, eps: float = 1e-8):
        self.rms = RunningMeanStd(())
        self.ret = np.zeros(num_envs, dtype=np.float64)
        self.gamma = gamma
        self.clip = clip
        self.eps = eps

    def __call__(self, reward: np.ndarray, done: np.ndarray) -> np.ndarray:
        self.ret = self.ret * self.gamma + reward
        self.rms.update(self.ret)
        self.ret[done] = 0.0
        scaled = reward / np.sqrt(self.rms.var + self.eps)
        return np.clip(scaled, -self.clip, self.clip)

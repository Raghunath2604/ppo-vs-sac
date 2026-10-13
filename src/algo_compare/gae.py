"""Generalised Advantage Estimation and the truncation-bootstrap helper."""

from __future__ import annotations

import numpy as np


def compute_gae(
    rewards: np.ndarray,  # (T, N)
    values: np.ndarray,  # (T, N)  V(s_t)
    dones: np.ndarray,  # (T, N)  1 if the episode ended *after* step t (terminated OR truncated)
    next_value: np.ndarray,  # (N,)   V(s_T) for bootstrapping the end of the rollout
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (advantages, returns) with shape (T, N).

    delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
    A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
    """
    T = rewards.shape[0]
    advantages = np.zeros_like(rewards, dtype=np.float64)
    last_gae = np.zeros_like(next_value, dtype=np.float64)
    for t in reversed(range(T)):
        next_v = next_value if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
        advantages[t] = last_gae
    return advantages, advantages + values


def bootstrap_truncated_rewards(
    rewards: np.ndarray, truncated: np.ndarray, terminated: np.ndarray, final_values: np.ndarray, gamma: float
) -> np.ndarray:
    """On a time-limit truncation the episode did not really end, so the value of the
    final state is added to the reward: r_t += gamma * V(s_final).

    Without this, truncation is treated as a true terminal and the critic learns biased values.
    """
    mask = np.logical_and(truncated, np.logical_not(terminated))
    return rewards + gamma * final_values * mask

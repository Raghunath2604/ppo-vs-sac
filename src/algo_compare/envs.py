"""Environment construction, including the multi-task "target velocity" variant of HalfCheetah.

Task design (a standard multi-task benchmark idea): the robot must run at a *target* forward velocity.

    reward = -|x_velocity - target| + reward_ctrl        (reward_ctrl is the usual control cost, <= 0)

A multi-task policy samples a target per episode and is told the target through an extra observation entry
(``condition=True``); a specialist trains on one fixed target and is not conditioned.
"""

from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import numpy as np


def parse_targets(spec: str) -> list[float]:
    """'1.0,2.5,4.0' -> [1.0, 2.5, 4.0]; '' -> []"""
    return [float(x) for x in spec.split(",") if x.strip()]


class TargetVelocityEnv(gym.Wrapper):
    """Re-defines the reward as velocity tracking and (optionally) appends the target to the observation."""

    def __init__(
        self,
        env: gym.Env,
        targets: Sequence[float],
        condition: bool = True,
        seed: int = 0,
        fixed_target: float | None = None,
    ):
        super().__init__(env)
        if not targets and fixed_target is None:
            raise ValueError("need at least one target velocity")
        self.targets = np.asarray(list(targets) if targets else [fixed_target], dtype=np.float64)
        self.condition = condition
        self.fixed_target = fixed_target
        self.rng = np.random.default_rng(seed)
        self.target = float(self.targets[0])
        if condition:
            space = env.observation_space
            lo = float(min(self.targets.min(), fixed_target if fixed_target is not None else np.inf))
            hi = float(max(self.targets.max(), fixed_target if fixed_target is not None else -np.inf))
            self.observation_space = gym.spaces.Box(
                low=np.append(space.low, lo).astype(space.dtype),
                high=np.append(space.high, hi).astype(space.dtype),
                dtype=space.dtype,
            )

    def _aug(self, obs: np.ndarray) -> np.ndarray:
        return np.append(obs, self.target).astype(obs.dtype) if self.condition else obs

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.fixed_target is not None:
            self.target = float(self.fixed_target)
        else:
            self.target = float(self.rng.choice(self.targets))
        obs, info = self.env.reset(seed=seed, options=options)
        return self._aug(obs), {**info, "target_velocity": self.target}

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        reward = -abs(float(info["x_velocity"]) - self.target) + float(info["reward_ctrl"])
        return self._aug(obs), reward, terminated, truncated, {**info, "target_velocity": self.target}


def make_env(
    env_id: str,
    targets: Sequence[float] = (),
    condition: bool = True,
    seed: int = 0,
    fixed_target: float | None = None,
) -> gym.Env:
    env = gym.make(env_id)
    if targets or fixed_target is not None:
        if "x_velocity" not in env.reset(seed=0)[1] and not _has_velocity(env):
            raise ValueError(f"{env_id} does not expose x_velocity; the target-velocity task needs e.g. HalfCheetah-v5")
        env = TargetVelocityEnv(env, targets, condition=condition, seed=seed, fixed_target=fixed_target)
    return env


def _has_velocity(env: gym.Env) -> bool:
    env.reset(seed=0)
    _, _, _, _, info = env.step(env.action_space.sample())
    env.reset(seed=0)
    return "x_velocity" in info and "reward_ctrl" in info


def make_env_for(cfg, seed: int = 0, fixed_target: float | None = None) -> gym.Env:
    """Builds the environment a config describes (works for PPOConfig and OffPolicyConfig)."""
    return make_env(
        cfg.env_id,
        parse_targets(getattr(cfg, "task_velocities", "")),
        condition=getattr(cfg, "task_condition", True),
        seed=seed,
        fixed_target=fixed_target,
    )

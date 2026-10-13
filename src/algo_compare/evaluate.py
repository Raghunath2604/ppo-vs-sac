"""Deterministic evaluation of a trained policy."""

from __future__ import annotations

import argparse
import json

import gymnasium as gym
import numpy as np
import torch

from .checkpoint import load_checkpoint
from .envs import make_env_for


def evaluate_policy(agent, obs_norm, env_id, act_low, act_high, episodes=10, seed=12345, env=None) -> dict:
    own_env = env is None
    env = gym.make(env_id) if own_env else env
    returns, lengths = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, ret, length = False, 0.0, 0
        while not done:
            x = obs_norm(obs[None, :]) if obs_norm is not None else obs[None, :].astype(np.float32)
            with torch.no_grad():
                a = agent.act_deterministic(torch.as_tensor(x, dtype=torch.float32)).numpy()[0]
            obs, r, term, trunc, _ = env.step(np.clip(a, act_low, act_high))
            ret += float(r)
            length += 1
            done = term or trunc
        returns.append(ret)
        lengths.append(length)
    if own_env:
        env.close()
    return {
        "episodes": episodes,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        "returns": returns,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a saved PPO checkpoint.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--env-id", default=None, help="defaults to the env the model was trained on")
    ap.add_argument("--episodes", type=int, default=10)
    args = ap.parse_args()
    agent, obs_norm, cfg, lo, hi = load_checkpoint(args.model)
    env = make_env_for(cfg, seed=12345)
    result = evaluate_policy(agent, obs_norm, args.env_id or cfg.env_id, lo, hi, args.episodes, env=env)
    print(json.dumps({k: v for k, v in result.items() if k != "returns"}, indent=2))


if __name__ == "__main__":
    main()

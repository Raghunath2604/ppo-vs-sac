"""PPO training loop + CLI.

Usage:
    python -m algo_compare.train --env-id HalfCheetah-v5 --total-timesteps 1000000 --seed 1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from collections import deque
from dataclasses import fields

import gymnasium as gym
import numpy as np
import torch

from .agent import ppo_update
from .checkpoint import save_checkpoint
from .config import PPOConfig
from .envs import make_env_for
from .evaluate import evaluate_policy
from .gae import bootstrap_truncated_rewards, compute_gae
from .networks import ActorCritic
from .normalize import ObsNormalizer, RewardScaler

CSV_COLUMNS = [
    "iteration",
    "global_step",
    "mean_return_10",
    "episodes_done",
    "learning_rate",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clipfrac",
    "explained_variance",
    "sps",
    "wall_time_s",
    "time_rollout_s",
    "time_update_s",
]


def make_envs(cfg: PPOConfig) -> gym.vector.SyncVectorEnv:
    # SAME_STEP autoreset: when an episode ends, the returned obs is already the reset obs and
    # the true last obs is available in info["final_obs"] (needed for truncation bootstrapping).
    return gym.vector.SyncVectorEnv(
        [(lambda i=i: make_env_for(cfg, seed=cfg.seed + i)) for i in range(cfg.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(cfg: PPOConfig) -> dict:
    os.makedirs(cfg.log_dir, exist_ok=True)
    torch.set_num_threads(cfg.torch_threads)
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    with open(os.path.join(cfg.log_dir, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    envs = make_envs(cfg)
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))
    act_low, act_high = envs.single_action_space.low, envs.single_action_space.high

    agent = ActorCritic(obs_dim, act_dim, cfg.hidden_size)
    optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)
    obs_norm = ObsNormalizer((obs_dim,)) if cfg.norm_obs else None
    rew_scaler = RewardScaler(cfg.num_envs, cfg.gamma) if cfg.norm_reward else None

    T, N = cfg.num_steps, cfg.num_envs
    obs_buf = np.zeros((T, N, obs_dim), dtype=np.float32)
    act_buf = np.zeros((T, N, act_dim), dtype=np.float32)
    logp_buf = np.zeros((T, N), dtype=np.float32)
    rew_buf = np.zeros((T, N), dtype=np.float64)
    done_buf = np.zeros((T, N), dtype=np.float64)
    val_buf = np.zeros((T, N), dtype=np.float32)

    def normalise(x: np.ndarray) -> np.ndarray:
        return obs_norm(x) if obs_norm is not None else x.astype(np.float32)

    raw_obs, _ = envs.reset(seed=cfg.seed)
    envs.action_space.seed(cfg.seed)
    if obs_norm is not None:
        obs_norm.update(raw_obs)

    ep_ret = np.zeros(N)
    recent_returns: deque[float] = deque(maxlen=10)
    episodes_done = 0
    global_step = 0
    start = time.perf_counter()
    t_roll = t_upd = 0.0

    csv_path = os.path.join(cfg.log_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for iteration in range(1, cfg.num_iterations + 1):
            if cfg.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / cfg.num_iterations
                optimizer.param_groups[0]["lr"] = frac * cfg.learning_rate

            # ---------------- rollout ----------------
            t_phase = time.perf_counter()
            for t in range(T):
                global_step += N
                x = normalise(raw_obs)
                obs_buf[t] = x
                with torch.no_grad():
                    action, logp, _, value = agent.get_action_and_value(torch.as_tensor(x))
                act_np = action.numpy()
                act_buf[t], logp_buf[t], val_buf[t] = act_np, logp.numpy(), value.numpy()

                raw_obs, reward, terminated, truncated, info = envs.step(np.clip(act_np, act_low, act_high))
                done = np.logical_or(terminated, truncated)

                ep_ret += reward
                for i in np.flatnonzero(done):
                    recent_returns.append(float(ep_ret[i]))
                    episodes_done += 1
                    ep_ret[i] = 0.0

                scaled = rew_scaler(reward, done) if rew_scaler is not None else reward.astype(np.float64)

                # time-limit truncation is not a real terminal: bootstrap from V(final_obs)
                trunc_only = np.logical_and(truncated, np.logical_not(terminated))
                if trunc_only.any():
                    final_values = np.zeros(N)
                    for i in np.flatnonzero(trunc_only):
                        fo = normalise(np.asarray(info["final_obs"][i])[None, :])
                        with torch.no_grad():
                            final_values[i] = agent.get_value(torch.as_tensor(fo)).item()
                    scaled = bootstrap_truncated_rewards(scaled, truncated, terminated, final_values, cfg.gamma)

                rew_buf[t] = scaled
                done_buf[t] = done.astype(np.float64)
                if obs_norm is not None:
                    obs_norm.update(raw_obs)

            t_roll += time.perf_counter() - t_phase
            t_phase = time.perf_counter()
            # ---------------- advantages ----------------
            with torch.no_grad():
                next_value = agent.get_value(torch.as_tensor(normalise(raw_obs))).numpy()
            adv, ret = compute_gae(rew_buf, val_buf.astype(np.float64), done_buf, next_value, cfg.gamma, cfg.gae_lambda)

            batch = {
                "obs": torch.as_tensor(obs_buf.reshape(-1, obs_dim)),
                "actions": torch.as_tensor(act_buf.reshape(-1, act_dim)),
                "logprobs": torch.as_tensor(logp_buf.reshape(-1)),
                "advantages": torch.as_tensor(adv.reshape(-1), dtype=torch.float32),
                "returns": torch.as_tensor(ret.reshape(-1), dtype=torch.float32),
                "values": torch.as_tensor(val_buf.reshape(-1)),
            }
            stats = ppo_update(agent, optimizer, batch, cfg, rng)
            t_upd += time.perf_counter() - t_phase

            row = {
                "iteration": iteration,
                "global_step": global_step,
                "mean_return_10": float(np.mean(recent_returns)) if recent_returns else float("nan"),
                "episodes_done": episodes_done,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "sps": int(global_step / max(1e-9, time.perf_counter() - start)),
                "wall_time_s": time.perf_counter() - start,
                "time_rollout_s": t_roll,
                "time_update_s": t_upd,
                **stats,
            }
            writer.writerow(row)
            fcsv.flush()
            if iteration == 1 or iteration % 10 == 0 or iteration == cfg.num_iterations:
                print(
                    f"[{cfg.env_id} seed={cfg.seed}] it {iteration}/{cfg.num_iterations} "
                    f"step {global_step} return10 {row['mean_return_10']:.1f} "
                    f"kl {stats['approx_kl']:.4f} ev {stats['explained_variance']:.2f} sps {row['sps']}",
                    flush=True,
                )

    envs.close()
    model_path = os.path.join(cfg.log_dir, "model.pt")
    save_checkpoint(model_path, agent, obs_norm, cfg, obs_dim, act_dim, act_low, act_high)

    agent.eval()
    result = evaluate_policy(
        agent, obs_norm, cfg.env_id, act_low, act_high, cfg.eval_episodes, env=make_env_for(cfg, seed=12345)
    )
    with open(os.path.join(cfg.log_dir, "final_eval.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"final eval over {cfg.eval_episodes} episodes: {result['mean_return']:.1f} +- {result['std_return']:.1f}")
    return result


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train PPO from scratch.")
    for f in fields(PPOConfig):
        default = f.default
        flag = "--" + f.name.replace("_", "-")
        if isinstance(default, bool):
            ap.add_argument(
                flag, type=lambda s: s.lower() in ("1", "true", "yes", "y"), default=default, metavar="{true,false}"
            )
        else:
            ap.add_argument(flag, type=type(default), default=default)
    return ap


def main() -> None:
    cfg = PPOConfig.from_dict(vars(build_parser().parse_args()))
    train(cfg)


if __name__ == "__main__":
    main()

"""SAC and TD3 agents plus the shared off-policy training loop.

python -m algo_compare.offpolicy --algo sac --env-id HalfCheetah-v5 --total-timesteps 300000 --seed 1
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import time
from collections import deque
from dataclasses import fields

import numpy as np
import torch
import torch.nn.functional as F

from .checkpoint import save_offpolicy_checkpoint
from .config import OffPolicyConfig
from .envs import make_env_for
from .evaluate import evaluate_policy
from .offpolicy_nets import SACActor, TD3Actor, TwinQ
from .replay import ReplayBuffer

CSV_COLUMNS = [
    "iteration",
    "global_step",
    "mean_return_10",
    "episodes_done",
    "critic_loss",
    "actor_loss",
    "alpha",
    "q_mean",
    "sps",
    "wall_time_s",
    "time_rollout_s",
    "time_update_s",
]


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters(), strict=True):
            tp.mul_(1 - tau).add_(sp, alpha=tau)


class SACAgent:
    """Soft Actor-Critic with twin critics and automatic entropy-temperature tuning."""

    def __init__(self, obs_dim, act_dim, act_low, act_high, cfg: OffPolicyConfig):
        self.cfg = cfg
        self.actor = SACActor(obs_dim, act_dim, cfg.hidden_size, act_low, act_high)
        self.critic = TwinQ(obs_dim, act_dim, cfg.hidden_size)
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.learning_rate)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.learning_rate)
        self.target_entropy = -float(act_dim)
        self.log_alpha = torch.tensor(float(np.log(cfg.init_alpha)), requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.learning_rate)

    @property
    def alpha(self) -> float:
        return float(self.log_alpha.exp().item())

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> np.ndarray:
        a, _ = self.actor(torch.as_tensor(obs, dtype=torch.float32)[None, :])
        return a[0].numpy()

    def update(self, b: dict[str, torch.Tensor]) -> dict[str, float]:
        cfg = self.cfg
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            next_a, next_logp = self.actor(b["next_obs"])
            tq1, tq2 = self.critic_target(b["next_obs"], next_a)
            target = b["rew"] + cfg.gamma * (1 - b["term"]) * (torch.min(tq1, tq2) - alpha * next_logp)
        q1, q2 = self.critic(b["obs"], b["act"])
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        a, logp = self.actor(b["obs"])
        aq1, aq2 = self.critic(b["obs"], a)
        actor_loss = (alpha * logp - torch.min(aq1, aq2)).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if cfg.autotune_alpha:
            alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()

        soft_update(self.critic_target, self.critic, cfg.tau)
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha,
            "q_mean": q1.mean().item(),
        }


class TD3Agent:
    """Twin Delayed DDPG: clipped double-Q, target-policy smoothing, delayed actor updates."""

    def __init__(self, obs_dim, act_dim, act_low, act_high, cfg: OffPolicyConfig):
        self.cfg = cfg
        self.actor = TD3Actor(obs_dim, act_dim, cfg.hidden_size, act_low, act_high)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = TwinQ(obs_dim, act_dim, cfg.hidden_size)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.learning_rate)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.learning_rate)
        self.low = torch.as_tensor(np.asarray(act_low, dtype=np.float32))
        self.high = torch.as_tensor(np.asarray(act_high, dtype=np.float32))
        self.updates = 0
        self.last_actor_loss = float("nan")

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> np.ndarray:
        a = self.actor(torch.as_tensor(obs, dtype=torch.float32)[None, :])[0]
        noise = torch.randn_like(a) * self.cfg.exploration_noise * self.actor.scale
        return torch.max(torch.min(a + noise, self.high), self.low).numpy()

    def update(self, b: dict[str, torch.Tensor]) -> dict[str, float]:
        cfg = self.cfg
        self.updates += 1
        with torch.no_grad():
            scale = self.actor.scale
            noise = (torch.randn_like(b["act"]) * cfg.target_noise * scale).clamp(
                -cfg.noise_clip * scale, cfg.noise_clip * scale
            )
            next_a = torch.max(torch.min(self.actor_target(b["next_obs"]) + noise, self.high), self.low)
            tq1, tq2 = self.critic_target(b["next_obs"], next_a)
            target = b["rew"] + cfg.gamma * (1 - b["term"]) * torch.min(tq1, tq2)
        q1, q2 = self.critic(b["obs"], b["act"])
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        if self.updates % cfg.policy_delay == 0:
            actor_loss = -self.critic(b["obs"], self.actor(b["obs"]))[0].mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()
            soft_update(self.actor_target, self.actor, cfg.tau)
            soft_update(self.critic_target, self.critic, cfg.tau)
            self.last_actor_loss = actor_loss.item()
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": self.last_actor_loss,
            "alpha": float("nan"),
            "q_mean": q1.mean().item(),
        }


def make_agent(cfg: OffPolicyConfig, obs_dim, act_dim, low, high):
    if cfg.algo == "sac":
        return SACAgent(obs_dim, act_dim, low, high, cfg)
    if cfg.algo == "td3":
        return TD3Agent(obs_dim, act_dim, low, high, cfg)
    raise ValueError(f"unknown algo {cfg.algo!r}; expected 'sac' or 'td3'")


def train_offpolicy(cfg: OffPolicyConfig) -> dict:
    os.makedirs(cfg.log_dir, exist_ok=True)
    torch.set_num_threads(cfg.torch_threads)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    with open(os.path.join(cfg.log_dir, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    env = make_env_for(cfg, seed=cfg.seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    low, high = env.action_space.low, env.action_space.high
    agent = make_agent(cfg, obs_dim, act_dim, low, high)
    buffer = ReplayBuffer(obs_dim, act_dim, min(cfg.buffer_size, cfg.total_timesteps), seed=cfg.seed)

    obs, _ = env.reset(seed=cfg.seed)
    env.action_space.seed(cfg.seed)
    ep_ret, episodes_done = 0.0, 0
    recent: deque[float] = deque(maxlen=10)
    t_roll = t_upd = 0.0
    start = time.perf_counter()
    stats = {"critic_loss": float("nan"), "actor_loss": float("nan"), "alpha": float("nan"), "q_mean": float("nan")}

    with open(os.path.join(cfg.log_dir, "metrics.csv"), "w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for step in range(1, cfg.total_timesteps + 1):
            t0 = time.perf_counter()
            action = env.action_space.sample() if step <= cfg.learning_starts else agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            buffer.add(obs, action, reward, next_obs, terminated)  # next_obs is the TRUE next state
            ep_ret += float(reward)
            if terminated or truncated:
                recent.append(ep_ret)
                episodes_done += 1
                ep_ret = 0.0
                obs, _ = env.reset()
            else:
                obs = next_obs
            t_roll += time.perf_counter() - t0

            if step >= max(cfg.learning_starts, cfg.batch_size):
                t1 = time.perf_counter()
                for _ in range(cfg.utd):
                    stats = agent.update(buffer.sample(cfg.batch_size))
                t_upd += time.perf_counter() - t1

            if step % cfg.log_interval == 0 or step == cfg.total_timesteps:
                wall = time.perf_counter() - start
                writer.writerow(
                    {
                        "iteration": step // cfg.log_interval,
                        "global_step": step,
                        "mean_return_10": float(np.mean(recent)) if recent else float("nan"),
                        "episodes_done": episodes_done,
                        **stats,
                        "sps": int(step / max(1e-9, wall)),
                        "wall_time_s": wall,
                        "time_rollout_s": t_roll,
                        "time_update_s": t_upd,
                    }
                )
                fcsv.flush()
                if step % (cfg.log_interval * 10) == 0 or step == cfg.total_timesteps:
                    print(
                        f"[{cfg.algo} {cfg.env_id} seed={cfg.seed}] step {step}/{cfg.total_timesteps} "
                        f"return10 {np.mean(recent) if recent else float('nan'):.1f} sps {int(step / wall)}",
                        flush=True,
                    )
    env.close()

    save_offpolicy_checkpoint(os.path.join(cfg.log_dir, "model.pt"), cfg, agent.actor, obs_dim, act_dim, low, high)
    agent.actor.eval()
    result = evaluate_policy(
        agent.actor, None, cfg.env_id, low, high, cfg.eval_episodes, env=make_env_for(cfg, seed=12345)
    )
    with open(os.path.join(cfg.log_dir, "final_eval.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"final eval over {cfg.eval_episodes} episodes: {result['mean_return']:.1f} +- {result['std_return']:.1f}")
    return result


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train SAC or TD3 from scratch.")
    for f in fields(OffPolicyConfig):
        flag = "--" + f.name.replace("_", "-")
        if isinstance(f.default, bool):
            ap.add_argument(
                flag, type=lambda s: s.lower() in ("1", "true", "yes", "y"), default=f.default, metavar="{true,false}"
            )
        else:
            ap.add_argument(flag, type=type(f.default), default=f.default)
    return ap


def main() -> None:
    train_offpolicy(OffPolicyConfig.from_dict(vars(build_parser().parse_args())))


if __name__ == "__main__":
    main()

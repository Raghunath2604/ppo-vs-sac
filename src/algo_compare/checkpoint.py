"""Save / load trained policies (tensor-only payload so torch.load(weights_only=True) works)."""

from __future__ import annotations

import numpy as np
import torch

from .config import OffPolicyConfig, PPOConfig
from .networks import ActorCritic
from .normalize import ObsNormalizer
from .offpolicy_nets import SACActor, TD3Actor


def save_checkpoint(path, agent, obs_norm, cfg, obs_dim, act_dim, act_low, act_high) -> None:
    payload = {
        "algo": "ppo",
        "config": cfg.to_dict(),
        "model_state": agent.state_dict(),
        "obs_dim": int(obs_dim),
        "act_dim": int(act_dim),
        "act_low": [float(x) for x in np.asarray(act_low).ravel()],
        "act_high": [float(x) for x in np.asarray(act_high).ravel()],
        "obs_norm": None,
    }
    if obs_norm is not None:
        sd = obs_norm.rms.state_dict()
        payload["obs_norm"] = {
            "mean": torch.as_tensor(sd["mean"]),
            "var": torch.as_tensor(sd["var"]),
            "count": sd["count"],
            "clip": float(obs_norm.clip),
        }
    torch.save(payload, path)


def save_offpolicy_checkpoint(path, cfg, actor, obs_dim, act_dim, act_low, act_high) -> None:
    """SAC/TD3 checkpoints store only the actor (that is all that is needed to act or deploy)."""
    torch.save(
        {
            "algo": cfg.algo,
            "config": cfg.to_dict(),
            "model_state": actor.state_dict(),
            "obs_dim": int(obs_dim),
            "act_dim": int(act_dim),
            "act_low": [float(x) for x in np.asarray(act_low).ravel()],
            "act_high": [float(x) for x in np.asarray(act_high).ravel()],
            "obs_norm": None,
        },
        path,
    )


def load_checkpoint(path):
    """Returns (agent, obs_norm_or_None, cfg, act_low, act_high). Works for PPO, SAC and TD3 checkpoints."""
    p = torch.load(path, map_location="cpu", weights_only=True)
    algo = p.get("algo", "ppo")
    low = np.array(p["act_low"], dtype=np.float32)
    high = np.array(p["act_high"], dtype=np.float32)
    if algo in ("sac", "td3"):
        cfg = OffPolicyConfig.from_dict(p["config"])
        cls = SACActor if algo == "sac" else TD3Actor
        agent = cls(p["obs_dim"], p["act_dim"], cfg.hidden_size, low, high)
        agent.load_state_dict(p["model_state"])
        agent.eval()
        return agent, None, cfg, low, high
    cfg = PPOConfig.from_dict(p["config"])
    agent = ActorCritic(p["obs_dim"], p["act_dim"], cfg.hidden_size)
    agent.load_state_dict(p["model_state"])
    agent.eval()
    obs_norm = None
    if p["obs_norm"] is not None:
        obs_norm = ObsNormalizer((p["obs_dim"],), clip=p["obs_norm"]["clip"])
        obs_norm.rms.load_state_dict(
            {
                "mean": p["obs_norm"]["mean"].numpy(),
                "var": p["obs_norm"]["var"].numpy(),
                "count": p["obs_norm"]["count"],
            }
        )
    return agent, obs_norm, cfg, np.array(p["act_low"], dtype=np.float32), np.array(p["act_high"], dtype=np.float32)

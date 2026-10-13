"""Export a trained policy as a self-contained TorchScript module.

The exported module takes a RAW observation and returns the clipped deterministic action,
i.e. observation normalisation and action clipping are baked in, so consumers need neither
this package nor the training stats.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from torch import nn

from .checkpoint import load_checkpoint


class DeployablePolicy(nn.Module):
    def __init__(self, actor_mean: nn.Module, mean, var, clip: float, low, high, normalise: bool):
        super().__init__()
        self.actor_mean = actor_mean
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("var", torch.as_tensor(var, dtype=torch.float32))
        self.register_buffer("low", torch.as_tensor(low, dtype=torch.float32))
        self.register_buffer("high", torch.as_tensor(high, dtype=torch.float32))
        self.clip = float(clip)
        self.normalise = bool(normalise)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.normalise:
            obs = torch.clamp((obs - self.mean) / torch.sqrt(self.var + 1e-8), -self.clip, self.clip)
        action = self.actor_mean(obs)
        return torch.max(torch.min(action, self.high), self.low)


def build_deployable(model_path: str) -> tuple[DeployablePolicy, dict]:
    agent, obs_norm, cfg, low, high = load_checkpoint(model_path)
    obs_dim = agent.obs_dim
    if obs_norm is not None:
        mean, var, clip = obs_norm.rms.mean, obs_norm.rms.var, obs_norm.clip
    else:
        mean, var, clip = np.zeros(obs_dim), np.ones(obs_dim), 10.0
    policy = DeployablePolicy(agent.actor_mean, mean, var, clip, low, high, obs_norm is not None).eval()
    meta = {"algo": getattr(cfg, "algo", "ppo"), "env_id": cfg.env_id, "obs_dim": int(obs_dim), "act_dim": len(low)}
    return policy, meta


def export_torchscript(model_path: str, out_dir: str) -> str:
    policy, meta = build_deployable(model_path)
    example = torch.zeros(1, meta["obs_dim"])
    scripted = torch.jit.trace(policy, example)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "policy.ts")
    scripted.save(out_path)
    with open(os.path.join(out_dir, "policy_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a PPO checkpoint to TorchScript.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", default="export")
    args = ap.parse_args()
    print("wrote", export_torchscript(args.model, args.out_dir))


if __name__ == "__main__":
    main()

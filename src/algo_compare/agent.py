"""The PPO policy/value update (clipped surrogate objective)."""

from __future__ import annotations

import numpy as np
import torch

from .config import PPOConfig
from .networks import ActorCritic


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    var_y = np.var(y_true)
    return float("nan") if var_y == 0 else float(1 - np.var(y_true - y_pred) / var_y)


def ppo_update(
    agent: ActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    cfg: PPOConfig,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Runs `update_epochs` passes of minibatch SGD on one rollout batch.

    `batch` holds flattened tensors: obs, actions, logprobs, advantages, returns, values.
    """
    b_size = batch["obs"].shape[0]
    mb_size = max(1, b_size // cfg.num_minibatches)
    clipfracs: list[float] = []
    pg_losses: list[float] = []
    v_losses: list[float] = []
    entropies: list[float] = []
    approx_kl = torch.tensor(0.0)

    for _epoch in range(cfg.update_epochs):
        idx = rng.permutation(b_size)
        for start in range(0, b_size - mb_size + 1, mb_size):
            mb = torch.as_tensor(idx[start : start + mb_size], dtype=torch.long)
            _, new_logp, entropy, new_value = agent.get_action_and_value(batch["obs"][mb], batch["actions"][mb])
            log_ratio = new_logp - batch["logprobs"][mb]
            ratio = log_ratio.exp()

            with torch.no_grad():
                # k3 estimator of KL(old || new) (low variance, non-negative)
                approx_kl = ((ratio - 1) - log_ratio).mean()
                clipfracs.append(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())

            mb_adv = batch["advantages"][mb]
            if cfg.norm_adv:
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

            # clipped surrogate (maximise => minimise the negative)
            pg_loss1 = -mb_adv * ratio
            pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # value loss (optionally clipped around the old value estimate)
            if cfg.clip_vloss:
                v_unclipped = (new_value - batch["returns"][mb]) ** 2
                v_clipped = batch["values"][mb] + torch.clamp(
                    new_value - batch["values"][mb], -cfg.clip_coef, cfg.clip_coef
                )
                v_loss = 0.5 * torch.max(v_unclipped, (v_clipped - batch["returns"][mb]) ** 2).mean()
            else:
                v_loss = 0.5 * ((new_value - batch["returns"][mb]) ** 2).mean()

            ent_loss = entropy.mean()
            loss = pg_loss - cfg.ent_coef * ent_loss + cfg.vf_coef * v_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
            optimizer.step()

            pg_losses.append(pg_loss.item())
            v_losses.append(v_loss.item())
            entropies.append(ent_loss.item())

        if cfg.target_kl > 0 and approx_kl.item() > cfg.target_kl:
            break

    stats: dict[str, float] = {
        "policy_loss": float(np.mean(pg_losses)),
        "value_loss": float(np.mean(v_losses)),
        "entropy": float(np.mean(entropies)),
    }
    stats["approx_kl"] = float(approx_kl.item())
    stats["clipfrac"] = float(np.mean(clipfracs)) if clipfracs else 0.0
    stats["explained_variance"] = explained_variance(batch["values"].numpy(), batch["returns"].numpy())
    return stats

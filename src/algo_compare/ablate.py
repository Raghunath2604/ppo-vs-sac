"""Multi-seed ablation runner.

    python -m algo_compare.ablate --study core --env-id HalfCheetah-v5 --seeds 5 \
        --total-timesteps 1000000 --out runs/ablation
"""

from __future__ import annotations

import argparse
import itertools
import multiprocessing as mp
import os

from .config import PPOConfig
from .train import train

# name -> overrides on top of the baseline PPOConfig
VARIANTS: dict[str, dict] = {
    "baseline": {},
    "no_adv_norm": {"norm_adv": False},
    "no_obs_norm": {"norm_obs": False},
    "no_reward_norm": {"norm_reward": False},
    "no_vloss_clip": {"clip_vloss": False},
    "no_lr_anneal": {"anneal_lr": False},
    "clip_0.1": {"clip_coef": 0.1},
    "clip_0.3": {"clip_coef": 0.3},
    "lr_1e-3": {"learning_rate": 1e-3},
    "lr_1e-4": {"learning_rate": 1e-4},
    "hidden_32": {"hidden_size": 32},
    "hidden_256": {"hidden_size": 256},
}

STUDIES: dict[str, list[str]] = {
    # the implementation details most often reported to matter for PPO
    "core": ["baseline", "no_adv_norm", "no_obs_norm", "no_reward_norm", "no_vloss_clip"],
    # core + optimisation hyper-parameters
    "full": list(VARIANTS),
    "tiny": ["baseline", "no_obs_norm"],  # used for fast pipeline checks
}


def _run(job: tuple[dict, str]) -> str:
    cfg_dict, out_dir = job
    if os.path.exists(os.path.join(out_dir, "final_eval.json")):
        return f"skip (done) {out_dir}"
    train(PPOConfig.from_dict(cfg_dict))
    return f"done {out_dir}"


def build_jobs(study: str, env_id: str, seeds: int, total_timesteps: int, out: str, base: dict | None = None):
    jobs = []
    for variant, seed in itertools.product(STUDIES[study], range(1, seeds + 1)):
        cfg = PPOConfig(env_id=env_id, total_timesteps=total_timesteps, seed=seed)
        for k, v in (base or {}).items():
            setattr(cfg, k, v)
        for k, v in VARIANTS[variant].items():
            setattr(cfg, k, v)
        cfg.log_dir = os.path.join(out, variant, f"seed_{seed}")
        jobs.append((cfg.to_dict(), cfg.log_dir))
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a multi-seed PPO ablation study.")
    ap.add_argument("--study", choices=list(STUDIES), default="core")
    ap.add_argument("--env-id", default="HalfCheetah-v5")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--total-timesteps", type=int, default=1_000_000)
    ap.add_argument("--out", default="runs/ablation")
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker processes (<= CPU cores)")
    ap.add_argument("--eval-episodes", type=int, default=10)
    args = ap.parse_args()

    jobs = build_jobs(
        args.study, args.env_id, args.seeds, args.total_timesteps, args.out, {"eval_episodes": args.eval_episodes}
    )
    print(f"{len(jobs)} runs ({len(STUDIES[args.study])} variants x {args.seeds} seeds) -> {args.out}")
    if args.jobs <= 1:
        for j in jobs:
            print(_run(j))
    else:
        with mp.get_context("spawn").Pool(args.jobs) as pool:
            for msg in pool.imap_unordered(_run, jobs):
                print(msg)


if __name__ == "__main__":
    main()

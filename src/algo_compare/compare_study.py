"""Run the algorithm-comparison and multi-task studies (algorithms x seeds, resumable, optional multiprocessing).

    # sample-efficiency / stability / wall-clock comparison
    python -m algo_compare.compare_study --study algos --env-id HalfCheetah-v5 --algos ppo,sac,td3 \
        --seeds 5 --total-timesteps 300000 --jobs 4 --out runs/algos_hc

    # multi-task (target velocity) vs specialists
    python -m algo_compare.compare_study --study multitask --env-id HalfCheetah-v5 --algos ppo,sac \
        --targets 1.0,2.5,4.0 --seeds 3 --total-timesteps 300000 --out runs/mt_hc
"""

from __future__ import annotations

import argparse
import itertools
import multiprocessing as mp
import os

from .config import OffPolicyConfig, PPOConfig
from .envs import parse_targets
from .offpolicy import train_offpolicy
from .train import train

ALGOS = ("ppo", "sac", "td3")


def _make_cfg(algo: str, env_id: str, seed: int, steps: int, log_dir: str, eval_episodes: int):
    if algo == "ppo":
        return PPOConfig(env_id=env_id, seed=seed, total_timesteps=steps, log_dir=log_dir, eval_episodes=eval_episodes)
    return OffPolicyConfig(
        algo=algo, env_id=env_id, seed=seed, total_timesteps=steps, log_dir=log_dir, eval_episodes=eval_episodes
    )


def apply_overrides(cfg, algo: str, overrides: list[str]) -> None:
    """Each override is 'algo.field=value' (algo in ppo|sac|td3|all)."""
    for text in overrides:
        target, rest = text.split(".", 1)
        field, value = rest.split("=", 1)
        if target not in (algo, "all"):
            continue
        if field not in cfg.__dataclass_fields__:
            if target == "all":
                continue  # e.g. an off-policy-only field in an 'all.' override
            raise ValueError(f"{type(cfg).__name__} has no field {field!r}")
        current = getattr(cfg, field)
        if isinstance(current, bool):
            parsed = value.lower() in ("1", "true", "yes", "y")
        else:
            parsed = type(current)(value)
        setattr(cfg, field, parsed)


def build_jobs(
    study: str,
    env_id: str,
    algos: list[str],
    seeds: int,
    total_timesteps: int,
    out: str,
    targets: list[float] | None = None,
    multi_factor: float = 1.0,
    overrides: list[str] | None = None,
    eval_episodes: int = 10,
) -> list[tuple[str, dict, str]]:
    """Returns jobs = (algo, config_dict, log_dir)."""
    for a in algos:
        if a not in ALGOS:
            raise ValueError(f"unknown algo {a!r}; expected a subset of {ALGOS}")
    if study == "algos":
        variants = [(a, a, {}, 1.0) for a in algos]
    elif study == "multitask":
        if not targets or len(targets) < 2:
            raise ValueError("--study multitask needs --targets with at least two velocities")
        tstr = ",".join(f"{t:g}" for t in targets)
        variants = []
        for a in algos:
            variants.append((f"{a}_multi", a, {"task_velocities": tstr, "task_condition": True}, multi_factor))
            for t in targets:
                variants.append((f"{a}_task_{t:g}", a, {"task_velocities": f"{t:g}", "task_condition": False}, 1.0))
    else:
        raise ValueError(f"unknown study {study!r}")
    jobs = []
    for (name, algo, extra, factor), seed in itertools.product(variants, range(1, seeds + 1)):
        log_dir = os.path.join(out, name, f"seed_{seed}")
        cfg = _make_cfg(algo, env_id, seed, int(total_timesteps * factor), log_dir, eval_episodes)
        for k, v in extra.items():
            setattr(cfg, k, v)
        apply_overrides(cfg, algo, overrides or [])
        jobs.append((algo, cfg.to_dict(), log_dir))
    return jobs


def _run(job: tuple[str, dict, str]) -> str:
    algo, cfg_dict, log_dir = job
    if os.path.exists(os.path.join(log_dir, "final_eval.json")):
        return f"skip (done) {log_dir}"
    if algo == "ppo":
        train(PPOConfig.from_dict(cfg_dict))
    else:
        train_offpolicy(OffPolicyConfig.from_dict(cfg_dict))
    return f"done {log_dir}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run algorithm-comparison / multi-task studies.")
    ap.add_argument("--study", choices=["algos", "multitask"], default="algos")
    ap.add_argument("--env-id", default="HalfCheetah-v5")
    ap.add_argument("--algos", default="ppo,sac,td3")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--total-timesteps", type=int, default=300_000)
    ap.add_argument("--targets", default="1.0,2.5,4.0", help="target velocities (multitask study)")
    ap.add_argument("--multi-factor", type=float, default=1.0, help="multiply the multi-task policy's step budget")
    ap.add_argument("--override", action="append", default=[], metavar="ALGO.FIELD=VALUE")
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker processes (<= CPU cores)")
    ap.add_argument("--out", default="runs/study")
    args = ap.parse_args()
    jobs = build_jobs(
        args.study,
        args.env_id,
        [a for a in args.algos.split(",") if a],
        args.seeds,
        args.total_timesteps,
        args.out,
        targets=parse_targets(args.targets),
        multi_factor=args.multi_factor,
        overrides=args.override,
        eval_episodes=args.eval_episodes,
    )
    print(f"{len(jobs)} runs -> {args.out}")
    if args.jobs <= 1:
        for j in jobs:
            print(_run(j))
    else:
        with mp.get_context("spawn").Pool(args.jobs) as pool:
            for msg in pool.imap_unordered(_run, jobs):
                print(msg)


if __name__ == "__main__":
    main()

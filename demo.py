#!/usr/bin/env python3
"""Interactive live demonstration of trained PPO, SAC, and TD3 policies.

Run a deterministic policy simulation on HalfCheetah-v5, benchmark inference latency,
compare policies side-by-side, and optionally render animated GIFs.

Usage:
    python demo.py                          # Run SAC on target-velocity HalfCheetah
    python demo.py --algo td3 --target 3.0  # Run TD3 targeting 3.0 m/s
    python demo.py --compare                # Side-by-side benchmark of SAC vs TD3 vs PPO
    python demo.py --gif assets/demo.gif    # Render animated simulation GIF
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import gymnasium as gym
import numpy as np
import torch
from PIL import Image, ImageDraw

from algo_compare.checkpoint import load_checkpoint
from algo_compare.envs import TargetVelocityEnv

DEFAULT_MODELS = {
    "sac": "runs/algos_preview/sac/seed_1/model.pt",
    "td3": "runs/algos_preview/td3/seed_1/model.pt",
    "ppo": "runs/algos_preview/ppo/seed_1/model.pt",
}

FALLBACK_MODELS = {
    "sac": "runs/algos_quick/sac/seed_1/model.pt",
    "td3": "runs/algos_quick/td3/seed_1/model.pt",
    "ppo": "runs/algos_quick/ppo/seed_1/model.pt",
}


def find_model(algo: str, user_path: str | None = None) -> str:
    if user_path and os.path.exists(user_path):
        return user_path
    if os.path.exists(DEFAULT_MODELS.get(algo, "")):
        return DEFAULT_MODELS[algo]
    if os.path.exists(FALLBACK_MODELS.get(algo, "")):
        return FALLBACK_MODELS[algo]
    for root, _, files in os.walk("runs"):
        for f in files:
            if f == "model.pt" and algo in root.lower():
                return os.path.join(root, f)
    raise FileNotFoundError(f"No trained model found for {algo}. Please specify --model PATH.")


def print_banner() -> None:
    print("\n" + "=" * 78)
    print("   PPO vs SAC vs TD3: Empirical RL Continuous Control Live Demonstration")
    print("=" * 78)


def run_simulation(
    algo: str,
    model_path: str,
    target_velocity: float = 2.5,
    steps: int = 200,
    seed: int = 42,
    gif_path: str | None = None,
) -> dict:
    print(f"\n[+] Loading {algo.upper()} checkpoint: {model_path}")
    agent, obs_norm, cfg, low, high = load_checkpoint(model_path)
    obs_dim = getattr(agent, "obs_dim", 17)

    render_mode = "rgb_array" if gif_path else None
    base_env = gym.make("HalfCheetah-v5", render_mode=render_mode)
    env = TargetVelocityEnv(base_env, targets=[target_velocity], condition=False, fixed_target=target_velocity)

    obs, _ = env.reset(seed=seed)
    frames: list[Image.Image] = []

    print(f"[+] Environment: HalfCheetah-v5 | Target Velocity: {target_velocity:.2f} m/s | Steps: {steps}")
    print("\n" + "-" * 78)
    print(
        f"{'Step':>6} | {'Velocity':>10} | {'Target':>8} | {'Error':>8} | "
        f"{'Reward':>9} | {'ActNorm':>8} | {'Latency':>8}"
    )
    print("-" * 78)

    velocities = []
    rewards = []
    latencies = []

    for step in range(1, steps + 1):
        x = obs_norm(obs[None, :]) if obs_norm is not None else obs[None, :].astype(np.float32)
        if x.shape[1] > obs_dim:
            x = x[:, :obs_dim]

        t0 = time.perf_counter()
        with torch.no_grad():
            a = agent.act_deterministic(torch.as_tensor(x, dtype=torch.float32)).numpy()[0]
        dt = (time.perf_counter() - t0) * 1000.0

        clipped_a = np.clip(a, low, high)
        obs, rew, terminated, truncated, info = env.step(clipped_a)

        vx = float(info.get("x_velocity", 0.0))
        err = abs(vx - target_velocity)
        norm_a = float(np.linalg.norm(clipped_a))

        velocities.append(vx)
        rewards.append(rew)
        latencies.append(dt)

        if step % 20 == 0 or step == 1 or step == steps:
            print(
                f"{step:>6} | {vx:>9.2f}m/s | {target_velocity:>7.2f}m/s | "
                f"{err:>7.2f}m/s | {rew:>9.2f} | {norm_a:>8.2f} | {dt:>6.2f}ms"
            )

        if gif_path and step % 2 == 0:
            frame_arr = env.render()
            if frame_arr is not None:
                img = Image.fromarray(frame_arr)
                draw = ImageDraw.Draw(img)
                text = (
                    f"{algo.upper()} | Step {step} | Vx: {vx:.2f} m/s | "
                    f"Target: {target_velocity:.2f} m/s | Latency: {dt:.2f}ms"
                )
                draw.rectangle([(10, 10), (img.width - 10, 35)], fill=(0, 0, 0, 180))
                draw.text((15, 14), text, fill=(255, 255, 255))
                frames.append(img)

        if terminated or truncated:
            obs, _ = env.reset(seed=seed + step)

    env.close()

    if gif_path and frames:
        os.makedirs(os.path.dirname(os.path.abspath(gif_path)), exist_ok=True)
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=40, loop=0)
        print(f"\n[+] Rendered animated simulation GIF ({len(frames)} frames) -> {gif_path}")

    mean_v = float(np.mean(velocities))
    mean_err = float(np.mean(np.abs(np.array(velocities) - target_velocity)))
    tot_rew = float(np.sum(rewards))
    mean_lat = float(np.mean(latencies))

    print("-" * 78)
    print(f"[Summary] Mean Velocity: {mean_v:.2f} m/s | Mean Tracking Error: {mean_err:.2f} m/s")
    print(f"[Summary] Cumulative Reward: {tot_rew:.2f} | Avg Step Latency: {mean_lat:.3f} ms")
    print("-" * 78)

    return {
        "algo": algo,
        "mean_velocity": mean_v,
        "mean_error": mean_err,
        "total_reward": tot_rew,
        "mean_latency": mean_lat,
    }


def compare_all(iters: int = 100) -> None:
    print("\n" + "=" * 78)
    print("   Side-by-Side Algorithm Comparison Benchmark (SAC vs TD3 vs PPO)")
    print("=" * 78)

    dummy_obs = np.random.randn(1, 17).astype(np.float32)
    t_obs = torch.as_tensor(dummy_obs)

    results = []
    for algo in ["sac", "td3", "ppo"]:
        try:
            path = find_model(algo)
            agent, obs_norm, cfg, low, high = load_checkpoint(path)
            param_count = sum(p.numel() for p in agent.parameters())

            # Warmup
            with torch.no_grad():
                _ = agent.act_deterministic(t_obs)

            # Benchmark 100 inferences
            times = []
            for _ in range(iters):
                t0 = time.perf_counter()
                with torch.no_grad():
                    act = agent.act_deterministic(t_obs).numpy()[0]
                times.append((time.perf_counter() - t0) * 1000.0)

            results.append({
                "algo": algo.upper(),
                "params": param_count,
                "latency_mean": np.mean(times),
                "latency_p95": np.percentile(times, 95),
                "action_sample": np.round(act[:3], 3),
                "path": path,
            })
        except Exception as e:
            results.append({"algo": algo.upper(), "error": str(e)})

    print("\n" + "-" * 78)
    print(
        f"{'Algorithm':<10} | {'Parameters':<12} | "
        f"{'Mean Latency':<12} | {'P95 Latency':<12} | {'First 3 Actions':<16}"
    )
    print("-" * 78)
    for r in results:
        if "error" in r:
            print(f"{r['algo']:<10} | ERROR: {r['error']}")
        else:
            act_str = f"[{r['action_sample'][0]:.2f}, {r['action_sample'][1]:.2f}, {r['action_sample'][2]:.2f}]"
            print(
                f"{r['algo']:<10} | {r['params']:<12,} | "
                f"{r['latency_mean']:>8.3f} ms | {r['latency_p95']:>8.3f} ms | {act_str:<16}"
            )
    print("-" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live policy demonstration for PPO, SAC, and TD3.")
    parser.add_argument("--algo", default="sac", choices=["sac", "td3", "ppo"], help="Algorithm to run")
    parser.add_argument("--model", default=None, help="Path to checkpoint model.pt")
    parser.add_argument("--target", type=float, default=2.5, help="Target forward velocity (m/s)")
    parser.add_argument("--steps", type=int, default=200, help="Simulation steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gif", default=None, help="Save simulation as animated GIF")
    parser.add_argument("--compare", action="store_true", help="Compare SAC, TD3, and PPO side-by-side")
    args = parser.parse_args()

    print_banner()

    if args.compare:
        compare_all()
        return

    try:
        model_path = find_model(args.algo, args.model)
        run_simulation(
            algo=args.algo,
            model_path=model_path,
            target_velocity=args.target,
            steps=args.steps,
            seed=args.seed,
            gif_path=args.gif,
        )
    except FileNotFoundError as e:
        print(f"[!] Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

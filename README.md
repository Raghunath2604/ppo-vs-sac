# ppo-vs-sac

**Project 3 of 3.** PPO, SAC and TD3 implemented from scratch in PyTorch, with a rigorous comparison harness
(sample efficiency, final performance, stability, wall-clock cost and a time profile), plus a **multi-task** study
(one conditioned policy vs. specialists on a target-velocity family of HalfCheetah tasks).

> **Start here:** [`docs/PROCESS.md`](docs/PROCESS.md): the complete guide from setup to write-up to deployment.  
> **Technical Report:** [`docs/RESEARCH_REPORT.md`](docs/RESEARCH_REPORT.md) | **Empirical Results:** [`docs/RESULTS_TEMPLATE.md`](docs/RESULTS_TEMPLATE.md)  
> **Interactive Dashboard:** Launch [`dashboard/index.html`](dashboard/index.html) (or `python -m http.server 8089 --directory dashboard`).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"

make test            # 59 tests (~35 s)
make algos-quick     # pipeline check on an easy task: ppo vs sac vs td3 -> report (15-25 min on a slow CPU)
```

The real studies are `make algos-all AJOBS=4` (HalfCheetah, PPO/SAC/TD3, 5 seeds) and `make mt-all AJOBS=4`
(multi-task, 3 seeds). **First measure your machine's speed**: off-policy training is much slower per step than PPO
(see the guide, step 2). No `make`? Every target is a plain `python -m algo_compare....` command in the guide.

## Pipeline

```
compare_study  ->  runs/<variant>/seed_<k>/{metrics.csv,model.pt,final_eval.json}   algorithms x seeds, resumable
compare_report ->  summary.md, curves_steps.png, curves_wall.png, profile.png        AUC, steps/time-to-threshold, stability, CIs
task_eval      ->  task_matrix.csv, task_comparison.csv, task_heatmap.png            multi-task vs specialists, cross-task transfer
export / serve ->  policy.ts, FastAPI /act                                           works for PPO, SAC and TD3 checkpoints
```

| Module | Purpose |
|---|---|
| `offpolicy.py`, `offpolicy_nets.py`, `replay.py` | SAC (twin Q, tanh-Gaussian actor, auto-tuned temperature), TD3 (twin Q, target smoothing, delayed updates), replay buffer |
| `agent.py`, `gae.py`, `train.py` | PPO (project 1), now also logging wall-clock and rollout-vs-update time |
| `envs.py` | target-velocity multi-task wrapper for HalfCheetah (conditioned or specialist) |
| `compare_study.py`, `compare_report.py` | study runner and the metric/plot report |
| `task_eval.py` | cross-task evaluation and multi-task report |

## Verified status (what was actually run)

Build environment: Python 3.12, Linux, a slow shared single CPU core; torch 2.14.1, gymnasium 1.4.0, mujoco 3.15.0.

- `pytest`: **59 passed** (24 from project 1 + 35 new); `ruff check` and `ruff format --check` clean. The new tests include:
  terminal transitions bootstrap nothing (Q converges to the reward), both agents learn a one-step task to the right
  action, TD3's actor updates are delayed, SAC's temperature is tuned only when enabled, actor outputs stay within action
  bounds, reproducible training, TorchScript export equals the actor, served SAC/TD3 policies, and metric definitions
  (including censoring when a seed never reaches the threshold).
- **Correctness check on a known benchmark** (Pendulum-v1, default hyper-parameters: 2 x 256 networks, batch 256, one
  seed each, no confidence interval): SAC's smoothed return went from about -1,070 to -1,390 early on to **-145 at 8,000
  steps** (deterministic evaluation -141.5 +- 112.9 over 5 episodes). TD3 went from about -1,470 to **-165 at 10,000 steps**
  (evaluation -148.9 +- 119.7). Both learn as expected, so the off-policy implementations are not obviously broken.
- **Mini algorithm study (pipeline check, not the science):** InvertedPendulum-v5, 2 seeds, 20,000 steps, with *small*
  off-policy networks (64 units, batch 64) so it could run in the build environment. Deterministic evaluation returns:
  PPO 574 and 813, SAC 185 and 83, TD3 58 and 55. **This setting does not show an off-policy advantage**, and it should not
  be read as evidence about the algorithms (tiny networks, a very short budget, an easy task, 2 seeds). What the study
  did show about *cost*: ~850-950 steps/s for PPO vs ~155 for SAC and ~265 for TD3; updates took ~44% of PPO's time and
  ~90% of SAC's and TD3's; a run took ~0.3 min (PPO) vs ~2.1 min (SAC) and ~1.3 min (TD3).
- **Multi-task pipeline** (HalfCheetah-v5, targets 1.0 and 3.0, PPO and SAC, 2 seeds, 2,048 steps each): 12 runs trained,
  all evaluated on both targets, and the report/heatmap produced. The policies are essentially untrained (returns about
  -1,000 and -3,000, i.e. standing still), so this validates the machinery only.
- In the build environment SAC with default 2 x 256 networks ran at only ~60 steps/s (including warm-up). Your machine
  will differ: measure it before planning budgets.

**Not run:** the real HalfCheetah/Hopper studies (that is your job, and the point of the project), the Docker build,
GitHub Actions, Windows/macOS.

## Layout

```
src/algo_compare/   package (offpolicy*, replay, envs, compare_*, task_eval + project-1 modules)
tests/              59 tests
docs/PROCESS.md     full guide;  docs/RESULTS_TEMPLATE.md   write-up skeleton
Makefile  Dockerfile  pyproject.toml  .github/workflows/ci.yml
```

# PPO vs SAC vs TD3, and a multi-task study: the complete process, end to end

Project 3 of 3. It builds on the PPO from project 1 (a self-contained copy is included; nothing else needs installing)
and adds two off-policy algorithms, **SAC** and **TD3**, written from scratch, plus a rigorous way to compare them.

> **Study A (algorithms).** On the same tasks and the same environment-step budget, how do PPO, SAC and TD3 differ in
> *sample efficiency*, *final performance*, *stability across seeds*, *wall-clock cost*, and *where the time goes*?
>
> **Study B (multi-task).** Can one policy, told the target, match specialists trained on each task? How task-specific
> is a specialist? Does the answer differ between PPO and SAC?

> **Honesty rule.** Only report numbers you measured. The algorithms ship with different "standard" hyper-parameters, so
> a comparison is only as fair as you make it; section 3d explains exactly how and what to state in your write-up.

---

## 0. Why this project matters

Scaling RL is mostly about the trade-off between **data** (environment steps, expensive in robotics) and **compute**
(gradient updates, wall-clock). On-policy and off-policy methods sit at opposite ends of it. This project lets you
*measure* that trade-off instead of reciting it.

| Skill | Evidence in this repo |
|---|---|
| Implement modern RL algorithms | `offpolicy.py` (SAC, TD3), `agent.py` (PPO), each with unit tests |
| Correctness engineering | terminal-vs-truncation handling, clipped double-Q, delayed updates, one-step-task learning tests |
| Rigorous comparison | `compare_report.py`: AUC, steps/time-to-threshold, stability, 95% CIs, Welch tests |
| Performance profiling | `time_rollout_s` vs `time_update_s` logged for every run, `profile.png` |
| Multi-task learning | `envs.py` target-velocity tasks, `task_eval.py` cross-task matrix and multi-vs-specialist comparison |
| Systems | export/serve for PPO, SAC and TD3 policies through one interface |

---

## 1. Setup

Python 3.10+ (3.11 recommended), ~3 GB disk, CPU only.

```bash
unzip ppo-vs-sac.zip && cd ppo-vs-sac
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu   # macOS: plain `pip install torch`
pip install -e ".[dev]"
python -c "import torch, gymnasium, mujoco; print(torch.__version__, gymnasium.__version__, mujoco.__version__)"
```

The package is called `algo_compare`. No `make`? Every Makefile target is a plain `python -m algo_compare....` command
shown in this guide.

---

## 2. Repository tour (new in this project)

```
src/algo_compare/
├── offpolicy.py        SACAgent, TD3Agent and the shared off-policy training loop (+ CLI)         NEW
├── offpolicy_nets.py   SAC squashed-Gaussian actor, TD3 deterministic actor, twin Q networks      NEW
├── replay.py           uniform replay buffer (stores *termination* only, see 3b)                  NEW
├── envs.py             target-velocity multi-task wrapper for HalfCheetah                         NEW
├── compare_study.py    algorithms x seeds and multi-task study runner (resumable, parallel)       NEW
├── compare_report.py   metrics, tables, curves vs steps and vs wall-clock, profile plot           NEW
├── task_eval.py        cross-task evaluation + multi-task-vs-specialist report                    NEW
├── config.py           + OffPolicyConfig; PPOConfig gains task_velocities / task_condition        EXTENDED
├── train.py            PPO loop now also logs wall-clock and rollout/update time                  EXTENDED
├── checkpoint.py       one loader for PPO, SAC and TD3 checkpoints                                EXTENDED
└── (agent, gae, networks, normalize, evaluate, export, serve, analysis, ablate: as in project 1)
```

---

## 3. Concepts

### 3a. Why the algorithms behave differently

| | PPO (on-policy) | SAC / TD3 (off-policy) |
|---|---|---|
| Data use | each batch is used for a few epochs, then discarded | every transition goes into a replay buffer and is reused many times |
| Updates per environment step | ~`update_epochs` x (batch passes) / batch, i.e. a handful of passes per rollout | `utd` (default 1) gradient updates *per step* |
| Expected strength | cheap per step, simple, stable | far fewer environment steps to reach a given return |
| Expected cost | many steps needed | much more compute per step; sensitive to hyper-parameters |

**SAC** learns a stochastic policy that maximises return *plus entropy*; it uses two critics (take the minimum to reduce
over-estimation), a slowly moving target critic, a tanh-squashed Gaussian policy, and **automatically tunes the entropy
temperature `alpha`** to hit a target entropy of `-action_dim`.

**TD3** learns a deterministic policy with (1) two critics and the minimum, (2) **target-policy smoothing** (clipped
noise added to the target action), (3) **delayed** actor and target updates (every 2nd critic update), and Gaussian
exploration noise when acting.

### 3b. Termination vs truncation in a replay buffer

A transition that ends because the robot *fell* (termination) has no future value; one that ends because the *time limit*
hit (truncation) does. The replay buffer therefore stores **only `terminated`** as the "done" flag, and stores the true
next observation. (PPO handles the same issue by bootstrapping `V(final_obs)`.) A test trains a critic on terminal
transitions and checks that Q converges to the reward alone.

### 3c. The multi-task design

HalfCheetah must run at a **target forward velocity**:

```
reward = -|x_velocity - target| + reward_ctrl
```

- **Specialist:** trained on one fixed target, not told the target (`--task-condition false`).
- **Multi-task policy:** a target is sampled per episode from a set and appended to the observation (`--task-condition true`).

Returns are negative (closer to 0 is better). Only environments exposing `x_velocity` and `reward_ctrl` in `info`
(HalfCheetah) are supported; anything else raises a clear error.

### 3d. Making the comparison fair (and saying what you did)

Defaults differ between algorithms, deliberately, because they are each algorithm's standard settings:

| Setting | PPO | SAC / TD3 |
|---|---|---|
| Network | 2 x 64 tanh, separate actor/critic | 2 x 256 ReLU |
| Observation / reward normalisation | running normalisation on | none |
| Acting | 1 environment (configurable `--num-envs`) | 1 environment |
| Updates | 10 epochs x 32 minibatches per 2048 steps | 1 update of batch 256 per step |

Three legitimate framings of "equal budget": equal **environment steps** (sample efficiency; the default here), equal
**wall-clock** (`curves_wall.png`), equal **gradient updates**. State which one each claim uses. Do not hand-tune one
algorithm more than the others; if you tune, give each the same tuning budget. Mention that these are default settings,
not tuned optima, and that results can change with tuning.

---

## 4. Experimental design

### Study A: algorithms (`--study algos`)

Variants: `ppo`, `sac`, `td3` on the same environment(s), same seeds, same step budget.

Hypotheses (write them down *before* looking at results):

- **A1.** SAC and TD3 reach a given return in fewer environment steps than PPO.
- **A2.** PPO is cheaper per step, and it may win on wall-clock for a *low* target return but lose for a *high* one.
- **A3.** The off-policy methods spend most of their time in gradient updates; PPO spends most of its time acting/simulating.
- **A4.** Stability (variance across seeds and within late training) differs between algorithms; there is no assumption about which is better.

Metrics (per seed, then averaged over seeds with a 95% t-interval):

| Metric | Definition | Caveat |
|---|---|---|
| `final_return` | mean smoothed training return over the last 10% of steps | training return includes exploration noise |
| `auc` | mean smoothed return over the whole run | rewards learning early; depends on budget length |
| `steps_to_threshold` | first logged step with smoothed return >= threshold | threshold = start + 80% of the best algorithm's improvement; seeds that never reach it are reported as `k/n` |
| `stability_std_last20` | std of the smoothed return over the last 20% | lower = steadier; depends on smoothing window (10 episodes) |
| `wall_minutes`, `sps` | total wall-clock and steps per second | machine-dependent; compare only within one machine |
| `update_fraction` | share of time spent in gradient updates | PPO's GAE counts as update time |
| `time_to_threshold` | wall-clock at first reaching the threshold | |
| Welch p vs reference | `final_return` and `auc` | small n; many comparisons; interpret cautiously |

### Study B: multi-task (`--study multitask`)

Variants per algorithm: `<algo>_multi` (conditioned, all targets) and `<algo>_task_<t>` (specialist for each target).

- **B1.** At an equal per-run step budget, the multi-task policy is worse than a specialist on its own task (interference).
- **B2.** Specialists perform badly on other targets.
- **B3.** The multi-task gap differs between PPO and SAC.

Budget fairness: specialists get `N` steps *each* (so `k*N` in total for `k` tasks); the multi-task policy gets `N` by
default. Use `--multi-factor k` to give it the same total budget, and say which you used.

---

## 5. Step-by-step

### Step 1. Tests

```bash
python -m pytest -q
```

Expect all to pass (see the README for the count). Highlights: replay wraparound; actor outputs always within action
bounds; soft-update interpolation; SAC temperature tuned only when enabled; TD3 actor updates delayed; **terminal
transitions bootstrap nothing**; **both agents learn a one-step task to the correct action**; reproducible training;
SAC/TD3 checkpoints export to TorchScript and serve; metric definitions; the report recovers planted differences.

### Step 2. Measure your machine's speed first

Off-policy training is far slower per step than PPO, and speed varies a lot between machines. Time 5,000 steps:

```bash
python -m algo_compare.offpolicy --algo sac --env-id HalfCheetah-v5 --total-timesteps 5000 --learning-starts 1000 --log-dir runs/speedtest
```

Use the printed `sps` to estimate budgets (`time = steps / sps`). In the build environment (a slow, shared single core)
SAC with 2 x 256 networks and batch 256 ran at only ~60 steps/s including the random warm-up, which is why the
verification study there used small networks (see section 6). Your machine will be faster; measure it.

### Step 3. Pipeline check on an easy task

```bash
make algos-quick       # ppo, sac, td3 on InvertedPendulum, 2 seeds, small off-policy networks
```

This verifies the machinery, not the science: InvertedPendulum is too easy to separate the algorithms reliably.

### Step 4. Study A for real

```bash
python -m algo_compare.compare_study --study algos --env-id HalfCheetah-v5 --algos ppo,sac,td3 \
    --seeds 5 --total-timesteps 300000 --jobs 4 --out runs/algos_hc
python -m algo_compare.compare_report --runs runs/algos_hc --out results/algos_hc --reference ppo
```

`--jobs` should be at most your CPU core count and each job uses one thread. The run is resumable (finished runs are
skipped). Preview with `--seeds 3 --total-timesteps 100000`. Consider also `Hopper-v5` or `Walker2d-v5` (note: these
terminate early, so return mixes performance with survival). 300k steps is a deliberately modest budget; PPO is expected
to need more, which is part of what you are measuring. Add a longer run for the final write-up if you have time.

Outputs in `results/algos_hc/`: `summary.md/.csv`, `per_run.csv`, `curves_steps.png` (return vs environment steps),
`curves_wall.png` (return vs wall-clock), `profile.png` (seconds per 1,000 steps split into acting vs updates).

### Step 5. Study B for real

```bash
python -m algo_compare.compare_study --study multitask --env-id HalfCheetah-v5 --algos ppo,sac \
    --targets 1.0,2.5,4.0 --seeds 3 --total-timesteps 300000 --jobs 4 --out runs/mt_hc
python -m algo_compare.task_eval --runs runs/mt_hc --targets 1.0,2.5,4.0 --episodes 5 --jobs 4 --out results/mt_hc/task_eval.csv
python -m algo_compare.task_eval --report results/mt_hc/task_eval.csv --out results/mt_hc
```

Outputs: `task_matrix.csv` (every policy on every target), `task_comparison.csv` (multi-task minus matching specialist,
with Welch p), `task_summary.md`, `task_heatmap.png`. Choose targets the robot can actually reach; a target far above
what the specialist can achieve makes every policy look equally bad.

### Step 6. Interpret honestly

- **Curves first.** Look at *when* curves separate. Sample-efficiency claims come from `curves_steps.png`; cost claims from
  `curves_wall.png` and `profile.png`.
- **Same machine, same time of day.** Wall-clock numbers are only comparable within one machine and one load level.
  Run parallel jobs with care: `--jobs` changes per-job speed if you oversubscribe cores.
- **Threshold-based metrics are censored.** If some seeds never reach the threshold, say so (`reached_threshold`), and do
  not average a missing value as if it were zero.
- **Small samples.** With 3-5 seeds the intervals are wide and Welch tests weak; with many metrics some will look
  significant by chance. Prefer large, consistent, visible effects.
- **Training vs evaluation return.** Training return includes exploration noise (PPO samples, SAC samples, TD3 adds noise);
  `eval_return` is deterministic.
- **What you cannot conclude:** anything beyond the environments, budgets, networks and default hyper-parameters you used.
- Report null and unexpected results.

Build the write-up from [`RESULTS_TEMPLATE.md`](RESULTS_TEMPLATE.md) using your own numbers.

---

## 6. What was actually verified when this was built

(Details in the README "Verified status".) 59 tests pass. SAC and TD3, with default hyper-parameters, learned **Pendulum-v1**
from about -1,400 to about -150 within 8-10k steps (one seed each), which is the expected behaviour of a correct
implementation. A small three-algorithm study on InvertedPendulum (2 seeds, 20k steps, *small* off-policy networks) ran
through study -> report; in that setting PPO was ahead on return (deterministic evaluation 574 and 813 vs 185 and 83 for SAC
and 58 and 55 for TD3), which is **not** evidence against off-policy methods: the networks were tiny, the budget very short and
the task easy. What it did measure reliably was the *cost* profile: PPO ran at ~900 steps/s with ~44% of the time in
updates, SAC at ~155 steps/s and TD3 at ~265 steps/s with ~90% of the time in updates. The multi-task machinery was run end
to end on untrained policies.

Treat that as a demonstration that the harness works. The questions in section 4 are yours to answer with the real runs.
Not run: full HalfCheetah/Hopper studies, Docker, GitHub Actions, Windows/macOS.

---

## 7. Deployment

SAC and TD3 policies deploy through the same path as PPO: one checkpoint format, one export, one server.

```bash
python -m algo_compare.export --model runs/algos_hc/sac/seed_1/model.pt --out-dir export
PPO_MODEL_PATH=runs/algos_hc/sac/seed_1/model.pt python -m uvicorn --factory algo_compare.serve:create_app --port 8000
curl localhost:8000/health        # reports algo, env_id, obs_dim, act_dim
curl -X POST localhost:8000/act -H 'content-type: application/json' -d '{"observation":[...17 numbers...]}'
```

The exported TorchScript policy takes a raw observation and returns the clipped deterministic action (SAC: tanh of the
mean; TD3: the actor output). A test asserts that the exported output equals the actor's. For multi-task policies the
observation includes the target as its last entry (18 numbers for HalfCheetah).

Docker and CI are as in project 1 (`Dockerfile`, `.github/workflows/ci.yml`). **The Docker build and CI workflow have not
been run in the build environment.**

---

## 8. Extending it

- **Update-to-data ratio.** Sweep `--utd` (1, 2, 4) for SAC: more updates per step buys sample efficiency at a compute cost; plot return vs steps *and* vs wall-clock.
- **Fairness.** Give PPO the larger network, or SAC the smaller one, and see how much of the gap is architecture.
- **Combine with project 2.** Run the robustness evaluation on SAC and TD3 policies: which family is more robust?
- **Bigger multi-task.** More targets, or condition on a learned task embedding; try a longer multi-task budget (`--multi-factor`).
- **Throughput.** Try PPO with `--num-envs 4/8`; profile again.

---

## 9. Interview talking points

1. **Why is SAC more sample-efficient than PPO?** Replay reuses every transition many times; PPO discards data after a few epochs.
2. **Why is it slower in wall-clock?** One gradient update per environment step dominates; I measured the update fraction.
3. **Why two critics?** Taking the minimum reduces Q over-estimation (clipped double-Q).
4. **What does SAC's entropy term do and why tune alpha?** Encourages exploration; tuning targets a fixed entropy so the temperature adapts to reward scale.
5. **What are TD3's three tricks?** Twin critics, target-policy smoothing, delayed policy updates.
6. **Termination vs truncation in replay?** Store only termination as done; truncation still bootstraps.
7. **How did you make the comparison fair?** Same seeds, steps, environment; stated defaults; three budget framings.
8. **What is censoring in steps-to-threshold?** Seeds that never reach it; report k/n and medians.
9. **What did multi-task cost?** Gap between the conditioned policy and specialists, with CIs.
10. **What can't you conclude?** Other environments, longer budgets, tuned hyper-parameters.

---

## 10. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| SAC/TD3 extremely slow | Expected: 1 update per step with 256-unit nets. Measure `sps`, reduce `--total-timesteps`, use `--jobs`, or smaller `--hidden-size`/`--batch-size` (state it) |
| SAC return stuck early | Needs `learning_starts` random steps first (default 5000); check `critic_loss` and `alpha` in `metrics.csv` |
| TD3 returns collapse | Try more `--learning-starts`, smaller `--exploration-noise`; check Q values (`q_mean`) for blow-up |
| `does not expose x_velocity` | Multi-task needs HalfCheetah-v5 |
| `--override` error | Format is `algo.field=value` with algo in `ppo|sac|td3|all` and a real config field |
| All variants look the same | Environment too easy (e.g. InvertedPendulum) or budget too short; use HalfCheetah |
| Parallel runs slower than expected | `--jobs` above your core count oversubscribes the CPU |
| Wide CIs | Expected at 3-5 seeds; add seeds rather than hiding the interval |

---

## 11. Limitations (state these)

- Default hyper-parameters per algorithm; no tuning; networks and normalisation differ between PPO and SAC/TD3.
- One or two environments, modest budgets, 3-5 seeds.
- Wall-clock depends on hardware and load; comparisons hold within one machine only.
- The multi-task benchmark is a single-robot velocity-tracking family, not a diverse multi-robot benchmark.
- Simulated, deterministic-dynamics tasks only; no real hardware.

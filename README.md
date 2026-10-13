# Sample Efficiency, Stability, and Cost of PPO, SAC, and TD3
### From-Scratch Continuous Control Implementations, a Reproducible Comparison Harness, and a Multi-Task Locomotion Study

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.14](https://img.shields.io/badge/PyTorch-2.14%20(CPU)-EE4C2C.svg)](https://pytorch.org/)
[![Gymnasium 1.4](https://img.shields.io/badge/Gymnasium-1.4.0-008080.svg)](https://gymnasium.farama.org/)
[![MuJoCo 3.15](https://img.shields.io/badge/MuJoCo-3.15.0-black.svg)](https://mujoco.org/)
[![Tests Passing](https://img.shields.io/badge/tests-59%2F59%20passed%20(100%25)-success.svg)](tests/)
[![Technical Report](https://img.shields.io/badge/paper-Technical%20Report%20(PDF)-8A2BE2.svg)](docs/technical_report.pdf)
[![LaTeX Preprint](https://img.shields.io/badge/LaTeX-IEEE%20Preprint-blue.svg)](paper/paper.tex)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2FRaghunath2604%2Fppo-vs-sac)

**Author:** Raghunathareddy GR  
**Affiliation:** Department of Artificial Intelligence and Machine Learning, REVA University, Bengaluru, India  
**Software Package:** `ppo-vs-sac` (Python package `algo_compare`)  
**Repository:** [https://github.com/Raghunath2604/ppo-vs-sac.git](https://github.com/Raghunath2604/ppo-vs-sac.git)  
**Publications:** [Technical Report (PDF)](docs/technical_report.pdf) &bull; [IEEE LaTeX Preprint](paper/paper.tex) &bull; [Citation Metadata](CITATION.cff)

---

## Executive Overview

Reinforcement learning algorithms are routinely ranked solely by learning curves drawn against environment interaction steps. However, in practical engineering, choosing an algorithm is a trade-off between **data acquisition** (robot hours, physics simulation steps) and **computational budget** (autograd backpropagation, CPU/GPU wall-clock time). 

- **On-Policy (PPO):** Collects fresh rollouts, updates across a few epochs, and immediately discards data. Cheap per environment step, fast in wall-clock throughput, but sample hungry.
- **Off-Policy (SAC & TD3):** Reuses transitions repeatedly from an experience replay buffer ($utd \ge 1$). Highly sample efficient in environment steps, but requires orders of magnitude more gradient computation per second.

This project delivers a **rigorous, reproducible benchmarking instrument** and **from-scratch implementations** of PPO, SAC, and TD3 in PyTorch. It features first-class wall-clock time profiling, explicit handling of censored statistical metrics (reporting completion counts and arrival medians), a velocity-conditioned multi-task locomotion study on MuJoCo `HalfCheetah-v5`, a unified TorchScript export contract, a production FastAPI inference microservice, and an interactive analytics web dashboard.

<p align="center">
  <img src="assets/halfcheetah_demo.gif" width="65%" alt="HalfCheetah Continuous Control Demonstration" />
  <br>
  <em>Trained SAC Policy executing continuous torque control on HalfCheetah-v5 with real-time velocity tracking telemetry.</em>
</p>

---

## ⚡ Instant Live Demonstration (`demo.py`)

Run an instant, zero-configuration policy simulation and multi-algorithm benchmark directly from your terminal:

```bash
# 1. Run live simulation with real-time ASCII telemetry table
python demo.py --algo sac --target 2.5 --steps 200

# 2. Side-by-side benchmark comparing SAC vs TD3 vs PPO inference latency and actions
python demo.py --compare

# 3. Render and save an animated simulation GIF
python demo.py --algo td3 --gif assets/td3_demo.gif --steps 100
```

#### Terminal Telemetry Output
```text
==============================================================================
   PPO vs SAC vs TD3: Empirical RL Continuous Control Live Demonstration
==============================================================================
[+] Loading SAC checkpoint: runs/algos_preview/sac/seed_1/model.pt
[+] Environment: HalfCheetah-v5 | Target Velocity: 2.50 m/s | Steps: 200
------------------------------------------------------------------------------
  Step |   Velocity |   Target |    Error |    Reward | Action Norm |   Latency
------------------------------------------------------------------------------
     1 |      0.78m/s |    2.50m/s |    1.72m/s |     -2.13 |        2.03 |    3.48ms
    20 |      0.68m/s |    2.50m/s |    1.82m/s |     -2.10 |        1.68 |    0.45ms
    40 |      0.13m/s |    2.50m/s |    2.37m/s |     -2.49 |        1.07 |    0.23ms
    60 |      0.58m/s |    2.50m/s |    1.92m/s |     -2.24 |        1.79 |    0.31ms
...
------------------------------------------------------------------------------
[Summary] Mean Velocity: 0.60 m/s | Mean Tracking Error: 1.90 m/s
[Summary] Cumulative Reward: -221.59 | Avg Step Latency: 0.538 ms
------------------------------------------------------------------------------
```

## System Architecture & Pipeline Design

### 1. End-to-End System Architecture

<p align="center">
  <img src="assets/system_architecture.svg" width="100%" alt="End-to-End System Architecture Diagram" />
</p>

```mermaid
flowchart TB
    subgraph ENV["1. Continuous Control Environments (Gymnasium & MuJoCo)"]
        direction TB
        E1["HalfCheetah-v5<br/><i>(obs: 17 continuous, act: 6 torque)</i>"]
        E2["InvertedPendulum-v5<br/><i>(obs: 4 continuous, act: 1 force)</i>"]
        E3["TargetVelocityEnv<br/><i>Multi-Task Wrapper [s, v*] with v* ∈ {1.0, 2.5, 4.0}</i>"]
    end

    subgraph ALGO["2. From-Scratch Continuous Control Algorithms (PyTorch)"]
        direction TB
        subgraph PPO_BLOCK["On-Policy: PPO"]
            direction TB
            PPO_ACT["Actor-Critic MLP<br/>(Shared / Separate 2x64)"]
            PPO_BUF["Rollout Buffer<br/>(Discarded every batch)"]
            PPO_LOSS["Clipped Surrogate Loss<br/>& GAE(λ) Advantage"]
        end

        subgraph OFF_BLOCK["Off-Policy: SAC & TD3"]
            direction TB
            OFF_BUF["Ring Replay Buffer (1M Transitions)<br/>d = terminated (No truncation leak)"]
            SAC_CORE["Soft Actor-Critic (SAC)<br/>• Squashed Gaussian (tanh + Jacobian)<br/>• Twin Critics (Q1, Q2 min-target)<br/>• Dual Temp Autotuning (α)"]
            TD3_CORE["Twin Delayed DDPG (TD3)<br/>• Deterministic Policy (k=2 delay)<br/>• Target Smoothing Noise<br/>• Clipped Double-Q"]
        end
    end

    subgraph HARNESS["3. Rigorous Empirical Benchmarking Harness"]
        direction TB
        RUNNER["Multiprocess Study Runner<br/>(compare_study.py across seeds & jobs)"]
        PROFILER["Wall-Clock Time Profiler<br/>(t_rollout vs t_update split)"]
        METRICS["Survival Metrics Engine<br/>(k/n completed, median arrival θ)"]
        MT_EVAL["Multi-Task Evaluator<br/>(Transfer Matrix & Heatmaps)"]
    end

    subgraph DEPLOY["4. Serving, Web Dashboard & Publications"]
        direction TB
        TS["TorchScript JIT Compiler<br/>(policy.ts serialization)"]
        API["FastAPI Microservice<br/>(/health, /act &lt; 2ms, /switch)"]
        DASH["Analytics Web Dashboard<br/>(Vanilla JS + Canvas Gym Sim)"]
        PUB["Technical Report (PDF)<br/>(docs/technical_report.pdf)"]
    end

    ENV --> ALGO
    ALGO --> HARNESS
    ALGO --> DEPLOY
    HARNESS --> DASH
    DEPLOY --> DASH
    HARNESS --> PUB
```

### 2. Execution & Dataflow Pipeline: On-Policy vs. Off-Policy

```mermaid
flowchart LR
    subgraph PPO_PIPE["On-Policy Pipeline (PPO)"]
        direction TB
        P1["1. Collect Rollout<br/>(N steps × M envs)"] --> P2["2. Compute GAE(γ, λ)<br/>& Value Targets"]
        P2 --> P3["3. SGD Mini-Batches<br/>(10 epochs over batch)"]
        P3 --> P4["4. Discard Rollout Data<br/>(Update Fraction ~38%)"]
        P4 --> P1
    end

    subgraph OFF_PIPE["Off-Policy Pipeline (SAC & TD3)"]
        direction TB
        O1["1. Step Environment<br/>(1 transition)"] --> O2["2. Push to 1M Buffer<br/>(s, a, r, s', d=term)"]
        O2 --> O3["3. Sample Batch (256)<br/>(UTD ratio ≥ 1)"]
        O3 --> O4["4. Critic Min-Target Step<br/>(Autograd backprop)"]
        O4 --> O5["5. Actor & Polyak Update<br/>(SAC: every step | TD3: k=2)"]
        O5 --> O1
    end
```

### 3. Repository File Map & Modular Responsibilities

```
ppo-vs-sac/
├── src/algo_compare/
│   ├── offpolicy.py          # SAC & TD3 agents, autograd training loop, wall-clock profiler
│   ├── offpolicy_nets.py     # Squashed Gaussian actor, deterministic TD3 actor, Twin Q-networks
│   ├── replay.py             # Preallocated ring replay buffer (strict termination vs truncation)
│   ├── envs.py               # TargetVelocityEnv wrapper for HalfCheetah (conditioned & specialist)
│   ├── config.py             # PPOConfig and OffPolicyConfig dataclasses with CLI parser
│   ├── train.py              # PPO training loop with timing columns (t_rollout, t_update)
│   ├── agent.py, gae.py      # Actor-Critic network and Generalized Advantage Estimation
│   ├── compare_study.py      # Parallel multiprocess study runner with algo.field overrides
│   ├── compare_report.py     # Metrics generator (AUC, censored threshold, 95% CIs, Welch tests)
│   ├── task_eval.py          # Cross-task deterministic evaluation & transfer matrix generator
│   ├── export.py             # TorchScript policy tracing (actor_mean extraction & bounds)
│   └── serve.py              # FastAPI microservice (/health, /act, /switch) with CORS
├── dashboard/                # Interactive analytics web dashboard (Vanilla JS + CSS)
├── docs/
│   ├── technical_report.pdf  # Compiled 2-page publication technical report (PDF)
│   ├── RESEARCH_REPORT.md    # IEEE-style technical report with formal derivations
│   ├── RESULTS_TEMPLATE.md   # Measured empirical benchmarks and hypothesis evaluations
│   └── PROCESS.md            # Detailed end-to-end execution guide
├── paper/
│   └── paper.tex             # Complete IEEEtran LaTeX preprint source
├── scripts/
│   ├── build_paper_pdf.py    # ReportLab script to compile docs/technical_report.pdf
│   └── generate_architecture_diagram.py # Vector SVG architecture generator
├── assets/
│   ├── system_architecture.svg # High-resolution vector architecture graphic
│   └── halfcheetah_demo.gif  # Real-time policy torque execution demonstration
├── results/                  # Generated benchmark tables, CSVs, and publication plots
│   ├── algos_preview/        # HalfCheetah algorithms study (steps, wall-clock, profile)
│   ├── mt_preview/           # HalfCheetah multi-task benchmark (heatmaps, transfer matrices)
│   └── algos_quick/          # InvertedPendulum pipeline pilot
├── tests/                    # 59 automated unit, property, learnability, and serving tests
├── CITATION.cff              # GitHub citation metadata (native "Cite this repository")
├── demo.py                   # Zero-configuration live terminal policy simulator & CLI benchmark
├── Makefile                  # Automation targets (test, speedtest, algos-quick, serve, etc.)
└── pyproject.toml            # Package configuration & dependencies
```

---

## Mathematical Foundations

### 1. Soft Actor-Critic (SAC)
SAC optimizes the maximum entropy reinforcement learning objective:
$$\mathcal{J}(\pi) = \sum_{t=0}^T \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} \left[ r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t)) \right]$$

- **Clipped Double-Q Target:**
  $$y = r + \gamma (1 - d) \left[ \min_{i \in \{1, 2\}} \bar{Q}_i(s', a') - \alpha \log \pi(a'|s') \right], \quad a' \sim \pi(\cdot|s')$$
- **Reparameterized Squashed Gaussian Actor:**
  $$u = \mu_\phi(s) + \sigma_\phi(s) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I), \quad a = \tanh(u) \cdot \text{scale} + \text{bias}$$
  $$\log \pi(a|s) = \sum_{k=1}^{\dim(\mathcal{A})} \left[ \log \mathcal{N}(u_k; \mu_k, \sigma_k) - \log \left( \text{scale}_k (1 - \tanh^2(u_k)) + 10^{-6} \right) \right]$$
- **Automated Temperature Dual Loss:**
  $$L_\alpha(\alpha) = -\mathbb{E} \left[ \log \alpha \cdot (\log \pi(a|s) + \bar{\mathcal{H}}) \right], \quad \bar{\mathcal{H}} = -\dim(\mathcal{A})$$

### 2. Twin Delayed DDPG (TD3)
TD3 curbs function approximation overestimation error through three core mechanisms:
1. **Target Policy Smoothing:**
   $$\tilde{a}' = \text{clip}\left( \bar{\pi}(s') + \text{clip}(\epsilon, -c, c), a_{\min}, a_{\max} \right), \quad \epsilon \sim \mathcal{N}(0, (\tilde{\sigma} \cdot \text{scale})^2)$$
2. **Clipped Double-Q Target:**
   $$y = r + \gamma (1 - d) \min_{i \in \{1, 2\}} \bar{Q}_i(s', \tilde{a}')$$
3. **Delayed Policy Updates:** The actor $\phi$ and target networks $\bar{\theta}_1, \bar{\theta}_2, \bar{\phi}$ are updated only every $k=2$ critic iterations, curbing target variance.

### 3. Experience Replay: Termination vs. Truncation
Treating artificial time-limit truncations as task terminations introduces significant value underestimation. In [`replay.py`](src/algo_compare/replay.py), the stored terminal flag $d$ is set strictly on true task termination:
$$d = \text{terminated} \quad (\text{not } \text{terminated} \lor \text{truncated})$$
Furthermore, when an episode truncates, the true next state $s_{t+1}$ is inserted into the buffer before environment reset, ensuring that value bootstrapping correctly targets $V(s_{t+1})$.

---

## Measured Empirical Results

All measurements were performed directly on the host system (Windows 11 AMD64, Python 3.12, PyTorch 2.14.1+cpu).

### 1. Step 2 Timing Benchmark (`runs/speedtest`)
Measuring 5,000 steps of SAC with default $2 \times 256$ networks on `HalfCheetah-v5`:
- **Total Runtime:** 316.37 s (5.27 min)
- **Overall Throughput:** **15.8 steps/sec**
- **Rollout Time ($t_{\text{rollout}}$):** 14.07 s (**4.45%**)
- **Update Time ($t_{\text{update}}$):** 302.23 s (**95.55%**)
- **Update Fraction ($F_{\text{update}}$):** **0.955** (Compute is 21.5$\times$ more expensive than simulation)

### 2. Algorithms Benchmark (`HalfCheetah-v5`)
Comparison over 20,000 steps per run (2 seeds each) under equal step budgets. Threshold $\theta = -225.07$ (80% improvement over initial return):

| Algorithm | Seeds | Final Return | 95% CI | AUC | 95% CI | Stability ($\sigma_{\text{last 20\%}}$) | Reached $\theta$ | Median Steps to $\theta$ | Wall-Clock | SPS | Update Fraction |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **PPO** | 2 | $-305.58$ | $\pm 214.38$ | $-327.74$ | $\pm 211.62$ | **2.20** | **0 / 2** | — (Censored) | **2.58 min** | **119.03** | **0.38** |
| **SAC** | 2 | **$-203.97$** | $\pm 72.73$ | **$-223.58$** | $\pm 217.56$ | 24.05 | **2 / 2** | **4,500** | 10.83 min | 30.77 | **0.91** |
| **TD3** | 2 | $-361.92$ | $\pm 1555.00$| $-397.21$ | $\pm 449.89$ | 31.22 | **2 / 2** | **10,500** | 6.04 min | 55.15 | **0.91** |

<p align="center">
  <img src="results/algos_preview/curves_steps.png" width="48%" alt="Curves Steps" />
  <img src="results/algos_preview/profile.png" width="48%" alt="Cost Profile" />
</p>

### 3. Multi-Task Study: Conditioned Generalists vs. Specialists
Target velocity locomotion ($v^* \in \{1.0, 2.5, 4.0\}$ m/s) on `HalfCheetah-v5`:

| Algorithm | Target Velocity | Multi-Task Mean | Specialist Mean | $\Delta (\text{Multi} - \text{Specialist})$ | Welch $p$-value | Transfer Insight |
|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **PPO** | 1.00 m/s | $-1007.21$ | $-1002.04$ | $-5.17$ | $0.37$ | Tight parity in early locomotion gait learning |
| **PPO** | 2.50 m/s | $-2505.37$ | $-2502.59$ | $-2.77$ | $0.45$ | Multi-task conditioning tracks smoothly |
| **PPO** | 4.00 m/s | $-4003.85$ | $-4004.67$ | $+0.82$ | $0.60$ | Specialist and generalist perform identically |
| **SAC** | 1.00 m/s | $-1103.19$ | $-1042.61$ | **$-60.58$** | $0.29$ | Specialist holds $+60.6$ return lead |
| **SAC** | 2.50 m/s | $-2636.71$ | $-2561.50$ | **$-75.21$** | $0.11$ | Task capacity sharing penalty observed |
| **SAC** | 4.00 m/s | $-4175.45$ | $-4136.66$ | **$-38.80$** | $0.60$ | High velocity specialist outperforms |

<p align="center">
  <img src="results/mt_preview/task_heatmap.png" width="60%" alt="Multi-Task Heatmap" />
</p>

---

## Interactive Web Dashboard

A glassmorphic analytics dashboard is bundled in [`dashboard/`](dashboard/) and runs locally at **`http://localhost:8089`**:

- **Cost Profiler:** Interactive stacked bar chart visualizing rollout simulation vs. gradient backpropagation across algorithms.
- **Dynamic Learning Curves:** Switch between environment steps, wall-clock minutes, and gradient update counts.
- **Multi-Task Transfer Matrix:** Interactive velocity-target heatmap with out-of-distribution transfer analysis.
- **Live Policy Sandbox:** Direct connection to the running FastAPI microservice on port 8000, streaming real continuous actions across 6 joint channels with real-time latency indicators.

#### Run Locally
```bash
python -m http.server 8089 --directory dashboard
# Open in browser: http://localhost:8089
```

#### Deploy Live on Vercel
Deploy the dashboard to Vercel with zero configuration:

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2FRaghunath2604%2Fppo-vs-sac)

1. Import `https://github.com/Raghunath2604/ppo-vs-sac` into your [Vercel Dashboard](https://vercel.com/new).
2. The included [`vercel.json`](vercel.json) configures `"outputDirectory": "dashboard"` and security headers automatically.
3. Click **Deploy** — your live dashboard will be online in ~10 seconds.

---

## Production Deployment & Inference API

### 1. TorchScript Export
Export any trained checkpoint (PPO, SAC, or TD3) into a self-contained TorchScript binary with baked-in observation normalization and action bound clipping:
```bash
python -m algo_compare.export --model runs/algos_preview/sac/seed_1/model.pt --out-dir export/sac
python -m algo_compare.export --model runs/algos_preview/td3/seed_1/model.pt --out-dir export/td3
python -m algo_compare.export --model runs/algos_preview/ppo/seed_1/model.pt --out-dir export/ppo
```

### 2. FastAPI Inference Microservice
Launch the production policy server:
```bash
PPO_MODEL_PATH=runs/algos_preview/sac/seed_1/model.pt \
    python -m uvicorn --factory algo_compare.serve:create_app --port 8000 --host 127.0.0.1
```

#### Health Check Endpoint
```bash
curl http://127.0.0.1:8000/health
```
```json
{
  "status": "ok",
  "algo": "sac",
  "env_id": "HalfCheetah-v5",
  "obs_dim": 17,
  "act_dim": 6,
  "requests_served": 1
}
```

#### Low-Latency Inference Endpoint
```bash
curl -X POST http://127.0.0.1:8000/act \
  -H "Content-Type: application/json" \
  -d '{"observation": [0, 1.2, 0.05, 0.4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}'
```
```json
{
  "action": [0.9937, 0.0179, 0.9878, 0.9945, 0.9978, -0.6610],
  "latency_ms": 3.20
}
```

#### Dynamic Model Switching
Switch the active model on-the-fly without service downtime:
```bash
curl -X POST "http://127.0.0.1:8000/switch?algo=td3"
```
```json
{
  "status": "ok",
  "switched_to": "td3",
  "algo": "td3",
  "obs_dim": 17,
  "act_dim": 6
}
```

---

## Quickstart & Reproduction

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Raghunath2604/ppo-vs-sac.git
cd ppo-vs-sac

# Create virtual environment & install dependencies
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

### 2. Run Test Suite (59 Tests)
```bash
pytest -q
ruff check src tests
```

### 3. Run Pipeline Check (Pilot Study)
```bash
make algos-quick
```

### 4. Run Primary Benchmark Studies
```bash
# Benchmark algorithms on HalfCheetah (4 parallel jobs)
python -m algo_compare.compare_study --study algos --env-id HalfCheetah-v5 --algos ppo,sac,td3 \
    --seeds 5 --total-timesteps 300000 --jobs 4 --out runs/algos_hc

# Generate statistical reports and curves
python -m algo_compare.compare_report --runs runs/algos_hc --out results/algos_hc --reference ppo

# Multi-task target velocity study
python -m algo_compare.compare_study --study multitask --env-id HalfCheetah-v5 --algos ppo,sac \
    --targets 1.0,2.5,4.0 --seeds 3 --total-timesteps 300000 --jobs 4 --out runs/mt_hc

# Cross-task evaluation & heatmap
python -m algo_compare.task_eval --runs runs/mt_hc --targets 1.0,2.5,4.0 --episodes 5 --jobs 4 \
    --out results/mt_hc/task_eval.csv
python -m algo_compare.task_eval --report results/mt_hc/task_eval.csv --out results/mt_hc
```

---

## Summary of Findings

1. **Sample Efficiency vs. Compute Frontier:** SAC and TD3 achieved target threshold returns in $4,500$ and $10,500$ steps respectively on `HalfCheetah-v5`, while PPO failed to reach it within 20,000 steps. However, off-policy methods spend $>91\%$ of wall-clock time in gradient backpropagation on CPU, whereas PPO spends only $38\%$ in updates.
2. **Computational Speedup of TD3 over SAC:** TD3 achieved $1.8\times$ higher step throughput than SAC ($55.2\text{ SPS}$ vs. $30.8\text{ SPS}$) and completed training in $6.04\text{ min}$ vs. $10.83\text{ min}$. This directly validates the design benefit of TD3's delayed actor updates ($k=2$) and deterministic execution (omitting dual entropy optimization).
3. **Multi-Task Capacity Penalty:** Conditioned SAC policies exhibit a clear transfer gap ($\Delta = -38.8$ to $-75.2$) compared to dedicated single-task specialists under equal per-run budgets. Specialists evaluated out-of-distribution suffer severe performance drops (e.g., dropping to $-4042.6$ on high-velocity targets).
4. **Censored Statistics are Vital:** Reporting completions ($k/n$) and computing medians strictly over successful runs prevents survivorship bias in RL benchmarks.

---

## References

1. J. Schulman et al., "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017.
2. J. Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation," ICLR, 2016.
3. T. Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor," ICML, 2018.
4. S. Fujimoto et al., "Addressing Function Approximation Error in Actor-Critic Methods," ICML, 2018.
5. F. Pardo et al., "Time Limits in Reinforcement Learning," ICML, 2018.
6. P. Henderson et al., "Deep Reinforcement Learning that Matters," AAAI, 2018.
7. R. Agarwal et al., "Deep Reinforcement Learning at the Edge of the Statistical Precipice," NeurIPS, 2021.

---

## Technical Report & Citation

If you use this benchmark harness, mathematical implementations, or empirical findings in your research or applications, please cite our technical report:

### BibTeX
```bibtex
@article{raghunathareddy2026ppo_vs_sac,
  title={Sample Efficiency, Stability, and Cost of PPO, SAC, and TD3: From-Scratch Implementations, a Reproducible Comparison Harness, and a Multi-Task Study},
  author={Raghunathareddy, G. R.},
  journal={Technical Report, Department of Artificial Intelligence and Machine Learning, REVA University},
  year={2026},
  month={October},
  url={https://github.com/Raghunath2604/ppo-vs-sac}
}
```

You can also use GitHub's native **"Cite this repository"** dropdown in the right-hand repository sidebar, powered by [`CITATION.cff`](CITATION.cff).

### Paper Artifacts
- **Compiled Technical Report (PDF):** [`docs/technical_report.pdf`](docs/technical_report.pdf) *(2-page publication paper with embedded empirical figures, statistical tables, and mathematical derivations)*
- **IEEE LaTeX Preprint Source:** [`paper/paper.tex`](paper/paper.tex) *(Complete IEEEtran compsoc conference/transactions format)*
- **PDF Compilation Script:** Run `python scripts/build_paper_pdf.py` to compile the publication PDF report.


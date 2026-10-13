"""Hyperparameters for PPO, as a single dataclass that the CLI is generated from."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields


@dataclass
class PPOConfig:
    # --- experiment ---
    env_id: str = "HalfCheetah-v5"
    seed: int = 1
    total_timesteps: int = 1_000_000
    log_dir: str = "runs/default"
    eval_episodes: int = 10

    # --- rollout ---
    num_envs: int = 1
    num_steps: int = 2048  # steps per env per rollout; batch = num_envs * num_steps

    # --- optimisation ---
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    update_epochs: int = 10
    num_minibatches: int = 32
    max_grad_norm: float = 0.5
    target_kl: float = 0.0  # 0 disables early stopping on KL

    # --- PPO objective ---
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    norm_adv: bool = True

    # --- implementation details that matter (ablated in the study) ---
    norm_obs: bool = True
    norm_reward: bool = True
    hidden_size: int = 64

    # --- multi-task (target-velocity HalfCheetah): empty = the standard environment ---
    task_velocities: str = ""  # e.g. "1.0,2.5,4.0"; sampled uniformly per episode
    task_condition: bool = True  # append the target to the observation (False = specialist policy)

    # --- misc ---
    torch_threads: int = 1

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches

    @property
    def num_iterations(self) -> int:
        return max(1, self.total_timesteps // self.batch_size)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> PPOConfig:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class OffPolicyConfig:
    """Hyper-parameters shared by SAC and TD3 (standard defaults from the original papers / reference code)."""

    algo: str = "sac"  # "sac" or "td3"
    env_id: str = "HalfCheetah-v5"
    seed: int = 1
    total_timesteps: int = 300_000
    log_dir: str = "runs/default"
    eval_episodes: int = 10
    log_interval: int = 2000  # environment steps between metric rows

    hidden_size: int = 256
    batch_size: int = 256
    buffer_size: int = 1_000_000
    learning_starts: int = 5000  # uniform-random actions before this many steps
    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    utd: int = 1  # gradient updates per environment step (update-to-data ratio)

    # SAC
    autotune_alpha: bool = True
    init_alpha: float = 1.0
    # TD3
    policy_delay: int = 2
    exploration_noise: float = 0.1  # std, as a fraction of the action range
    target_noise: float = 0.2
    noise_clip: float = 0.5

    task_velocities: str = ""
    task_condition: bool = True
    torch_threads: int = 1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> OffPolicyConfig:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

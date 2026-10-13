import numpy as np
import pytest

from algo_compare.envs import TargetVelocityEnv, make_env, make_env_for, parse_targets

ENV = "HalfCheetah-v5"


def test_parse_targets():
    assert parse_targets("1.0, 2.5,4") == [1.0, 2.5, 4.0]
    assert parse_targets("") == []


def test_reward_is_velocity_tracking_plus_control_cost():
    env = make_env(ENV, [2.0], condition=False)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(20):
        _, r, _, _, info = env.step(rng.uniform(-1, 1, size=6).astype(np.float32))
        assert np.isclose(r, -abs(info["x_velocity"] - 2.0) + info["reward_ctrl"])
        assert info["target_velocity"] == 2.0


def test_conditioning_appends_target_to_observation():
    plain = make_env(ENV, [1.0, 3.0], condition=False)
    cond = make_env(ENV, [1.0, 3.0], condition=True)
    obs_p, _ = plain.reset(seed=0)
    obs_c, info = cond.reset(seed=0)
    assert obs_c.shape[0] == obs_p.shape[0] + 1 and cond.observation_space.shape == obs_c.shape
    assert obs_c[-1] == info["target_velocity"]
    assert cond.observation_space.contains(obs_c)
    obs_c2, *_ = cond.step(np.zeros(6, dtype=np.float32))
    assert obs_c2[-1] == info["target_velocity"]  # constant within an episode


def test_targets_are_sampled_per_episode_and_reproducibly():
    env = make_env(ENV, [1.0, 2.5, 4.0], seed=0)
    seen = {env.reset()[1]["target_velocity"] for _ in range(60)}
    assert seen == {1.0, 2.5, 4.0}
    a = [make_env(ENV, [1.0, 2.5, 4.0]).reset(seed=s)[1]["target_velocity"] for s in range(5)]
    b = [make_env(ENV, [1.0, 2.5, 4.0]).reset(seed=s)[1]["target_velocity"] for s in range(5)]
    assert a == b


def test_fixed_target_overrides_sampling():
    env = make_env(ENV, [1.0, 2.5, 4.0], fixed_target=2.5)
    assert {env.reset()[1]["target_velocity"] for _ in range(20)} == {2.5}


def test_unsupported_environment_is_rejected():
    with pytest.raises(ValueError):
        make_env("InvertedPendulum-v5", [1.0])
    with pytest.raises(ValueError):
        TargetVelocityEnv(make_env(ENV), [])


def test_make_env_for_uses_config_fields():
    from algo_compare.config import OffPolicyConfig, PPOConfig

    plain = make_env_for(PPOConfig(env_id=ENV))
    assert plain.observation_space.shape == (17,)
    multi = make_env_for(OffPolicyConfig(env_id=ENV, task_velocities="1,2", task_condition=True))
    assert multi.observation_space.shape == (18,)
    spec = make_env_for(PPOConfig(env_id=ENV, task_velocities="1", task_condition=False))
    assert spec.observation_space.shape == (17,)

import numpy as np

from algo_compare.normalize import ObsNormalizer, RewardScaler, RunningMeanStd


def test_running_stats_match_numpy():
    rng = np.random.default_rng(0)
    chunks = [rng.normal(3.0, 2.0, size=(n, 4)) for n in (5, 17, 100, 1)]
    rms = RunningMeanStd((4,), epsilon=1e-12)
    for c in chunks:
        rms.update(c)
    allx = np.concatenate(chunks)
    np.testing.assert_allclose(rms.mean, allx.mean(0), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(rms.var, allx.var(0), rtol=1e-6, atol=1e-8)


def test_obs_normalizer_standardises_and_clips():
    rng = np.random.default_rng(1)
    norm = ObsNormalizer((3,), clip=5.0)
    data = rng.normal(10.0, 3.0, size=(5000, 3))
    norm.update(data)
    out = norm(data)
    assert abs(out.mean()) < 0.05 and abs(out.std() - 1.0) < 0.05
    assert norm(np.full((1, 3), 1e9)).max() == 5.0


def test_reward_scaler_keeps_sign_and_shrinks_scale():
    scaler = RewardScaler(num_envs=2, gamma=0.99)
    rng = np.random.default_rng(2)
    outs = []
    for _ in range(500):
        r = rng.normal(100.0, 10.0, size=2)
        outs.append(scaler(r, np.zeros(2, dtype=bool)))
    outs = np.array(outs)
    assert (outs > 0).all()
    assert outs[-50:].mean() < 100.0 / 5  # much smaller than the raw reward scale

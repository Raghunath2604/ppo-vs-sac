import numpy as np

from algo_compare.gae import bootstrap_truncated_rewards, compute_gae


def naive_gae(rewards, values, dones, next_value, gamma, lam):
    """Direct O(T^2) definition: A_t = sum_l (gamma*lam)^l * delta_{t+l}, cut at episode ends."""
    T, N = rewards.shape
    adv = np.zeros((T, N))
    for n in range(N):
        for t in range(T):
            total, disc = 0.0, 1.0
            for k in range(t, T):
                nv = next_value[n] if k == T - 1 else values[k + 1, n]
                delta = rewards[k, n] + gamma * nv * (1 - dones[k, n]) - values[k, n]
                total += disc * delta
                disc *= gamma * lam
                if dones[k, n]:
                    break
            adv[t, n] = total
    return adv


def test_matches_naive_definition():
    rng = np.random.default_rng(0)
    T, N = 40, 3
    r, v = rng.normal(size=(T, N)), rng.normal(size=(T, N))
    d = (rng.random((T, N)) < 0.1).astype(float)
    nv = rng.normal(size=N)
    adv, ret = compute_gae(r, v, d, nv, 0.99, 0.95)
    np.testing.assert_allclose(adv, naive_gae(r, v, d, nv, 0.99, 0.95), atol=1e-9)
    np.testing.assert_allclose(ret, adv + v)


def test_lambda_one_is_discounted_return_minus_value():
    rng = np.random.default_rng(1)
    T = 10
    r, v = rng.normal(size=(T, 1)), rng.normal(size=(T, 1))
    nv = np.array([0.7])
    gamma = 0.9
    adv, ret = compute_gae(r, v, np.zeros((T, 1)), nv, gamma, 1.0)
    expected = np.array(
        [sum(gamma**k * r[t + k, 0] for k in range(T - t)) + gamma ** (T - t) * nv[0] for t in range(T)]
    )
    np.testing.assert_allclose(ret[:, 0], expected, atol=1e-9)


def test_lambda_zero_is_one_step_td():
    rng = np.random.default_rng(2)
    T = 8
    r, v = rng.normal(size=(T, 2)), rng.normal(size=(T, 2))
    nv = rng.normal(size=2)
    adv, _ = compute_gae(r, v, np.zeros((T, 2)), nv, 0.99, 0.0)
    next_v = np.vstack([v[1:], nv[None]])
    np.testing.assert_allclose(adv, r + 0.99 * next_v - v, atol=1e-9)


def test_terminal_blocks_bootstrapping():
    r = np.array([[1.0], [2.0]])
    v = np.array([[0.5], [0.5]])
    d = np.array([[1.0], [0.0]])
    adv, _ = compute_gae(r, v, d, np.array([100.0]), 0.99, 0.95)
    assert abs(adv[0, 0] - (1.0 - 0.5)) < 1e-9  # no bootstrap across the episode boundary


def test_truncation_bootstrap_only_for_pure_truncation():
    rewards = np.array([1.0, 1.0, 1.0])
    truncated = np.array([True, True, False])
    terminated = np.array([False, True, False])
    out = bootstrap_truncated_rewards(rewards, truncated, terminated, np.array([10.0, 10.0, 10.0]), 0.5)
    np.testing.assert_allclose(out, [6.0, 1.0, 1.0])

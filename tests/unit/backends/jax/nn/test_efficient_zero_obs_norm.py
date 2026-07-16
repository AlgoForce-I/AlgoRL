"""Observation normalization stability tests."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from algorl.backends.jax.nn.efficientzero.obs_norm import (
    compute_tentative_obs_stats,
    compute_tentative_obs_stats_jax,
    update_mean_var_count_from_moments,
    update_representation_obs_stats,
    update_representation_obs_stats_jax,
    with_representation_obs_stats,
)


def test_obs_norm_count_stays_accurate_past_float32_integer_limit() -> None:
    """Simulate ~1M batched updates without float32 count precision loss."""
    obs_dim = 17
    mean = np.zeros((obs_dim,), dtype=np.float32)
    var = np.ones((obs_dim,), dtype=np.float32)
    count = 1e3
    batch_count = 1536

    for _ in range(1_000):
        batch_mean = np.random.randn(obs_dim).astype(np.float32) * 0.01
        batch_var = np.full((obs_dim,), 0.05, dtype=np.float32)
        mean, var, count = update_mean_var_count_from_moments(
            mean,
            var,
            count,
            batch_mean,
            batch_var,
            batch_count,
        )

    expected_count = 1e3 + 1_000 * batch_count
    assert count == expected_count
    assert np.all(np.isfinite(mean))
    assert np.all(np.isfinite(var))
    assert float(np.min(var)) > 0.0


def test_obs_norm_keeps_exact_python_count() -> None:
    params = {
        "representation_model": {
            "running_mean": np.zeros((4,), dtype=np.float32),
            "running_var": np.ones((4,), dtype=np.float32),
        }
    }
    obs = np.ones((8, 4), dtype=np.float32)
    updated, count = update_representation_obs_stats(params, obs, count=1_500_000_000)
    assert count == 1_500_000_008
    assert "running_count" not in updated["representation_model"]
    assert np.all(np.isfinite(updated["representation_model"]["running_var"]))


def test_jax_obs_norm_preserves_int32_count_in_carry() -> None:
    params = {
        "representation_model": {
            "running_mean": jnp.zeros((4,), dtype=jnp.float32),
            "running_var": jnp.ones((4,), dtype=jnp.float32),
        }
    }
    obs = jnp.ones((8, 4), dtype=jnp.float32)
    updated, count = update_representation_obs_stats_jax(params, obs, count=jnp.int32(1_500_000_000))
    assert int(count) == 1_500_000_008
    assert count.dtype == jnp.int32
    assert "running_count" not in updated["representation_model"]


def test_tentative_obs_stats_match_committed_update() -> None:
    params = {
        "representation_model": {
            "running_mean": np.zeros((4,), dtype=np.float32),
            "running_var": np.ones((4,), dtype=np.float32),
        }
    }
    obs = np.random.randn(16, 4).astype(np.float32)
    tentative_mean, tentative_var, tentative_count = compute_tentative_obs_stats(
        params,
        obs,
        count=1000,
    )
    updated, committed_count = update_representation_obs_stats(params, obs, count=1000)
    assert committed_count == tentative_count
    np.testing.assert_allclose(
        updated["representation_model"]["running_mean"],
        tentative_mean,
    )
    np.testing.assert_allclose(
        updated["representation_model"]["running_var"],
        tentative_var,
    )


def test_with_representation_obs_stats_does_not_mutate_input() -> None:
    params = {
        "representation_model": {
            "running_mean": np.zeros((4,), dtype=np.float32),
            "running_var": np.ones((4,), dtype=np.float32),
        }
    }
    tentative_mean = np.full((4,), 2.0, dtype=np.float32)
    tentative_var = np.full((4,), 3.0, dtype=np.float32)
    updated = with_representation_obs_stats(
        params,
        mean=tentative_mean,
        var=tentative_var,
    )
    np.testing.assert_allclose(params["representation_model"]["running_mean"], 0.0)
    np.testing.assert_allclose(updated["representation_model"]["running_mean"], 2.0)
    np.testing.assert_allclose(updated["representation_model"]["running_var"], 3.0)


def test_jax_tentative_obs_stats_match_committed_update() -> None:
    params = {
        "representation_model": {
            "running_mean": jnp.zeros((4,), dtype=jnp.float32),
            "running_var": jnp.ones((4,), dtype=jnp.float32),
        }
    }
    obs = jnp.ones((8, 4), dtype=jnp.float32)
    tentative_mean, tentative_var, tentative_count = compute_tentative_obs_stats_jax(
        params,
        obs,
        count=jnp.int32(1000),
    )
    updated, committed_count = update_representation_obs_stats_jax(
        params,
        obs,
        count=jnp.int32(1000),
    )
    assert int(committed_count) == int(tentative_count)
    np.testing.assert_allclose(
        updated["representation_model"]["running_mean"],
        tentative_mean,
    )
    np.testing.assert_allclose(
        updated["representation_model"]["running_var"],
        tentative_var,
    )

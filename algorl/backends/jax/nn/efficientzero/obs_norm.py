"""Running observation statistics for EfficientZero-V2-aligned representation nets."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np


def _observation_batch_matrix(observations: np.ndarray) -> np.ndarray:
    if observations.ndim == 3:
        return observations.reshape(-1, observations.shape[-1])
    if observations.ndim == 2:
        return observations
    if observations.ndim == 1:
        return observations.reshape(1, -1)
    raise ValueError(f"Unsupported observation batch shape {observations.shape!r}.")


def update_mean_var_count_from_moments(
    mean: np.ndarray,
    var: np.ndarray,
    count: float,
    batch_mean: np.ndarray,
    batch_var: np.ndarray,
    batch_count: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Online observation mean/variance update (Welford merge)."""
    mean64 = np.asarray(mean, dtype=np.float64)
    var64 = np.asarray(var, dtype=np.float64)
    count64 = float(count)
    batch_count64 = float(batch_count)
    batch_mean64 = np.asarray(batch_mean, dtype=np.float64)
    batch_var64 = np.asarray(batch_var, dtype=np.float64)

    delta = batch_mean64 - mean64
    tot_count = count64 + batch_count64
    new_mean = mean64 + delta * batch_count64 / tot_count
    m_a = var64 * count64
    m_b = batch_var64 * batch_count64
    m2 = m_a + m_b + np.square(delta) * count64 * batch_count64 / tot_count
    new_var = m2 / tot_count
    return new_mean.astype(np.float32), new_var.astype(np.float32), float(tot_count)


def compute_tentative_obs_stats(
    params: dict[str, Any],
    observations: np.ndarray,
    *,
    count: float | int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Merge a batch into current obs stats without mutating ``params``."""
    obs = np.asarray(observations, dtype=np.float32)
    flat = np.asarray(_observation_batch_matrix(obs), dtype=np.float32)
    if flat.shape[0] == 0:
        rep_params = params["representation_model"]
        mean = np.asarray(
            rep_params.get("running_mean", np.zeros(flat.shape[1], dtype=np.float32))
        )
        var = np.asarray(rep_params.get("running_var", np.ones(flat.shape[1], dtype=np.float32)))
        return mean, var, int(count)

    rep_params = params["representation_model"]
    mean = np.asarray(rep_params.get("running_mean", np.zeros(flat.shape[1], dtype=np.float32)))
    var = np.asarray(rep_params.get("running_var", np.ones(flat.shape[1], dtype=np.float32)))
    batch_mean = flat.mean(axis=0)
    batch_var = flat.var(axis=0)
    new_mean, new_var, new_count = update_mean_var_count_from_moments(
        mean,
        var,
        float(count),
        batch_mean,
        batch_var,
        int(flat.shape[0]),
    )
    return new_mean, np.maximum(new_var, np.float32(0.0)), int(new_count)


def with_representation_obs_stats(
    params: dict[str, Any],
    *,
    mean: np.ndarray | jnp.ndarray,
    var: np.ndarray | jnp.ndarray,
) -> dict[str, Any]:
    """Return ``params`` with the given observation normalization stats."""
    rep_params = dict(params["representation_model"])
    updated = dict(params)
    updated["representation_model"] = {
        **rep_params,
        "running_mean": mean,
        "running_var": var,
    }
    return updated


def update_representation_obs_stats(
    params: dict[str, Any],
    observations: np.ndarray,
    *,
    count: float | int,
) -> tuple[dict[str, Any], int]:
    """Update ``running_mean`` / ``running_var`` from a training batch of observations."""
    new_mean, new_var, new_count = compute_tentative_obs_stats(
        params,
        observations,
        count=count,
    )
    return with_representation_obs_stats(params, mean=new_mean, var=new_var), new_count


def _observation_batch_matrix_jax(observations: jnp.ndarray) -> jnp.ndarray:
    if observations.ndim == 3:
        return observations.reshape(-1, observations.shape[-1])
    if observations.ndim == 2:
        return observations
    return observations.reshape(1, -1)


def _merge_obs_stats_jax(
    mean: jnp.ndarray,
    var: jnp.ndarray,
    count: jnp.ndarray,
    flat: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    batch_count = jnp.int32(flat.shape[0])
    batch_mean = jnp.mean(flat, axis=0)
    batch_var = jnp.var(flat, axis=0)

    tot_count = count + batch_count
    tot_count_f = tot_count.astype(jnp.float32)
    batch_count_f = batch_count.astype(jnp.float32)
    count_f = count.astype(jnp.float32)

    delta = batch_mean - mean
    new_mean = mean + delta * batch_count_f / tot_count_f
    m_a = var * count_f
    m_b = batch_var * batch_count_f
    m2 = m_a + m_b + jnp.square(delta) * count_f * batch_count_f / tot_count_f
    new_var = jnp.maximum(m2 / tot_count_f, jnp.float32(0.0))
    return new_mean, new_var, tot_count


def compute_tentative_obs_stats_jax(
    params: dict[str, Any],
    observations: jnp.ndarray,
    count: jnp.ndarray | int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Merge a batch into current obs stats without mutating ``params``."""
    rep_params = params["representation_model"]
    flat = _observation_batch_matrix_jax(observations)
    mean = jnp.asarray(rep_params["running_mean"], dtype=jnp.float32)
    var = jnp.asarray(rep_params["running_var"], dtype=jnp.float32)
    count_i = jnp.asarray(count, dtype=jnp.int32)
    return _merge_obs_stats_jax(mean, var, count_i, flat)


def update_representation_obs_stats_jax(
    params: dict[str, Any],
    observations: jnp.ndarray,
    *,
    count: jnp.ndarray | int,
) -> tuple[dict[str, Any], jnp.ndarray]:
    """JAX obs stats update; ``count`` is carried separately from trainable params."""
    new_mean, new_var, new_count = compute_tentative_obs_stats_jax(
        params,
        observations,
        count,
    )
    return with_representation_obs_stats(params, mean=new_mean, var=new_var), new_count

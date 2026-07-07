"""Running observation statistics for EfficientZero-V2-aligned representation nets."""

from __future__ import annotations

from typing import Any

import numpy as np


def update_mean_var_count_from_moments(
    mean: np.ndarray,
    var: np.ndarray,
    count: float,
    batch_mean: np.ndarray,
    batch_var: np.ndarray,
    batch_count: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Online observation mean/variance update (Welford merge)."""
    delta = batch_mean - mean
    tot_count = count + batch_count
    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = m2 / tot_count
    return new_mean.astype(np.float32), new_var.astype(np.float32), float(tot_count)


def update_representation_obs_stats(
    params: dict[str, Any],
    observations: np.ndarray,
    *,
    initial_count: float = 1e3,
) -> dict[str, Any]:
    """Update ``running_mean`` / ``running_var`` from a training batch of observations."""
    obs = np.asarray(observations, dtype=np.float32)
    if obs.ndim == 3:
        flat = obs.reshape(-1, obs.shape[-1])
    elif obs.ndim == 2:
        flat = obs
    elif obs.ndim == 1:
        flat = obs.reshape(1, -1)
    else:
        raise ValueError(f"Unsupported observation batch shape {obs.shape!r}.")

    if flat.shape[0] == 0:
        return params

    rep_params = dict(params["representation_model"])
    mean = np.asarray(rep_params.get("running_mean", np.zeros(flat.shape[1], dtype=np.float32)))
    var = np.asarray(rep_params.get("running_var", np.ones(flat.shape[1], dtype=np.float32)))
    count = float(rep_params.get("running_count", initial_count))

    batch_mean = flat.mean(axis=0)
    batch_var = flat.var(axis=0)
    new_mean, new_var, new_count = update_mean_var_count_from_moments(
        mean,
        var,
        count,
        batch_mean,
        batch_var,
        int(flat.shape[0]),
    )

    updated = dict(params)
    updated["representation_model"] = {
        **rep_params,
        "running_mean": new_mean,
        "running_var": new_var,
        "running_count": np.asarray(new_count, dtype=np.float32),
    }
    return updated

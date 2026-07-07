"""Batch reanalyze for EfficientZero training (policy MCTS + value inference)."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from algorl.agents.configs import EfficientZeroConfig
from algorl.backends.jax.nn.efficientzero.model import EfficientZero as EfficientZeroNetwork
from algorl.backends.jax.nn.efficientzero.model import Params
from algorl.backends.jax.planners.efficientzero import EfficientZeroPlanner
from algorl.backends.jax.planners.mcts.core import validate_search_batch_size


def effective_reanalyze_search_batch_size(config: EfficientZeroConfig) -> int:
    """MCTS width for training-time policy reanalyze (independent of rollout width)."""
    if config.reanalyze_search_batch_size is not None:
        return validate_search_batch_size(config.reanalyze_search_batch_size)
    return validate_search_batch_size(config.search_batch_size)


def mcts_temperature(config: EfficientZeroConfig, trained_steps: int) -> float:
    """MCTS temperature schedule for reanalyze."""
    if not config.change_temperature:
        return 1.0
    total = max(1, config.total_training_steps)
    if trained_steps < 0.5 * total:
        return 1.0
    if trained_steps < 0.75 * total:
        return 0.5
    return 0.25


def batch_initial_values(
    model: EfficientZeroNetwork,
    params: Params,
    observations: np.ndarray,
    *,
    rng_key: jax.Array,
    mini_batch_size: int = 256,
) -> np.ndarray:
    """Batched initial-inference value extraction for reanalyze."""
    obs = np.asarray(observations, dtype=np.float32)
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    batch_size = obs.shape[0]
    outputs: list[np.ndarray] = []
    slices = int(np.ceil(batch_size / mini_batch_size))
    key = rng_key
    for slice_index in range(slices):
        start = mini_batch_size * slice_index
        end = mini_batch_size * (slice_index + 1)
        chunk = jnp.asarray(obs[start:end], dtype=jnp.float32)
        key, infer_key = jax.random.split(key)
        infer_keys = jax.random.split(infer_key, chunk.shape[0])

        def infer_one(single_obs: jnp.ndarray, subkey: jnp.ndarray) -> jnp.ndarray:
            _, value, _ = model.initial_inference(
                params,
                single_obs,
                training=False,
                rng=subkey,
            )
            return jnp.asarray(value, dtype=jnp.float32)

        # Vector networks reshape obs to a flat vector and only support one sample
        # per forward pass; vmap batches the JIT call without changing numerics.
        values = np.asarray(jax.vmap(infer_one)(chunk, infer_keys), dtype=np.float32)
        outputs.append(values.reshape(-1))
    return np.concatenate(outputs, axis=0)


def _as_host_array(value: object, *, dtype: np.dtype = np.float32) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


def _as_candidate_matrix(candidates: object) -> np.ndarray:
    root_candidates = _as_host_array(candidates)
    if root_candidates.ndim == 1:
        return root_candidates.reshape(-1, 1)
    if root_candidates.ndim == 2:
        return root_candidates
    return root_candidates.reshape(root_candidates.shape[0], -1)


def _extract_search_outputs(
    result: object,
    index: int,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    single = result.to_single(index)
    weights = _as_host_array(single.action_weights).reshape(-1)
    candidates = _as_candidate_matrix(single.root_candidates)
    best = _as_host_array(single.action).reshape(-1)
    return weights, float(single.root_value), candidates, best


def _pad_trajectory_steps(
    step_candidates: list[np.ndarray],
    step_policies: list[np.ndarray],
    step_best: list[np.ndarray],
    step_values: list[float],
    *,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    max_candidates = max(candidate.shape[0] for candidate in step_candidates)
    action_dim = step_candidates[0].shape[1]
    padded_candidates = np.zeros((window, max_candidates, action_dim), dtype=np.float32)
    padded_policies = np.zeros((window, max_candidates), dtype=np.float32)
    padded_best = np.zeros((window, action_dim), dtype=np.float32)
    search_values = np.zeros((window,), dtype=np.float32)

    for step_index, candidates in enumerate(step_candidates):
        count_i = candidates.shape[0]
        padded_candidates[step_index, :count_i] = candidates
        if count_i < max_candidates:
            padded_candidates[step_index, count_i:] = candidates[-1]
        policy = step_policies[step_index]
        if policy.shape[0] == count_i:
            padded_policies[step_index, :count_i] = policy
        elif policy.shape[0] == max_candidates:
            padded_policies[step_index] = policy
        else:
            padded_policies[step_index, : policy.shape[0]] = policy
        padded_best[step_index] = step_best[step_index]
        search_values[step_index] = step_values[step_index]

    return padded_policies, search_values, padded_candidates, padded_best


def reanalyze_policy_batch(
    planner: EfficientZeroPlanner,
    observations: np.ndarray,
    *,
    params: Params,
    reanalyze_count: int,
    temperature: float = 1.0,
    search_batch_size: int | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run fresh MCTS on the first ``reanalyze_count`` batch items (policy reanalyze only).

    Returns ``policy_targets``, ``search_values``, ``policy_candidates``, and
    ``best_actions`` with shape ``[B, K+1, ...]``.
    """
    batch_size, window, _ = observations.shape
    count = min(int(reanalyze_count), batch_size)
    if count <= 0:
        raise ValueError("reanalyze_count must be > 0 when reanalyze is enabled.")

    if search_batch_size is None:
        search_batch_size = max(1, planner.search_batch_size)
    else:
        search_batch_size = max(1, validate_search_batch_size(search_batch_size))
    work_items = [(batch_index, step_index) for batch_index in range(count) for step_index in range(window)]
    total_progress = len(work_items)

    step_policies: dict[tuple[int, int], np.ndarray] = {}
    step_values: dict[tuple[int, int], float] = {}
    step_candidates: dict[tuple[int, int], np.ndarray] = {}
    step_best: dict[tuple[int, int], np.ndarray] = {}
    completed_progress = 0

    for chunk_start in range(0, total_progress, search_batch_size):
        chunk_items = work_items[chunk_start : chunk_start + search_batch_size]
        obs_chunk = np.stack(
            [_as_host_array(observations[batch_index, step_index]) for batch_index, step_index in chunk_items],
            axis=0,
        )
        result = planner.search_batch(
            obs_chunk,
            deterministic=False,
            temperature=temperature,
            params=params,
            use_self_play=False,
        )
        for lane, (batch_index, step_index) in enumerate(chunk_items):
            weights, value, candidates, best = _extract_search_outputs(result, lane)
            key = (batch_index, step_index)
            step_policies[key] = weights
            step_values[key] = value
            step_candidates[key] = candidates
            step_best[key] = best

        completed_progress += len(chunk_items)
        if on_progress is not None:
            on_progress(completed_progress, total_progress)

    policy_targets = np.zeros((batch_size, window, 0), dtype=np.float32)
    search_values = np.zeros((batch_size, window), dtype=np.float32)
    policy_candidates: list[np.ndarray] = []
    best_actions = np.zeros((batch_size, window, 0), dtype=np.float32)

    for batch_index in range(count):
        traj_policies = [step_policies[(batch_index, step_index)] for step_index in range(window)]
        traj_values = [step_values[(batch_index, step_index)] for step_index in range(window)]
        traj_candidates = [step_candidates[(batch_index, step_index)] for step_index in range(window)]
        traj_best = [step_best[(batch_index, step_index)] for step_index in range(window)]
        padded_policies, padded_values, padded_candidates, padded_best = _pad_trajectory_steps(
            traj_candidates,
            traj_policies,
            traj_best,
            traj_values,
            window=window,
        )
        search_values[batch_index] = padded_values
        policy_candidates.append(padded_candidates)
        if policy_targets.shape[2] == 0:
            policy_targets = np.zeros(
                (batch_size, window, padded_policies.shape[1]),
                dtype=np.float32,
            )
            best_actions = np.zeros((batch_size, window, padded_best.shape[1]), dtype=np.float32)
        policy_targets[batch_index] = padded_policies
        best_actions[batch_index] = padded_best

    if not policy_candidates:
        return (
            policy_targets,
            search_values,
            np.zeros((batch_size, window, 1, 1), dtype=np.float32),
            best_actions,
        )

    max_candidates = max(item.shape[1] for item in policy_candidates)
    action_dim = policy_candidates[0].shape[2]
    stacked = np.zeros((batch_size, window, max_candidates, action_dim), dtype=np.float32)
    for batch_index, candidates in enumerate(policy_candidates):
        stacked[batch_index, :, : candidates.shape[1], :] = candidates
    return policy_targets, search_values, stacked, best_actions


def reanalyze_training_batch(
    planner: EfficientZeroPlanner,
    observations: np.ndarray,
    *,
    params: Params,
    reanalyze_count: int,
    temperature: float = 1.0,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Backward-compatible alias for :func:`reanalyze_policy_batch`."""
    return reanalyze_policy_batch(
        planner,
        observations,
        params=params,
        reanalyze_count=reanalyze_count,
        temperature=temperature,
        on_progress=on_progress,
    )

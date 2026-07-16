"""Batch reanalyze for EfficientZero training (policy MCTS + value inference)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from algorl.agents.configs import EfficientZeroConfig
from algorl.backends.jax.nn.efficientzero.model import EfficientZero as EfficientZeroNetwork
from algorl.backends.jax.nn.efficientzero.model import Params
from algorl.backends.jax.planners.efficientzero import EfficientZeroPlanner
from algorl.backends.jax.planners.mcts.core import validate_search_batch_size


# Auto-width bounds for training-time reanalyze MCTS. The fallback matches the
# previous fixed preset; the max bounds compile time on very large GPUs.
_AUTO_WIDTH_FALLBACK = 10_240
_AUTO_WIDTH_MIN = 1_024
_AUTO_WIDTH_MAX = 65_536
_AUTO_WIDTH_MEMORY_FRACTION = 0.4


def _estimate_search_root_bytes(config: EfficientZeroConfig, *, action_dim: int) -> int:
    """Rough per-root device memory for one continuous-MCTS search lane."""
    from algorl.backends.jax.planners.efficientzero.planner import (
        continuous_search_config_from_agent,
    )

    search = continuous_search_config_from_agent(config)
    nodes = search.num_simulations + 1
    num_candidates = search.num_sampled_actions
    latent = max(config.hidden_shape, config.dyn_shape, config.rep_net_shape)
    floats_per_root = nodes * (
        6  # node scalars: visits, values, raw values, parents, actions, depth
        + 5 * num_candidates  # children index/prior/visits/rewards/values
        + latent  # latent embedding per node
        + num_candidates * action_dim  # candidate actions per node
        + num_candidates  # prior logits
    )
    transient_floats = 64 * latent  # NN activations and mctx temporaries
    # float32 storage with 3x headroom for allocator fragmentation and
    # double-buffered intermediates.
    return 4 * (floats_per_root + transient_floats) * 3


def resolve_reanalyze_search_width(
    config: EfficientZeroConfig,
    *,
    action_dim: int = 8,
) -> int:
    """Resolve the reanalyze MCTS width, sizing ``"auto"`` from free GPU memory.

    Wider search batches raise GPU occupancy without changing any search
    result (roots are independent and use one frozen params snapshot), so
    ``"auto"`` targets a fraction of currently-free device memory and falls
    back to the historical fixed width when memory stats are unavailable
    (e.g. CPU-only runs).
    """
    value = config.reanalyze_search_batch_size
    if value is None:
        return validate_search_batch_size(config.search_batch_size)
    if isinstance(value, str):
        if value != "auto":
            raise ValueError(
                f"reanalyze_search_batch_size must be an int, None, or 'auto'; got {value!r}."
            )
        from algorl.backends.jax.memory import gpu_available_memory_bytes

        available = gpu_available_memory_bytes()
        if available is None:
            return _AUTO_WIDTH_FALLBACK
        per_root = _estimate_search_root_bytes(config, action_dim=action_dim)
        width = int(available * _AUTO_WIDTH_MEMORY_FRACTION) // max(1, per_root)
        return int(np.clip(width, _AUTO_WIDTH_MIN, _AUTO_WIDTH_MAX))
    return validate_search_batch_size(int(value))


def effective_reanalyze_search_batch_size(config: EfficientZeroConfig) -> int:
    """MCTS width for training-time policy reanalyze (independent of rollout width)."""
    return resolve_reanalyze_search_width(config)


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


ValueInferenceFn = Callable[[Params, jnp.ndarray, jax.Array], jnp.ndarray]


def make_batched_value_inference(model: EfficientZeroNetwork) -> ValueInferenceFn:
    """Build a reusable JIT'd batched initial-inference value function.

    Building this once (e.g. at learner init) avoids re-tracing the network
    on every reanalyze value refresh; ``params`` are traced arguments so
    weight updates never trigger recompilation.
    """

    def infer(params: Params, obs: jnp.ndarray, keys: jax.Array) -> jnp.ndarray:
        def infer_one(single_obs: jnp.ndarray, subkey: jnp.ndarray) -> jnp.ndarray:
            _, value, _ = model.initial_inference(
                params,
                single_obs,
                training=False,
                rng=subkey,
            )
            return jnp.asarray(value, dtype=jnp.float32)

        # Vector networks reshape obs to a flat vector and only support one sample
        # per forward pass; vmap batches the call without changing numerics.
        return jax.vmap(infer_one)(obs, keys)

    return jax.jit(infer)


def batch_initial_values(
    model: EfficientZeroNetwork,
    params: Params,
    observations: np.ndarray,
    *,
    rng_key: jax.Array,
    mini_batch_size: int = 256,
    infer_fn: ValueInferenceFn | None = None,
) -> np.ndarray:
    """Batched initial-inference value extraction for reanalyze."""
    if infer_fn is None:
        infer_fn = make_batched_value_inference(model)
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
        values = np.asarray(infer_fn(params, chunk, infer_keys), dtype=np.float32)
        outputs.append(values.reshape(-1))
    return np.concatenate(outputs, axis=0)


def _batched_search_outputs(
    result: object,
    valid_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract MCTS outputs for a whole search chunk with one host transfer per array."""
    weights = np.asarray(result.action_weights, dtype=np.float32)[:valid_count]
    values = np.asarray(result.root_values, dtype=np.float32).reshape(-1)[:valid_count]
    candidates = np.asarray(result.root_candidates, dtype=np.float32)[:valid_count]
    if candidates.ndim == 2:
        candidates = candidates[:, :, None]
    elif candidates.ndim > 3:
        candidates = candidates.reshape(candidates.shape[0], candidates.shape[1], -1)
    best = np.asarray(result.actions, dtype=np.float32)[:valid_count]
    if best.ndim == 1:
        best = best[:, None]
    else:
        best = best.reshape(best.shape[0], -1)
    return weights, values, candidates, best


def _pad_search_observations(
    observations: np.ndarray,
    *,
    search_batch_size: int,
) -> tuple[np.ndarray, int]:
    """Pad MCTS roots to a fixed width so JAX search compiles once."""
    chunk = np.asarray(observations, dtype=np.float32)
    valid_count = int(chunk.shape[0])
    if valid_count == search_batch_size:
        return chunk, valid_count
    if valid_count > search_batch_size:
        raise ValueError(
            f"search chunk size {valid_count} exceeds configured width {search_batch_size}."
        )
    if valid_count == 0:
        raise ValueError("search chunk must contain at least one observation.")
    pad = np.repeat(chunk[-1:], search_batch_size - valid_count, axis=0)
    return np.concatenate([chunk, pad], axis=0), valid_count


def _execute_reanalyze_mcts(
    planner: EfficientZeroPlanner,
    flat_observations: np.ndarray,
    *,
    params: Params,
    temperature: float,
    search_batch_size: int,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the flat reanalyze root queue through fixed-width JIT'd searches."""
    total = int(flat_observations.shape[0])
    chunk_weights: list[np.ndarray] = []
    chunk_values: list[np.ndarray] = []
    chunk_candidates: list[np.ndarray] = []
    chunk_best: list[np.ndarray] = []

    for chunk_start in range(0, total, search_batch_size):
        obs_chunk = flat_observations[chunk_start : chunk_start + search_batch_size]
        padded_chunk, valid_count = _pad_search_observations(
            obs_chunk,
            search_batch_size=search_batch_size,
        )
        result = planner.search_batch(
            padded_chunk,
            deterministic=False,
            temperature=temperature,
            params=params,
            use_self_play=False,
        )
        weights, values, candidates, best = _batched_search_outputs(result, valid_count)
        chunk_weights.append(weights)
        chunk_values.append(values)
        chunk_candidates.append(candidates)
        chunk_best.append(best)

        if on_progress is not None:
            on_progress(min(chunk_start + search_batch_size, total), total)

    return (
        np.concatenate(chunk_weights, axis=0),
        np.concatenate(chunk_values, axis=0),
        np.concatenate(chunk_candidates, axis=0),
        np.concatenate(chunk_best, axis=0),
    )


def reanalyze_fused_policy_batches(
    planner: EfficientZeroPlanner,
    observation_batches: Sequence[np.ndarray],
    *,
    params: Params,
    reanalyze_count: int,
    temperature: float = 1.0,
    search_batch_size: int | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Run fresh MCTS for multiple training batches through one wide search queue."""
    if not observation_batches:
        return []
    if int(reanalyze_count) <= 0:
        raise ValueError("reanalyze_count must be > 0 when reanalyze is enabled.")

    if search_batch_size is None:
        search_batch_size = max(1, planner.search_batch_size)
    else:
        search_batch_size = max(1, validate_search_batch_size(search_batch_size))

    # Flatten all reanalyze roots into one queue: batches are contiguous, each
    # contributing its first ``count`` samples over the full window.
    segments: list[tuple[int, int, int]] = []  # (batch_size, count, window)
    flat_parts: list[np.ndarray] = []
    for observations in observation_batches:
        batch_size, window, obs_dim = observations.shape
        count = min(int(reanalyze_count), batch_size)
        segments.append((batch_size, count, window))
        flat_parts.append(
            np.ascontiguousarray(
                np.asarray(observations, dtype=np.float32)[:count]
            ).reshape(count * window, obs_dim)
        )
    flat_observations = np.concatenate(flat_parts, axis=0)
    # Searching wider than the queue only pads with duplicate roots.
    search_batch_size = min(search_batch_size, int(flat_observations.shape[0]))

    weights, values, candidates, best = _execute_reanalyze_mcts(
        planner,
        flat_observations,
        params=params,
        temperature=temperature,
        search_batch_size=search_batch_size,
        on_progress=on_progress,
    )

    outputs: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    offset = 0
    num_candidates = int(weights.shape[1])
    action_dim = int(candidates.shape[2])
    for batch_size, count, window in segments:
        length = count * window
        policy_targets = np.zeros((batch_size, window, num_candidates), dtype=np.float32)
        search_values = np.zeros((batch_size, window), dtype=np.float32)
        policy_candidates = np.zeros(
            (batch_size, window, num_candidates, action_dim),
            dtype=np.float32,
        )
        best_actions = np.zeros((batch_size, window, action_dim), dtype=np.float32)
        policy_targets[:count] = weights[offset : offset + length].reshape(
            count, window, num_candidates
        )
        search_values[:count] = values[offset : offset + length].reshape(count, window)
        policy_candidates[:count] = candidates[offset : offset + length].reshape(
            count, window, num_candidates, action_dim
        )
        best_actions[:count] = best[offset : offset + length].reshape(
            count, window, action_dim
        )
        outputs.append((policy_targets, search_values, policy_candidates, best_actions))
        offset += length
    return outputs


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
    results = reanalyze_fused_policy_batches(
        planner,
        [observations],
        params=params,
        reanalyze_count=reanalyze_count,
        temperature=temperature,
        search_batch_size=search_batch_size,
        on_progress=on_progress,
    )
    return results[0]


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

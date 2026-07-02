"""Batch MCTS reanalyze for EfficientZero training.

Reanalyze runs as eager Python orchestration around :meth:`EfficientZeroPlanner.search_batch`.
Planner outputs may be JAX arrays, but padding/merging here uses NumPy on the host.
The learner converts the returned arrays to ``jnp.ndarray`` immediately before the
JIT-compiled update.
"""

from __future__ import annotations

import numpy as np

from algorl.backends.jax.planners.efficientzero import EfficientZeroPlanner


def _as_host_array(value: object, *, dtype: np.dtype = np.float32) -> np.ndarray:
    """Materialize planner outputs to a host NumPy array."""
    return np.asarray(value, dtype=dtype)


def _as_candidate_matrix(candidates: object) -> np.ndarray:
    """Normalize root candidate actions to ``[num_candidates, action_dim]``."""
    root_candidates = _as_host_array(candidates)
    if root_candidates.ndim == 1:
        return root_candidates.reshape(-1, 1)
    if root_candidates.ndim == 2:
        return root_candidates
    return root_candidates.reshape(root_candidates.shape[0], -1)


def reanalyze_training_batch(
    planner: EfficientZeroPlanner,
    observations: np.ndarray,
    *,
    reanalyze_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run fresh MCTS on the first ``reanalyze_count`` batch items.

    Returns updated ``policy_targets``, ``search_values``, ``policy_candidates``,
    and ``best_actions`` with shape ``[B, K+1, ...]``.
    """
    batch_size, window, _ = observations.shape
    count = min(int(reanalyze_count), batch_size)
    if count <= 0:
        raise ValueError("reanalyze_count must be > 0 when reanalyze is enabled.")

    policy_targets = np.zeros((batch_size, window, 0), dtype=np.float32)
    search_values = np.zeros((batch_size, window), dtype=np.float32)
    policy_candidates: list[np.ndarray] = []
    best_actions = np.zeros((batch_size, window, 0), dtype=np.float32)

    for batch_index in range(count):
        step_candidates: list[np.ndarray] = []
        step_policies: list[np.ndarray] = []
        step_values: list[float] = []
        step_best: list[np.ndarray] = []

        for step_index in range(window):
            obs = _as_host_array(observations[batch_index, step_index])
            result = planner.search_batch(obs, deterministic=True)
            single = result.to_single(0)
            weights = _as_host_array(single.action_weights).reshape(-1)
            candidates = _as_candidate_matrix(single.root_candidates)
            best = _as_host_array(single.action).reshape(-1)

            step_policies.append(weights)
            step_values.append(float(single.root_value))
            step_candidates.append(candidates)
            step_best.append(best)

        max_candidates = max(candidate.shape[0] for candidate in step_candidates)
        action_dim = step_candidates[0].shape[1]
        padded_candidates = np.zeros((window, max_candidates, action_dim), dtype=np.float32)
        padded_policies = np.zeros((window, max_candidates), dtype=np.float32)
        padded_best = np.zeros((window, action_dim), dtype=np.float32)
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
            search_values[batch_index, step_index] = step_values[step_index]

        policy_candidates.append(padded_candidates)
        if policy_targets.shape[2] == 0:
            policy_targets = np.zeros((batch_size, window, max_candidates), dtype=np.float32)
            best_actions = np.zeros((batch_size, window, action_dim), dtype=np.float32)
        policy_targets[batch_index] = padded_policies
        best_actions[batch_index] = padded_best

    if not policy_candidates:
        return policy_targets, search_values, np.zeros((batch_size, window, 1, 1), dtype=np.float32), best_actions

    max_candidates = max(item.shape[1] for item in policy_candidates)
    action_dim = policy_candidates[0].shape[2]
    stacked = np.zeros((batch_size, window, max_candidates, action_dim), dtype=np.float32)
    for batch_index, candidates in enumerate(policy_candidates):
        stacked[batch_index, :, : candidates.shape[1], :] = candidates
    return policy_targets, search_values, stacked, best_actions

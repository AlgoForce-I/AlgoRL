"""Bootstrapped and GAE value targets for EfficientZero replay."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def bootstrapped_values(
    rewards: np.ndarray,
    values: np.ndarray,
    *,
    discount: float,
    td_steps: int,
    episodic: bool = False,
) -> np.ndarray:
    """HyperCEZ ``GameTrajectory.get_bootstrapped_value`` for vector trajectories."""
    traj_len = int(rewards.shape[0])
    bt_values = np.zeros((traj_len,), dtype=np.float32)
    td_steps = max(1, int(td_steps))

    for idx in range(traj_len):
        if idx + td_steps < traj_len:
            bt_value = (discount**td_steps) * float(values[idx + td_steps])
            for step in range(td_steps):
                bt_value += (discount**step) * float(rewards[idx + step])
        else:
            overflow = idx + td_steps - traj_len + 1
            if episodic:
                bt_value = 0.0
            else:
                bt_value = (discount ** (td_steps - overflow)) * float(values[-1])
            for step in range(td_steps - overflow):
                bt_value += (discount**step) * float(rewards[idx + step])
        bt_values[idx] = bt_value
    return bt_values


def gae_values(
    rewards: np.ndarray,
    values: np.ndarray,
    *,
    discount: float,
    td_steps: int,
    td_lambda: float,
    gae_max_steps: int,
    episodic: bool = False,
    index: int | None = None,
    collected_transitions: int | None = None,
    auto_td_steps: int = 30_000,
) -> np.ndarray:
    """HyperCEZ ``GameTrajectory.get_gae_value`` (scalar trajectory)."""
    traj_len = int(rewards.shape[0])
    if traj_len == 0:
        return np.zeros((0,), dtype=np.float32)

    if index is None or collected_transitions is None:
        effective_lambda = td_lambda
    else:
        delta_lambda = 0.1 * (collected_transitions - index) / float(auto_td_steps)
        effective_lambda = float(np.clip(td_lambda - delta_lambda, 0.65, td_lambda))

    lens = int(gae_max_steps)
    gae = np.zeros((traj_len,), dtype=np.float32)

    for start in range(traj_len):
        delta = np.zeros((lens,), dtype=np.float32)
        advantage = np.zeros((lens + 1,), dtype=np.float32)
        cursor = lens - 1
        for idx in reversed(range(start, start + lens)):
            if idx + td_steps < traj_len:
                bt_value = (discount**td_steps) * float(values[idx + td_steps])
                for step in range(td_steps):
                    bt_value += (discount**step) * float(rewards[idx + step])
            else:
                overflow = idx + td_steps - traj_len + 1
                if episodic:
                    bt_value = 0.0
                else:
                    bt_value = (discount ** (td_steps - overflow)) * float(values[-1])
                for step in range(td_steps - overflow):
                    if idx + step < traj_len:
                        bt_value += (discount**step) * float(rewards[idx + step])

            if idx < traj_len:
                delta[cursor] = bt_value - float(values[idx])
            else:
                delta[cursor] = 0.0
            advantage[cursor] = delta[cursor] + discount * effective_lambda * advantage[cursor + 1]
            cursor -= 1

        local_values = np.asarray(values[start : start + lens], dtype=np.float32)
        if local_values.shape[0] < lens:
            local_values = np.concatenate(
                [local_values, np.zeros((lens - local_values.shape[0],), dtype=np.float32)]
            )
        gae[start] = advantage[:lens][0] + local_values[0]

    return gae


def mix_value_targets(
    bootstrapped: np.ndarray,
    search: np.ndarray,
    *,
    use_search_mask: np.ndarray,
) -> np.ndarray:
    """Blend bootstrapped and search targets (HyperCEZ ``mixed`` value target)."""
    mask = use_search_mask.astype(np.float32)
    return bootstrapped * mask + search * (1.0 - mask)


def adaptive_td_steps(
    base_td_steps: int,
    *,
    sample_index: int,
    collected_transitions: int,
    auto_td_steps: int = 30_000,
) -> int:
    delta = (collected_transitions - sample_index) // auto_td_steps
    return int(np.clip(base_td_steps - delta, 1, base_td_steps))


def gae_extra_steps(config: object) -> int:
    """HyperCEZ GAE lookahead beyond the unroll window (``extra`` in ``prepare_reward_value_gae``)."""
    from algorl.agents.configs import EfficientZeroConfig

    if not isinstance(config, EfficientZeroConfig):
        raise TypeError("gae_extra_steps requires EfficientZeroConfig.")
    return max(
        0,
        min(
            int(1 / (1 - config.td_lambda)),
            config.gae_max_steps,
        )
        - config.unroll_steps
        - 1,
    )


def trajectory_padding_gap(config: object) -> int:
    """Steps of tail context to pad across trajectory blocks (HyperCEZ ``gap_step``)."""
    from algorl.agents.configs import EfficientZeroConfig

    if not isinstance(config, EfficientZeroConfig):
        raise TypeError("trajectory_padding_gap requires EfficientZeroConfig.")
    if config.model_value_target == "bootstrapped":
        return config.n_stack + config.td_steps
    return config.n_stack + 1 + gae_extra_steps(config) + 1


def extended_target_window(config: object) -> int:
    """Observation/reward window needed for full-horizon value targets.

    HyperCEZ computes value targets on the stored trajectory, so every unroll
    position can bootstrap ``td_steps`` (or GAE ``extra + 1``) transitions ahead.
    The replay buffer must therefore expose that many steps beyond the
    ``unroll_steps + 1`` training window.
    """
    from algorl.agents.configs import EfficientZeroConfig

    if not isinstance(config, EfficientZeroConfig):
        raise TypeError("extended_target_window requires EfficientZeroConfig.")
    if config.model_value_target == "bootstrapped":
        return config.unroll_steps + 1 + config.td_steps
    return config.unroll_steps + gae_extra_steps(config) + 2


def prepare_bootstrapped_batch_values(
    observations: np.ndarray,
    rewards: np.ndarray,
    sample_indices: np.ndarray,
    *,
    valid_lengths: np.ndarray,
    bootstrap_limits: np.ndarray | None = None,
    total_transitions: int,
    config: object,
    infer_values: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """HyperCEZ ``prepare_reward_value`` targets for a sampled training batch.

    ``observations``/``rewards`` cover the extended window
    (:func:`extended_target_window`); ``valid_lengths[b]`` is the number of core
    trajectory transitions remaining from the sampled position (HyperCEZ
    ``traj_len - state_index``), which caps the TD horizon near trajectory ends.
    """
    from algorl.agents.configs import EfficientZeroConfig

    if not isinstance(config, EfficientZeroConfig):
        raise TypeError("prepare_bootstrapped_batch_values requires EfficientZeroConfig.")

    batch_size, window, obs_dim = observations.shape
    unroll_positions = config.unroll_steps + 1
    if window < unroll_positions + 1:
        raise ValueError(f"observation window {window} < unroll_positions + 1.")
    if bootstrap_limits is None:
        bootstrap_limits = valid_lengths

    inferred = np.asarray(
        infer_values(observations.reshape(batch_size * window, obs_dim)),
        dtype=np.float32,
    ).reshape(batch_size, window)

    targets = np.zeros((batch_size, unroll_positions), dtype=np.float32)
    for batch_index in range(batch_size):
        valid_len = int(valid_lengths[batch_index])
        limit = int(bootstrap_limits[batch_index])
        sample_index = int(sample_indices[batch_index])

        # Off-policy correction: shorter horizon of td steps. Disabled for
        # ``mixed``/``max`` value targets, exactly as in HyperCEZ.
        if config.value_target in ("mixed", "max"):
            td_steps = config.td_steps
        else:
            td_steps = adaptive_td_steps(
                config.td_steps,
                sample_index=sample_index,
                collected_transitions=total_transitions,
                auto_td_steps=config.auto_td_steps,
            )
        td_steps = int(np.clip(min(valid_len, td_steps), 1, config.td_steps))

        for position in range(unroll_positions):
            td_steps = max(1, min(valid_len - position, td_steps))
            bootstrap_position = position + td_steps

            target = 0.0
            if bootstrap_position <= limit and bootstrap_position < window:
                target = float(config.discount**td_steps) * float(
                    inferred[batch_index, bootstrap_position]
                )
            for step in range(position, min(bootstrap_position, window)):
                target += float(config.discount ** (step - position)) * float(
                    rewards[batch_index, step]
                )
            # Positions past the trajectory end train against zero (masked out).
            targets[batch_index, position] = target if position <= valid_len else 0.0
    return targets


def prepare_gae_batch_values(
    observations: np.ndarray,
    rewards: np.ndarray,
    sample_indices: np.ndarray,
    *,
    valid_lengths: np.ndarray,
    bootstrap_limits: np.ndarray | None = None,
    total_transitions: int,
    config: object,
    infer_values: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """GAE value targets with fresh reanalyze-model inference (HyperCEZ ``prepare_reward_value_gae``)."""
    from algorl.agents.configs import EfficientZeroConfig

    if not isinstance(config, EfficientZeroConfig):
        raise TypeError("prepare_gae_batch_values requires EfficientZeroConfig.")

    batch_size, window, obs_dim = observations.shape
    unroll_positions = config.unroll_steps + 1
    extra = gae_extra_steps(config)
    span = config.unroll_steps + 1 + extra
    if window < span + 1:
        raise ValueError(f"observation window {window} < GAE span + 1 ({span + 1}).")
    if bootstrap_limits is None:
        bootstrap_limits = valid_lengths

    inferred = np.asarray(
        infer_values(observations.reshape(batch_size * window, obs_dim)),
        dtype=np.float32,
    ).reshape(batch_size, window)

    targets = np.zeros((batch_size, unroll_positions), dtype=np.float32)
    for batch_index in range(batch_size):
        valid_len = int(valid_lengths[batch_index])
        limit = int(bootstrap_limits[batch_index])
        sample_index = int(sample_indices[batch_index])

        # HyperCEZ checks ``model["value_target"]`` (never 'mixed'/'max') here,
        # so the lambda age-decay is always active on the GAE path.
        delta_lambda = 0.1 * (total_transitions - sample_index) / float(config.auto_td_steps)
        td_lambda = float(np.clip(config.td_lambda - delta_lambda, 0.65, config.td_lambda))

        advantage = np.zeros((span + 1,), dtype=np.float32)
        for position in reversed(range(span)):
            bootstrap_position = position + 1
            value = 0.0
            if bootstrap_position <= limit:
                value = float(config.discount) * float(inferred[batch_index, bootstrap_position])
            if position < window:
                value += float(rewards[batch_index, position])
            current = float(inferred[batch_index, position]) if position < valid_len else 0.0
            delta = value - current
            advantage[position] = delta + float(config.discount) * td_lambda * advantage[position + 1]

        for position in range(unroll_positions):
            current = float(inferred[batch_index, position]) if position < valid_len else 0.0
            target = advantage[position] + current
            targets[batch_index, position] = target if position <= valid_len else 0.0
    return targets

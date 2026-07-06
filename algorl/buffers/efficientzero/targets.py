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


def trajectory_padding_gap(config: object) -> int:
    """Steps of tail context to pad across trajectory blocks (HyperCEZ ``gap_step``)."""
    from algorl.agents.configs import EfficientZeroConfig

    if not isinstance(config, EfficientZeroConfig):
        raise TypeError("trajectory_padding_gap requires EfficientZeroConfig.")
    if config.model_value_target == "bootstrapped":
        return config.n_stack + config.td_steps
    extra = max(
        0,
        min(
            int(1 / (1 - config.td_lambda)),
            config.gae_max_steps,
        )
        - config.unroll_steps
        - 1,
    )
    return config.n_stack + 1 + extra + 1


def prepare_bootstrapped_batch_values(
    observations: np.ndarray,
    rewards: np.ndarray,
    sample_indices: np.ndarray,
    *,
    total_transitions: int,
    config: object,
    infer_values: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """HyperCEZ ``prepare_reward_value`` targets for a sampled training batch."""
    from algorl.agents.configs import EfficientZeroConfig

    if not isinstance(config, EfficientZeroConfig):
        raise TypeError("prepare_bootstrapped_batch_values requires EfficientZeroConfig.")

    batch_size, window, obs_dim = observations.shape
    unroll_positions = config.unroll_steps + 1
    if window < unroll_positions:
        raise ValueError(f"observation window {window} < unroll_positions {unroll_positions}.")

    zero_obs = np.zeros((obs_dim,), dtype=np.float32)
    value_obs: list[np.ndarray] = []
    td_steps_flat: list[int] = []
    value_mask: list[float] = []
    layout: list[tuple[int, int, int]] = []

    for batch_index in range(batch_size):
        sample_index = int(sample_indices[batch_index])
        td_steps = adaptive_td_steps(
            config.td_steps,
            sample_index=sample_index,
            collected_transitions=total_transitions,
            auto_td_steps=config.auto_td_steps,
        )
        if config.value_target in ("mixed", "max"):
            td_steps = config.td_steps

        for position in range(unroll_positions):
            local_td = min(window - 1 - position, td_steps)
            local_td = max(1, int(local_td))
            bootstrap_position = position + local_td
            if bootstrap_position < window:
                value_obs.append(observations[batch_index, bootstrap_position])
                value_mask.append(1.0)
            else:
                value_obs.append(zero_obs)
                value_mask.append(0.0)
            td_steps_flat.append(local_td)
            layout.append((batch_index, position, local_td))

    inferred = infer_values(np.stack(value_obs, axis=0).astype(np.float32))
    inferred = np.asarray(inferred, dtype=np.float32).reshape(-1)
    value_mask_arr = np.asarray(value_mask, dtype=np.float32)
    discounted = inferred * (config.discount ** np.asarray(td_steps_flat, dtype=np.float32))
    discounted = discounted * value_mask_arr

    targets = np.zeros((batch_size, window), dtype=np.float32)
    cursor = 0
    for batch_index, position, local_td in layout:
        bootstrap_position = position + local_td
        target = float(discounted[cursor])
        for step, reward in enumerate(
            rewards[batch_index, position:bootstrap_position],
            start=0,
        ):
            target += float(config.discount**step) * float(reward)
        if position < window:
            targets[batch_index, position] = target
        cursor += 1
    return targets[:, :unroll_positions]


def prepare_gae_batch_values(
    observations: np.ndarray,
    rewards: np.ndarray,
    sample_indices: np.ndarray,
    *,
    total_transitions: int,
    config: object,
    infer_values: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """GAE value targets with fresh reanalyze-model inference (HyperCEZ ``prepare_reward_value_gae``)."""
    from algorl.agents.configs import EfficientZeroConfig

    if not isinstance(config, EfficientZeroConfig):
        raise TypeError("prepare_gae_batch_values requires EfficientZeroConfig.")

    batch_size, window, obs_dim = observations.shape
    flat_obs = observations.reshape(batch_size * window, obs_dim)
    flat_values = np.asarray(infer_values(flat_obs), dtype=np.float32).reshape(batch_size, window)

    targets = np.zeros((batch_size, window), dtype=np.float32)
    for batch_index in range(batch_size):
        gae = gae_values(
            rewards[batch_index],
            flat_values[batch_index],
            discount=config.discount,
            td_steps=config.td_steps,
            td_lambda=config.td_lambda,
            gae_max_steps=config.gae_max_steps,
            index=int(sample_indices[batch_index]),
            collected_transitions=total_transitions,
            auto_td_steps=config.auto_td_steps,
        )
        targets[batch_index, : gae.shape[0]] = gae[:window]
    return targets[:, : config.unroll_steps + 1]

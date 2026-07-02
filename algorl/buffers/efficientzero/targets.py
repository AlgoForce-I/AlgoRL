"""Bootstrapped and GAE value targets for EfficientZero replay."""

from __future__ import annotations

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
) -> np.ndarray:
    """HyperCEZ ``GameTrajectory.get_gae_value`` (scalar trajectory)."""
    traj_len = int(rewards.shape[0])
    if traj_len == 0:
        return np.zeros((0,), dtype=np.float32)

    if index is None or collected_transitions is None:
        effective_lambda = td_lambda
    else:
        delta_lambda = 0.1 * (collected_transitions - index) / 30_000.0
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
) -> int:
    delta = (collected_transitions - sample_index) // 30_000
    return int(np.clip(base_td_steps - delta, 1, base_td_steps))

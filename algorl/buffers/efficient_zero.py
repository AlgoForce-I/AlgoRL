"""EfficientZero replay buffer with trajectory chunks and MCTS targets."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import numpy as np

from algorl.core.replay_buffer import ReplayBuffer
from algorl.core.types import Action, Batch, Observation, Transition

POLICY_TARGET_INFO_KEY = "policy_target"
SEARCH_VALUE_INFO_KEY = "search_value"
PRED_VALUE_INFO_KEY = "pred_value"

INFO_KEYS = (
    POLICY_TARGET_INFO_KEY,
    SEARCH_VALUE_INFO_KEY,
    PRED_VALUE_INFO_KEY,
)


@dataclass(frozen=True)
class EfficientZeroStep:
    """One environment step plus search targets for unrolled training."""

    observation: np.ndarray
    action: np.ndarray
    reward: float
    policy_target: np.ndarray
    pred_value: float
    search_value: float
    done: bool


@dataclass
class EfficientZeroTrajectory:
    """Fixed-size trajectory block aligned with HyperCEZ ``GameTrajectory``."""

    max_size: int
    steps: list[EfficientZeroStep] = field(default_factory=list)

    def append(self, step: EfficientZeroStep) -> bool:
        self.steps.append(step)
        return self.is_full() or step.done

    def is_full(self) -> bool:
        return len(self.steps) >= self.max_size

    def clear(self) -> list[EfficientZeroStep]:
        committed = self.steps
        self.steps = []
        return committed


def search_fields_from_transition_info(
    info: dict[str, object],
    *,
    default_policy_dim: int = 0,
) -> tuple[np.ndarray, float, float]:
    """Extract MCTS targets stored on a :class:`Transition`."""
    policy = info.get(POLICY_TARGET_INFO_KEY)
    if policy is None:
        policy_target = np.zeros((default_policy_dim,), dtype=np.float32)
    else:
        policy_target = np.asarray(policy, dtype=np.float32).reshape(-1)

    pred_value = float(info.get(PRED_VALUE_INFO_KEY, 0.0))
    search_value = float(info.get(SEARCH_VALUE_INFO_KEY, 0.0))
    return policy_target, pred_value, search_value


def step_from_transition(transition: Transition) -> EfficientZeroStep:
    """Convert a rollout transition into a stored EfficientZero step."""
    policy_target, pred_value, search_value = search_fields_from_transition_info(transition.info)
    return EfficientZeroStep(
        observation=np.asarray(transition.observation, dtype=np.float32),
        action=_action_array(transition.action),
        reward=float(transition.reward),
        policy_target=policy_target,
        pred_value=pred_value,
        search_value=search_value,
        done=bool(transition.done),
    )


class EfficientZeroReplayBuffer(ReplayBuffer):
    """Trajectory replay storage for EfficientZero / MuZero-style training.

    Rollouts are grouped into trajectory blocks (``trajectory_size``). Each step
    stores MCTS policy/value targets via ``Transition.info``:

    - ``policy_target``: improved root policy (candidate-slot distribution)
    - ``search_value``: root value after search
    - ``pred_value``: network value before search (optional)
    """

    def __init__(
        self,
        capacity: int,
        *,
        unroll_steps: int = 5,
        trajectory_size: int = 100,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}.")
        if unroll_steps <= 0:
            raise ValueError(f"unroll_steps must be > 0, got {unroll_steps}.")
        if trajectory_size <= 0:
            raise ValueError(f"trajectory_size must be > 0, got {trajectory_size}.")

        self.capacity = capacity
        self.unroll_steps = unroll_steps
        self.trajectory_size = trajectory_size
        self._trajectories: Deque[list[EfficientZeroStep]] = deque()
        self._lookup: list[tuple[int, int]] = []
        self._base_traj_idx = 0
        self._active = EfficientZeroTrajectory(max_size=trajectory_size)

    def add(self, transition: Transition) -> None:
        step = step_from_transition(transition)
        if self._active.append(step):
            self._commit_trajectory(self._active.clear())
            if not step.done:
                self._active = EfficientZeroTrajectory(max_size=self.trajectory_size)

    def sample(self, batch_size: int) -> Batch:
        if batch_size > len(self._lookup):
            raise ValueError(
                f"Requested batch size {batch_size} exceeds stored transitions {len(self._lookup)}."
            )

        rng = np.random.default_rng()
        indices = rng.choice(len(self._lookup), size=batch_size, replace=False)

        observations: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        rewards: list[np.ndarray] = []
        policy_targets: list[np.ndarray] = []
        pred_values: list[np.ndarray] = []
        search_values: list[np.ndarray] = []
        dones: list[np.ndarray] = []
        sample_indices: list[int] = []

        window = self.unroll_steps + 1
        for flat_index in indices:
            traj_idx, step_idx = self._lookup[int(flat_index)]
            traj = self._trajectories[traj_idx - self._base_traj_idx]
            end = step_idx + window
            if end > len(traj):
                raise RuntimeError(
                    "Invalid replay lookup while sampling an unroll window. "
                    f"traj_len={len(traj)}, step_idx={step_idx}, window={window}."
                )

            chunk = traj[step_idx:end]
            observations.append(
                np.stack([step.observation for step in chunk], axis=0).astype(np.float32)
            )
            actions.append(
                np.stack([step.action for step in chunk[:-1]], axis=0).astype(np.float32)
            )
            rewards.append(np.asarray([step.reward for step in chunk], dtype=np.float32))
            policy_targets.append(
                np.stack([step.policy_target for step in chunk], axis=0).astype(np.float32)
            )
            pred_values.append(
                np.asarray([step.pred_value for step in chunk], dtype=np.float32)
            )
            search_values.append(
                np.asarray([step.search_value for step in chunk], dtype=np.float32)
            )
            dones.append(np.asarray([step.done for step in chunk], dtype=np.bool_))
            sample_indices.append(int(flat_index))

        return Batch(
            data={
                "observations": np.stack(observations, axis=0),
                "actions": np.stack(actions, axis=0),
                "rewards": np.stack(rewards, axis=0),
                "policy_targets": np.stack(policy_targets, axis=0),
                "pred_values": np.stack(pred_values, axis=0),
                "search_values": np.stack(search_values, axis=0),
                "dones": np.stack(dones, axis=0),
                "indices": np.asarray(sample_indices, dtype=np.int32),
            }
        )

    def __len__(self) -> int:
        return len(self._lookup)

    def _commit_trajectory(self, steps: list[EfficientZeroStep]) -> None:
        if not steps:
            return

        traj_idx = self._base_traj_idx + len(self._trajectories)
        self._trajectories.append(steps)
        for step_pos in range(len(steps)):
            if step_pos + self.unroll_steps < len(steps):
                self._lookup.append((traj_idx, step_pos))

        self._trim_to_capacity()

    def _trim_to_capacity(self) -> None:
        while len(self._lookup) > self.capacity:
            if not self._trajectories:
                self._lookup.clear()
                break
            dropped = self._trajectories.popleft()
            dropped_len = len(dropped)
            self._base_traj_idx += 1
            self._lookup = [
                (traj_idx, step_idx)
                for traj_idx, step_idx in self._lookup
                if traj_idx >= self._base_traj_idx
            ]
            if dropped_len == 0 and len(self._lookup) > self.capacity:
                continue


def _action_array(action: Action) -> np.ndarray:
    array = np.asarray(action, dtype=np.float32)
    if array.ndim == 0:
        return array.reshape(1)
    return array.reshape(-1)

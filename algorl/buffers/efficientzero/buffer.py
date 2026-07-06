"""EfficientZero replay buffer with trajectory chunks and MCTS targets."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import numpy as np

from algorl.agents.configs import EfficientZeroConfig
from algorl.buffers.efficientzero.targets import (
    bootstrapped_values,
    gae_values,
    mix_value_targets,
    trajectory_padding_gap,
)
from algorl.core.replay_buffer import ReplayBuffer
from algorl.core.types import Action, Batch, Transition

POLICY_TARGET_INFO_KEY = "policy_target"
SEARCH_VALUE_INFO_KEY = "search_value"
PRED_VALUE_INFO_KEY = "pred_value"
ROOT_CANDIDATES_INFO_KEY = "root_candidates"
BEST_ACTION_INFO_KEY = "best_action"

INFO_KEYS = (
    POLICY_TARGET_INFO_KEY,
    SEARCH_VALUE_INFO_KEY,
    PRED_VALUE_INFO_KEY,
    ROOT_CANDIDATES_INFO_KEY,
    BEST_ACTION_INFO_KEY,
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
    root_candidates: np.ndarray
    best_action: np.ndarray
    done: bool


@dataclass
class EfficientZeroTrajectory:
    """Fixed-size trajectory block aligned with HyperCEZ ``GameTrajectory``."""

    max_size: int
    steps: list[EfficientZeroStep] = field(default_factory=list)
    bootstrapped_values: np.ndarray | None = None
    gae_values: np.ndarray | None = None

    def append(self, step: EfficientZeroStep) -> bool:
        self.steps.append(step)
        return self.is_full() or step.done

    def is_full(self) -> bool:
        return len(self.steps) >= self.max_size

    def clear(self) -> list[EfficientZeroStep]:
        committed = self.steps
        self.steps = []
        self.bootstrapped_values = None
        self.gae_values = None
        return committed

    def pad_over(
        self,
        tail_steps: list[EfficientZeroStep],
    ) -> None:
        """Append tail context from the next trajectory block (HyperCEZ ``pad_over``)."""
        for step in tail_steps:
            self.steps.append(step)

    def finalize_targets(self, config: EfficientZeroConfig) -> None:
        if not self.steps:
            return
        rewards = np.asarray([step.reward for step in self.steps], dtype=np.float32)
        pred_values = np.asarray([step.pred_value for step in self.steps], dtype=np.float32)
        search_values = np.asarray([step.search_value for step in self.steps], dtype=np.float32)
        value_source = pred_values if config.model_value_target == "bootstrapped" else search_values
        if config.model_value_target == "GAE":
            self.gae_values = gae_values(
                rewards,
                value_source,
                discount=config.discount,
                td_steps=config.td_steps,
                td_lambda=config.td_lambda,
                gae_max_steps=config.gae_max_steps,
                auto_td_steps=config.auto_td_steps,
            )
        self.bootstrapped_values = bootstrapped_values(
            rewards,
            value_source,
            discount=config.discount,
            td_steps=config.td_steps,
        )


def search_fields_from_transition_info(
    info: dict[str, object],
    *,
    default_policy_dim: int = 0,
    default_action: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
    """Extract MCTS targets stored on a :class:`Transition`."""
    policy = info.get(POLICY_TARGET_INFO_KEY)
    if policy is None:
        policy_target = np.zeros((default_policy_dim,), dtype=np.float32)
    else:
        policy_target = np.asarray(policy, dtype=np.float32).reshape(-1)

    pred_value = float(info.get(PRED_VALUE_INFO_KEY, 0.0))
    search_value = float(info.get(SEARCH_VALUE_INFO_KEY, 0.0))

    candidates = info.get(ROOT_CANDIDATES_INFO_KEY)
    if candidates is None:
        root_candidates = np.zeros((0, 0), dtype=np.float32)
    else:
        root_candidates = np.asarray(candidates, dtype=np.float32)
        if root_candidates.ndim == 1:
            root_candidates = root_candidates.reshape(-1, 1)
        elif root_candidates.ndim > 2:
            root_candidates = root_candidates.reshape(root_candidates.shape[0], -1)

    best = info.get(BEST_ACTION_INFO_KEY, default_action)
    if best is None:
        best_action = np.zeros((0,), dtype=np.float32)
    else:
        best_action = np.asarray(best, dtype=np.float32).reshape(-1)

    return policy_target, pred_value, search_value, root_candidates, best_action


def step_from_transition(transition: Transition) -> EfficientZeroStep:
    """Convert a rollout transition into a stored EfficientZero step."""
    action = _action_array(transition.action)
    policy_target, pred_value, search_value, root_candidates, best_action = (
        search_fields_from_transition_info(
            transition.info,
            default_action=action,
        )
    )
    if best_action.size == 0:
        best_action = action
    return EfficientZeroStep(
        observation=np.asarray(transition.observation, dtype=np.float32),
        action=action,
        reward=float(transition.reward),
        policy_target=policy_target,
        pred_value=pred_value,
        search_value=search_value,
        root_candidates=root_candidates,
        best_action=best_action,
        done=bool(transition.done),
    )


class EfficientZeroReplayBuffer(ReplayBuffer):
    """Trajectory replay storage for EfficientZero / MuZero-style training."""

    def __init__(
        self,
        capacity: int,
        *,
        config: EfficientZeroConfig | None = None,
        unroll_steps: int = 5,
        trajectory_size: int = 100,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}.")
        if unroll_steps <= 0:
            raise ValueError(f"unroll_steps must be > 0, got {unroll_steps}.")
        if trajectory_size <= 0:
            raise ValueError(f"trajectory_size must be > 0, got {trajectory_size}.")

        self.config = config or EfficientZeroConfig(unroll_steps=unroll_steps, trajectory_size=trajectory_size)
        self.capacity = capacity
        self.unroll_steps = unroll_steps
        self.trajectory_size = trajectory_size
        self._trajectories: Deque[EfficientZeroTrajectory] = deque()
        self._stored_steps: Deque[list[EfficientZeroStep]] = deque()
        self._lookup: list[tuple[int, int]] = []
        self._priorities: list[float] = []
        self._base_traj_idx = 0
        self._active = EfficientZeroTrajectory(max_size=trajectory_size)
        self._pending_commit: EfficientZeroTrajectory | None = None
        self._total_commits = 0

    def add(self, transition: Transition) -> None:
        step = step_from_transition(transition)
        if self._active.append(step):
            self._commit_active(step.done)

    def sample(
        self,
        batch_size: int,
        *,
        beta: float = 1.0,
        trained_steps: int = 0,
    ) -> Batch:
        if batch_size > len(self._lookup):
            raise ValueError(
                f"Requested batch size {batch_size} exceeds stored transitions {len(self._lookup)}."
            )

        indices = self._sample_indices(batch_size, beta=beta)
        return self._build_batch(indices, trained_steps=trained_steps)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        for index, priority in zip(indices.reshape(-1), priorities.reshape(-1), strict=True):
            flat_index = int(index)
            if 0 <= flat_index < len(self._priorities):
                self._priorities[flat_index] = float(priority)

    def __len__(self) -> int:
        return len(self._lookup)

    @property
    def total_transitions(self) -> int:
        return len(self._lookup)

    def _commit_active(self, done: bool) -> None:
        steps = self._active.clear()
        if not steps:
            return

        current = EfficientZeroTrajectory(max_size=self.trajectory_size)
        current.steps = list(steps)

        if self._pending_commit is not None:
            gap = trajectory_padding_gap(self.config)
            tail = current.steps[:gap]
            self._pending_commit.pad_over(tail)
            self._pending_commit.finalize_targets(self.config)
            self._store_trajectory(self._pending_commit.steps, self._pending_commit)
            self._pending_commit = None

        if len(current.steps) >= self.trajectory_size and not done:
            self._pending_commit = current
        else:
            current.finalize_targets(self.config)
            self._store_trajectory(current.steps, current)

        if done and self._pending_commit is not None:
            self._pending_commit.finalize_targets(self.config)
            self._store_trajectory(self._pending_commit.steps, self._pending_commit)
            self._pending_commit = None

        if not done:
            self._active = EfficientZeroTrajectory(max_size=self.trajectory_size)

    def _store_trajectory(
        self,
        steps: list[EfficientZeroStep],
        traj_meta: EfficientZeroTrajectory,
    ) -> None:
        if not steps:
            return

        traj_idx = self._base_traj_idx + len(self._stored_steps)
        self._stored_steps.append(steps)
        self._trajectories.append(traj_meta)
        rewards = np.asarray([step.reward for step in steps], dtype=np.float32)
        pred_values = np.asarray([step.pred_value for step in steps], dtype=np.float32)
        bootstrapped = traj_meta.bootstrapped_values
        if bootstrapped is None:
            bootstrapped = bootstrapped_values(
                rewards,
                pred_values,
                discount=self.config.discount,
                td_steps=self.config.td_steps,
            )

        # HyperCEZ ``save_trajectory``: new transitions all get the buffer-wide max
        # priority (optimistic init) so fresh data is sampled promptly; per-step
        # priorities are refreshed once the samples pass through training.
        if self.config.use_priority:
            traj_priorities = (
                np.abs(pred_values[: len(steps)] - np.asarray(bootstrapped[: len(steps)], dtype=np.float32))
                + self.config.min_prior
            )
            max_prior = max(self._priorities) if self._priorities else 1.0
            new_priority = float(max(max_prior, float(traj_priorities.max())))
        else:
            new_priority = 1.0

        for step_pos in range(len(steps)):
            if step_pos + self.unroll_steps < len(steps):
                self._lookup.append((traj_idx, step_pos))
                self._priorities.append(new_priority)

        self._total_commits += 1
        self._trim_to_capacity()

    def _sample_indices(self, batch_size: int, *, beta: float) -> np.ndarray:
        total = len(self._lookup)
        if not self.config.use_priority:
            rng = np.random.default_rng()
            return rng.choice(total, size=batch_size, replace=False)

        # HyperCEZ: permanently zero priorities outside the top-transitions window.
        if total > int(self.config.top_transitions):
            cutoff = total - int(self.config.top_transitions)
            for index in range(cutoff):
                self._priorities[index] = 0.0

        priorities = np.asarray(self._priorities[:total], dtype=np.float64)
        probs = priorities**self.config.priority_prob_alpha
        total_prob = probs.sum()
        if total_prob <= 0.0:
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / total_prob

        rng = np.random.default_rng()
        indices = rng.choice(total, size=batch_size, replace=False, p=probs)
        weights = (total * probs[indices]) ** (-beta)
        weights = weights / max(weights.max(), 1e-8)
        weights = np.clip(weights, 0.1, 1.0)
        self._last_weights = weights.astype(np.float32)
        return indices

    def _build_batch(self, indices: np.ndarray, *, trained_steps: int) -> Batch:
        window = self.unroll_steps + 1
        observations: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        rewards: list[np.ndarray] = []
        policy_targets: list[np.ndarray] = []
        pred_values: list[np.ndarray] = []
        search_values: list[np.ndarray] = []
        value_targets: list[np.ndarray] = []
        policy_candidates: list[np.ndarray] = []
        best_actions: list[np.ndarray] = []
        dones: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        mix_masks: list[np.ndarray] = []
        sample_indices: list[int] = []
        weights: list[float] = []

        for offset, flat_index in enumerate(indices):
            traj_idx, step_idx = self._lookup[int(flat_index)]
            traj_steps = self._stored_steps[traj_idx - self._base_traj_idx]
            traj_meta = self._trajectories[traj_idx - self._base_traj_idx]
            end = step_idx + window
            if end > len(traj_steps):
                raise RuntimeError(
                    "Invalid replay lookup while sampling an unroll window. "
                    f"traj_len={len(traj_steps)}, step_idx={step_idx}, window={window}."
                )

            chunk = traj_steps[step_idx:end]
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

            if traj_meta.gae_values is not None:
                bootstrapped = traj_meta.gae_values[step_idx:end]
            elif traj_meta.bootstrapped_values is not None:
                bootstrapped = traj_meta.bootstrapped_values[step_idx:end]
            else:
                bootstrapped = pred_values[-1]

            search = search_values[-1]
            if self.config.value_target == "search":
                targets = search
                mix_mask = np.zeros((window,), dtype=np.float32)
            elif self.config.value_target == "mixed" and trained_steps >= self.config.start_use_mix_training_steps:
                recent = int(flat_index > self.total_transitions - self.config.mixed_value_threshold)
                mix_mask = np.full((window,), recent, dtype=np.float32)
                targets = mix_value_targets(bootstrapped, search, use_search_mask=mix_mask)
            else:
                targets = bootstrapped
                mix_mask = np.ones((window,), dtype=np.float32)
            value_targets.append(np.asarray(targets, dtype=np.float32))
            mix_masks.append(mix_mask)

            candidates = self._stack_candidates(chunk)
            policy_candidates.append(candidates)
            best_actions.append(
                np.stack([step.best_action for step in chunk], axis=0).astype(np.float32)
            )

            valid_steps = min(self.unroll_steps, len(chunk) - 1)
            mask = np.zeros((self.unroll_steps,), dtype=np.float32)
            mask[:valid_steps] = 1.0
            for step_i in range(valid_steps):
                if chunk[step_i].done:
                    mask[step_i:] = 0.0
                    break
            masks.append(mask)
            sample_indices.append(int(flat_index))
            if hasattr(self, "_last_weights"):
                weights.append(float(self._last_weights[offset]))
            else:
                weights.append(1.0)

        batch_data = {
            "observations": np.stack(observations, axis=0),
            "actions": np.stack(actions, axis=0),
            "rewards": np.stack(rewards, axis=0),
            "policy_targets": np.stack(policy_targets, axis=0),
            "pred_values": np.stack(pred_values, axis=0),
            "search_values": np.stack(search_values, axis=0),
            "value_targets": np.stack(value_targets, axis=0),
            "policy_candidates": self._stack_batch_candidates(policy_candidates),
            "best_actions": np.stack(best_actions, axis=0),
            "dones": np.stack(dones, axis=0),
            "masks": np.stack(masks, axis=0),
            "mix_masks": np.stack(mix_masks, axis=0),
            "indices": np.asarray(sample_indices, dtype=np.int32),
            "weights": np.asarray(weights, dtype=np.float32),
        }
        return Batch(data=batch_data)

    @staticmethod
    def _stack_batch_candidates(policy_candidates: list[np.ndarray]) -> np.ndarray:
        max_window = max(item.shape[0] for item in policy_candidates)
        max_candidates = max(item.shape[1] for item in policy_candidates)
        action_dim = policy_candidates[0].shape[2]
        stacked = np.zeros(
            (len(policy_candidates), max_window, max_candidates, action_dim),
            dtype=np.float32,
        )
        for batch_index, candidates in enumerate(policy_candidates):
            window, count, dim = candidates.shape
            stacked[batch_index, :window, :count, :dim] = candidates
            if count < max_candidates:
                stacked[batch_index, :window, count:, :dim] = candidates[:, -1:, :]
        return stacked

    def _trim_to_capacity(self) -> None:
        while len(self._lookup) > self.capacity:
            if not self._stored_steps:
                self._lookup.clear()
                self._priorities.clear()
                break
            self._stored_steps.popleft()
            self._trajectories.popleft()
            self._base_traj_idx += 1
            self._lookup = [
                (traj_idx, step_idx)
                for traj_idx, step_idx in self._lookup
                if traj_idx >= self._base_traj_idx
            ]
            drop_count = len(self._priorities) - len(self._lookup)
            if drop_count > 0:
                self._priorities = self._priorities[drop_count:]

    @staticmethod
    def _stack_candidates(chunk: list[EfficientZeroStep]) -> np.ndarray:
        arrays: list[np.ndarray] = []
        for step in chunk:
            candidates = step.root_candidates
            if candidates.size == 0:
                candidates = step.best_action.reshape(1, -1)
            arrays.append(candidates.astype(np.float32))
        max_candidates = max(array.shape[0] for array in arrays)
        action_dim = arrays[0].shape[1]
        padded = np.zeros((len(arrays), max_candidates, action_dim), dtype=np.float32)
        for index, candidates in enumerate(arrays):
            padded[index, : candidates.shape[0]] = candidates
            if candidates.shape[0] < max_candidates:
                padded[index, candidates.shape[0] :] = candidates[-1]
        return padded


def _action_array(action: Action) -> np.ndarray:
    array = np.asarray(action, dtype=np.float32)
    if array.ndim == 0:
        return array.reshape(1)
    return array.reshape(-1)

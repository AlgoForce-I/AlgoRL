"""EfficientZero replay buffer with trajectory chunks and MCTS targets."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque

import numpy as np

from algorl.agents.configs import EfficientZeroConfig
from algorl.buffers.efficientzero.targets import (
    bootstrapped_values,
    extended_target_window,
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
ENV_ID_INFO_KEY = "env_id"

# Matches ``clip_inference_values`` in the learner priority computation.
_PRIORITY_VALUE_CLIP = 1e5


def _sanitize_priority(value: float, *, min_prior: float) -> float:
    """Map non-finite PER priorities to a large but sampleable finite weight."""
    if not np.isfinite(value):
        return _PRIORITY_VALUE_CLIP + min_prior
    return float(np.clip(value, min_prior, _PRIORITY_VALUE_CLIP + min_prior))


def _finite_max_priority(priorities: list[float], *, default: float = 1.0) -> float:
    finite = [priority for priority in priorities if np.isfinite(priority)]
    if not finite:
        return default
    return float(max(finite))


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
    """Fixed-size trajectory block with optional tail padding for value targets.

    ``core_len`` is the number of transitions in this block; tail steps after
    ``core_len`` are context from the next block. ``final_observation`` is the
    terminal next-observation for done episodes (used when bootstrapping values).
    """

    max_size: int
    steps: list[EfficientZeroStep] = field(default_factory=list)
    core_len: int | None = None
    final_observation: np.ndarray | None = None
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
        self.core_len = None
        self.final_observation = None
        self.bootstrapped_values = None
        self.gae_values = None
        return committed

    def pad_over(
        self,
        tail_steps: list[EfficientZeroStep],
    ) -> None:
        """Append tail context from the next trajectory block."""
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
        # Per-env-lane trajectory assembly: parallel rollouts interleave
        # transitions from independent envs, so each lane accumulates its own
        # temporally-coherent trajectory before committing to shared storage.
        self._active: dict[int, EfficientZeroTrajectory] = {}
        self._pending_commit: dict[int, EfficientZeroTrajectory] = {}
        self._total_commits = 0

    def add(self, transition: Transition) -> None:
        env_id = int(transition.info.get(ENV_ID_INFO_KEY, 0))
        step = step_from_transition(transition)
        active = self._active.get(env_id)
        if active is None:
            active = EfficientZeroTrajectory(max_size=self.trajectory_size)
            self._active[env_id] = active
        if active.append(step):
            final_observation = None
            if step.done and transition.next_observation is not None:
                final_observation = np.asarray(transition.next_observation, dtype=np.float32)
            self._commit_active(env_id, step.done, final_observation=final_observation)
        else:
            self._maybe_release_pending(env_id)

    def _maybe_release_pending(self, env_id: int) -> None:
        """Store a pending block as soon as its tail context exists.

        The pending block only needs ``trajectory_padding_gap`` steps from the
        next block for full-horizon value targets. Releasing it immediately
        (instead of waiting for the next block to fill completely) makes new
        data sampleable ~one block earlier, which matters for parallel envs
        where a full block spans ``trajectory_size * num_envs`` global steps.
        """
        pending = self._pending_commit.get(env_id)
        if pending is None:
            return
        active = self._active.get(env_id)
        gap = trajectory_padding_gap(self.config)
        if active is None or len(active.steps) < gap:
            return
        del self._pending_commit[env_id]
        pending.pad_over(active.steps[:gap])
        pending.finalize_targets(self.config)
        self._store_trajectory(pending.steps, pending)

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

    def clear(self) -> None:
        """Drop all stored and in-flight data (e.g. at a continual-learning task switch)."""
        self._trajectories.clear()
        self._stored_steps.clear()
        self._lookup.clear()
        self._priorities.clear()
        self._base_traj_idx = 0
        self._active.clear()
        self._pending_commit.clear()

    def save(self, directory: str | Path) -> None:
        """Persist stored and in-flight trajectories to a directory (no pickle)."""
        from algorl.buffers.efficientzero.checkpoint import save_efficient_zero_buffer

        save_efficient_zero_buffer(self, directory)

    def load(self, directory: str | Path) -> None:
        """Restore buffer contents from :meth:`save` into this instance."""
        from algorl.buffers.efficientzero.checkpoint import load_efficient_zero_buffer

        load_efficient_zero_buffer(self, directory)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        min_prior = float(self.config.min_prior)
        for index, priority in zip(indices.reshape(-1), priorities.reshape(-1), strict=True):
            flat_index = int(index)
            if 0 <= flat_index < len(self._priorities):
                self._priorities[flat_index] = _sanitize_priority(
                    float(priority),
                    min_prior=min_prior,
                )

    def __len__(self) -> int:
        return len(self._lookup)

    @property
    def total_transitions(self) -> int:
        return len(self._lookup)

    def _commit_active(
        self,
        env_id: int,
        done: bool,
        *,
        final_observation: np.ndarray | None = None,
    ) -> None:
        steps = self._active[env_id].clear()
        if not steps:
            return

        current = EfficientZeroTrajectory(max_size=self.trajectory_size)
        current.steps = list(steps)
        current.core_len = len(steps)
        if done:
            current.final_observation = final_observation

        pending = self._pending_commit.pop(env_id, None)
        if pending is not None:
            gap = trajectory_padding_gap(self.config)
            tail = current.steps[:gap]
            pending.pad_over(tail)
            pending.finalize_targets(self.config)
            self._store_trajectory(pending.steps, pending)

        if len(current.steps) >= self.trajectory_size and not done:
            self._pending_commit[env_id] = current
        else:
            current.finalize_targets(self.config)
            self._store_trajectory(current.steps, current)

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
        core_len = traj_meta.core_len if traj_meta.core_len is not None else len(steps)
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

        # New transitions inherit the buffer-wide max priority (optimistic PER init).
        if self.config.use_priority:
            pred_for_prior = np.asarray(pred_values[:core_len], dtype=np.float32)
            boot_for_prior = np.asarray(bootstrapped[:core_len], dtype=np.float32)
            if self.config.clip_inference_values:
                pred_for_prior = np.clip(pred_for_prior, 0.0, _PRIORITY_VALUE_CLIP)
                boot_for_prior = np.clip(boot_for_prior, 0.0, _PRIORITY_VALUE_CLIP)
            traj_priorities = np.abs(pred_for_prior - boot_for_prior) + self.config.min_prior
            max_prior = _finite_max_priority(self._priorities)
            new_priority = _sanitize_priority(
                max(max_prior, float(traj_priorities.max())),
                min_prior=self.config.min_prior,
            )
        else:
            new_priority = 1.0

        # Every core transition is a valid sample position.
        for step_pos in range(core_len):
            self._lookup.append((traj_idx, step_pos))
            self._priorities.append(new_priority)

        self._total_commits += 1
        self._trim_to_capacity()

    def _sample_indices(self, batch_size: int, *, beta: float) -> np.ndarray:
        total = len(self._lookup)
        if not self.config.use_priority:
            rng = np.random.default_rng()
            return rng.choice(total, size=batch_size, replace=False)

        # EfficientZero-V2: permanently zero priorities outside the top-transitions window.
        if total > int(self.config.top_transitions):
            cutoff = total - int(self.config.top_transitions)
            for index in range(cutoff):
                self._priorities[index] = 0.0

        priorities = np.asarray(self._priorities[:total], dtype=np.float64)
        priorities = np.nan_to_num(priorities, nan=0.0, posinf=0.0, neginf=0.0)
        priorities = np.maximum(priorities, 0.0)
        probs = priorities**self.config.priority_prob_alpha
        total_prob = probs.sum()
        if total_prob <= 0.0 or not np.isfinite(total_prob):
            probs = np.ones_like(priorities) / len(priorities)
        else:
            probs = probs / total_prob
            if not np.all(np.isfinite(probs)):
                probs = np.ones_like(priorities) / len(priorities)

        rng = np.random.default_rng()
        indices = rng.choice(total, size=batch_size, replace=False, p=probs)
        weights = (total * probs[indices]) ** (-beta)
        weights = weights / max(weights.max(), 1e-8)
        weights = np.clip(weights, 0.1, 1.0)
        self._last_weights = weights.astype(np.float32)
        return indices

    def _build_batch(self, indices: np.ndarray, *, trained_steps: int) -> Batch:
        window = self.unroll_steps + 1
        # EfficientZero-V2 computes value targets on the stored trajectory; expose the
        # extra tail so every unroll position keeps its full TD / GAE horizon.
        ext_window = max(window, extended_target_window(self.config))
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
        valid_lengths: list[int] = []
        bootstrap_limits: list[int] = []
        sample_indices: list[int] = []
        weights: list[float] = []

        for offset, flat_index in enumerate(indices):
            traj_idx, step_idx = self._lookup[int(flat_index)]
            traj_steps = self._stored_steps[traj_idx - self._base_traj_idx]
            traj_meta = self._trajectories[traj_idx - self._base_traj_idx]
            core_len = traj_meta.core_len if traj_meta.core_len is not None else len(traj_steps)
            if step_idx >= core_len:
                raise RuntimeError(
                    "Invalid replay lookup while sampling an unroll window. "
                    f"core_len={core_len}, step_idx={step_idx}."
                )
            valid_len = core_len - step_idx

            chunk = traj_steps[step_idx : step_idx + ext_window]

            obs_rows = [step.observation.astype(np.float32) for step in chunk]
            has_terminal_obs = False
            if (
                len(obs_rows) < ext_window
                and step_idx + len(chunk) == len(traj_steps)
                and traj_meta.final_observation is not None
            ):
                obs_rows.append(traj_meta.final_observation.astype(np.float32))
                has_terminal_obs = True
            while len(obs_rows) < ext_window:
                # Repeat the last frame when padding observations.
                obs_rows.append(obs_rows[-1])
            observations.append(np.stack(obs_rows[:ext_window], axis=0))

            reward_row = np.zeros((ext_window,), dtype=np.float32)
            reward_row[: len(chunk)] = [step.reward for step in chunk]
            rewards.append(reward_row)

            # Observation one step past the last core transition: available from
            # tail padding or the stored terminal observation (EfficientZero-V2 always
            # has ``obs_lst[traj_len]``, so ``bootstrap_index <= traj_len``).
            if len(chunk) > valid_len or (len(chunk) == valid_len and has_terminal_obs):
                bootstrap_limit = valid_len
            else:
                bootstrap_limit = valid_len - 1
            valid_lengths.append(valid_len)
            bootstrap_limits.append(bootstrap_limit)

            window_chunk = chunk[:window]
            n_window = len(window_chunk)

            action_row = np.zeros((self.unroll_steps, window_chunk[0].action.shape[0]), dtype=np.float32)
            for k, step in enumerate(chunk[: self.unroll_steps]):
                action_row[k] = step.action
            actions.append(action_row)

            policy_row = np.zeros((window, window_chunk[0].policy_target.shape[0]), dtype=np.float32)
            pred_row = np.zeros((window,), dtype=np.float32)
            search_row = np.zeros((window,), dtype=np.float32)
            done_row = np.zeros((window,), dtype=np.bool_)
            best_row = np.zeros((window, window_chunk[0].best_action.shape[0]), dtype=np.float32)
            for k, step in enumerate(window_chunk):
                policy_row[k] = step.policy_target
                pred_row[k] = step.pred_value
                search_row[k] = step.search_value
                done_row[k] = step.done
                best_row[k] = step.best_action
            for k in range(n_window, window):
                best_row[k] = best_row[n_window - 1]
            policy_targets.append(policy_row)
            pred_values.append(pred_row)
            search_values.append(search_row)
            dones.append(done_row)
            best_actions.append(best_row)

            candidates = self._stack_candidates(window_chunk)
            if candidates.shape[0] < window:
                pad = np.repeat(candidates[-1:], window - candidates.shape[0], axis=0)
                candidates = np.concatenate([candidates, pad], axis=0)
            policy_candidates.append(candidates)

            if traj_meta.gae_values is not None:
                stored = traj_meta.gae_values
            elif traj_meta.bootstrapped_values is not None:
                stored = traj_meta.bootstrapped_values
            else:
                stored = pred_row
            bootstrapped = np.zeros((window,), dtype=np.float32)
            source = np.asarray(stored, dtype=np.float32)[step_idx : step_idx + window]
            bootstrapped[: source.shape[0]] = source
            if 0 < source.shape[0] < window:
                bootstrapped[source.shape[0] :] = source[-1]

            search = search_row
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

            # Train unroll step k only while the next position is inside the trajectory.
            mask = np.zeros((self.unroll_steps,), dtype=np.float32)
            mask[: max(0, min(self.unroll_steps, valid_len - 1))] = 1.0
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
            "valid_lengths": np.asarray(valid_lengths, dtype=np.int32),
            "bootstrap_limits": np.asarray(bootstrap_limits, dtype=np.int32),
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

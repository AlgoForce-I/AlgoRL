"""Episode-level return and success tracking for training logs."""

from __future__ import annotations

import re
from dataclasses import dataclass


def sanitize_task_name(task_name: str) -> str:
    """Make task names safe for TensorBoard scalar tags."""
    return re.sub(r"[^A-Za-z0-9_\-./]+", "_", task_name)


def task_name_from_info(info: dict[str, object]) -> str:
    task_name = info.get("task_name")
    if isinstance(task_name, str) and task_name:
        return task_name
    task_index = info.get("task_index", info.get("seq_idx"))
    if task_index is not None:
        return f"task_{int(task_index)}"
    return "default"


def task_index_from_info(info: dict[str, object]) -> int | None:
    for key in ("task_index", "seq_idx"):
        value = info.get(key)
        if value is not None:
            return int(value)
    return None


@dataclass(frozen=True)
class EpisodeEndEvent:
    """One completed training episode or continual-learning task segment."""

    task_name: str
    task_index: int | None
    episode_return: float
    episode_length: int
    success: bool | None


def episode_metrics_from_event(event: EpisodeEndEvent) -> dict[str, float]:
    safe_name = sanitize_task_name(event.task_name)
    metrics = {
        "train/episode_return": event.episode_return,
        "train/episode_length": float(event.episode_length),
        f"train/task/{safe_name}/episode_return": event.episode_return,
        f"train/task/{safe_name}/episode_length": float(event.episode_length),
    }
    if event.success is not None:
        success_value = float(event.success)
        metrics["train/success"] = success_value
        metrics[f"train/task/{safe_name}/success"] = success_value
    return metrics


class EpisodeMetricsTracker:
    """Accumulate per-episode return and success, attributed to the active task."""

    def __init__(self, *, success_threshold: float = 0.5) -> None:
        self._success_threshold = success_threshold
        self._task_name = "default"
        self._task_index: int | None = None
        self._episode_return = 0.0
        self._episode_length = 0
        self._episode_success = False
        self._has_success_signal = False

    def begin_episode(self, info: dict[str, object]) -> None:
        self._task_name = task_name_from_info(info)
        self._task_index = task_index_from_info(info)
        self._episode_return = 0.0
        self._episode_length = 0
        self._episode_success = False
        self._has_success_signal = False

    def observe_step(
        self,
        reward: float,
        done: bool,
        info: dict[str, object],
    ) -> EpisodeEndEvent | None:
        self._episode_return += float(reward)
        self._episode_length += 1
        self._observe_success(info)

        if not done:
            return None

        return EpisodeEndEvent(
            task_name=self._task_name,
            task_index=self._task_index,
            episode_return=self._episode_return,
            episode_length=self._episode_length,
            success=self._episode_success if self._has_success_signal else None,
        )

    def metrics_from_event(self, event: EpisodeEndEvent) -> dict[str, float]:
        return episode_metrics_from_event(event)

    def _observe_success(self, info: dict[str, object]) -> None:
        if "success" not in info:
            return
        self._has_success_signal = True
        if float(info["success"]) >= self._success_threshold:
            self._episode_success = True


class BatchedEpisodeMetricsTracker:
    """Independent episode trackers for each parallel env lane."""

    def __init__(self, num_envs: int, *, success_threshold: float = 0.5) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self._num_envs = num_envs
        self._trackers = [
            EpisodeMetricsTracker(success_threshold=success_threshold)
            for _ in range(num_envs)
        ]
        self._started = [False] * num_envs

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def begin_episode(self, lane: int, info: dict[str, object]) -> None:
        self._trackers[lane].begin_episode(info)
        self._started[lane] = True

    def observe_step(
        self,
        lane: int,
        reward: float,
        done: bool,
        info: dict[str, object],
    ) -> EpisodeEndEvent | None:
        if not 0 <= lane < self._num_envs:
            raise IndexError(f"lane {lane} out of range for {self._num_envs} envs")
        if not self._started[lane]:
            self.begin_episode(lane, info)
        event = self._trackers[lane].observe_step(reward, done, info)
        if event is not None:
            self.begin_episode(lane, info)
        return event

    def metrics_from_event(self, event: EpisodeEndEvent) -> dict[str, float]:
        return episode_metrics_from_event(event)


def batched_episode_summary_metrics(events: list[EpisodeEndEvent]) -> dict[str, float]:
    """Aggregate per-lane episode completions from one vector-env step."""
    if not events:
        return {}
    returns = [float(event.episode_return) for event in events]
    lengths = [float(event.episode_length) for event in events]
    metrics: dict[str, float] = {
        "train/batched/mean_episode_return": float(sum(returns) / len(returns)),
        "train/batched/mean_episode_length": float(sum(lengths) / len(lengths)),
        "train/batched/episode_completions": float(len(events)),
    }
    successes = [event.success for event in events if event.success is not None]
    if successes:
        metrics["train/batched/episode_success_frac"] = float(sum(bool(s) for s in successes) / len(successes))
    return metrics

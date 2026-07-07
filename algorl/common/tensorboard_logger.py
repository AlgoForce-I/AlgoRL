"""TensorBoard-backed training logger."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from algorl.common.episode_metrics import EpisodeEndEvent, episode_metrics_from_event, sanitize_task_name
from algorl.common.logger import Logger


class _ScalarWriter:
    """Minimal TensorBoard scalar writer without a PyTorch dependency."""

    def __init__(self, log_dir: str) -> None:
        from tensorboard.compat.proto import event_pb2, summary_pb2
        from tensorboard.summary.writer.event_file_writer import EventFileWriter

        self._summary_pb2 = summary_pb2
        self._event_pb2 = event_pb2
        self._writer = EventFileWriter(log_dir)

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        summary_value = self._summary_pb2.Summary.Value(tag=tag, simple_value=float(value))
        summary = self._summary_pb2.Summary(value=[summary_value])
        event = self._event_pb2.Event(
            wall_time=time.time(),
            step=int(step),
            summary=summary,
        )
        self._writer.add_event(event)

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()


class TensorboardLogger(Logger):
    """Logger that mirrors scalar metrics to TensorBoard event files."""

    def __init__(self, log_dir: str) -> None:
        super().__init__()
        self.log_dir = log_dir
        self._writer = _ScalarWriter(log_dir)
        self._closed = False
        self._task_success_totals: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
        self._global_success_total = 0
        self._global_success_count = 0

    def record(self, step: int, metrics: dict[str, Any]) -> None:
        super().record(step, metrics)
        for key, value in metrics.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                self._writer.add_scalar(key, float(value), step)

    def flush(self) -> None:
        """Flush pending TensorBoard events so dashboards update promptly."""
        self._writer.flush()

    def record_episode(self, step: int, event: EpisodeEndEvent) -> dict[str, float]:
        """Log episode metrics and update per-task success rates."""
        metrics = episode_metrics_from_event(event)
        if event.success is not None:
            safe_name = sanitize_task_name(event.task_name)
            task_wins, task_total = self._task_success_totals[event.task_name]
            task_wins += int(event.success)
            task_total += 1
            self._task_success_totals[event.task_name] = (task_wins, task_total)
            task_rate = task_wins / task_total
            metrics[f"train/task/{safe_name}/success_rate"] = task_rate

            self._global_success_total += int(event.success)
            self._global_success_count += 1
            metrics["train/success_rate"] = self._global_success_total / self._global_success_count

        self.record(step, metrics)
        return metrics

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        self._writer.flush()
        self._writer.close()

"""Helpers / callbacks for HyperCEZ continual-learning retention metrics."""

from __future__ import annotations

from typing import Any


class HyperCEZRetentionCallback:
    """Copy learner fix-target retention metrics into the training step info.

    The learner already emits ``retention/*`` scalars every
    ``retention_log_interval`` train steps. This callback additionally pulls
    ``retention_target_metrics()`` onto ``info`` every ``retention_every_steps``
    env steps so TensorBoard keeps those scalars even when the train step did
    not emit them. It does not run environment evaluation; use
    ``agent.learn(eval_period=...)`` for that.
    """

    def __init__(self, learner: Any, *, retention_every_steps: int = 50_000) -> None:
        self.learner = learner
        self.retention_every_steps = max(0, int(retention_every_steps))
        self._last_retention_step = -1

    def on_step(self, step: int, info: dict[str, Any]) -> None:
        if self.retention_every_steps <= 0:
            return
        if step - self._last_retention_step < self.retention_every_steps:
            return
        metrics_fn = getattr(self.learner, "retention_target_metrics", None)
        if not callable(metrics_fn):
            return
        self._last_retention_step = step
        for key, value in metrics_fn().items():
            info.setdefault(key if key.startswith("train/") else f"train/{key}", value)

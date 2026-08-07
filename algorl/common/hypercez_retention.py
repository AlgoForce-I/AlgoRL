"""Helpers / callbacks for HyperCEZ continual-learning retention metrics."""

from __future__ import annotations

from typing import Any


class HyperCEZRetentionCallback:
    """Log learner fix-target retention metrics into the training step info.

    The learner already emits ``retention/*`` scalars every
    ``retention_log_interval`` train steps. This callback additionally copies
    any such keys already present on ``info`` (no-op) and can be extended to
    trigger heavier eval rollouts.
    """

    def __init__(self, learner: Any, *, eval_every_steps: int = 50_000) -> None:
        self.learner = learner
        self.eval_every_steps = max(0, int(eval_every_steps))
        self._last_eval_step = -1

    def on_step(self, step: int, info: dict[str, Any]) -> None:
        if self.eval_every_steps <= 0:
            return
        if step - self._last_eval_step < self.eval_every_steps:
            return
        metrics_fn = getattr(self.learner, "retention_target_metrics", None)
        if not callable(metrics_fn):
            return
        self._last_eval_step = step
        for key, value in metrics_fn().items():
            info.setdefault(key if key.startswith("train/") else f"train/{key}", value)

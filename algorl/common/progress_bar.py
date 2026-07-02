"""Training progress display backed by tqdm."""

from __future__ import annotations

from typing import Any


class TqdmProgressBar:
    """Step counter with optional metric postfix for :class:`~algorl.core.training_loop.TrainingLoop`."""

    def __init__(self, *, desc: str = "train", **kwargs: Any) -> None:
        self._desc = desc
        self._kwargs = kwargs
        self._bar: Any | None = None

    def start(self, total: int, **kwargs: Any) -> None:
        from tqdm import tqdm

        merged = {**self._kwargs, **kwargs}
        self._bar = tqdm(total=total, desc=self._desc, **merged)

    def update(self, step_info: dict[str, Any] | None = None, *, n: int = 1) -> None:
        if self._bar is None:
            return
        increment = max(0, int(n))
        if increment <= 0:
            return
        postfix = _postfix_from_step_info(step_info)
        if postfix:
            self._bar.set_postfix(postfix, refresh=True)
        self._bar.update(increment)

    def pulse(self, step_info: dict[str, Any] | None = None) -> None:
        """Refresh postfix without advancing the timestep counter."""
        if self._bar is None:
            return
        postfix = _postfix_from_step_info(step_info)
        if postfix:
            self._bar.set_postfix(postfix, refresh=True)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None


def _postfix_from_step_info(step_info: dict[str, Any] | None) -> dict[str, str]:
    if not step_info:
        return {}

    labels = {
        "train/reward": "reward",
        "train/loss": "loss",
        "loss": "loss",
        "policy_loss": "pi_loss",
        "value_loss": "v_loss",
        "reward_loss": "r_loss",
        "consistency_loss": "cons_loss",
        "train/episode_return": "ep_ret",
        "train/episode_success": "success",
    }
    postfix: dict[str, str] = {}
    phase = step_info.get("phase")
    if isinstance(phase, str) and phase:
        postfix["phase"] = phase
    reanalyze = step_info.get("reanalyze")
    if isinstance(reanalyze, str) and reanalyze:
        postfix["reanalyze"] = reanalyze
    for key, label in labels.items():
        value = step_info.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            postfix[label] = f"{float(value):.3f}"
    task = step_info.get("task_name")
    if isinstance(task, str) and task:
        postfix["task"] = task
    return postfix

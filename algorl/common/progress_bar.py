"""Training progress display backed by tqdm."""

from __future__ import annotations

from typing import Any


class TqdmProgressBar:
    """Step counter with optional metric postfix for :class:`~algorl.core.training_loop.TrainingLoop`."""

    def __init__(self, *, desc: str = "train", **kwargs: Any) -> None:
        self._desc = desc
        self._kwargs = kwargs
        self._bar: Any | None = None

    def start(self, total: int) -> None:
        from tqdm import tqdm

        self._bar = tqdm(total=total, desc=self._desc, **self._kwargs)

    def update(self, step_info: dict[str, Any] | None = None) -> None:
        if self._bar is None:
            return
        postfix = _postfix_from_step_info(step_info)
        if postfix:
            self._bar.set_postfix(postfix, refresh=False)
        self._bar.update(1)

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
        "train/episode_return": "ep_ret",
        "train/episode_success": "success",
    }
    postfix: dict[str, str] = {}
    for key, label in labels.items():
        value = step_info.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            postfix[label] = f"{float(value):.3f}"
    return postfix

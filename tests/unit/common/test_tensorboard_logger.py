"""TensorBoard logger tests."""

from __future__ import annotations

import tempfile

from algorl.common.episode_metrics import EpisodeEndEvent
from algorl.common.tensorboard_logger import TensorboardLogger


def test_tensorboard_logger_records_episode_and_success_rate() -> None:
    with tempfile.TemporaryDirectory() as log_dir:
        logger = TensorboardLogger(log_dir)
        event = EpisodeEndEvent(
            task_name="hammer-v3",
            task_index=0,
            episode_return=12.5,
            episode_length=20,
            success=True,
        )
        metrics = logger.record_episode(10, event)

        assert metrics["train/episode_return"] == 12.5
        assert metrics["train/task/hammer-v3/success_rate"] == 1.0
        assert metrics["train/success_rate"] == 1.0
        assert logger.history[-1]["train/episode_return"] == 12.5

        event_2 = EpisodeEndEvent(
            task_name="hammer-v3",
            task_index=0,
            episode_return=3.0,
            episode_length=5,
            success=False,
        )
        metrics_2 = logger.record_episode(20, event_2)
        assert metrics_2["train/task/hammer-v3/success_rate"] == 0.5
        logger.close()

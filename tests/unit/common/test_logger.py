"""Logger history retention tests."""

from __future__ import annotations

from algorl.common.logger import Logger


def test_logger_history_is_bounded() -> None:
    logger = Logger(history_limit=10)
    for step in range(1_000):
        logger.record(step, {"train/reward": float(step)})

    # Trimming happens in blocks, so the bound is the limit, not an exact size.
    assert 10 <= len(logger.history) <= 20
    assert logger.history[-1]["step"] == 999


def test_logger_history_unbounded_when_limit_is_none() -> None:
    logger = Logger(history_limit=None)
    for step in range(100):
        logger.record(step, {"train/reward": 0.0})

    assert len(logger.history) == 100

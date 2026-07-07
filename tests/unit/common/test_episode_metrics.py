"""Episode metrics tracker tests."""

from algorl.common.episode_metrics import EpisodeMetricsTracker


def test_episode_tracker_logs_task_segment_return() -> None:
    tracker = EpisodeMetricsTracker()
    tracker.begin_episode({"task_name": "hammer-v3", "seq_idx": 0})

    assert tracker.observe_step(1.0, False, {"success": 0.0}) is None
    event = tracker.observe_step(2.0, True, {"success": 1.0, "task_name": "push-wall-v3"})

    assert event is not None
    assert event.task_name == "hammer-v3"
    assert event.episode_return == 3.0
    assert event.episode_length == 2
    assert event.success is True

    metrics = tracker.metrics_from_event(event)
    assert metrics["train/task/hammer-v3/episode_return"] == 3.0
    assert metrics["train/task/hammer-v3/success"] == 1.0


def test_episode_tracker_defaults_without_task_info() -> None:
    tracker = EpisodeMetricsTracker()
    tracker.begin_episode({})
    event = tracker.observe_step(0.5, True, {})

    assert event is not None
    assert event.task_name == "default"
    assert event.success is None

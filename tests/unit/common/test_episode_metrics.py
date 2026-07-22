"""Episode metrics tracker tests."""

from algorl.common.episode_metrics import (
    BatchedEpisodeMetricsTracker,
    EpisodeMetricsTracker,
    batched_episode_summary_metrics,
)
from algorl.common.episode_metrics import EpisodeEndEvent


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


def test_batched_episode_tracker_isolates_lanes() -> None:
    tracker = BatchedEpisodeMetricsTracker(2)

    assert tracker.observe_step(0, 1.0, False, {}) is None
    assert tracker.observe_step(1, 10.0, False, {}) is None

    event_lane_0 = tracker.observe_step(0, 2.0, True, {})
    assert event_lane_0 is not None
    assert event_lane_0.episode_return == 3.0
    assert event_lane_0.episode_length == 2

    assert tracker.observe_step(1, 5.0, False, {}) is None
    event_lane_1 = tracker.observe_step(1, 7.0, True, {})
    assert event_lane_1 is not None
    assert event_lane_1.episode_return == 22.0
    assert event_lane_1.episode_length == 3


def test_episode_tracker_records_search_return_gap() -> None:
    tracker = EpisodeMetricsTracker()
    tracker.begin_episode({})
    assert tracker.observe_step(1.0, False, {"search_value": 10.0, "pred_value": 8.0}) is None
    event = tracker.observe_step(2.0, True, {})

    assert event is not None
    assert event.search_value_start == 10.0
    assert event.pred_value_start == 8.0

    metrics = tracker.metrics_from_event(event)
    assert metrics["train/search_return_gap"] == 7.0
    assert metrics["train/search_return_abs_gap"] == 7.0
    assert metrics["train/pred_return_gap"] == 5.0
    assert metrics["train/pred_return_abs_gap"] == 5.0


def test_batched_episode_summary_metrics() -> None:
    events = [
        EpisodeEndEvent("default", None, 3.0, 2, None, search_value_start=10.0),
        EpisodeEndEvent("default", None, 9.0, 4, True, search_value_start=20.0, pred_value_start=18.0),
    ]
    metrics = batched_episode_summary_metrics(events)
    assert metrics["train/batched/mean_episode_return"] == 6.0
    assert metrics["train/batched/mean_episode_length"] == 3.0
    assert metrics["train/batched/episode_completions"] == 2.0
    assert metrics["train/batched/episode_success_frac"] == 1.0
    assert metrics["train/batched/mean_search_return_gap"] == 9.0
    assert metrics["train/batched/mean_search_return_abs_gap"] == 9.0
    assert metrics["train/batched/mean_pred_return_gap"] == 9.0
    assert metrics["train/batched/mean_pred_return_abs_gap"] == 9.0

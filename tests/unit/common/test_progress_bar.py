"""Tests for training progress display."""

from __future__ import annotations

from unittest import mock

from algorl.common.progress_bar import TqdmProgressBar


def test_tqdm_progress_bar_uses_realtime_refresh_defaults() -> None:
    bar = TqdmProgressBar()
    fake_tqdm = mock.Mock()
    with mock.patch("tqdm.tqdm", fake_tqdm):
        bar.start(10)
    fake_tqdm.assert_called_once()
    kwargs = fake_tqdm.call_args.kwargs
    assert kwargs["mininterval"] == 0
    assert kwargs["miniters"] == 1
    assert kwargs["smoothing"] == 0
    assert kwargs["initial"] == 0


def test_tqdm_progress_bar_starts_at_resumed_step() -> None:
    bar = TqdmProgressBar()
    fake_tqdm = mock.Mock()
    with mock.patch("tqdm.tqdm", fake_tqdm):
        bar.start(10_000_000, initial=1_000_000)
    kwargs = fake_tqdm.call_args.kwargs
    assert kwargs["total"] == 10_000_000
    assert kwargs["initial"] == 1_000_000

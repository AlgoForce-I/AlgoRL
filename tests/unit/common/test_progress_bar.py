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

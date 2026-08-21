"""Tests for training progress display."""

from __future__ import annotations

from unittest import mock

from algorl.common.progress_bar import TqdmProgressBar, _postfix_from_step_info


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


def test_tqdm_progress_bar_postfix_includes_eval_phase() -> None:
    postfix = _postfix_from_step_info(
        {
            "phase": "eval",
            "eval": "hammer-v3 4/10",
            "eval_step": "80/200",
            "eval/mean_return": 12.5,
        }
    )
    assert postfix["phase"] == "eval"
    assert postfix["eval"] == "hammer-v3 4/10"
    assert postfix["eval_step"] == "80/200"
    assert postfix["eval_ret"] == "12.500"

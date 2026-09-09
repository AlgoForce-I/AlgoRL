"""Tests for JAX memory / multiprocessing helpers."""

from __future__ import annotations

import multiprocessing as mp
import os
from unittest import mock

from algorl.backends.jax.memory import (
    collect_device_memory_metrics,
    collect_memory_metrics,
    collect_ram_metrics,
    collect_vram_metrics,
    configure_jax_gpu_memory,
)


def test_collect_ram_metrics_reports_rss() -> None:
    metrics = collect_ram_metrics()
    assert "system/ram_rss_gb" in metrics
    assert metrics["system/ram_rss_gb"] > 0.0


def test_collect_memory_metrics_includes_ram() -> None:
    metrics = collect_memory_metrics()
    assert "system/ram_rss_gb" in metrics
    # VRAM keys are optional on CPU-only / unavailable devices.
    assert all(key.startswith("system/") for key in metrics)
    _ = collect_vram_metrics()  # smoke; may be empty


def test_configure_jax_gpu_memory_sets_allocator_env() -> None:
    with mock.patch.dict(os.environ, {}, clear=True):
        configure_jax_gpu_memory(preallocate=False, memory_fraction=0.5)
        assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
        assert os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.5"


def test_configure_jax_gpu_memory_reserves_room_for_mjx() -> None:
    with mock.patch.dict(os.environ, {}, clear=True):
        with mock.patch(
            "algorl.backends.jax.memory._device_total_memory_bytes",
            return_value=32 * 1024**3,
        ):
            configure_jax_gpu_memory(memory_fraction=0.85, reserve_gb=8.0)
        assert float(os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]) == 0.75


def test_configure_jax_gpu_memory_reserve_never_raises_the_fraction() -> None:
    with mock.patch.dict(os.environ, {}, clear=True):
        with mock.patch(
            "algorl.backends.jax.memory._device_total_memory_bytes",
            return_value=32 * 1024**3,
        ):
            configure_jax_gpu_memory(memory_fraction=0.5, reserve_gb=1.0)
        assert float(os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]) == 0.5


def test_configure_jax_gpu_memory_reserve_falls_back_without_a_device() -> None:
    with mock.patch.dict(os.environ, {}, clear=True):
        with mock.patch(
            "algorl.backends.jax.memory._device_total_memory_bytes",
            return_value=None,
        ):
            configure_jax_gpu_memory(memory_fraction=0.85, reserve_gb=8.0)
        assert os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.85"


def test_collect_device_memory_metrics_is_empty_without_warp() -> None:
    with mock.patch.dict("sys.modules", {}, clear=False) as modules:
        modules.pop("warp", None)
        assert collect_device_memory_metrics() == {}


def test_configure_jax_gpu_memory_uses_cpu_in_spawn_workers() -> None:
    with mock.patch.dict(os.environ, {}, clear=True):
        with mock.patch(
            "algorl.backends.jax.memory.mp.current_process",
            return_value=mock.Mock(name="SpawnProcess-1"),
        ):
            configure_jax_gpu_memory()
        assert os.environ["JAX_PLATFORMS"] == "cpu"
        assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


def _spawn_worker_probe(result_queue: mp.Queue[str]) -> None:
    configure_jax_gpu_memory()
    result_queue.put(os.environ.get("JAX_PLATFORMS", ""))


def test_configure_jax_gpu_memory_in_real_spawn_child() -> None:
    ctx = mp.get_context("spawn")
    queue: mp.Queue[str] = ctx.Queue()
    process = ctx.Process(target=_spawn_worker_probe, args=(queue,))
    process.start()
    process.join(timeout=30)
    assert process.exitcode == 0
    assert queue.get(timeout=5) == "cpu"

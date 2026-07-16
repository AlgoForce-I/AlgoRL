"""Tests for JAX memory / multiprocessing helpers."""

from __future__ import annotations

import multiprocessing as mp
import os
from unittest import mock

from algorl.backends.jax.memory import configure_jax_gpu_memory


def test_configure_jax_gpu_memory_sets_allocator_env() -> None:
    with mock.patch.dict(os.environ, {}, clear=True):
        configure_jax_gpu_memory(preallocate=False, memory_fraction=0.5)
        assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
        assert os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.5"


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

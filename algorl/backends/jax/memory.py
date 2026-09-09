"""JAX GPU memory helpers.

Call :func:`configure_jax_gpu_memory` before importing JAX when possible
(``example.py`` does this at module top). It also pins spawned Gymnasium
vector-env workers to CPU JAX so re-importing user scripts does not claim GPU
memory in every subprocess.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import resource
import subprocess
import sys
from typing import Any

_GIB = 1024.0**3


def _configure_jax_for_subprocess_workers() -> None:
    if mp.current_process().name == "MainProcess":
        return
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def _bytes_to_gb(num_bytes: int | float) -> float:
    return float(num_bytes) / _GIB


def _read_proc_status_kb(key: str) -> int | None:
    """Read a kB field from ``/proc/self/status`` (Linux)."""
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith(key):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1])
    except (OSError, ValueError):
        return None
    return None


def _read_meminfo_kb(key: str) -> int | None:
    """Read a kB field from ``/proc/meminfo`` (Linux)."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith(key):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1])
    except (OSError, ValueError):
        return None
    return None


def collect_ram_metrics() -> dict[str, float]:
    """Host RAM metrics for TensorBoard (process RSS + system totals when available)."""
    metrics: dict[str, float] = {}
    rss_kb = _read_proc_status_kb("VmRSS:")
    if rss_kb is not None:
        metrics["system/ram_rss_gb"] = rss_kb / (1024.0**2)
    else:
        # Linux: ru_maxrss is kB; macOS: bytes. Prefer /proc above on Linux.
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if usage > 1 << 32:  # likely bytes (macOS)
            metrics["system/ram_rss_gb"] = _bytes_to_gb(usage)
        else:
            metrics["system/ram_rss_gb"] = usage / (1024.0**2)

    mem_total = _read_meminfo_kb("MemTotal:")
    mem_available = _read_meminfo_kb("MemAvailable:")
    if mem_total is not None:
        metrics["system/ram_system_total_gb"] = mem_total / (1024.0**2)
    if mem_available is not None:
        metrics["system/ram_system_available_gb"] = mem_available / (1024.0**2)
    if mem_total is not None and mem_available is not None:
        used_kb = mem_total - mem_available
        metrics["system/ram_system_used_gb"] = used_kb / (1024.0**2)
        metrics["system/ram_system_percent"] = 100.0 * used_kb / max(1, mem_total)
    return metrics


def collect_vram_metrics() -> dict[str, float]:
    """Device allocator stats from the default JAX accelerator (GB)."""
    metrics: dict[str, float] = {}
    try:
        import jax

        device = jax.local_devices()[0]
        if device.platform == "cpu":
            return metrics
        stats: dict[str, Any] | None = device.memory_stats()
        if not stats:
            return metrics
    except Exception:  # pragma: no cover - platform-specific probing
        return metrics

    mapping = (
        ("bytes_in_use", "system/vram_bytes_in_use_gb"),
        ("peak_bytes_in_use", "system/vram_peak_bytes_in_use_gb"),
        ("bytes_reserved", "system/vram_bytes_reserved_gb"),
        ("peak_bytes_reserved", "system/vram_peak_bytes_reserved_gb"),
        ("bytes_limit", "system/vram_bytes_limit_gb"),
        ("bytes_reservable_limit", "system/vram_bytes_reservable_limit_gb"),
    )
    for src, dest in mapping:
        if src in stats and stats[src] is not None:
            metrics[dest] = _bytes_to_gb(int(stats[src]))

    in_use = stats.get("bytes_in_use")
    limit = stats.get("bytes_limit") or stats.get("bytes_reservable_limit")
    if in_use is not None and limit:
        metrics["system/vram_utilization_percent"] = 100.0 * float(in_use) / float(limit)
    return metrics


def collect_device_memory_metrics() -> dict[str, float]:
    """Driver-level device memory, including bytes allocated outside JAX.

    MJX's warp backend allocates through its own CUDA allocator, so none of the
    physics memory shows up in :func:`collect_vram_metrics`. Those allocations
    are the ones that fail first once the JAX pool has grown to fill the card,
    which makes ``system/gpu_free_gb`` the scalar to watch for MJX runs.
    """
    warp = sys.modules.get("warp")
    if warp is None:
        return {}
    try:
        device = warp.get_device("cuda:0")
        total = int(device.total_memory)
        free = int(device.free_memory)
    except Exception:  # pragma: no cover - platform-specific probing
        return {}
    metrics = {
        "system/gpu_total_gb": _bytes_to_gb(total),
        "system/gpu_free_gb": _bytes_to_gb(free),
        "system/gpu_used_gb": _bytes_to_gb(total - free),
    }
    try:
        metrics["system/gpu_warp_pool_gb"] = _bytes_to_gb(
            warp.get_mempool_used_mem_current(device)
        )
        metrics["system/gpu_warp_pool_peak_gb"] = _bytes_to_gb(
            warp.get_mempool_used_mem_high(device)
        )
    except Exception:  # pragma: no cover - optional warp API
        pass
    return metrics


def collect_memory_metrics() -> dict[str, float]:
    """Combined host RAM + device VRAM scalars for logging."""
    return {
        **collect_ram_metrics(),
        **collect_vram_metrics(),
        **collect_device_memory_metrics(),
    }


def _device_total_memory_bytes() -> int | None:
    """Total memory of GPU 0, queried without creating a CUDA context."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return None
    try:
        return int(float(lines[0])) * 1024 * 1024
    except ValueError:
        return None


def gpu_available_memory_bytes() -> int | None:
    """Free device memory on the default JAX accelerator, or ``None`` if unknown.

    Uses the allocator's own view (``bytes_limit`` respects
    ``XLA_PYTHON_CLIENT_MEM_FRACTION``), so results account for memory already
    claimed by compiled programs and live buffers.
    """
    try:
        import jax

        device = jax.local_devices()[0]
        if device.platform == "cpu":
            return None
        stats = device.memory_stats()
        if not stats:
            return None
        limit = stats.get("bytes_limit") or stats.get("bytes_reservable_limit")
        if limit is None:
            return None
        in_use = stats.get("bytes_in_use", 0)
        return max(0, int(limit) - int(in_use))
    except Exception:  # pragma: no cover - platform-specific probing
        return None


def configure_jax_gpu_memory(
    *,
    preallocate: bool = False,
    memory_fraction: float | None = 0.85,
    reserve_gb: float | None = None,
) -> None:
    """Tune XLA GPU allocator for long EZ runs (EfficientZero-V2-like PyTorch footprint).

    JAX defaults to preallocating nearly all GPU memory, which makes OOM look
    sudden and leaves little room for MJX + MCTS + backward peaks.

    ``reserve_gb`` caps the JAX pool so that many gigabytes stay free for
    allocators JAX does not manage. MJX's warp backend is the important one: it
    allocates models, simulation data, and collision workspaces with its own
    CUDA allocator, so a JAX pool that grows to its fraction limit starves the
    physics engine and env construction fails with warp's own
    ``Failed to allocate N bytes on device 'cuda:0'``. CW10 with 32 training
    lanes plus cached eval envs needs roughly 2.5 GB there, and peaks higher
    while stepping, so reserve a few GB when running MJX.

    When Gymnasium ``AsyncVectorEnv`` uses ``spawn``, worker processes re-import
    the user script. Call this function before any JAX import so workers only
    initialize CPU JAX and avoid GPU OOM during env startup.
    """
    _configure_jax_for_subprocess_workers()
    if not preallocate:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if reserve_gb is not None:
        total_bytes = _device_total_memory_bytes()
        if total_bytes:
            reserved = max(0.0, float(reserve_gb)) * _GIB
            capped = max(0.05, (total_bytes - reserved) / total_bytes)
            memory_fraction = min(
                capped, 1.0 if memory_fraction is None else float(memory_fraction)
            )
    if memory_fraction is not None:
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(memory_fraction))

"""Throughput instrumentation for implementation-vs-implementation benchmarks.

Records the quantities a speed comparison has to report together: throughput,
the replay ratio it was achieved at, where the wall-clock went, and how loaded
the device was. Reporting throughput without the replay ratio is meaningless,
because collecting more env steps per gradient step raises steps/s while doing
less work per sample.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from collections import deque
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

_HARDWARE_QUERY = (
    "nvidia-smi",
    "--query-gpu=utilization.gpu,utilization.memory,memory.used,clocks.sm",
    "--format=csv,noheader,nounits",
)


def _gpu_sample() -> dict[str, float]:
    """GPU utilisation for device 0, or ``{}`` when nvidia-smi is unavailable."""
    try:
        result = subprocess.run(_HARDWARE_QUERY, capture_output=True, text=True, timeout=15, check=True)
    except (OSError, subprocess.SubprocessError):
        return {}
    line = next((ln for ln in result.stdout.splitlines() if ln.strip()), "")
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 4:
        return {}
    try:
        return {
            "bench/gpu_util_pct": float(parts[0]),
            "bench/gpu_mem_util_pct": float(parts[1]),
            "bench/gpu_mem_used_mb": float(parts[2]),
            "bench/gpu_sm_clock_mhz": float(parts[3]),
        }
    except ValueError:
        return {}


def _cpu_sample() -> dict[str, float]:
    try:
        import psutil
    except ImportError:
        return {}
    process = psutil.Process()
    return {
        "bench/cpu_util_pct": float(psutil.cpu_percent(interval=None)),
        "bench/proc_cpu_util_pct": float(process.cpu_percent(interval=None)),
        "bench/proc_rss_gb": float(process.memory_info().rss) / 2**30,
    }


class ThroughputCallback:
    """Log throughput, replay ratio, phase split and device load during training.

    Attaches through the ordinary callback hook, so it works for any agent and
    adds no cost to the training path beyond a dictionary update per env step.
    Device sampling shells out to ``nvidia-smi`` and is rate-limited by
    ``hardware_every_s`` rather than run per step.

    The learner's train-step counter drives two of the metrics: the replay
    ratio (gradient steps per env step, which a sweep must hold fixed) and the
    phase split (an interval where the counter advanced is a training burst).
    """

    def __init__(
        self,
        learner: Any,
        *,
        log_every_steps: int = 100,
        window_steps: int = 2_000,
        hardware_every_s: float = 10.0,
    ) -> None:
        self.learner = learner
        self.log_every_steps = max(1, int(log_every_steps))
        self.window_steps = max(2, int(window_steps))
        self.hardware_every_s = float(hardware_every_s)
        self._created_at = time.perf_counter()
        self._first_step_at: float | None = None
        self._first_step: int | None = None
        self._history: deque[tuple[float, int, int]] = deque(maxlen=self.window_steps)
        self._last_sample: tuple[float, int, int] | None = None
        self._hardware: dict[str, float] = {}
        self._hardware_at = 0.0
        self._train_seconds = 0.0
        self._rollout_seconds = 0.0
        self._last_logged_step = -1
        self._train_began_at: float | None = None
        self._train_began_step: int | None = None

    def _train_steps(self) -> int:
        for attribute in ("_train_steps", "train_steps"):
            value = getattr(self.learner, attribute, None)
            if value is not None:
                return int(value)
        return 0

    def on_step(self, step: int, info: dict[str, Any]) -> None:
        now = time.perf_counter()
        trained = self._train_steps()

        if self._first_step_at is None:
            # Everything before the first env step: env construction, model
            # build and the first JIT compilations.
            self._first_step_at = now
            self._first_step = int(step)
            info["bench/startup_s"] = now - self._created_at

        if trained > 0 and self._train_began_at is None:
            # Throughput before the first gradient step is rollout-only and far
            # higher than the steady state; the sweep must compare steady states.
            self._train_began_at = now
            self._train_began_step = int(step)

        if self._last_sample is not None:
            previous_time, _, previous_trained = self._last_sample
            elapsed = now - previous_time
            if trained > previous_trained:
                self._train_seconds += elapsed
            else:
                self._rollout_seconds += elapsed
        self._last_sample = (now, int(step), trained)
        self._history.append((now, int(step), trained))

        if step - self._last_logged_step < self.log_every_steps:
            return
        self._last_logged_step = int(step)
        info.update(self.metrics(now=now, step=step, trained=trained))

    def metrics(self, *, now: float, step: int, trained: int) -> dict[str, float]:
        out: dict[str, float] = {}
        if self._first_step_at is None or self._first_step is None:
            return out

        total_elapsed = now - self._first_step_at
        total_steps = step - self._first_step
        if total_elapsed > 0 and total_steps > 0:
            out["bench/env_steps_per_s_avg"] = total_steps / total_elapsed
        out["bench/wall_clock_s"] = total_elapsed
        out["bench/env_steps"] = float(total_steps)
        out["bench/train_steps"] = float(trained)
        if total_steps > 0:
            # The control variable of a throughput sweep: throughput is only
            # comparable between runs that do the same work per env step.
            out["bench/grad_steps_per_env_step"] = trained / total_steps

        if self._train_began_at is not None and self._train_began_step is not None:
            steady_elapsed = now - self._train_began_at
            steady_steps = step - self._train_began_step
            if steady_elapsed > 0 and steady_steps > 0:
                out["bench/env_steps_per_s_steady"] = steady_steps / steady_elapsed
            out["bench/warmup_env_steps"] = float(self._train_began_step - self._first_step)

        oldest_time, oldest_step, _ = self._history[0]
        window_elapsed = now - oldest_time
        window_steps = step - oldest_step
        if window_elapsed > 0 and window_steps > 0:
            out["bench/env_steps_per_s"] = window_steps / window_elapsed

        accounted = self._train_seconds + self._rollout_seconds
        if accounted > 0:
            out["bench/train_time_frac"] = self._train_seconds / accounted
            out["bench/rollout_time_frac"] = self._rollout_seconds / accounted

        if now - self._hardware_at >= self.hardware_every_s:
            self._hardware = {**_gpu_sample(), **_cpu_sample()}
            self._hardware_at = now
        out.update(self._hardware)
        return out

    def summary(self) -> dict[str, Any]:
        """Final numbers for the benchmark table."""
        now = time.perf_counter()
        trained = self._train_steps()
        step = self._last_sample[1] if self._last_sample else 0
        out: dict[str, Any] = {
            "startup_s": (self._first_step_at - self._created_at) if self._first_step_at else None,
            "total_wall_clock_s": now - self._created_at,
            "train_seconds": self._train_seconds,
            "rollout_seconds": self._rollout_seconds,
        }
        out.update(self.metrics(now=now, step=step, trained=trained))
        return out


def hardware_description() -> dict[str, Any]:
    """Host and device identity, so a number can be attributed to a machine."""
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        pass
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=True,
        )
        info["gpu"] = result.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
    for module in ("jax", "jaxlib", "torch"):
        try:
            info[module] = __import__(module).__version__
        except Exception:  # noqa: BLE001 - version probing is best effort
            pass
    try:
        info["git_commit"] = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def write_benchmark_record(
    directory: str | Path,
    *,
    name: str,
    config: Any,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write one JSON record describing a benchmark point (config + hardware)."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "name": name,
        "hardware": hardware_description(),
        "config": json.loads(json.dumps(asdict(config) if is_dataclass(config) else vars(config), default=str)),
    }
    if extra:
        payload.update(extra)
    path = target / "benchmark.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path

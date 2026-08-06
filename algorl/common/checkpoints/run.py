"""Orchestrate multi-file run checkpoints (agent + loop + env + buffer)."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np

from algorl.common.checkpoints.atomic import commit_staging_dir, prepare_staging_dir
from algorl.common.checkpoints.manifest import (
    build_manifest,
    read_json,
    read_manifest,
    write_json,
    write_manifest,
)


def _config_to_jsonable(config: Any) -> dict[str, Any]:
    if is_dataclass(config):
        raw = asdict(config)
    elif hasattr(config, "__dict__"):
        raw = dict(config.__dict__)
    else:
        raise TypeError(f"Cannot serialize config type {type(config)!r}.")
    return json.loads(json.dumps(raw, default=str))


def _find_continual_env(env: Any) -> Any | None:
    for candidate in (
        env,
        getattr(env, "raw", None),
        getattr(env, "unwrapped", None),
        getattr(getattr(env, "raw", None), "jax_env", None),
    ):
        if candidate is None:
            continue
        if hasattr(candidate, "curriculum_checkpoint_state") or hasattr(
            candidate, "_seq_idx"
        ):
            return candidate
    return None


def save_run_checkpoint(
    *,
    directory: str | Path,
    agent_name: str,
    step: int,
    config: Any,
    learner: Any,
    replay_buffer: Any,
    env: Any,
    loop_state: dict[str, Any],
    logger_history: list[dict[str, Any]] | None = None,
    tag: str | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    """Atomically write a full training-run checkpoint directory."""
    final_dir = Path(directory)
    staging = prepare_staging_dir(final_dir)
    artifacts: dict[str, str] = {
        "config": "config.json",
        "loop": "loop.json",
    }

    write_json(staging / "config.json", _config_to_jsonable(config))
    write_json(staging / "loop.json", loop_state)

    if logger_history is not None:
        write_json(staging / "logger.json", {"history": logger_history})
        artifacts["logger"] = "logger.json"

    if hasattr(learner, "save"):
        learner.save(staging / "learner")
        artifacts["learner"] = "learner"
    else:
        raise TypeError(f"Learner {type(learner)!r} has no save().")

    if hasattr(replay_buffer, "save"):
        replay_buffer.save(staging / "buffer")
        artifacts["buffer"] = "buffer"

    continual = _find_continual_env(env)
    if continual is not None:
        if hasattr(continual, "curriculum_checkpoint_state"):
            env_state = continual.curriculum_checkpoint_state()
        else:
            env_state = {
                "seq_idx": int(getattr(continual, "_seq_idx", 0)),
                "global_step": int(getattr(continual, "_global_step", 0)),
            }
        write_json(staging / "env.json", env_state)
        artifacts["env"] = "env.json"

    task_id = loop_state.get("task_id")
    if task_id is None:
        task_id = getattr(learner, "task_id", None)
    manifest = build_manifest(
        agent=agent_name,
        step=step,
        task_id=None if task_id is None else int(task_id),
        artifacts=artifacts,
        tag=tag,
        extra=extra_meta,
    )
    write_manifest(staging, manifest)
    return commit_staging_dir(staging, final_dir)


def load_run_checkpoint(
    *,
    directory: str | Path,
    learner: Any,
    replay_buffer: Any,
    env: Any,
    rng_key: jax.Array | None = None,
) -> dict[str, Any]:
    """Restore learner/buffer/env from a run checkpoint; return loop state."""
    directory = Path(directory)
    manifest = read_manifest(directory)
    artifacts = manifest["artifacts"]

    if "learner" in artifacts:
        if not hasattr(learner, "load"):
            raise TypeError(f"Learner {type(learner)!r} has no load().")
        learner.load(directory / artifacts["learner"])

    if "buffer" in artifacts and hasattr(replay_buffer, "load"):
        replay_buffer.load(directory / artifacts["buffer"])

    loop_state = read_json(directory / artifacts.get("loop", "loop.json"))
    logger_payload = None
    if "logger" in artifacts:
        logger_payload = read_json(directory / artifacts["logger"])

    continual = _find_continual_env(env)
    if continual is not None and "env" in artifacts:
        env_state = read_json(directory / artifacts["env"])
        key = rng_key if rng_key is not None else jax.random.PRNGKey(0)
        if hasattr(continual, "load_curriculum_checkpoint_state"):
            continual.load_curriculum_checkpoint_state(env_state, key=key)
        else:
            continual._seq_idx = int(env_state["seq_idx"])
            continual._global_step = int(env_state["global_step"])
            if hasattr(continual, "_make_task_vector_env"):
                continual._vector_env = continual._make_task_vector_env(continual._seq_idx)
                continual._state = continual._vector_env.reset(key)

    return {
        "manifest": manifest,
        "loop": loop_state,
        "logger": logger_payload,
        "config": read_json(directory / artifacts.get("config", "config.json")),
    }


def prune_step_checkpoints(checkpoint_dir: str | Path, *, keep_last: int) -> None:
    """Keep only the newest ``step_*`` directories under ``checkpoint_dir``."""
    if keep_last is None or keep_last <= 0:
        return
    root = Path(checkpoint_dir)
    if not root.is_dir():
        return
    step_dirs = sorted(
        [path for path in root.iterdir() if path.is_dir() and path.name.startswith("step_")],
        key=lambda path: path.name,
    )
    for stale in step_dirs[:-keep_last]:
        shutil.rmtree(stale, ignore_errors=True)


def write_resume_overrides(directory: str | Path, overrides: dict[str, Any]) -> None:
    write_json(Path(directory) / "resume_overrides.json", overrides)

"""Serialize :class:`EfficientZeroReplayBuffer` without pickling live objects."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from algorl.buffers.efficientzero.buffer import (
    EfficientZeroStep,
    EfficientZeroTrajectory,
)
from algorl.common.checkpoints import (
    load_pytree,
    read_json,
    save_pytree,
    write_json,
)

BUFFER_FORMAT = "efficient_zero_replay_v1"


def _step_to_dict(step: EfficientZeroStep) -> dict[str, Any]:
    return {
        "observation": np.asarray(step.observation, dtype=np.float32),
        "action": np.asarray(step.action, dtype=np.float32),
        "reward": float(step.reward),
        "policy_target": np.asarray(step.policy_target, dtype=np.float32),
        "pred_value": float(step.pred_value),
        "search_value": float(step.search_value),
        "root_candidates": np.asarray(step.root_candidates, dtype=np.float32),
        "best_action": np.asarray(step.best_action, dtype=np.float32),
        "done": bool(step.done),
    }


def _step_from_dict(payload: dict[str, Any]) -> EfficientZeroStep:
    return EfficientZeroStep(
        observation=np.asarray(payload["observation"], dtype=np.float32),
        action=np.asarray(payload["action"], dtype=np.float32),
        reward=float(payload["reward"]),
        policy_target=np.asarray(payload["policy_target"], dtype=np.float32),
        pred_value=float(payload["pred_value"]),
        search_value=float(payload["search_value"]),
        root_candidates=np.asarray(payload["root_candidates"], dtype=np.float32),
        best_action=np.asarray(payload["best_action"], dtype=np.float32),
        done=bool(payload["done"]),
    )


def _traj_to_dict(traj: EfficientZeroTrajectory) -> dict[str, Any]:
    return {
        "max_size": int(traj.max_size),
        "core_len": None if traj.core_len is None else int(traj.core_len),
        "final_observation": (
            None
            if traj.final_observation is None
            else np.asarray(traj.final_observation, dtype=np.float32)
        ),
        "bootstrapped_values": (
            None
            if traj.bootstrapped_values is None
            else np.asarray(traj.bootstrapped_values, dtype=np.float32)
        ),
        "gae_values": (
            None
            if traj.gae_values is None
            else np.asarray(traj.gae_values, dtype=np.float32)
        ),
        "steps": [_step_to_dict(step) for step in traj.steps],
    }


def _traj_from_dict(payload: dict[str, Any]) -> EfficientZeroTrajectory:
    traj = EfficientZeroTrajectory(max_size=int(payload["max_size"]))
    traj.core_len = (
        None if payload["core_len"] is None else int(payload["core_len"])
    )
    final_obs = payload.get("final_observation")
    traj.final_observation = (
        None if final_obs is None else np.asarray(final_obs, dtype=np.float32)
    )
    boot = payload.get("bootstrapped_values")
    traj.bootstrapped_values = (
        None if boot is None else np.asarray(boot, dtype=np.float32)
    )
    gae = payload.get("gae_values")
    traj.gae_values = None if gae is None else np.asarray(gae, dtype=np.float32)
    traj.steps = [_step_from_dict(step) for step in payload["steps"]]
    return traj


def _lane_map_to_dict(
    lanes: dict[int, EfficientZeroTrajectory],
) -> dict[int, dict[str, Any]]:
    return {int(env_id): _traj_to_dict(traj) for env_id, traj in lanes.items()}


def _lane_map_from_dict(
    payload: dict[Any, Any],
) -> dict[int, EfficientZeroTrajectory]:
    return {
        int(env_id): _traj_from_dict(traj_payload)
        for env_id, traj_payload in payload.items()
    }


def save_efficient_zero_buffer(buffer: Any, directory: str | Path) -> None:
    """Write buffer contents under ``directory`` (meta JSON + pytree payload)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    # Stored trajectories keep steps in ``_stored_steps``; meta objects in
    # ``_trajectories`` may share or diverge from that list — persist both.
    stored = []
    for steps, traj_meta in zip(buffer._stored_steps, buffer._trajectories, strict=True):
        meta_dict = _traj_to_dict(traj_meta)
        # Prefer the stored step list (source of truth for sampling).
        meta_dict["steps"] = [_step_to_dict(step) for step in steps]
        stored.append(meta_dict)

    payload = {
        "lookup": np.asarray(buffer._lookup, dtype=np.int64).reshape(-1, 2)
        if buffer._lookup
        else np.zeros((0, 2), dtype=np.int64),
        "priorities": np.asarray(buffer._priorities, dtype=np.float64),
        "stored": stored,
        "active": _lane_map_to_dict(buffer._active),
        "pending_commit": _lane_map_to_dict(buffer._pending_commit),
    }
    write_json(
        directory / "meta.json",
        {
            "format": BUFFER_FORMAT,
            "capacity": int(buffer.capacity),
            "unroll_steps": int(buffer.unroll_steps),
            "trajectory_size": int(buffer.trajectory_size),
            "base_traj_idx": int(buffer._base_traj_idx),
            "total_commits": int(buffer._total_commits),
            "num_lookup": len(buffer._lookup),
            "num_trajectories": len(buffer._stored_steps),
            "use_priority": bool(buffer.config.use_priority),
        },
    )
    save_pytree(directory / "data", payload)


def load_efficient_zero_buffer(buffer: Any, directory: str | Path) -> None:
    """Restore buffer contents from ``directory`` into an existing instance."""
    directory = Path(directory)
    meta = read_json(directory / "meta.json")
    if meta.get("format") != BUFFER_FORMAT:
        raise ValueError(
            f"Unsupported buffer format {meta.get('format')!r}; "
            f"expected {BUFFER_FORMAT!r}."
        )
    if int(meta["capacity"]) != int(buffer.capacity):
        raise ValueError(
            f"Buffer capacity mismatch: checkpoint={meta['capacity']} "
            f"live={buffer.capacity}."
        )
    if int(meta["unroll_steps"]) != int(buffer.unroll_steps):
        raise ValueError(
            f"unroll_steps mismatch: checkpoint={meta['unroll_steps']} "
            f"live={buffer.unroll_steps}."
        )
    if int(meta["trajectory_size"]) != int(buffer.trajectory_size):
        raise ValueError(
            f"trajectory_size mismatch: checkpoint={meta['trajectory_size']} "
            f"live={buffer.trajectory_size}."
        )

    payload = load_pytree(directory / "data")
    buffer.clear()
    buffer._base_traj_idx = int(meta["base_traj_idx"])
    buffer._total_commits = int(meta["total_commits"])

    stored_steps = []
    trajectories = []
    for traj_payload in payload["stored"]:
        traj = _traj_from_dict(traj_payload)
        trajectories.append(traj)
        stored_steps.append(list(traj.steps))
    buffer._stored_steps = deque(stored_steps)
    buffer._trajectories = deque(trajectories)

    lookup = np.asarray(payload["lookup"], dtype=np.int64).reshape(-1, 2)
    buffer._lookup = [(int(row[0]), int(row[1])) for row in lookup]
    priorities = np.asarray(payload["priorities"], dtype=np.float64).reshape(-1)
    buffer._priorities = [float(value) for value in priorities.tolist()]
    if len(buffer._priorities) != len(buffer._lookup):
        raise ValueError(
            "Corrupt buffer checkpoint: "
            f"len(priorities)={len(buffer._priorities)} != len(lookup)={len(buffer._lookup)}."
        )

    buffer._active = _lane_map_from_dict(payload.get("active", {}))
    buffer._pending_commit = _lane_map_from_dict(payload.get("pending_commit", {}))

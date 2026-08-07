"""Serialize :class:`UniformReplayBuffer` without pickling live objects."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from algorl.common.checkpoints import load_pytree, read_json, save_pytree, write_json
from algorl.core.types import Transition

BUFFER_FORMAT = "uniform_replay_v1"


def _transition_to_dict(transition: Transition) -> dict[str, Any]:
    info: dict[str, Any] = {}
    for key, value in transition.info.items():
        str_key = str(key)
        if isinstance(value, (bool, int, float, str)) or value is None:
            info[str_key] = value
        else:
            info[str_key] = np.asarray(value)
    return {
        "observation": np.asarray(transition.observation),
        "action": np.asarray(transition.action),
        "reward": float(transition.reward),
        "next_observation": np.asarray(transition.next_observation),
        "done": bool(transition.done),
        "info": info,
    }


def _transition_from_dict(payload: dict[str, Any]) -> Transition:
    info_raw = payload.get("info", {})
    info: dict[str, object] = {}
    for key, value in info_raw.items():
        info[str(key)] = value
    return Transition(
        observation=np.asarray(payload["observation"]),
        action=np.asarray(payload["action"]),
        reward=float(payload["reward"]),
        next_observation=np.asarray(payload["next_observation"]),
        done=bool(payload["done"]),
        info=info,
    )


def save_uniform_buffer(buffer: Any, directory: str | Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(
        directory / "meta.json",
        {
            "format": BUFFER_FORMAT,
            "capacity": int(buffer._capacity),
            "size": len(buffer),
        },
    )
    save_pytree(
        directory / "data",
        {"transitions": [_transition_to_dict(item) for item in buffer._storage]},
    )


def load_uniform_buffer(buffer: Any, directory: str | Path) -> None:
    directory = Path(directory)
    meta = read_json(directory / "meta.json")
    if meta.get("format") != BUFFER_FORMAT:
        raise ValueError(
            f"Unsupported buffer format {meta.get('format')!r}; "
            f"expected {BUFFER_FORMAT!r}."
        )
    if int(meta["capacity"]) != int(buffer._capacity):
        raise ValueError(
            f"Buffer capacity mismatch: checkpoint={meta['capacity']} "
            f"live={buffer._capacity}."
        )
    payload = load_pytree(directory / "data")
    buffer._storage = deque(
        (_transition_from_dict(item) for item in payload["transitions"]),
        maxlen=buffer._capacity,
    )

"""Directory checkpoint I/O for :class:`EfficientZeroLearner`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from algorl.common.checkpoints import (
    load_pytree_as_jax,
    read_json,
    save_pytree,
    write_json,
)

LEARNER_FORMAT = "efficient_zero_learner_v1"


def _rng_to_array(key: Any) -> np.ndarray:
    if hasattr(jax.random, "key_data"):
        return np.asarray(jax.random.key_data(key))
    return np.asarray(key)


def _rng_from_array(data: np.ndarray) -> jax.Array:
    array = np.asarray(data)
    wrap = getattr(jax.random, "wrap_key_data", None)
    if wrap is not None and array.dtype == np.uint32:
        return wrap(array)
    return jnp.asarray(array)


def efficient_zero_checkpoint_state(learner: Any) -> dict[str, Any]:
    return {
        "params": learner.params,
        "opt_state": learner._opt_state,
        "self_play_params": learner._self_play_params,
        "reanalyze_params": learner._reanalyze_params,
        "recent_reanalyze_params": learner._recent_reanalyze_params,
        "rng_key": _rng_to_array(learner._rng_key),
    }


def apply_efficient_zero_learner_state(
    learner: Any,
    *,
    meta: dict[str, Any],
    data: dict[str, Any],
) -> None:
    from algorl.backends.jax.learners.efficientzero.learner import _strip_obs_running_count
    from algorl.backends.jax.planners.efficientzero import EfficientZeroPlanner

    learner.params = _strip_obs_running_count(data["params"])
    learner._opt_state = data["opt_state"]
    learner._self_play_params = _strip_obs_running_count(data["self_play_params"])
    learner._reanalyze_params = _strip_obs_running_count(data["reanalyze_params"])
    learner._recent_reanalyze_params = _strip_obs_running_count(
        data["recent_reanalyze_params"]
    )
    learner._rng_key = _rng_from_array(data["rng_key"])
    learner._train_steps = int(meta["train_steps"])
    learner._obs_running_count = int(meta["obs_running_count"])
    learner._sync_params()
    if isinstance(learner.planner, EfficientZeroPlanner):
        learner.planner.self_play_params = learner._self_play_params
        learner.planner.params = learner.params


def save_efficient_zero_learner(learner: Any, directory: str | Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(
        directory / "meta.json",
        {
            "format": LEARNER_FORMAT,
            "train_steps": int(learner._train_steps),
            "obs_running_count": int(learner._obs_running_count),
        },
    )
    save_pytree(directory / "data", efficient_zero_checkpoint_state(learner))


def load_efficient_zero_learner(learner: Any, directory: str | Path) -> None:
    directory = Path(directory)
    meta = read_json(directory / "meta.json")
    if meta.get("format") != LEARNER_FORMAT:
        raise ValueError(
            f"Unsupported learner format {meta.get('format')!r}; "
            f"expected {LEARNER_FORMAT!r}."
        )
    data = load_pytree_as_jax(directory / "data")
    apply_efficient_zero_learner_state(learner, meta=meta, data=data)

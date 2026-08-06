"""Directory checkpoint I/O for :class:`HyperCEZLearner`."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from algorl.backends.jax.learners.efficientzero.checkpoint import (
    _rng_from_array,
    _rng_to_array,
)
from algorl.common.checkpoints import (
    load_pytree_as_jax,
    read_json,
    save_pytree,
    write_json,
)

LEARNER_FORMAT = "hyper_cez_learner_v1"


def _host_tree(tree: Any) -> Any:
    """Copy array leaves to durable NumPy buffers before serialization."""
    import numpy as np

    def _leaf(value: Any) -> Any:
        if hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "__array__"):
            return np.array(value)
        return value

    return jax.tree.map(_leaf, tree)


def hypercez_checkpoint_state(learner: Any) -> dict[str, Any]:
    reg_targets = learner._reg_targets
    return {
        "train_state": _host_tree(learner.train_state),
        "frozen_ez": _host_tree(learner.frozen_ez),
        "params": _host_tree(learner.params),
        "opt_state": _host_tree(learner._opt_state),
        "reg_hnet_opt_state": _host_tree(learner._reg_hnet_opt_state),
        "reg_alpha_opt_state": _host_tree(learner._reg_alpha_opt_state),
        "self_play_params": _host_tree(learner._self_play_params),
        "reanalyze_params": _host_tree(learner._reanalyze_params),
        "recent_reanalyze_params": _host_tree(learner._recent_reanalyze_params),
        "self_play_hnets": _host_tree(learner._self_play_hnets),
        "reanalyze_hnets": _host_tree(learner._reanalyze_hnets),
        "recent_reanalyze_hnets": _host_tree(learner._recent_reanalyze_hnets),
        "shared_snapshots": _host_tree(learner._shared_snapshots),
        "reg_targets": {} if reg_targets is None else _host_tree(reg_targets),
        "ema_reg_per_task": _host_tree(learner._ema_reg_per_task),
        "rng_key": _rng_to_array(learner._rng_key),
        "has_reg_targets": reg_targets is not None,
    }


def apply_hypercez_learner_state(
    learner: Any,
    *,
    meta: dict[str, Any],
    data: dict[str, Any],
) -> None:
    from algorl.backends.jax.learners.efficientzero.learner import _strip_obs_running_count
    from algorl.backends.jax.learners.hypercez.learner import materialize_from_train_state
    from algorl.backends.jax.planners.efficientzero import EfficientZeroPlanner

    learner.train_state = data["train_state"]
    learner.frozen_ez = data["frozen_ez"]
    learner.params = _strip_obs_running_count(data["params"])
    learner._opt_state = data["opt_state"]
    learner._reg_hnet_opt_state = data["reg_hnet_opt_state"]
    learner._reg_alpha_opt_state = data["reg_alpha_opt_state"]
    learner._self_play_params = _strip_obs_running_count(data["self_play_params"])
    learner._reanalyze_params = _strip_obs_running_count(data["reanalyze_params"])
    learner._recent_reanalyze_params = _strip_obs_running_count(
        data["recent_reanalyze_params"]
    )
    learner._self_play_hnets = data["self_play_hnets"]
    learner._reanalyze_hnets = data["reanalyze_hnets"]
    learner._recent_reanalyze_hnets = data["recent_reanalyze_hnets"]
    learner._shared_snapshots = data["shared_snapshots"]
    if bool(data["has_reg_targets"]):
        learner._reg_targets = data["reg_targets"]
    else:
        learner._reg_targets = None
    learner._ema_reg_per_task = jnp.asarray(data["ema_reg_per_task"], dtype=jnp.float32)
    learner._rng_key = _rng_from_array(data["rng_key"])

    learner._train_steps = int(meta["train_steps"])
    learner._task_train_steps = int(meta["task_train_steps"])
    learner.task_id = int(meta["task_id"])
    learner._obs_running_count = int(meta["obs_running_count"])
    learner._ema_task_loss = meta.get("ema_task_loss")
    learner._ema_reg_loss = meta.get("ema_reg_loss")

    learner.world_model.set_task_id(learner.task_id)
    learner._jit_materialize = jax.jit(
        partial(
            materialize_from_train_state,
            frozen_ez=learner.frozen_ez,
            hnet_modules=learner.hnet_modules,
            hnet_components=learner.config.hnet_components,
            num_tasks=learner.config.num_tasks,
            alpha_max=learner.config.alpha_max,
        ),
        static_argnums=(1,),
    )
    learner._recompile_train_kernels()
    learner._sync_params()
    if isinstance(learner.planner, EfficientZeroPlanner):
        learner.planner.self_play_params = learner._self_play_params
        learner.planner.params = learner.params


def save_hypercez_learner(learner: Any, directory: str | Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(
        directory / "meta.json",
        {
            "format": LEARNER_FORMAT,
            "train_steps": int(learner._train_steps),
            "task_train_steps": int(learner._task_train_steps),
            "task_id": int(learner.task_id),
            "obs_running_count": int(learner._obs_running_count),
            "ema_task_loss": learner._ema_task_loss,
            "ema_reg_loss": learner._ema_reg_loss,
        },
    )
    save_pytree(directory / "data", hypercez_checkpoint_state(learner))


def load_hypercez_learner(learner: Any, directory: str | Path) -> None:
    directory = Path(directory)
    meta = read_json(directory / "meta.json")
    if meta.get("format") != LEARNER_FORMAT:
        raise ValueError(
            f"Unsupported learner format {meta.get('format')!r}; "
            f"expected {LEARNER_FORMAT!r}."
        )
    data = load_pytree_as_jax(directory / "data")
    apply_hypercez_learner_state(learner, meta=meta, data=data)

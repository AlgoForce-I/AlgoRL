"""Helpers to pack HyperCEZ trainable state for Optax."""

from __future__ import annotations

from typing import Any

from algorl.backends.jax.nn.efficientzero.model import Params
from algorl.backends.jax.nn.hypercez.shapes import merge_params, partition_params

# Train-state pytree (all leaves are JAX arrays):
# {
#   "hnets": {component: hnet_params},
#   "alphas": {task_id: {component: scalar}},
#   "shared": {component: shared_leaves},
#   "projections": {component: full_params},
# }


def split_live_for_train(
    live_ez: Params,
    hnet_components: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split live EZ params into shared (hnet comps) and projections (rest)."""
    hnet_set = set(hnet_components)
    shared: dict[str, Any] = {}
    projections: dict[str, Any] = {}
    for name, params in live_ez.items():
        if name in hnet_set:
            _, shared[name] = partition_params(params)
        else:
            projections[name] = params
    return shared, projections


def join_live_ez(
    *,
    shared: dict[str, Any],
    projections: dict[str, Any],
    frozen_ez: Params,
    hnet_components: tuple[str, ...],
) -> Params:
    """Rebuild a full live EZ params tree for materialize / world-model sync."""
    live: dict[str, Any] = dict(projections)
    for name in hnet_components:
        frozen_generated, _ = partition_params(frozen_ez[name])
        live[name] = merge_params(frozen_generated, shared[name])
    return live


def build_train_state(
    *,
    hnet_params: dict[str, Any],
    alphas: dict[int, dict[str, Any]],
    live_ez: Params,
    hnet_components: tuple[str, ...],
) -> dict[str, Any]:
    shared, projections = split_live_for_train(live_ez, hnet_components)
    return {
        "hnets": hnet_params,
        "alphas": alphas,
        "shared": shared,
        "projections": projections,
    }

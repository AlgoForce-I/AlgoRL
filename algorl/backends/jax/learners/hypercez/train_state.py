"""Helpers to pack HyperCEZ trainable state for Optax."""

from __future__ import annotations

from typing import Any

from algorl.backends.jax.nn.efficientzero.model import Params
from algorl.backends.jax.nn.hypercez.shapes import merge_params, partition_params

# Train-state pytree (all leaves are JAX arrays):
# {
#   "hnets": {component: hnet_params},
#   "alphas": {task_id: {component: scalar}},
#   "base": {component: generated_W0_leaves},
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


def extract_base_generated(
    ez_params: Params,
    hnet_components: tuple[str, ...],
) -> dict[str, Any]:
    """Generated (W0) partition for each hypernet-wrapped component."""
    return {
        name: partition_params(ez_params[name])[0]
        for name in hnet_components
    }


def join_live_ez(
    *,
    shared: dict[str, Any],
    projections: dict[str, Any],
    base: dict[str, Any],
    hnet_components: tuple[str, ...],
) -> Params:
    """Rebuild a full live EZ params tree for materialize / world-model sync."""
    live: dict[str, Any] = dict(projections)
    for name in hnet_components:
        live[name] = merge_params(base[name], shared[name])
    return live


def rebuild_frozen_ez(
    *,
    base: dict[str, Any],
    template_ez: Params,
    hnet_components: tuple[str, ...],
) -> Params:
    """Write updated W0 generated leaves into a full EZ params tree.

    Shared leaves in ``template_ez`` are kept (materialize reads shared from
    live state separately); only generated base weights are replaced.
    """
    out: dict[str, Any] = dict(template_ez)
    for name in hnet_components:
        _, shared = partition_params(template_ez[name])
        out[name] = merge_params(base[name], shared)
    return out


def build_train_state(
    *,
    hnet_params: dict[str, Any],
    alphas: dict[int, dict[str, Any]],
    live_ez: Params,
    hnet_components: tuple[str, ...],
    base_ez: Params | None = None,
) -> dict[str, Any]:
    """Pack Optax train state.

    ``base`` holds generated W0 leaves (from ``base_ez`` if given, else
    ``live_ez``). When ``frozen_base_weights`` is True the learner still
    stores ``base`` but applies ``optax.set_to_zero`` so it does not move.
    """
    shared, projections = split_live_for_train(live_ez, hnet_components)
    source = live_ez if base_ez is None else base_ez
    return {
        "hnets": hnet_params,
        "alphas": alphas,
        "base": extract_base_generated(source, hnet_components),
        "shared": shared,
        "projections": projections,
    }

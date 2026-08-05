"""Materialize EfficientZero params from frozen base + hypernet deltas."""

from __future__ import annotations

from typing import Any, Mapping

import jax
import jax.numpy as jnp

from algorl.agents.configs import DEFAULT_HYPERCEZ_HNET_COMPONENTS
from algorl.backends.jax.nn.hypercez.shapes import merge_params, partition_params


def materialize_delta_component(
    *,
    delta_tree: Any,
    frozen_base: Any,
    shared_live: Any,
    alpha: jnp.ndarray | float,
    alpha_max: float = 0.2,
) -> Any:
    """Build one component: ``W = W0_gen + alpha_max * tanh(alpha) * dW``.

    Shared leaves (LayerNorm, running stats) come from ``shared_live``.
    Generated leaves come from the frozen base plus the scaled hypernet delta.
    """
    frozen_generated, _ = partition_params(frozen_base)
    scale = jnp.asarray(alpha_max, dtype=jnp.float32) * jnp.tanh(
        jnp.asarray(alpha, dtype=jnp.float32)
    )
    generated = jax.tree_util.tree_map(
        lambda weight0, delta: weight0 + scale * delta,
        frozen_generated,
        delta_tree,
    )
    return merge_params(generated, shared_live)


def materialize_ez_params(
    *,
    component_deltas: Mapping[str, Any],
    frozen_ez: Mapping[str, Any],
    live_ez: Mapping[str, Any],
    alphas: Mapping[str, jnp.ndarray | float],
    hnet_components: tuple[str, ...] = DEFAULT_HYPERCEZ_HNET_COMPONENTS,
    alpha_max: float = 0.2,
) -> dict[str, Any]:
    """Build a full EfficientZero params dict for one task.

    Non-hypernet components (e.g. projections) are copied from ``live_ez``.
    Hypernet components use frozen generated weights + live shared leaves.
    """
    out: dict[str, Any] = {}
    hnet_set = set(hnet_components)
    for name, live in live_ez.items():
        if name not in hnet_set:
            out[name] = live
            continue
        if name not in component_deltas:
            raise KeyError(f"Missing hypernet delta for component {name!r}")
        if name not in frozen_ez:
            raise KeyError(f"Missing frozen base for component {name!r}")
        if name not in alphas:
            raise KeyError(f"Missing alpha for component {name!r}")
        _, shared_live = partition_params(live)
        out[name] = materialize_delta_component(
            delta_tree=component_deltas[name],
            frozen_base=frozen_ez[name],
            shared_live=shared_live,
            alpha=alphas[name],
            alpha_max=alpha_max,
        )
    return out

"""Null-space projection of hypernetwork updates (exact retention, full plasticity).

A previous task ``j`` reaches the unchunked hypernetwork through one frozen
input, its embedding ``e_j``, so every layer sees exactly one activation per
previous task. Folding the bias in as a constant input, a Dense layer's output
for task ``j`` changes by ``[x_j, 1] · [ΔK; Δb]``. Removing the part of the
update that lies in ``span{[x_j, 1]}`` therefore leaves that output, every
downstream activation, and task ``j``'s generated weights unchanged. The
current task keeps all directions orthogonal to that span (the hypernet's free
output directions) at the full learning rate, so no regularizer is needed.

LayerNorm is diagonal: coordinate ``i`` maps ``n_{j,i}`` to ``n_{j,i}·s_i + b_i``,
so each coordinate is projected against ``span{[n_{j,i}, 1]}`` on its own.

The projection is applied to the optimizer *update* (after Adam), because
Adam's per-coordinate scaling would move a projected gradient back out of the
free subspace.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from algorl.backends.jax.nn.hypercez.hyper_model import hypernetwork_layer_inputs

# component -> layer key ("hidden_{i}", "trunk_norm", "heads") -> basis rows
NullspaceBases = dict[str, dict[str, jnp.ndarray]]

EMBEDDINGS_KEY = "task_embeddings"
TRUNK_NORM_KEY = "trunk_norm"
HEADS_KEY = "heads"


def _dense_row_basis(inputs: np.ndarray, rel_tol: float) -> np.ndarray:
    """Orthonormal rows spanning ``{[x_j, 1]}``; shape ``(rank, dim + 1)``."""
    augmented = np.concatenate([inputs, np.ones((inputs.shape[0], 1))], axis=1)
    _, singular, vt = np.linalg.svd(augmented, full_matrices=False)
    return vt[singular > rel_tol * singular.max()]


def _layer_norm_basis(normalized: np.ndarray, rel_tol: float) -> np.ndarray:
    """Per-coordinate orthonormal rows spanning ``{[n_{j,i}, 1]}``; shape ``(dim, 2, 2)``.

    Rows outside a coordinate's rank are zero, so they project nothing.
    """
    pairs = np.stack([normalized.T, np.ones_like(normalized.T)], axis=-1)
    _, singular, vt = np.linalg.svd(pairs, full_matrices=False)
    keep = singular > rel_tol * singular.max(axis=-1, keepdims=True)
    basis = np.zeros((normalized.shape[1], 2, 2))
    basis[:, : vt.shape[1]] = vt * keep[..., None]
    return basis


def build_nullspace_bases(
    hnet_params: dict[str, Any],
    hnet_modules: dict[str, Any],
    hnet_components: tuple[str, ...],
    protected_task_ids: Iterable[int],
    *,
    rel_tol: float = 1e-6,
) -> NullspaceBases | None:
    """Bases of the activations the protected tasks feed each hypernet layer.

    Returns ``None`` when there is nothing to protect. SVDs run in float64 on
    the host; bases are stored as float32 closure constants for the kernels.
    """
    task_ids = sorted({int(task) for task in protected_task_ids})
    if not task_ids:
        return None
    bases: NullspaceBases = {}
    for component in hnet_components:
        inputs = hypernetwork_layer_inputs(
            hnet_modules[component], hnet_params[component], task_ids
        )
        component_bases: dict[str, jnp.ndarray] = {}
        for layer, activations in inputs.items():
            array = np.asarray(activations, dtype=np.float64)
            basis = (
                _layer_norm_basis(array, rel_tol)
                if layer == TRUNK_NORM_KEY
                else _dense_row_basis(array, rel_tol)
            )
            component_bases[layer] = jnp.asarray(basis, dtype=jnp.float32)
        bases[component] = component_bases
    return bases


def _project_dense(update: Any, basis: jnp.ndarray) -> dict[str, jnp.ndarray]:
    kernel, bias = update["kernel"], update["bias"]
    weights, bias_weights = basis[:, :-1], basis[:, -1]
    coeff = weights @ kernel + bias_weights[:, None] * bias[None, :]
    return {
        **update,
        "kernel": kernel - weights.T @ coeff,
        "bias": bias - bias_weights @ coeff,
    }


def _project_layer_norm(update: Any, basis: jnp.ndarray) -> dict[str, jnp.ndarray]:
    stacked = jnp.stack([update["scale"], update["bias"]], axis=-1)
    coeff = jnp.einsum("irc,ic->ir", basis, stacked)
    stacked = stacked - jnp.einsum("irc,ir->ic", basis, coeff)
    return {**update, "scale": stacked[:, 0], "bias": stacked[:, 1]}


def project_hnet_updates(
    hnet_updates: dict[str, Any],
    bases: NullspaceBases | None,
    *,
    task_id: int,
    hnet_components: tuple[str, ...],
) -> dict[str, Any]:
    """Keep only the current embedding row and the free directions of every layer."""
    projected = dict(hnet_updates)
    for component in hnet_components:
        updates = dict(hnet_updates[component])
        embeddings = updates[EMBEDDINGS_KEY]
        updates[EMBEDDINGS_KEY] = (
            jnp.zeros_like(embeddings).at[task_id].set(embeddings[task_id])
        )
        if bases is not None:
            component_bases = bases[component]
            for name, leaf in list(updates.items()):
                if name == EMBEDDINGS_KEY:
                    continue
                if name == TRUNK_NORM_KEY:
                    updates[name] = _project_layer_norm(leaf, component_bases[name])
                elif name.startswith("head_"):
                    updates[name] = _project_dense(leaf, component_bases[HEADS_KEY])
                elif name in component_bases:
                    updates[name] = _project_dense(leaf, component_bases[name])
                else:
                    raise KeyError(
                        f"No null-space basis for hypernet parameter {component}/{name}."
                    )
        projected[component] = updates
    return projected


def mask_alpha_updates(
    alpha_updates: dict[int, dict[str, jnp.ndarray]],
    *,
    task_id: int,
) -> dict[int, dict[str, jnp.ndarray]]:
    """Zero every task's α update except the current task's."""
    return {
        task: leaves if task == task_id else jax.tree.map(jnp.zeros_like, leaves)
        for task, leaves in alpha_updates.items()
    }


def free_direction_fraction(
    hnet_params: dict[str, Any],
    hnet_modules: dict[str, Any],
    hnet_components: tuple[str, ...],
    bases: NullspaceBases,
    task_id: int,
) -> dict[str, float]:
    """Share of the current task's head input outside the protected span.

    1.0 means the heads can move this task's generated weights in any
    direction; values near 0 mean the hypernet has run out of room for it.
    """
    fractions: dict[str, float] = {}
    for component in hnet_components:
        head_input = np.asarray(
            hypernetwork_layer_inputs(
                hnet_modules[component], hnet_params[component], [task_id]
            )[HEADS_KEY][0],
            dtype=np.float64,
        )
        augmented = np.append(head_input, 1.0)
        basis = np.asarray(bases[component][HEADS_KEY], dtype=np.float64)
        residual = augmented - basis.T @ (basis @ augmented)
        fractions[component] = float(np.linalg.norm(residual) / np.linalg.norm(augmented))
    return fractions

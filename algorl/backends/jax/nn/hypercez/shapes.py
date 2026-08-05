"""Identify Flax param leaves that hypernets should generate.

HyperCEZ keeps LayerNorm parameters (and AlgoRL obs-norm running stats) on the
main network. Hypernetworks only emit the remaining weight leaves.
"""

from __future__ import annotations

from typing import Any

import jax
from flax.traverse_util import flatten_dict, unflatten_dict

# Shared / non-generated leaves (HyperCEZ LayerNorm exclusion + AlgoRL obs norm).
_SHARED_LEAF_SUBSTRINGS = (
    "LayerNorm",
    "running_mean",
    "running_var",
)


def is_shared_leaf(path: tuple[Any, ...]) -> bool:
    """Return True if this flattened path stays on the main net."""
    text = "/".join(str(part) for part in path)
    return any(token in text for token in _SHARED_LEAF_SUBSTRINGS)


def partition_params(params: Any) -> tuple[Any, Any]:
    """Split a Flax params tree into ``(generated, shared)``.

    ``generated`` leaves are what one hypernet should output for this component.
    ``shared`` leaves (LayerNorm, running stats) stay on the main net.
    """
    flat = flatten_dict(params)
    generated_flat: dict[tuple[Any, ...], Any] = {}
    shared_flat: dict[tuple[Any, ...], Any] = {}
    for path, value in flat.items():
        if is_shared_leaf(path):
            shared_flat[path] = value
        else:
            generated_flat[path] = value
    return unflatten_dict(generated_flat), unflatten_dict(shared_flat)


def target_shapes(params: Any) -> tuple[tuple[int, ...], ...]:
    """Ordered shapes of generated leaves (hypernet head layout)."""
    generated, _ = partition_params(params)
    leaves, _ = jax.tree_util.tree_flatten(generated)
    return tuple(tuple(int(s) for s in leaf.shape) for leaf in leaves)


def generated_treedef(params: Any) -> Any:
    """TreeDef of the generated partition (for reshaping hypernet outputs)."""
    generated, _ = partition_params(params)
    _, treedef = jax.tree_util.tree_flatten(generated)
    return treedef


def merge_params(generated: Any, shared: Any) -> Any:
    """Recombine partitions into a full component params tree."""
    generated_flat = flatten_dict(generated)
    shared_flat = flatten_dict(shared)
    overlap = set(generated_flat) & set(shared_flat)
    if overlap:
        raise ValueError(f"generated/shared partitions overlap on paths: {sorted(overlap)}")
    return unflatten_dict({**generated_flat, **shared_flat})


def path_strings(params: Any) -> list[str]:
    """Sorted ``a/b/c`` path strings for debugging partitions."""
    return sorted("/".join(str(part) for part in path) for path in flatten_dict(params))

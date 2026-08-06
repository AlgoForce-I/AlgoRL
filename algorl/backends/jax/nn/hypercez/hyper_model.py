"""Flax task-conditioned hypernetwork for EfficientZero weight deltas."""

from __future__ import annotations

import math
from typing import Any, Literal, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn

from algorl.backends.jax.nn.hypercez.shapes import generated_treedef, target_shapes

HyperNetModule = Any  # HyperNetwork | ChunkedHyperNetwork


class HyperNetwork(nn.Module):
    """Unchunked hypernetwork: task embedding → per-leaf weight tensors.

    Matches the HyperCEZ / HyperCL unchunked design: an MLP trunk on a learned
    task embedding, then one linear head per target parameter tensor.
    ``num_tasks`` is fixed at construction (v1); growing the bank comes later.
    """

    target_shapes: tuple[tuple[int, ...], ...]
    hidden_dims: tuple[int, ...] = (100, 100)
    emb_size: int = 10
    num_tasks: int = 10
    emb_init_std: float = 1.0
    # Exact-zero heads + open α make the first Adam step apply a full-scale
    # residual (policy collapse). Tiny Gaussian keeps W≈W0 but softens that jump.
    head_init_std: float = 1e-3

    def setup(self) -> None:
        if not self.target_shapes:
            raise ValueError("HyperNetwork requires at least one target shape.")
        if self.num_tasks < 1:
            raise ValueError(f"num_tasks must be >= 1, got {self.num_tasks}.")
        if self.emb_size < 1:
            raise ValueError(f"emb_size must be >= 1, got {self.emb_size}.")
        if self.head_init_std < 0.0:
            raise ValueError(f"head_init_std must be >= 0, got {self.head_init_std}.")

    @nn.compact
    def __call__(self, task_id: int | jnp.ndarray) -> tuple[jnp.ndarray, ...]:
        embeddings = self.param(
            "task_embeddings",
            lambda rng, shape: jax.random.normal(rng, shape) * self.emb_init_std,
            (self.num_tasks, self.emb_size),
        )
        emb = embeddings[jnp.asarray(task_id, dtype=jnp.int32)]
        h = emb
        for index, width in enumerate(self.hidden_dims):
            h = nn.Dense(width, name=f"hidden_{index}")(h)
            h = nn.relu(h)
        h = nn.LayerNorm(name="trunk_norm")(h)

        head_std = self.head_init_std
        # Unchunked heads map a width-H trunk onto full weight tensors. After one
        # Adam step, ΔW ≈ η‖h‖²·g_ΔW with ‖h‖²~H (LayerNorm), so without this
        # scale the first update collapses the policy for large H (e.g. 128).
        head_out_scale = jnp.asarray(
            1.0 / math.sqrt(self.hidden_dims[-1]) if self.hidden_dims else 1.0,
            dtype=jnp.float32,
        )
        outputs: list[jnp.ndarray] = []
        for index, shape in enumerate(self.target_shapes):
            flat_size = math.prod(shape)
            flat = nn.Dense(
                flat_size,
                kernel_init=(
                    nn.initializers.zeros
                    if head_std == 0.0
                    else nn.initializers.normal(stddev=head_std)
                ),
                bias_init=nn.initializers.zeros,
                name=f"head_{index}",
            )(h)
            outputs.append(jnp.reshape(flat * head_out_scale, shape))
        return tuple(outputs)


class ChunkedHyperNetwork(nn.Module):
    """Chunked hypernetwork: shared MLP emits fixed-size weight chunks.

    Matches HyperCL ``ChunkedHyperNetworkHandler``: flatten target weights into
    chunks of size ``chunk_dim``, each conditioned on ``(task_emb, chunk_emb)``.
    Chunk embeddings are part of theta (shared across tasks).
    """

    target_shapes: tuple[tuple[int, ...], ...]
    hidden_dims: tuple[int, ...] = (20, 20)
    emb_size: int = 10
    num_tasks: int = 10
    emb_init_std: float = 1.0
    chunk_dim: int = 2000
    cemb_size: int = 20
    cemb_init_std: float = 1.0
    head_init_std: float = 1e-3

    def setup(self) -> None:
        if not self.target_shapes:
            raise ValueError("ChunkedHyperNetwork requires at least one target shape.")
        if self.num_tasks < 1:
            raise ValueError(f"num_tasks must be >= 1, got {self.num_tasks}.")
        if self.emb_size < 1:
            raise ValueError(f"emb_size must be >= 1, got {self.emb_size}.")
        if self.chunk_dim < 1:
            raise ValueError(f"chunk_dim must be >= 1, got {self.chunk_dim}.")
        if self.cemb_size < 1:
            raise ValueError(f"cemb_size must be >= 1, got {self.cemb_size}.")
        if self.head_init_std < 0.0:
            raise ValueError(f"head_init_std must be >= 0, got {self.head_init_std}.")

    @property
    def num_outputs(self) -> int:
        return sum(math.prod(shape) for shape in self.target_shapes)

    @property
    def num_chunks(self) -> int:
        return int(math.ceil(self.num_outputs / self.chunk_dim))

    @nn.compact
    def __call__(self, task_id: int | jnp.ndarray) -> tuple[jnp.ndarray, ...]:
        task_embeddings = self.param(
            "task_embeddings",
            lambda rng, shape: jax.random.normal(rng, shape) * self.emb_init_std,
            (self.num_tasks, self.emb_size),
        )
        chunk_embeddings = self.param(
            "chunk_embeddings",
            lambda rng, shape: jax.random.normal(rng, shape) * self.cemb_init_std,
            (self.num_chunks, self.cemb_size),
        )
        task_emb = task_embeddings[jnp.asarray(task_id, dtype=jnp.int32)]
        task_tiled = jnp.broadcast_to(
            task_emb[None, :],
            (self.num_chunks, self.emb_size),
        )
        # Batch over chunks with shared Dense weights (HyperCL-style).
        h = jnp.concatenate([task_tiled, chunk_embeddings], axis=-1)
        for index, width in enumerate(self.hidden_dims):
            h = nn.Dense(width, name=f"hidden_{index}")(h)
            h = nn.relu(h)
        h = nn.LayerNorm(name="trunk_norm")(h)
        head_std = self.head_init_std
        chunks = nn.Dense(
            self.chunk_dim,
            kernel_init=(
                nn.initializers.zeros
                if head_std == 0.0
                else nn.initializers.normal(stddev=head_std)
            ),
            bias_init=nn.initializers.zeros,
            name="chunk_head",
        )(h)
        flat = jnp.reshape(chunks, (-1,))[: self.num_outputs]

        outputs: list[jnp.ndarray] = []
        offset = 0
        for shape in self.target_shapes:
            flat_size = math.prod(shape)
            outputs.append(jnp.reshape(flat[offset : offset + flat_size], shape))
            offset += flat_size
        return tuple(outputs)


def build_hypernetwork_for_component(
    component_params: Any,
    *,
    hidden_dims: tuple[int, ...] = (100, 100),
    emb_size: int = 10,
    num_tasks: int = 10,
    emb_init_std: float = 1.0,
    head_init_std: float = 1e-3,
    hnet_type: Literal["unchunked", "chunked"] = "unchunked",
    chunk_dim: int = 2000,
    cemb_size: int = 20,
    cemb_init_std: float = 1.0,
) -> HyperNetwork | ChunkedHyperNetwork:
    """Build a hypernet whose outputs match the generated leaves of ``component_params``."""
    shapes = target_shapes(component_params)
    if hnet_type == "chunked":
        return ChunkedHyperNetwork(
            target_shapes=shapes,
            hidden_dims=hidden_dims,
            emb_size=emb_size,
            num_tasks=num_tasks,
            emb_init_std=emb_init_std,
            chunk_dim=chunk_dim,
            cemb_size=cemb_size,
            cemb_init_std=cemb_init_std,
            head_init_std=head_init_std,
        )
    if hnet_type != "unchunked":
        raise ValueError(
            f"hnet_type must be 'unchunked' or 'chunked', got {hnet_type!r}."
        )
    return HyperNetwork(
        target_shapes=shapes,
        hidden_dims=hidden_dims,
        emb_size=emb_size,
        num_tasks=num_tasks,
        emb_init_std=emb_init_std,
        head_init_std=head_init_std,
    )


def init_hypernetwork_params(
    module: HyperNetModule,
    rng: jax.Array,
    *,
    task_id: int = 0,
) -> Any:
    """Initialize hypernet params with a scalar task id."""
    return module.init(rng, jnp.asarray(task_id, dtype=jnp.int32))["params"]


def apply_hypernetwork(
    module: HyperNetModule,
    params: Any,
    task_id: int | jnp.ndarray,
    *,
    dtheta: Any | None = None,
) -> tuple[jnp.ndarray, ...]:
    """Forward with optional lookahead ``dtheta`` added to all hypernet params."""
    apply_params = params
    if dtheta is not None:
        apply_params = jax.tree_util.tree_map(lambda weight, delta: weight + delta, params, dtheta)
    return module.apply({"params": apply_params}, jnp.asarray(task_id, dtype=jnp.int32))


def outputs_to_tree(outputs: Sequence[jnp.ndarray], treedef: Any) -> Any:
    """Reshape a hypernet output tuple into the generated params pytree."""
    leaves = list(outputs)
    if treedef.num_leaves != len(leaves):
        raise ValueError(
            f"treedef has {treedef.num_leaves} leaves but got {len(leaves)} outputs"
        )
    return jax.tree_util.tree_unflatten(treedef, leaves)


def component_outputs_to_tree(outputs: Sequence[jnp.ndarray], component_params: Any) -> Any:
    """Map hypernet outputs onto the generated partition of ``component_params``."""
    return outputs_to_tree(outputs, generated_treedef(component_params))

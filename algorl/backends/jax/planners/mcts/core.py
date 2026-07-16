"""Shared MCTX search utilities and batched MCTS planner base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any, Iterator

import jax
import jax.numpy as jnp
import mctx
import numpy as np

from algorl.agents.configs import BaseAgentConfig, SearchAgentConfig
from algorl.core.component_context import ComponentContext
from algorl.core.planner import BatchedPlanner
from algorl.core.types import Action, Observation, PolicyTarget, ValueTarget

ObservationBatch = Observation | jnp.ndarray | np.ndarray | list[Observation]
InvalidActionsMask = jnp.ndarray | None


# ---------------------------------------------------------------------------
# Recurrent dynamics (MCTX)
# ---------------------------------------------------------------------------


class RecurrentFn(ABC):
    """Object-oriented dynamics object for MCTX tree expansion.

    MCTX types this contract as ``mctx.RecurrentFn`` (a callable alias).
    Subclasses implement :meth:`apply`; :meth:`__call__` and :meth:`as_mctx``
    satisfy the MCTX ``(params, rng_key, action, embedding)`` interface.
    """

    def __call__(
        self,
        params: Any,
        rng_key: Any,
        action: jnp.ndarray,
        embedding: Any,
    ) -> tuple[mctx.RecurrentFnOutput, Any]:
        return self.apply(params, rng_key, action, embedding)

    def as_mctx(self) -> mctx.RecurrentFn:
        """Return this object for MCTX APIs expecting ``mctx.RecurrentFn``."""
        return self  # type: ignore[return-value]

    @abstractmethod
    def apply(
        self,
        params: Any,
        rng_key: Any,
        action: jnp.ndarray,
        embedding: Any,
    ) -> tuple[mctx.RecurrentFnOutput, Any]:
        """Run one batched expansion step used inside MCTS."""


# ---------------------------------------------------------------------------
# Config and results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MCTSConfig:
    """MCTX search hyperparameters."""

    num_simulations: int
    max_num_considered_actions: int | None = None
    temperature: float = 1.0
    dirichlet_fraction: float = 0.0
    dirichlet_alpha: float = 0.3
    gumbel_scale: float = 1.0
    use_gumbel: bool = False


@dataclass(frozen=True)
class MCTSResult:
    """Single-environment MCTS output."""

    action: Action
    action_weights: PolicyTarget
    root_value: ValueTarget
    search_tree: Any | None = None


@dataclass(frozen=True)
class MCTSBatchedResult:
    """Batched MCTS output with JAX array fields shaped for batch size ``B``."""

    actions: jnp.ndarray
    action_weights: jnp.ndarray
    root_values: jnp.ndarray
    search_tree: Any

    @property
    def batch_size(self) -> int:
        return int(self.actions.shape[0])

    def to_single(self, index: int = 0) -> MCTSResult:
        """Extract one environment result for Gymnasium-style APIs."""
        if not 0 <= index < self.batch_size:
            raise IndexError(f"batch index {index} out of range for batch size {self.batch_size}")

        return MCTSResult(
            action=_action_from_batch(self.actions, index),
            action_weights=jnp.asarray(self.action_weights[index]),
            root_value=float(self.root_values[index]),
            search_tree=_slice_search_tree(self.search_tree, index),
        )


@dataclass(frozen=True)
class NormalizedObservationBatch:
    """Canonical observation batch passed to planner hooks.

    - ``is_stacked=False``: ``items`` is a Python list with one entry per env.
    - ``is_stacked=True``: ``items`` is a JAX array with shape ``[B, ...]``.
    - A rank-1 JAX array is treated as one vector/scalar observation, not a batch.
    """

    items: list[Observation] | jnp.ndarray
    is_stacked: bool

    @property
    def batch_size(self) -> int:
        if self.is_stacked:
            return int(jnp.asarray(self.items).shape[0])
        return len(self.items)


@dataclass(frozen=True)
class _SearchOptions:
    invalid_actions: InvalidActionsMask = None
    max_depth: int | None = None
    mcts_config: MCTSConfig | None = None

    @classmethod
    def from_kwargs(cls, kwargs: dict[str, Any]) -> _SearchOptions:
        return cls(
            invalid_actions=kwargs.get("invalid_actions"),
            max_depth=kwargs.get("max_depth"),
            mcts_config=kwargs.get("mcts_config"),
        )


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------


def normalize_observation_batch(observations: ObservationBatch) -> NormalizedObservationBatch:
    """Normalize single, list, or pre-stacked observations into one batch object."""
    if isinstance(observations, list):
        return NormalizedObservationBatch(items=observations, is_stacked=False)
    if isinstance(observations, (jnp.ndarray, np.ndarray)):
        array = jnp.asarray(observations, dtype=jnp.float32)
        # Only treat rank-2 arrays as pre-stacked batches.
        # Higher-rank arrays are ambiguous (e.g. PGX board matrices/tensors),
        # and for AlgoRL we treat those as a *single* observation.
        if array.ndim == 2:
            return NormalizedObservationBatch(items=array, is_stacked=True)
        return NormalizedObservationBatch(items=[array], is_stacked=False)
    return NormalizedObservationBatch(items=[observations], is_stacked=False)


def chunk_observation_batch(
    observations: list[Observation],
    *,
    chunk_size: int,
) -> list[list[Observation]]:
    """Split a list of observations into fixed-size MCTS batches."""
    validate_search_batch_size(chunk_size)
    return [observations[start : start + chunk_size] for start in range(0, len(observations), chunk_size)]


def validate_search_batch_size(batch_size: int) -> int:
    """Validate planner/search chunk sizes."""
    if batch_size <= 0:
        raise ValueError(f"search_batch_size must be > 0, got {batch_size}.")
    return batch_size


def mcts_config_from_agent(config: BaseAgentConfig, *, use_gumbel: bool = False) -> MCTSConfig:
    """Build planner MCTS settings from an agent config dataclass."""
    if isinstance(config, SearchAgentConfig):
        return MCTSConfig(
            num_simulations=config.mcts_simulations,
            max_num_considered_actions=config.max_num_considered_actions,
            temperature=config.mcts_temperature,
            dirichlet_fraction=config.dirichlet_fraction,
            dirichlet_alpha=config.dirichlet_alpha,
            gumbel_scale=config.gumbel_scale,
            use_gumbel=use_gumbel,
        )

    return MCTSConfig(
        num_simulations=int(getattr(config, "mcts_simulations", 50)),
        max_num_considered_actions=getattr(config, "max_num_considered_actions", None),
        temperature=float(getattr(config, "mcts_temperature", 1.0)),
        dirichlet_fraction=float(getattr(config, "dirichlet_fraction", 0.0)),
        dirichlet_alpha=float(getattr(config, "dirichlet_alpha", 0.3)),
        gumbel_scale=float(getattr(config, "gumbel_scale", 1.0)),
        use_gumbel=use_gumbel,
    )


def infer_batch_size(root: mctx.RootFnOutput) -> int:
    """Return the MCTX batch dimension from a root structure."""
    return int(jax.tree_util.tree_leaves(root.embedding)[0].shape[0])


def validate_root_batch(root: mctx.RootFnOutput) -> int:
    """Validate that root tensors share the same batch dimension."""
    batch_size = infer_batch_size(root)
    if batch_size <= 0:
        raise ValueError(f"MCTS root batch size must be > 0, got {batch_size}.")

    prior_batch = int(root.prior_logits.shape[0])
    value_batch = int(root.value.shape[0])
    if prior_batch != batch_size or value_batch != batch_size:
        raise ValueError(
            "MCTS root batch dimensions are inconsistent: "
            f"embedding={batch_size}, prior_logits={prior_batch}, value={value_batch}."
        )
    return batch_size


# ---------------------------------------------------------------------------
# Search entry points
# ---------------------------------------------------------------------------


def run_muzero_search(
    *,
    params: Any,
    rng_key: Any,
    root: mctx.RootFnOutput,
    recurrent_fn: RecurrentFn,
    config: MCTSConfig,
    invalid_actions: InvalidActionsMask = None,
    max_depth: int | None = None,
) -> MCTSBatchedResult:
    """Run batched MuZero or Gumbel MuZero search.

    ``root`` must be batch-native with shapes ``[B, num_actions]``, ``[B]``,
    and ``[B, ...]`` for ``prior_logits``, ``value``, and ``embedding``.
    """
    if config.num_simulations <= 0:
        raise ValueError(f"num_simulations must be > 0, got {config.num_simulations}.")

    validate_root_batch(root)
    policy_output = _run_mctx_policy(
        params=params,
        rng_key=rng_key,
        root=root,
        recurrent_fn=recurrent_fn,
        config=config,
        invalid_actions=invalid_actions,
        max_depth=max_depth,
    )
    return _policy_output_to_batched_result(policy_output)


def run_muzero_search_single(
    *,
    params: Any,
    rng_key: Any,
    root: mctx.RootFnOutput,
    recurrent_fn: RecurrentFn,
    config: MCTSConfig,
    invalid_actions: InvalidActionsMask = None,
    max_depth: int | None = None,
    index: int = 0,
) -> MCTSResult:
    """Convenience wrapper for ``B == 1`` or extracting one item from a batch."""
    return run_muzero_search(
        params=params,
        rng_key=rng_key,
        root=root,
        recurrent_fn=recurrent_fn,
        config=config,
        invalid_actions=invalid_actions,
        max_depth=max_depth,
    ).to_single(index)


# ---------------------------------------------------------------------------
# Planner base class
# ---------------------------------------------------------------------------


class BaseMCTSPlanner(BatchedPlanner):
    """Batched MCTS planner base class.

    Subclasses implement ``build_root`` and ``make_recurrent_fn`` for a
    :class:`NormalizedObservationBatch`. ``search_batch`` is the high-performance
    entry point; ``search`` wraps a single observation as ``B = 1``.
    """

    def __init__(self, context: ComponentContext, *, use_gumbel: bool = False) -> None:
        self.backend = context.backend
        self.config = context.config
        self.env = context.env
        self.world_model = context.world_model
        self.params: Any = None
        self.mcts_config = mcts_config_from_agent(context.config, use_gumbel=use_gumbel)
        self._rng_key = self.backend.random_key(context.config.seed)
        self.last_result: MCTSBatchedResult | None = None

    @property
    def search_batch_size(self) -> int:
        configured = self.config.search_batch_size if isinstance(self.config, SearchAgentConfig) else 1
        return validate_search_batch_size(configured)

    @abstractmethod
    def build_root(self, observations: NormalizedObservationBatch) -> mctx.RootFnOutput:
        """Build a batched MCTX root from normalized observations."""

    @abstractmethod
    def make_recurrent_fn(self, observations: NormalizedObservationBatch) -> RecurrentFn:
        """Build the recurrent dynamics object used during tree expansion."""

    def iter_search_chunks(self, observations: list[Observation]) -> Iterator[list[Observation]]:
        """Yield observation chunks sized for ``search_batch_size``."""
        yield from chunk_observation_batch(observations, chunk_size=self.search_batch_size)

    def search_batch(
        self,
        observations: ObservationBatch,
        *,
        deterministic: bool = True,
        **kwargs: Any,
    ) -> MCTSBatchedResult:
        """Run MCTS for a batch of observations."""
        self._ensure_params_initialized()
        options = _SearchOptions.from_kwargs(kwargs)
        config = self._resolve_mcts_config(options.mcts_config, deterministic=deterministic)
        self._rng_key, subkey = jax.random.split(self._rng_key)

        normalized = normalize_observation_batch(observations)
        result = run_muzero_search(
            params=self.params,
            rng_key=subkey,
            root=self.build_root(normalized),
            recurrent_fn=self.make_recurrent_fn(normalized),
            config=config,
            invalid_actions=options.invalid_actions,
            max_depth=options.max_depth,
        )
        self.last_result = result
        return result

    def search(
        self,
        observation: Observation,
        *,
        deterministic: bool = True,
        **kwargs: Any,
    ) -> Action:
        """Run MCTS for one observation by delegating to ``search_batch``."""
        return self.search_batch(observation, deterministic=deterministic, **kwargs).to_single(0).action

    def _ensure_params_initialized(self) -> None:
        if self.params is None:
            raise RuntimeError(
                f"{type(self).__name__} params are not initialized. "
                "Set ``planner.params`` before calling search."
            )

    def _resolve_mcts_config(self, override: MCTSConfig | None, *, deterministic: bool) -> MCTSConfig:
        config = override or self.mcts_config
        if not deterministic:
            return config

        updates: dict[str, float] = {
            "temperature": 0.0,
            "dirichlet_fraction": 0.0,
        }
        if config.use_gumbel:
            updates["gumbel_scale"] = 0.0
        return replace(config, **updates)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _run_mctx_policy(
    *,
    params: Any,
    rng_key: Any,
    root: mctx.RootFnOutput,
    recurrent_fn: RecurrentFn,
    config: MCTSConfig,
    invalid_actions: InvalidActionsMask,
    max_depth: int | None,
) -> mctx.PolicyOutput[Any]:
    mctx_recurrent_fn = recurrent_fn.as_mctx()
    if config.use_gumbel:
        return mctx.gumbel_muzero_policy(
            params,
            rng_key,
            root,
            mctx_recurrent_fn,
            config.num_simulations,
            invalid_actions=invalid_actions,
            max_depth=max_depth,
            max_num_considered_actions=config.max_num_considered_actions or 16,
            gumbel_scale=config.gumbel_scale,
        )

    return mctx.muzero_policy(
        params,
        rng_key,
        root,
        mctx_recurrent_fn,
        config.num_simulations,
        invalid_actions=invalid_actions,
        max_depth=max_depth,
        temperature=config.temperature,
        dirichlet_fraction=config.dirichlet_fraction,
        dirichlet_alpha=config.dirichlet_alpha,
    )


def _policy_output_to_batched_result(policy_output: mctx.PolicyOutput[Any]) -> MCTSBatchedResult:
    summary = policy_output.search_tree.summary()
    return MCTSBatchedResult(
        actions=jnp.asarray(policy_output.action),
        action_weights=jnp.asarray(policy_output.action_weights),
        root_values=jnp.asarray(summary.value),
        search_tree=policy_output.search_tree,
    )


def _action_from_batch(actions: jnp.ndarray, index: int) -> Action:
    selected = actions[index]
    if actions.ndim == 1:
        return int(selected)
    return jnp.asarray(selected)


def _slice_search_tree(tree: Any, index: int) -> Any:
    return jax.tree.map(
        lambda leaf: leaf[index] if hasattr(leaf, "shape") and len(leaf.shape) > 0 else leaf,
        tree,
    )

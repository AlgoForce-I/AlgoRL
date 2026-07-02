"""EfficientZero planner: learned world model + continuous candidate-set MCTS."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from algorl.agents.configs import EfficientZeroConfig
from algorl.backends.jax.nn.efficientzero.model import Params
from algorl.backends.jax.planners.mcts.continuous import (
    ContinuousSearchConfig,
    ContinuousSearchResult,
    JittedContinuousSearchFn,
    build_continuous_root_from_model,
    make_jitted_continuous_search,
    make_model_continuous_recurrent_fn,
    run_continuous_search,
)
from algorl.backends.jax.planners.mcts.core import (
    NormalizedObservationBatch,
    ObservationBatch,
    normalize_observation_batch,
    validate_search_batch_size,
)
from algorl.backends.jax.world_models.efficientzero import EfficientZeroWorldModel
from algorl.core.component_context import ComponentContext
from algorl.core.planner import BatchedPlanner
from algorl.core.types import Action, Observation

_DISCRETE_SEARCH_MODEL_TYPES = frozenset({"atari"})


@dataclass(frozen=True)
class EfficientZeroSearchResult:
    """Single-environment continuous-search output."""

    action: Action
    action_index: int
    action_weights: jnp.ndarray
    root_value: float
    search_tree: Any
    root_candidates: jnp.ndarray


@dataclass(frozen=True)
class EfficientZeroBatchedResult:
    """Batched continuous-search output for parallel env rollouts."""

    actions: jnp.ndarray
    action_indices: jnp.ndarray
    action_weights: jnp.ndarray
    root_values: jnp.ndarray
    root_candidates: jnp.ndarray
    search_tree: Any | None = None
    pred_values: jnp.ndarray | None = None

    @property
    def batch_size(self) -> int:
        return int(self.actions.shape[0])

    def without_search_tree(self) -> EfficientZeroBatchedResult:
        """Drop MCTS tree payloads; training only needs root policy/value tensors."""
        if self.search_tree is None:
            return self
        return replace(self, search_tree=None)

    def as_training_snapshot(self) -> EfficientZeroBatchedResult:
        """Host NumPy snapshot for replay targets without retaining GPU search trees."""
        return EfficientZeroBatchedResult(
            actions=np.asarray(self.actions, dtype=np.float32),
            action_indices=np.asarray(self.action_indices, dtype=np.int32),
            action_weights=np.asarray(self.action_weights, dtype=np.float32),
            root_values=np.asarray(self.root_values, dtype=np.float32),
            search_tree=None,
            root_candidates=np.asarray(self.root_candidates, dtype=np.float32),
            pred_values=(
                None
                if self.pred_values is None
                else np.asarray(self.pred_values, dtype=np.float32)
            ),
        )

    @classmethod
    def from_continuous_result(cls, result: ContinuousSearchResult) -> EfficientZeroBatchedResult:
        return cls(
            actions=result.actions,
            action_indices=result.action_indices,
            action_weights=result.action_weights,
            root_values=result.root_values,
            search_tree=result.search_tree,
            root_candidates=result.root_candidates,
        )

    def to_single(self, index: int = 0) -> EfficientZeroSearchResult:
        if not 0 <= index < self.batch_size:
            raise IndexError(f"batch index {index} out of range for batch size {self.batch_size}")
        return EfficientZeroSearchResult(
            action=_action_from_batch(self.actions, index),
            action_index=int(self.action_indices[index]),
            action_weights=jnp.asarray(self.action_weights[index]),
            root_value=float(self.root_values[index]),
            search_tree=(
                None
                if self.search_tree is None
                else _slice_search_tree(self.search_tree, index)
            ),
            root_candidates=jnp.asarray(self.root_candidates[index]),
        )


def uses_continuous_search(
    config: EfficientZeroConfig,
    *,
    resolved_model_type: str | None = None,
) -> bool:
    """Return whether planning should use candidate-set ``search_continuous``.

    HyperCEZ routes Atari through discrete Gumbel ``search``; DMC uses
    ``search_continuous``. ``config.use_gumbel`` applies only to the discrete path.
    """
    model_type = config.model_type
    if model_type == "auto":
        if resolved_model_type is None:
            raise ValueError("resolved_model_type is required when config.model_type is 'auto'.")
        model_type = resolved_model_type
    if model_type in _DISCRETE_SEARCH_MODEL_TYPES:
        return False
    if config.policy_distribution == "discrete":
        return False
    return True


def continuous_search_config_from_agent(
    config: EfficientZeroConfig,
    *,
    search_config: ContinuousSearchConfig | None = None,
) -> ContinuousSearchConfig:
    """Map agent hyperparameters to :class:`ContinuousSearchConfig`.

    Continuous search never enables gumbel halving noise (HyperCEZ
    ``use_gumble_noise=False``). Root exploration uses ``add_noise`` at build time.
    ``config.use_gumbel`` is reserved for the discrete Atari search path.
    """
    num_actions = (
        config.max_num_considered_actions
        if config.max_num_considered_actions is not None
        else ContinuousSearchConfig().num_sampled_actions
    )
    base = search_config or ContinuousSearchConfig()
    return replace(
        base,
        num_simulations=config.mcts_simulations,
        num_sampled_actions=num_actions,
        num_top_actions=num_actions,
        gumbel_scale=0.0,
        use_gumbel_noise=False,
        lstm_horizon_len=config.lstm_horizon_len,
    )


class EfficientZeroPlanner(BatchedPlanner):
    """EfficientZero-V2 planner.

    DMC / continuous-control environments use :mod:`continuous` candidate-set MCTS.
    Atari requires discrete Gumbel ``search`` and is not supported here yet.
    """

    def __init__(
        self,
        context: ComponentContext,
        *,
        world_model: EfficientZeroWorldModel | None = None,
        search_config: ContinuousSearchConfig | None = None,
        use_jit: bool = True,
    ) -> None:
        if not isinstance(context.config, EfficientZeroConfig):
            raise TypeError(
                "EfficientZeroPlanner requires EfficientZeroConfig, "
                f"got {type(context.config)!r}."
            )

        self.backend = context.backend
        self.config: EfficientZeroConfig = context.config
        self.env = context.env
        self.world_model = world_model or self._require_world_model(context)
        self.model = self.world_model.model
        self.params: Params = self.world_model.params
        self._search_config_override = search_config
        self._use_jit = use_jit
        self._rng_key = self.backend.random_key(context.config.seed)
        self.last_result: EfficientZeroBatchedResult | None = None

        if not uses_continuous_search(
            self.config,
            resolved_model_type=self.world_model.resolved_model_type,
        ):
            self._recurrent_fn = None
            self._jitted_search = None
            return

        self._recurrent_fn = make_model_continuous_recurrent_fn(
            self.model,
            config=self.search_config,
        )
        self._jitted_search: JittedContinuousSearchFn | None = None
        if use_jit:
            self._jitted_search = make_jitted_continuous_search(
                self.search_config,
                self._recurrent_fn,
            )

    @property
    def search_config(self) -> ContinuousSearchConfig:
        if self._search_config_override is not None:
            return self._search_config_override
        return continuous_search_config_from_agent(self.config)

    @property
    def search_batch_size(self) -> int:
        return validate_search_batch_size(self.config.search_batch_size)

    def search_batch(
        self,
        observations: ObservationBatch,
        *,
        deterministic: bool = False,
        **kwargs: Any,
    ) -> EfficientZeroBatchedResult:
        """Run continuous MCTS for a batch of observations."""
        del kwargs
        self._ensure_params_initialized()
        self._ensure_continuous_search_supported()

        search_config = self._resolve_search_config()
        add_noise = not deterministic
        normalized = normalize_observation_batch(observations)
        obs_batch = _observations_to_array(normalized)

        self._rng_key, build_key, search_key = jax.random.split(self._rng_key, 3)
        root, root_candidates, extra_data = build_continuous_root_from_model(
            self.model,
            self.params,
            obs_batch,
            config=search_config,
            rng=build_key,
            add_noise=add_noise,
        )
        pred_values = jnp.asarray(root.value, dtype=jnp.float32)

        if self._use_jit and self._jitted_search is not None:
            result = self._jitted_search(
                self.params,
                search_key,
                root,
                extra_data,
                root_candidates,
            )
        else:
            result = run_continuous_search(
                params=self.params,
                rng_key=search_key,
                root=root,
                recurrent_fn=make_model_continuous_recurrent_fn(self.model, config=search_config),
                config=search_config,
                extra_data=extra_data,
                root_candidates=root_candidates,
            )

        batched = EfficientZeroBatchedResult.from_continuous_result(result)
        if pred_values is not None:
            batched = EfficientZeroBatchedResult(
                actions=batched.actions,
                action_indices=batched.action_indices,
                action_weights=batched.action_weights,
                root_values=batched.root_values,
                search_tree=batched.search_tree,
                root_candidates=batched.root_candidates,
                pred_values=pred_values,
            )
        training_snapshot = batched.as_training_snapshot()
        self.last_result = training_snapshot
        return training_snapshot

    def search(
        self,
        observation: Observation,
        *,
        deterministic: bool = False,
        **kwargs: Any,
    ) -> Action:
        """Plan for one observation via ``search_batch`` with ``B = 1``."""
        return self.search_batch(
            observation,
            deterministic=deterministic,
            **kwargs,
        ).to_single(0).action

    def _resolve_search_config(self) -> ContinuousSearchConfig:
        base = self._search_config_override or ContinuousSearchConfig()
        return continuous_search_config_from_agent(self.config, search_config=base)

    def _ensure_params_initialized(self) -> None:
        if self.params is None:
            raise RuntimeError(
                "EfficientZeroPlanner params are not initialized. "
                "Set ``planner.params`` before calling search."
            )

    def _ensure_continuous_search_supported(self) -> None:
        if uses_continuous_search(
            self.config,
            resolved_model_type=self.world_model.resolved_model_type,
        ):
            return
        raise NotImplementedError(
            "EfficientZeroPlanner currently implements candidate-set search_continuous "
            f"for continuous-control presets only. Model type "
            f"{self.world_model.resolved_model_type!r} requires discrete Gumbel search "
            "(HyperCEZ ``mcts.search``), which is not wired yet."
        )

    @staticmethod
    def _require_world_model(context: ComponentContext) -> EfficientZeroWorldModel:
        world_model = context.world_model
        if not isinstance(world_model, EfficientZeroWorldModel):
            raise TypeError(
                "EfficientZeroPlanner requires EfficientZeroWorldModel in the build context, "
                f"got {type(world_model)!r}."
            )
        return world_model


def build_efficient_zero_planner(context: ComponentContext) -> EfficientZeroPlanner:
    """Factory registered as JAX planner kind ``mcts`` for EfficientZero."""
    return EfficientZeroPlanner(context)


def _observations_to_array(observations: NormalizedObservationBatch) -> jnp.ndarray:
    if observations.is_stacked:
        return jnp.asarray(observations.items, dtype=jnp.float32)
    if len(observations.items) == 0:
        raise ValueError("EfficientZeroPlanner.search_batch received an empty observation batch.")
    first = observations.items[0]
    if isinstance(first, Mapping):
        raise TypeError("EfficientZeroPlanner does not support Dict observations yet.")
    arrays = [jnp.asarray(observation, dtype=jnp.float32) for observation in observations.items]
    return jnp.stack(arrays, axis=0)


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

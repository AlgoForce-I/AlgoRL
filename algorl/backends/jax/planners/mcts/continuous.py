"""EfficientZero-V2-style continuous candidate-set MCTS using MCTX tree primitives.

All search primitives are JAX-traceable. Use :func:`make_jitted_continuous_search` to
obtain a ``jax.jit``-compiled search function for training and inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Protocol, TypeAlias

import jax
import jax.numpy as jnp
import mctx
from mctx import RecurrentState
from mctx._src import search as mctx_search
from mctx._src.action_selection import switching_action_selection_wrapper
from mctx._src.base import InteriorActionSelectionFn

from algorl.backends.jax.nn.efficient_zero.model import EfficientZero as EfficientZeroNetwork
from algorl.backends.jax.nn.efficient_zero.model import Params
from algorl.backends.jax.planners.mcts.core import RecurrentFn
from algorl.backends.jax.world_models.efficient_zero import EfficientZeroLatentState

Array: TypeAlias = jax.Array
NodeIndex: TypeAlias = Array
ContinuousSearchParams: TypeAlias = Params

InitialStepFn = Callable[
    [Array],
    tuple[EfficientZeroLatentState, Array, Array],
]
RecurrentStepFn = Callable[
    [EfficientZeroLatentState, Array],
    tuple[EfficientZeroLatentState, Array, Array, Array],
]


# ---------------------------------------------------------------------------
# Config, state, and results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContinuousSearchConfig:
    """EfficientZero-V2 ``search_continuous`` settings from ``ez_hparams.json``."""

    num_simulations: int = 32
    num_sampled_actions: int = 16
    leaf_action_num: int = 2
    policy_action_num: int = 4
    random_action_num: int = 12
    std_magnification: float = 3.0
    discount: float = 0.997
    value_minmax_delta: float = 0.01
    c_visit: int = 50
    c_scale: float = 0.1
    num_top_actions: int = 16
    use_gumbel_noise: bool = True
    gumbel_scale: float = 1.0
    lstm_horizon_len: int = 5


@dataclass(frozen=True)
class ContinuousSearchState:
    """MCTX embedding for EfficientZero-V2 continuous search (array leaves only)."""

    latent_state: Array
    candidates: Array
    depth: Array


@dataclass(frozen=True)
class ContinuousSearchExtraData:
    """Per-batch search state mirroring EfficientZero-V2 ``ptree`` sequential halving."""

    gumbel: Array
    min_max_maximum: Array
    min_max_minimum: Array
    selected_children: Array
    num_selected: Array
    current_num_top_actions: Array
    current_phase: Array
    visit_num_for_next_phase: Array
    used_visit_num: Array


jax.tree_util.register_dataclass(
    ContinuousSearchState,
    data_fields=["latent_state", "candidates", "depth"],
    meta_fields=[],
)
jax.tree_util.register_dataclass(
    ContinuousSearchExtraData,
    data_fields=[
        "gumbel",
        "min_max_maximum",
        "min_max_minimum",
        "selected_children",
        "num_selected",
        "current_num_top_actions",
        "current_phase",
        "visit_num_for_next_phase",
        "used_visit_num",
    ],
    meta_fields=[],
)

ContinuousMCTSTree: TypeAlias = mctx.Tree[ContinuousSearchExtraData]


@dataclass(frozen=True)
class ContinuousSearchResult:
    """Output of :func:`run_continuous_search`."""

    actions: Array
    action_indices: Array
    action_weights: Array
    root_values: Array
    search_tree: ContinuousMCTSTree
    root_candidates: Array


jax.tree_util.register_dataclass(
    ContinuousSearchResult,
    data_fields=[
        "actions",
        "action_indices",
        "action_weights",
        "root_values",
        "search_tree",
        "root_candidates",
    ],
    meta_fields=[],
)

JittedContinuousSearchFn = Callable[
    [ContinuousSearchParams, Array, mctx.RootFnOutput, ContinuousSearchExtraData, Array],
    ContinuousSearchResult,
]
SimulationStepFn = Callable[
    [Array, tuple[Array, ContinuousMCTSTree, ContinuousSearchExtraData]],
    tuple[Array, ContinuousMCTSTree, ContinuousSearchExtraData],
]


# ---------------------------------------------------------------------------
# World-model bindings (eager helpers for tests and debugging)
# ---------------------------------------------------------------------------


class SupportsRecurrentStep(Protocol):
    def recurrent_step(
        self,
        latent_state: EfficientZeroLatentState,
        action: Array,
        *,
        training: bool = False,
    ) -> tuple[EfficientZeroLatentState, Array, Array, Array]:
        ...

    def initial_step(
        self,
        observation: Array,
        *,
        training: bool = False,
    ) -> tuple[EfficientZeroLatentState, Array, Array]:
        ...


def recurrent_step_fn_from_world_model(world_model: SupportsRecurrentStep) -> RecurrentStepFn:
    """Bind :meth:`EfficientZeroWorldModel.recurrent_step` as a plain callable."""

    def step(
        latent_state: EfficientZeroLatentState,
        action_vector: Array,
    ) -> tuple[EfficientZeroLatentState, Array, Array, Array]:
        return world_model.recurrent_step(latent_state, action_vector, training=False)

    return step


def initial_step_fn_from_world_model(world_model: SupportsRecurrentStep) -> InitialStepFn:
    def step(observation: Array) -> tuple[EfficientZeroLatentState, Array, Array]:
        return world_model.initial_step(observation, training=False)

    return step


# ---------------------------------------------------------------------------
# Candidate sampling and root construction
# ---------------------------------------------------------------------------


def _squashed_normal_log_prob(mean: Array, std: Array, actions: Array) -> Array:
    clipped = jnp.clip(actions, -0.999, 0.999)
    pre_tanh = jnp.arctanh(clipped)
    safe_std = jnp.maximum(std, 1e-6)
    var = safe_std**2
    log_gaussian = -0.5 * jnp.sum(
        ((pre_tanh - mean) ** 2) / var + 2.0 * jnp.log(safe_std) + jnp.log(2.0 * jnp.pi),
        axis=-1,
    )
    log_det = jnp.sum(jnp.log(1.0 - jnp.tanh(pre_tanh) ** 2 + 1e-6), axis=-1)
    return log_gaussian - log_det


def sample_actions(
    policy: Array,
    rng: Array,
    *,
    config: ContinuousSearchConfig,
    add_noise: bool = True,
    sample_nums: int | None = None,
) -> tuple[Array, Array]:
    """Sample candidate continuous actions (EfficientZero-V2 ``MCTS_base.sample_actions``)."""
    batch_size, policy_dim = policy.shape
    action_dim = policy_dim // 2
    num_sampled = config.num_sampled_actions if sample_nums is None else sample_nums

    n_policy = config.policy_action_num
    n_random = config.random_action_num
    if sample_nums is not None:
        n_policy = int(math.ceil(sample_nums / 2))
        n_random = sample_nums - n_policy
    if not add_noise:
        n_policy = num_sampled
        n_random = 0

    mean = policy[:, :action_dim]
    std = policy[:, action_dim:]
    safe_std = jnp.maximum(std, 1e-6)

    rng, policy_key, random_key = jax.random.split(rng, 3)
    policy_eps = jax.random.normal(policy_key, (batch_size, n_policy, action_dim))
    policy_actions = jnp.tanh(mean[:, None, :] + safe_std[:, None, :] * policy_eps)

    if add_noise:
        random_std = config.std_magnification * safe_std
        random_eps = jax.random.normal(random_key, (batch_size, n_random, action_dim))
        random_actions = jnp.tanh(mean[:, None, :] + random_std[:, None, :] * random_eps)
        random_log_prob_all = _squashed_normal_log_prob(
            mean[:, None, :],
            random_std[:, None, :],
            jnp.clip(
                jnp.concatenate([policy_actions, random_actions], axis=1),
                -0.999,
                0.999,
            ),
        )
    else:
        random_actions = policy_actions[:, :0, :]
        random_log_prob_all = jnp.zeros((batch_size, num_sampled), dtype=jnp.float32)

    all_actions = jnp.clip(
        jnp.concatenate([policy_actions, random_actions], axis=1),
        -0.999,
        0.999,
    )
    policy_log_prob = _squashed_normal_log_prob(
        mean[:, None, :],
        safe_std[:, None, :],
        all_actions,
    )
    ratio = n_policy / float(num_sampled)
    blended = policy_log_prob - (
        ratio * policy_log_prob + (1.0 - ratio) * random_log_prob_all
    )
    return all_actions, blended


def _ensure_obs_batch(observation: Array) -> Array:
    obs = jnp.asarray(observation, dtype=jnp.float32)
    return jnp.expand_dims(obs, axis=0) if obs.ndim == 1 else obs


def build_continuous_root_from_model(
    model: EfficientZeroNetwork,
    params: ContinuousSearchParams,
    observation: Array,
    *,
    config: ContinuousSearchConfig,
    rng: Array,
    add_noise: bool = True,
) -> tuple[mctx.RootFnOutput, Array, ContinuousSearchExtraData]:
    """JIT-traceable batched root build using Flax model ``params``.

    ``observation`` may be a single vector ``[obs_dim]`` or a batch ``[B, obs_dim]``.
    Returns MCTX batch-native roots with ``root.value.shape == (B,)``.
    """
    obs = _ensure_obs_batch(observation)
    batch_size = obs.shape[0]
    infer_key, rng = jax.random.split(rng)
    infer_keys = jax.random.split(infer_key, batch_size)

    def infer_one(single_obs: Array, key: Array) -> tuple[Array, Array, Array]:
        return model.initial_inference(
            params,
            single_obs,
            training=False,
            rng=key,
        )

    states, values, policies = jax.vmap(infer_one)(obs, infer_keys)
    return _build_continuous_roots_batch(
        states,
        values,
        policies,
        config=config,
        rng=rng,
        add_noise=add_noise,
    )


def build_continuous_root(
    observation: Array,
    *,
    initial_step_fn: InitialStepFn,
    config: ContinuousSearchConfig,
    rng: Array,
    add_noise: bool = True,
) -> tuple[mctx.RootFnOutput, Array, ContinuousSearchExtraData]:
    """Build batched MCTX roots via a bound world-model callable."""
    obs = _ensure_obs_batch(observation)

    def infer_one(single_obs: Array) -> tuple[Array, Array, Array]:
        latent, value, policy = initial_step_fn(single_obs)
        return latent.state, value, policy

    states, values, policies = jax.vmap(infer_one)(obs)
    return _build_continuous_roots_batch(
        states,
        values,
        policies,
        config=config,
        rng=rng,
        add_noise=add_noise,
    )


def _build_continuous_roots_batch(
    latent_states: Array,
    values: Array,
    policies: Array,
    *,
    config: ContinuousSearchConfig,
    rng: Array,
    add_noise: bool,
) -> tuple[mctx.RootFnOutput, Array, ContinuousSearchExtraData]:
    """Build ``B`` independent roots for MCTX search."""
    batch_size = latent_states.shape[0]
    num_actions = config.num_sampled_actions
    sample_keys = jax.random.split(rng, batch_size)

    def sample_for_env(policy: Array, key: Array) -> Array:
        candidates, _ = sample_actions(
            policy[None, ...],
            key,
            config=config,
            add_noise=add_noise,
        )
        return _pad_candidates(candidates[0], num_actions)

    candidates = jax.vmap(sample_for_env)(policies, sample_keys)
    embedding = ContinuousSearchState(
        latent_state=latent_states,
        candidates=candidates,
        depth=jnp.zeros((batch_size,), dtype=jnp.int32),
    )
    if config.use_gumbel_noise:
        rng, gumbel_key = jax.random.split(rng)
        gumbel = config.gumbel_scale * jax.random.gumbel(
            gumbel_key,
            shape=(batch_size, num_actions),
            dtype=jnp.float32,
        )
    else:
        gumbel = jnp.zeros((batch_size, num_actions), dtype=jnp.float32)
    extra_data = _initial_extra_data(gumbel, config)
    root = mctx.RootFnOutput(
        prior_logits=jnp.zeros((batch_size, num_actions), dtype=jnp.float32),
        value=jnp.asarray(values, dtype=jnp.float32).reshape(batch_size),
        embedding=_to_recurrent_embedding(embedding),
    )
    return root, candidates, extra_data


# ---------------------------------------------------------------------------
# MCTX adapters (fully traceable)
# ---------------------------------------------------------------------------


class ModelContinuousRecurrentFn(RecurrentFn):
    """Batched MCTX expansion using :meth:`EfficientZeroNetwork.recurrent_inference`."""

    def __init__(self, model: EfficientZeroNetwork, config: ContinuousSearchConfig) -> None:
        self._model = model
        self._config = config

    def apply(
        self,
        params: Any,
        rng_key: Any,
        action: jnp.ndarray,
        embedding: Any,
    ) -> tuple[mctx.RecurrentFnOutput, Any]:
        return _batched_model_recurrent_step(
            self._model,
            params,
            rng_key,
            action,
            embedding,
            config=self._config,
        )


class CallableContinuousRecurrentFn(RecurrentFn):
    """Batched MCTX expansion from a single-env recurrent callable (tests/stubs)."""

    def __init__(
        self,
        recurrent_step_fn: RecurrentStepFn,
        config: ContinuousSearchConfig,
    ) -> None:
        self._recurrent_step_fn = recurrent_step_fn
        self._config = config

    def apply(
        self,
        params: Any,
        rng_key: Any,
        action: jnp.ndarray,
        embedding: Any,
    ) -> tuple[mctx.RecurrentFnOutput, Any]:
        del params
        return _batched_callable_recurrent_step(
            self._recurrent_step_fn,
            rng_key,
            action,
            embedding,
            config=self._config,
        )


class ContinuousActionSelection:
    """EfficientZero-V2 root/interior action selection for MCTX ``simulate``."""

    def __init__(self, config: ContinuousSearchConfig) -> None:
        self._config = config

    def root(
        self,
        rng_key: Array,
        tree: ContinuousMCTSTree,
        node_index: NodeIndex,
    ) -> Array:
        del rng_key
        extra = tree.extra_data
        selected = extra.selected_children
        num_selected = extra.num_selected
        children_visits = _tree_children_visits(tree, node_index)
        max_selected = self._config.num_top_actions
        candidate_indices = selected[:max_selected]
        active = jnp.arange(max_selected) < num_selected
        visit_counts = jnp.where(
            active,
            children_visits[candidate_indices],
            jnp.iinfo(jnp.int32).max,
        )
        return candidate_indices[jnp.argmin(visit_counts)]

    def interior(
        self,
        rng_key: Array,
        tree: ContinuousMCTSTree,
        node_index: NodeIndex,
        depth: NodeIndex,
    ) -> Array:
        del rng_key, depth
        transformed_q = _transformed_completed_q(
            tree,
            node_index,
            tree.extra_data,
            self._config,
            num_children=self._config.leaf_action_num,
        )
        children_prior = _tree_children_prior_logits(tree, node_index)[: self._config.leaf_action_num]
        logits = children_prior + transformed_q[: self._config.leaf_action_num]
        policy = jax.nn.softmax(logits)
        node_visit = _tree_node_visits(tree, node_index)
        children_visits = _tree_children_visits(tree, node_index)[: self._config.leaf_action_num]
        scores = policy - children_visits / (1.0 + node_visit)
        return jnp.argmax(scores)

    def as_mctx(self) -> InteriorActionSelectionFn:
        return switching_action_selection_wrapper(self.root, self.interior)


def make_model_continuous_recurrent_fn(
    model: EfficientZeroNetwork,
    *,
    config: ContinuousSearchConfig,
) -> mctx.RecurrentFn:
    """Build a JIT-traceable recurrent fn backed by an EfficientZero Flax module."""
    return ModelContinuousRecurrentFn(model, config).as_mctx()


def make_continuous_recurrent_fn(
    recurrent_step_fn: RecurrentStepFn,
    *,
    config: ContinuousSearchConfig,
) -> mctx.RecurrentFn:
    """Build a JIT-traceable recurrent fn from a single-step callable (tests/stubs)."""
    return CallableContinuousRecurrentFn(recurrent_step_fn, config).as_mctx()


def _batched_model_recurrent_step(
    model: EfficientZeroNetwork,
    params: ContinuousSearchParams,
    rng_key: Array,
    action: Array,
    embedding: ContinuousSearchState,
    *,
    config: ContinuousSearchConfig,
) -> tuple[mctx.RecurrentFnOutput, RecurrentState]:
    batch_size = action.shape[0]
    batch_indices = jnp.arange(batch_size)
    action_vectors = embedding.candidates[batch_indices, action]
    keys = jax.random.split(rng_key, batch_size)

    def step(
        latent_state: Array,
        action_vector: Array,
        depth: Array,
        key: Array,
    ) -> tuple[Array, Array, Array, Array, Array]:
        next_state, reward, value, policy, _ = model.recurrent_inference(
            params,
            latent_state,
            action_vector,
            None,
            training=False,
            rng=key,
        )
        sampled, _ = sample_actions(
            policy[None, ...],
            key,
            config=config,
            add_noise=False,
            sample_nums=config.leaf_action_num,
        )
        next_candidates = _pad_candidates(sampled[0], config.num_sampled_actions)
        return next_state, reward, value, next_candidates, depth + 1

    next_states, rewards, values, next_candidates, next_depths = jax.vmap(step)(
        embedding.latent_state,
        action_vectors,
        embedding.depth,
        keys,
    )
    return _recurrent_output(
        rewards,
        values,
        next_states,
        next_candidates,
        next_depths,
        config=config,
    )


def _batched_callable_recurrent_step(
    recurrent_step_fn: RecurrentStepFn,
    rng_key: Array,
    action: Array,
    embedding: ContinuousSearchState,
    *,
    config: ContinuousSearchConfig,
) -> tuple[mctx.RecurrentFnOutput, RecurrentState]:
    batch_size = action.shape[0]
    batch_indices = jnp.arange(batch_size)
    action_vectors = embedding.candidates[batch_indices, action]
    keys = jax.random.split(rng_key, batch_size)

    def step(
        latent_state: Array,
        action_vector: Array,
        depth: Array,
        key: Array,
    ) -> tuple[Array, Array, Array, Array, Array]:
        next_latent, reward, value, policy = recurrent_step_fn(
            EfficientZeroLatentState(state=latent_state),
            action_vector,
        )
        sampled, _ = sample_actions(
            policy[None, ...],
            key,
            config=config,
            add_noise=False,
            sample_nums=config.leaf_action_num,
        )
        next_candidates = _pad_candidates(sampled[0], config.num_sampled_actions)
        return next_latent.state, reward, value, next_candidates, depth + 1

    next_states, rewards, values, next_candidates, next_depths = jax.vmap(step)(
        embedding.latent_state,
        action_vectors,
        embedding.depth,
        keys,
    )
    return _recurrent_output(
        rewards,
        values,
        next_states,
        next_candidates,
        next_depths,
        config=config,
    )


def _recurrent_output(
    rewards: Array,
    values: Array,
    next_states: Array,
    next_candidates: Array,
    next_depths: Array,
    *,
    config: ContinuousSearchConfig,
) -> tuple[mctx.RecurrentFnOutput, RecurrentState]:
    batch_size = rewards.shape[0]
    next_embedding = ContinuousSearchState(
        latent_state=next_states,
        candidates=next_candidates,
        depth=next_depths,
    )
    prior_logits = jnp.zeros((batch_size, config.num_sampled_actions), dtype=jnp.float32)
    return (
        mctx.RecurrentFnOutput(
            reward=rewards,
            discount=jnp.full((batch_size,), config.discount, dtype=jnp.float32),
            prior_logits=prior_logits,
            value=values,
        ),
        _to_recurrent_embedding(next_embedding),
    )


# ---------------------------------------------------------------------------
# Search loop
# ---------------------------------------------------------------------------


def _simulation_body(
    *,
    params: ContinuousSearchParams,
    recurrent_fn: mctx.RecurrentFn,
    action_selection_fn: InteriorActionSelectionFn,
    config: ContinuousSearchConfig,
    simulation_idx: Array,
    rng_key: Array,
    tree: ContinuousMCTSTree,
    extra_data: ContinuousSearchExtraData,
) -> tuple[Array, ContinuousMCTSTree, ContinuousSearchExtraData]:
    batch_size = tree.node_values.shape[0]
    rng_key, simulate_key, expand_key = jax.random.split(rng_key, 3)
    simulate_keys = jax.random.split(simulate_key, batch_size)
    parent_index, action = mctx_search.simulate(
        simulate_keys,
        tree,
        action_selection_fn,
        config.num_simulations,
    )
    batch_range = jnp.arange(batch_size)
    next_node_index = tree.children_index[batch_range, parent_index, action]
    next_node_index = jnp.where(
        next_node_index == mctx.Tree.UNVISITED,
        simulation_idx + 1,
        next_node_index,
    )
    tree = mctx_search.expand(
        params,
        expand_key,
        tree,
        recurrent_fn,
        parent_index,
        action,
        next_node_index,
    )
    tree = mctx_search.backward(tree, next_node_index)
    extra_data = tree.extra_data
    extra_data = _update_min_max_stats(tree, extra_data, next_node_index)
    extra_data = _maybe_sequential_halving(
        extra_data,
        tree,
        simulation_idx=simulation_idx,
        config=config,
    )
    return rng_key, tree.replace(extra_data=extra_data), extra_data


def _make_jitted_simulation_step(
    *,
    params: ContinuousSearchParams,
    recurrent_fn: mctx.RecurrentFn,
    action_selection_fn: InteriorActionSelectionFn,
    config: ContinuousSearchConfig,
) -> SimulationStepFn:
    """``fori_loop`` step with a per-search ``jax.jit`` kernel (params closed)."""

    @jax.jit
    def run_one_simulation(
        simulation_idx: Array,
        rng_key: Array,
        tree: ContinuousMCTSTree,
        extra_data: ContinuousSearchExtraData,
    ) -> tuple[Array, ContinuousMCTSTree, ContinuousSearchExtraData]:
        return _simulation_body(
            params=params,
            recurrent_fn=recurrent_fn,
            action_selection_fn=action_selection_fn,
            config=config,
            simulation_idx=simulation_idx,
            rng_key=rng_key,
            tree=tree,
            extra_data=extra_data,
        )

    def simulation_step(
        simulation_idx: Array,
        carry: tuple[Array, ContinuousMCTSTree, ContinuousSearchExtraData],
    ) -> tuple[Array, ContinuousMCTSTree, ContinuousSearchExtraData]:
        rng_key, tree, extra_data = carry
        return run_one_simulation(simulation_idx, rng_key, tree, extra_data)

    return simulation_step


def make_jitted_continuous_search(
    config: ContinuousSearchConfig,
    recurrent_fn: mctx.RecurrentFn,
) -> JittedContinuousSearchFn:
    """Return a search function with a ``jax.jit``-compiled simulation kernel.

    Each MCTS simulation is compiled separately. A single outer ``jit`` over the
    full loop currently hits XLA issues when composed with EfficientZero expand.
    """
    action_selection_fn = ContinuousActionSelection(config).as_mctx()

    def search(
        params: ContinuousSearchParams,
        rng_key: Array,
        root: mctx.RootFnOutput,
        extra_data: ContinuousSearchExtraData,
        root_candidates: Array,
    ) -> ContinuousSearchResult:
        simulation_step = _make_jitted_simulation_step(
            params=params,
            recurrent_fn=recurrent_fn,
            action_selection_fn=action_selection_fn,
            config=config,
        )
        return _run_continuous_search_impl(
            params=params,
            rng_key=rng_key,
            root=root,
            action_selection_fn=action_selection_fn,
            config=config,
            extra_data=extra_data,
            root_candidates=root_candidates,
            simulation_step=simulation_step,
        )

    return search


def run_continuous_search(
    *,
    params: ContinuousSearchParams,
    rng_key: Array,
    root: mctx.RootFnOutput,
    recurrent_fn: mctx.RecurrentFn,
    config: ContinuousSearchConfig,
    extra_data: ContinuousSearchExtraData,
    root_candidates: Array,
) -> ContinuousSearchResult:
    """Run EfficientZero-V2 ``search_continuous`` using MCTX tree primitives."""
    action_selection_fn = ContinuousActionSelection(config).as_mctx()
    simulation_step = _make_jitted_simulation_step(
        params=params,
        recurrent_fn=recurrent_fn,
        action_selection_fn=action_selection_fn,
        config=config,
    )
    return _run_continuous_search_impl(
        params=params,
        rng_key=rng_key,
        root=root,
        action_selection_fn=action_selection_fn,
        config=config,
        extra_data=extra_data,
        root_candidates=root_candidates,
        simulation_step=simulation_step,
    )


def _run_continuous_search_impl(
    *,
    params: ContinuousSearchParams,
    rng_key: Array,
    root: mctx.RootFnOutput,
    action_selection_fn: InteriorActionSelectionFn,
    config: ContinuousSearchConfig,
    extra_data: ContinuousSearchExtraData,
    root_candidates: Array,
    simulation_step: SimulationStepFn,
) -> ContinuousSearchResult:
    batch_size = root.value.shape[0]
    num_actions = config.num_sampled_actions
    invalid_actions = jnp.zeros((batch_size, num_actions), dtype=jnp.float32)

    tree: ContinuousMCTSTree = mctx_search.instantiate_tree_from_root(
        root,
        config.num_simulations,
        root_invalid_actions=invalid_actions,
        extra_data=extra_data,
    )
    tree = tree.replace(extra_data=_initialize_root_selection(tree, extra_data, config))

    rng_key, tree, extra_data = jax.lax.fori_loop(
        0,
        config.num_simulations,
        simulation_step,
        (rng_key, tree, extra_data),
    )

    action_indices = extra_data.selected_children[:, 0]
    root_candidates = _normalize_root_candidates(root_candidates, batch_size)
    actions = _select_root_actions(root_candidates, action_indices)
    return ContinuousSearchResult(
        actions=actions,
        action_indices=action_indices,
        action_weights=_root_improved_policy(tree, extra_data, config),
        root_values=tree.node_values[:, mctx.Tree.ROOT_INDEX],
        search_tree=tree,
        root_candidates=root_candidates,
    )


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def _to_recurrent_embedding(state: ContinuousSearchState) -> RecurrentState:
    return state  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# EfficientZero-V2 selection and Q transforms
# ---------------------------------------------------------------------------


def _initial_extra_data(
    gumbel: Array,
    config: ContinuousSearchConfig,
) -> ContinuousSearchExtraData:
    batch_size = gumbel.shape[0]
    current_m = min(config.num_top_actions, config.num_sampled_actions)
    visit_num_for_next_phase = max(
        math.floor(config.num_simulations / (math.log2(config.num_top_actions) * current_m)),
        1,
    ) * current_m
    return ContinuousSearchExtraData(
        gumbel=gumbel,
        min_max_maximum=jnp.full((batch_size,), -jnp.inf, dtype=jnp.float32),
        min_max_minimum=jnp.full((batch_size,), jnp.inf, dtype=jnp.float32),
        selected_children=jnp.full((batch_size, config.num_sampled_actions), -1, dtype=jnp.int32),
        num_selected=jnp.zeros((batch_size,), dtype=jnp.int32),
        current_num_top_actions=jnp.full((batch_size,), current_m, dtype=jnp.int32),
        current_phase=jnp.zeros((batch_size,), dtype=jnp.int32),
        visit_num_for_next_phase=jnp.full(
            (batch_size,),
            visit_num_for_next_phase,
            dtype=jnp.int32,
        ),
        used_visit_num=jnp.zeros((batch_size,), dtype=jnp.int32),
    )


def _initialize_root_selection(
    tree: ContinuousMCTSTree,
    extra: ContinuousSearchExtraData,
    config: ContinuousSearchConfig,
) -> ContinuousSearchExtraData:
    scores = extra.gumbel + tree.children_prior_logits[:, mctx.Tree.ROOT_INDEX]
    top_m = min(config.num_top_actions, config.num_sampled_actions)
    top_idx = jnp.argsort(-scores, axis=-1)[:, :top_m]
    pad_width = config.num_sampled_actions - top_m
    selected = jnp.pad(top_idx, ((0, 0), (0, pad_width)), constant_values=-1)
    return replace(
        extra,
        selected_children=selected,
        num_selected=jnp.full((scores.shape[0],), top_m, dtype=jnp.int32),
    )


def _transformed_completed_q(
    tree: ContinuousMCTSTree,
    node_index: NodeIndex,
    extra: ContinuousSearchExtraData,
    config: ContinuousSearchConfig,
    *,
    num_children: int,
) -> Array:
    v_mix = _v_mix(tree, node_index, num_children, config)
    normalized_v_mix = _normalize_value(v_mix, extra, config=config)
    children_index = _tree_children_index(tree, node_index)[:num_children]
    rewards = _tree_children_slice(tree.children_rewards, node_index, num_children)
    child_values = _tree_gather_node_values(tree, children_index)
    is_expanded = children_index != mctx.Tree.UNVISITED
    q_sa = rewards + config.discount * child_values
    normalized = _normalize_value(q_sa, extra, config=config)
    completed = jnp.where(is_expanded, normalized, normalized_v_mix)
    child_visits = _tree_children_slice(tree.children_visits, node_index, num_children)
    scale = (config.c_visit + jnp.max(child_visits)) * config.c_scale
    return completed * scale


def _v_mix(
    tree: ContinuousMCTSTree,
    node_index: NodeIndex,
    num_actions: int,
    config: ContinuousSearchConfig,
) -> Array:
    priors = jax.nn.softmax(_tree_children_prior_logits(tree, node_index)[:num_actions])
    rewards = _tree_children_slice(tree.children_rewards, node_index, num_actions)
    values = _tree_children_slice(tree.children_values, node_index, num_actions)
    q_values = rewards + config.discount * values
    children_index = _tree_children_index(tree, node_index)[:num_actions]
    expanded = children_index != mctx.Tree.UNVISITED
    pi_sum = jnp.sum(jnp.where(expanded, priors, 0.0))
    pi_q_sum = jnp.sum(jnp.where(expanded, priors * q_values, 0.0))
    node_visit = _tree_node_visits(tree, node_index)
    node_value = _tree_node_values(tree, node_index)
    return jnp.where(
        pi_sum < 1e-6,
        node_value,
        (1.0 / (1.0 + node_visit)) * (node_value + node_visit * pi_q_sum / pi_sum),
    )


def _normalize_value(
    value: Array,
    extra: ContinuousSearchExtraData,
    *,
    config: ContinuousSearchConfig,
) -> Array:
    maximum = extra.min_max_maximum
    minimum = extra.min_max_minimum
    delta = maximum - minimum
    norm = jnp.where(
        delta > 0,
        jnp.where(
            delta < config.value_minmax_delta,
            (value - minimum) / config.value_minmax_delta,
            (value - minimum) / delta,
        ),
        value,
    )
    return jnp.clip(norm, 0.0, 1.0)


def _update_min_max_stats_unbatched(
    tree: ContinuousMCTSTree,
    extra: ContinuousSearchExtraData,
    leaf_index: Array,
) -> ContinuousSearchExtraData:
    """Mirror HyperCEZ ``back_propagate`` min-max updates along the backup path."""

    def cond_fun(state: tuple[ContinuousSearchExtraData, Array, Array]) -> Array:
        _, index, _ = state
        return index != mctx.Tree.ROOT_INDEX

    def body_fun(
        state: tuple[ContinuousSearchExtraData, Array, Array],
    ) -> tuple[ContinuousSearchExtraData, Array, Array]:
        extra_data, index, bootstrap = state
        parent = tree.parents[index]
        action = tree.action_from_parent[index]
        reward = tree.children_rewards[parent, action]
        discount = tree.children_discounts[parent, action]
        bootstrap = reward + discount * bootstrap
        return (
            replace(
                extra_data,
                min_max_maximum=jnp.maximum(extra_data.min_max_maximum, bootstrap),
                min_max_minimum=jnp.minimum(extra_data.min_max_minimum, bootstrap),
            ),
            parent,
            bootstrap,
        )

    bootstrap = tree.node_values[leaf_index]
    extra_data, _, _ = jax.lax.while_loop(
        cond_fun,
        body_fun,
        (extra, leaf_index, bootstrap),
    )
    return extra_data


@jax.vmap
def _update_min_max_stats(
    tree: ContinuousMCTSTree,
    extra: ContinuousSearchExtraData,
    leaf_index: Array,
) -> ContinuousSearchExtraData:
    return _update_min_max_stats_unbatched(tree, extra, leaf_index)


def _maybe_sequential_halving_unbatched(
    extra: ContinuousSearchExtraData,
    tree: ContinuousMCTSTree,
    *,
    simulation_idx: Array,
    config: ContinuousSearchConfig,
) -> ContinuousSearchExtraData:
    should_halve = (simulation_idx + 1) >= extra.visit_num_for_next_phase
    num_selected = extra.num_selected
    can_halve = num_selected > 1

    def halve(extra_data: ContinuousSearchExtraData) -> ContinuousSearchExtraData:
        selected = extra_data.selected_children
        children_prior = tree.children_prior_logits[mctx.Tree.ROOT_INDEX]
        transformed_q = _transformed_completed_q(
            tree,
            jnp.array(mctx.Tree.ROOT_INDEX, dtype=jnp.int32),
            extra_data,
            config,
            num_children=config.num_sampled_actions,
        )
        gumbel = extra_data.gumbel
        slot_indices = jnp.arange(config.num_top_actions)
        active = slot_indices < num_selected
        actions_at_slots = selected[slot_indices]
        scores = (
            gumbel[actions_at_slots]
            + children_prior[actions_at_slots]
            + transformed_q[actions_at_slots]
        )
        scores = jnp.where(active, scores, -jnp.inf)
        sorted_slots = jnp.argsort(-scores)
        new_count = jnp.maximum(1, num_selected // 2)

        def fill_selected(i: int, arr: Array) -> Array:
            slot = sorted_slots[i]
            return arr.at[i].set(actions_at_slots[slot])

        new_selected = jax.lax.fori_loop(
            0,
            new_count,
            fill_selected,
            jnp.full((config.num_sampled_actions,), -1, dtype=jnp.int32),
        )

        current_m = jnp.maximum(1, extra_data.current_num_top_actions // 2)
        current_phase = extra_data.current_phase + 1
        log_top = jnp.float32(math.log2(config.num_top_actions))
        extra_visit = jax.lax.cond(
            current_m > 2,
            lambda: jnp.asarray(
                jnp.maximum(
                    jnp.floor(
                        jnp.float32(config.num_simulations) / (log_top * current_m)
                    ),
                    1.0,
                )
                * current_m,
                dtype=jnp.int32,
            ),
            lambda: jnp.asarray(
                config.num_simulations - extra_data.used_visit_num,
                dtype=jnp.int32,
            ),
        )
        used_visit_num = extra_data.used_visit_num + extra_visit
        visit_num_for_next_phase = jnp.minimum(
            extra_data.visit_num_for_next_phase + extra_visit,
            config.num_simulations,
        )
        return replace(
            extra_data,
            selected_children=new_selected,
            num_selected=new_count,
            current_num_top_actions=current_m,
            current_phase=current_phase,
            visit_num_for_next_phase=visit_num_for_next_phase,
            used_visit_num=used_visit_num,
        )

    return jax.lax.cond(
        jnp.logical_and(should_halve, can_halve),
        halve,
        lambda extra_data: extra_data,
        extra,
    )


def _maybe_sequential_halving(
    extra: ContinuousSearchExtraData,
    tree: ContinuousMCTSTree,
    *,
    simulation_idx: Array,
    config: ContinuousSearchConfig,
) -> ContinuousSearchExtraData:
    return jax.vmap(
        lambda e, t: _maybe_sequential_halving_unbatched(
            e,
            t,
            simulation_idx=simulation_idx,
            config=config,
        )
    )(extra, tree)


def _root_improved_policy_unbatched(
    tree: ContinuousMCTSTree,
    extra: ContinuousSearchExtraData,
    config: ContinuousSearchConfig,
) -> Array:
    root_index = jnp.array(mctx.Tree.ROOT_INDEX, dtype=jnp.int32)
    transformed_q = _transformed_completed_q(
        tree,
        root_index,
        extra,
        config,
        num_children=config.num_sampled_actions,
    )
    children_prior = _tree_children_prior_logits(tree, root_index)
    return jax.nn.softmax(children_prior + transformed_q)


def _root_improved_policy(
    tree: ContinuousMCTSTree,
    extra: ContinuousSearchExtraData,
    config: ContinuousSearchConfig,
) -> Array:
    return jax.vmap(
        lambda t, e: _root_improved_policy_unbatched(t, e, config)
    )(tree, extra)


def _pad_candidates(candidates: Array, num_actions: int) -> Array:
    action_dim = candidates.shape[-1]
    pad_len = num_actions - candidates.shape[0]
    padded = jnp.pad(candidates, ((0, pad_len), (0, 0)))
    return padded[:num_actions]


def _normalize_root_candidates(root_candidates: Array, batch_size: int) -> Array:
    """Ensure ``[B, num_actions, action_dim]`` layout for MCTX batch size ``B``."""
    if root_candidates.ndim == 3:
        return root_candidates
    if root_candidates.ndim == 2:
        if batch_size > 1:
            raise ValueError(
                "batched search requires root_candidates with shape "
                f"[{batch_size}, num_actions, action_dim], got {root_candidates.shape}"
            )
        return root_candidates[None, ...]
    raise ValueError(
        f"root_candidates must be rank 2 or 3, got shape {root_candidates.shape}"
    )


def _select_root_actions(
    root_candidates: Array,
    action_indices: Array,
) -> Array:
    """Pick chosen root actions from ``[B, num_actions, action_dim]`` candidates."""
    batch_indices = jnp.arange(action_indices.shape[0])
    return root_candidates[batch_indices, action_indices]


# ---------------------------------------------------------------------------
# Tree accessors (unbatched MCTX simulate slices)
# ---------------------------------------------------------------------------


def _tree_gather_node_values(tree: ContinuousMCTSTree, node_indices: Array) -> Array:
    return tree.node_values[node_indices]


def _tree_children_slice(
    values: Array,
    node_index: NodeIndex,
    num_children: int,
) -> Array:
    return values[node_index, :num_children]


def _tree_node_values(tree: ContinuousMCTSTree, node_index: NodeIndex) -> Array:
    return tree.node_values[node_index]


def _tree_children_rewards(tree: ContinuousMCTSTree, node_index: NodeIndex, action: int) -> Array:
    return _tree_children_slice(tree.children_rewards, node_index, action + 1)[action]


def _tree_children_values(tree: ContinuousMCTSTree, node_index: NodeIndex, action: int) -> Array:
    return _tree_children_slice(tree.children_values, node_index, action + 1)[action]


def _tree_children_index(tree: ContinuousMCTSTree, node_index: NodeIndex) -> Array:
    return tree.children_index[node_index]


def _tree_children_prior_logits(tree: ContinuousMCTSTree, node_index: NodeIndex) -> Array:
    return tree.children_prior_logits[node_index]


def _tree_node_visits(tree: ContinuousMCTSTree, node_index: NodeIndex) -> Array:
    return tree.node_visits[node_index]


def _tree_children_visits(tree: ContinuousMCTSTree, node_index: NodeIndex) -> Array:
    return _tree_children_slice(tree.children_visits, node_index, tree.children_visits.shape[-1])

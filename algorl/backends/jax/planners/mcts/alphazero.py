"""AlphaZero planner: exact dynamics + policy-value network + MCTX search."""

from __future__ import annotations

from typing import Any, Callable, Iterator

import jax
import jax.numpy as jnp
import mctx

from algorl.backends.jax.envs.factory import search_env_from_context
from algorl.backends.jax.envs.core import SearchEnvironment
from algorl.backends.jax.planners.mcts.core import (
    BaseMCTSPlanner,
    NormalizedObservationBatch,
    RecurrentFn,
    normalize_observation_batch,
)
from algorl.core.component_context import ComponentContext
from algorl.core.types import Observation

EvaluateFn = Callable[[Any, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]


def _default_evaluate(params: Any, observations: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    raise RuntimeError(
        "AlphaZeroPlanner.evaluate is not configured. "
        "Inject evaluate= or implement backends/jax/nn/alphazero.py."
    )


def _mask_invalid_logits(logits: jnp.ndarray, invalid_actions: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(invalid_actions, -jnp.inf, logits)


def _batched_step(
    search_env: SearchEnvironment,
    states: Any,
    actions: jnp.ndarray,
) -> tuple[Any, jnp.ndarray]:
    """Vectorize single-state ``step`` over batch axis 0.

    ``jax.vmap`` only receives traceable positional args (state, action). The
    ``search_env`` is closed over in the lambda — never passed as a vmap argument.
    """
    return jax.vmap(
        lambda state, action: search_env.step(state, action),
        in_axes=(0, 0),
    )(states, actions)


def _iter_observations(observations: NormalizedObservationBatch) -> Iterator[Observation]:
    if observations.is_stacked:
        items = jnp.asarray(observations.items)
        for index in range(items.shape[0]):
            yield items[index]
        return

    yield from observations.items


def _add_batch_dim(state: Any) -> Any:
    return jax.tree.map(
        lambda leaf: jnp.asarray(leaf)[None, ...] if hasattr(leaf, "shape") else leaf,
        state,
    )


def _observations_to_states(
    search_env: SearchEnvironment,
    observations: NormalizedObservationBatch,
) -> Any:
    states = [search_env.initial_state(observation) for observation in _iter_observations(observations)]
    if len(states) == 1:
        return _add_batch_dim(states[0])
    return jax.tree.map(lambda *leaves: jnp.stack(leaves, axis=0), *states)


class AlphaZeroRecurrentFn(RecurrentFn):
    """AlphaZero tree-expansion dynamics backed by exact search-env rules."""

    def __init__(self, search_env: SearchEnvironment, evaluate: EvaluateFn) -> None:
        self._search_env = search_env
        self._evaluate = evaluate

    def apply(
        self,
        params: Any,
        rng_key: Any,
        action: jnp.ndarray,
        embedding: Any,
    ) -> tuple[mctx.RecurrentFnOutput, Any]:
        del rng_key

        next_state, reward = _batched_step(self._search_env, embedding, action)
        terminated = self._search_env.is_terminal(next_state)

        obs = self._search_env.canonical_observation(next_state)
        prior_logits, value = self._evaluate(params, obs)

        invalid = self._search_env.invalid_actions(next_state)
        prior_logits = _mask_invalid_logits(prior_logits, invalid)

        discount = jnp.where(terminated, 0.0, -1.0)
        value = jnp.where(terminated, 0.0, value)

        output = mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=prior_logits,
            value=value,
        )
        return output, next_state


class AlphaZeroPlanner(BaseMCTSPlanner):
    """MCTS with true env dynamics and a shared policy-value network.

    ``params`` must be set before ``search`` / ``search_batch`` (typically by the
    learner). ``evaluate`` is injected so this file stays independent of the
    eventual Flax module in ``backends/jax/nn/alphazero.py``.
    """

    def __init__(
        self,
        context: ComponentContext,
        *,
        search_env: SearchEnvironment | None = None,
        evaluate: EvaluateFn | None = None,
        use_gumbel: bool = False,
    ) -> None:
        super().__init__(context, use_gumbel=use_gumbel)
        self._context = context
        self._search_env_override = search_env
        self._search_env_cached: SearchEnvironment | None = None
        self.evaluate = evaluate or _default_evaluate

    @property
    def search_env(self) -> SearchEnvironment:
        if self._search_env_override is not None:
            return self._search_env_override
        if self._search_env_cached is None:
            self._search_env_cached = search_env_from_context(self._context)
        return self._search_env_cached

    def build_root(self, observations: NormalizedObservationBatch) -> mctx.RootFnOutput:
        states = _observations_to_states(self.search_env, observations)

        obs = self.search_env.canonical_observation(states)
        prior_logits, value = self.evaluate(self.params, obs)

        invalid = self.search_env.invalid_actions(states)
        prior_logits = _mask_invalid_logits(prior_logits, invalid)

        return mctx.RootFnOutput(
            prior_logits=prior_logits,
            value=value,
            embedding=states,
        )

    def make_recurrent_fn(self, observations: NormalizedObservationBatch) -> RecurrentFn:
        del observations
        return AlphaZeroRecurrentFn(self.search_env, self.evaluate)

    def search_batch(self, observations, *, deterministic=True, **kwargs):
        if kwargs.get("invalid_actions") is None:
            self._ensure_params_initialized()
            normalized = normalize_observation_batch(observations)
            states = _observations_to_states(self.search_env, normalized)
            kwargs["invalid_actions"] = self.search_env.invalid_actions(states)
        return super().search_batch(observations, deterministic=deterministic, **kwargs)


def build_alphazero_planner(context: ComponentContext) -> AlphaZeroPlanner:
    return AlphaZeroPlanner(context)

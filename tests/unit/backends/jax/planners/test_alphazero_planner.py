"""AlphaZero planner tests."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import mctx
import pytest

from algorl.agents.configs import AlphaZeroConfig
from algorl.backends.jax.backend import JAXBackend
from algorl.backends.jax.envs import PgxGymEnv
from algorl.backends.jax.planners.mcts.alphazero import (
    AlphaZeroPlanner,
    AlphaZeroRecurrentFn,
    build_alphazero_planner,
)
from algorl.backends.jax.planners.mcts.core import NormalizedObservationBatch, RecurrentFn
from algorl.core.component_context import ComponentContext


class FakeState(NamedTuple):
    board: jnp.ndarray
    terminated: jnp.ndarray


class FakeSearchEnvironment:
    def __init__(self, num_actions: int = 3) -> None:
        self.num_actions = num_actions

    def initial_state(self, observation: Any) -> FakeState:
        del observation
        return FakeState(board=jnp.zeros((3,), dtype=jnp.float32), terminated=jnp.array(False))

    def step(self, state: FakeState, action: jnp.ndarray) -> tuple[FakeState, jnp.ndarray]:
        del action
        return (
            FakeState(board=state.board + 1.0, terminated=jnp.array(False)),
            jnp.array(0.0, dtype=jnp.float32),
        )

    def is_terminal(self, state: FakeState) -> jnp.ndarray:
        return jnp.asarray(state.terminated)

    def canonical_observation(self, state: FakeState) -> jnp.ndarray:
        return state.board

    def invalid_actions(self, state: FakeState) -> jnp.ndarray:
        mask = jnp.array([False, True, False])
        if state.board.ndim > 1:
            return jnp.broadcast_to(mask, (state.board.shape[0], self.num_actions))
        return mask


def _fake_evaluate(params: dict[str, int], observations: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    batch_size = observations.shape[0]
    num_actions = params["num_actions"]
    logits = jnp.zeros((batch_size, num_actions), dtype=jnp.float32)
    value = jnp.zeros((batch_size,), dtype=jnp.float32)
    return logits, value


@pytest.fixture
def fake_planner(cartpole_env) -> AlphaZeroPlanner:
    context = ComponentContext(
        backend=JAXBackend(),
        config=AlphaZeroConfig(require_implemented=False),
        env=cartpole_env,
    )
    return AlphaZeroPlanner(
        context,
        search_env=FakeSearchEnvironment(),
        evaluate=_fake_evaluate,
    )


def test_build_alphazero_planner_factory(cartpole_env) -> None:
    context = ComponentContext(
        backend=JAXBackend(),
        config=AlphaZeroConfig(require_implemented=False),
        env=cartpole_env,
    )
    planner = build_alphazero_planner(context)
    assert isinstance(planner, AlphaZeroPlanner)


def test_build_root_shapes(fake_planner: AlphaZeroPlanner) -> None:
    fake_planner.params = {"num_actions": 3}
    observations = NormalizedObservationBatch(items=[jnp.zeros(3)], is_stacked=False)
    root = fake_planner.build_root(observations)

    assert root.prior_logits.shape == (1, 3)
    assert root.value.shape == (1,)
    assert root.prior_logits[0, 1] == -jnp.inf


def test_make_recurrent_fn_returns_mctx_output(fake_planner: AlphaZeroPlanner) -> None:
    fake_planner.params = {"num_actions": 3}
    observations = NormalizedObservationBatch(items=[jnp.zeros(3)], is_stacked=False)
    recurrent_fn = fake_planner.make_recurrent_fn(observations)
    assert isinstance(recurrent_fn, RecurrentFn)
    assert isinstance(recurrent_fn, AlphaZeroRecurrentFn)
    root = fake_planner.build_root(observations)

    output, next_state = recurrent_fn.as_mctx()(
        fake_planner.params,
        jax.random.PRNGKey(0),
        jnp.array([0], dtype=jnp.int32),
        root.embedding,
    )
    assert isinstance(output, mctx.RecurrentFnOutput)
    assert next_state.board.shape == (1, 3)


def test_search_batch_requires_params(fake_planner: AlphaZeroPlanner) -> None:
    with pytest.raises(RuntimeError, match="params are not initialized"):
        fake_planner.search_batch([jnp.zeros(3)])


def test_search_batch_runs_with_fake_env(fake_planner: AlphaZeroPlanner) -> None:
    fake_planner.params = {"num_actions": 3}
    result = fake_planner.search_batch([jnp.zeros(3)], deterministic=True)
    assert result.batch_size == 1
    assert int(result.actions.shape[0]) == 1


def test_search_env_resolves_from_pgx_gym_env() -> None:
    env = PgxGymEnv("tic_tac_toe")
    env.reset(seed=0)
    context = ComponentContext(
        backend=JAXBackend(),
        config=AlphaZeroConfig(require_implemented=False),
        env=env,
    )
    planner = AlphaZeroPlanner(context, evaluate=_fake_evaluate)
    planner.params = {"num_actions": env.action_space.n}

    obs, _ = env.reset(seed=0)
    result = planner.search_batch(obs, deterministic=True)
    assert result.batch_size == 1

    env.close()

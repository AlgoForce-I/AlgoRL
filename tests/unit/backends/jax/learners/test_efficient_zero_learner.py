"""EfficientZero learner tests."""

from __future__ import annotations

import copy

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from algorl.agents.configs import EfficientZeroConfig
from algorl.backends.jax.learners.efficient_zero import (
    EfficientZeroLearner,
    build_efficient_zero_learner,
)
from algorl.backends.jax.planners.mcts.efficientzero import build_efficient_zero_planner
from algorl.backends.jax.world_models.efficient_zero import build_efficient_zero_world_model
from algorl.buffers.efficient_zero import (
    POLICY_TARGET_INFO_KEY,
    SEARCH_VALUE_INFO_KEY,
    EfficientZeroReplayBuffer,
)
from algorl.core.component_context import ComponentContext
from algorl.core.factory import get_backend
from algorl.core.types import Transition
from algorl.envs.training_env import TrainingEnv


@pytest.fixture
def cartpole_training_env() -> TrainingEnv:
    return TrainingEnv.from_gymnasium(gym.make("CartPole-v1"))


@pytest.fixture
def learner_context(cartpole_training_env: TrainingEnv) -> ComponentContext:
    config = EfficientZeroConfig.for_dmc_state(
        require_implemented=False,
        batch_size=2,
        unroll_steps=2,
        trajectory_size=4,
        learning_starts=0,
        mcts_simulations=2,
    )
    context = ComponentContext(
        backend=get_backend("jax"),
        config=config,
        env=cartpole_training_env,
    )
    context.world_model = build_efficient_zero_world_model(context)
    context.planner = build_efficient_zero_planner(context)
    return context


def _fill_buffer(buffer: EfficientZeroReplayBuffer, *, policy_dim: int = 4) -> None:
    for index in range(8):
        policy = np.full((policy_dim,), 1.0 / policy_dim, dtype=np.float32)
        buffer.add(
            Transition(
                observation=np.full((4,), float(index), dtype=np.float32),
                action=np.asarray([0.1], dtype=np.float32),
                reward=float(index) * 0.1,
                next_observation=np.full((4,), float(index + 1), dtype=np.float32),
                done=index % 4 == 3,
                info={
                    POLICY_TARGET_INFO_KEY: policy,
                    SEARCH_VALUE_INFO_KEY: float(index) * 0.05,
                },
            )
        )


def test_build_efficient_zero_learner(learner_context: ComponentContext) -> None:
    learner = build_efficient_zero_learner(learner_context)
    assert isinstance(learner, EfficientZeroLearner)


def test_efficient_zero_learner_train_step_updates_params(learner_context: ComponentContext) -> None:
    learner = build_efficient_zero_learner(learner_context)
    buffer = EfficientZeroReplayBuffer(
        capacity=100,
        unroll_steps=learner_context.config.unroll_steps,
        trajectory_size=learner_context.config.trajectory_size,
    )
    _fill_buffer(buffer)

    params_before = copy.deepcopy(jax.tree.map(np.asarray, learner.params))
    metrics = learner.train_step(buffer)

    assert np.isfinite(metrics["loss"])
    assert metrics["value_loss"] >= 0.0
    assert metrics["policy_loss"] >= 0.0
    assert learner.world_model.params is learner.params
    assert learner.planner.params is learner.params

    params_after = jax.tree.map(np.asarray, learner.params)
    changed = any(
        not np.allclose(before, after)
        for before, after in zip(
            jax.tree.leaves(params_before),
            jax.tree.leaves(params_after),
            strict=True,
        )
    )
    assert changed

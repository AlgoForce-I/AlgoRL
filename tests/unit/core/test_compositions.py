"""Composition contract tests."""

from __future__ import annotations

import gymnasium as gym
import pytest

from algorl.agents import compositions
from algorl.agents._compose import compose_agent
from algorl.agents.configs import BaseAgentConfig
from algorl.backends.jax.factory import JAXComponentFactory


@pytest.fixture
def cartpole_env() -> gym.Env:
    return gym.make("CartPole-v1")


def test_all_compositions_define_required_slots() -> None:
    required = set(compositions.BUILD_ORDER)
    for kind in compositions.registry.kinds:
        composition = compositions.registry.create(kind)
        assert required == set(composition.components)


def test_all_compositions_resolve_on_jax(cartpole_env: gym.Env) -> None:
    config = BaseAgentConfig(require_implemented=False)
    factory = JAXComponentFactory()

    for kind in compositions.registry.kinds:
        composition = compositions.registry.create(kind)
        components = compose_agent(kind, env=cartpole_env, config=config)
        assert components.backend.name == "jax"
        assert components.world_model is not None
        assert components.planner is not None
        assert components.learner is not None
        assert components.replay_buffer is not None

        for component_type, component_kind in composition.ordered_slots():
            assert not factory.is_stub(component_type, component_kind) or config.require_implemented is False


def test_compose_agent_rejects_stubs_when_required(cartpole_env: gym.Env) -> None:
    config = BaseAgentConfig(require_implemented=True)
    with pytest.raises(NotImplementedError, match="Stub component"):
        compose_agent("dreamer_v3", env=cartpole_env, config=config)


def test_planner_receives_world_model(cartpole_env: gym.Env) -> None:
    config = BaseAgentConfig(require_implemented=False)
    components = compose_agent("muzero", env=cartpole_env, config=config)
    assert components.planner.world_model is components.world_model


def test_learner_receives_dependencies(cartpole_env: gym.Env) -> None:
    config = BaseAgentConfig(require_implemented=False)
    components = compose_agent("dreamer_v3", env=cartpole_env, config=config)
    assert components.learner.world_model is components.world_model
    assert components.learner.planner is components.planner

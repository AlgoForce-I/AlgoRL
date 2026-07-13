"""EfficientZero model construction tests."""

from __future__ import annotations

import gymnasium as gym
import jax
import jax.numpy as jnp
import pytest

from algorl.agents.configs import EfficientZeroConfig
from algorl.backends.jax.nn import (
    build_efficient_zero_model,
    build_efficient_zero_model_from_env,
    infer_model_type,
    init_efficient_zero_params_from_env,
    init_efficient_zero_params_from_model,
)
from algorl.backends.jax.nn.efficientzero.support import vector_to_scalar
from algorl.envs.training_env import TrainingEnv


@pytest.fixture
def cartpole_env() -> TrainingEnv:
    return TrainingEnv.from_gymnasium(gym.make("CartPole-v1"))


def test_infer_model_type_vector() -> None:
    assert infer_model_type(4) == "dmc_state"
    assert infer_model_type((4,)) == "dmc_state"


def test_infer_model_type_image() -> None:
    assert infer_model_type((3, 84, 84)) == "dmc_image"


def test_efficientzero_v2_dmc_state_vector_preset() -> None:
    config = EfficientZeroConfig.for_dmc_state(require_implemented=False)
    assert config.model_type == "dmc_state"
    assert config.hidden_shape == 128
    assert config.num_blocks == 2
    assert config.rep_net_shape == 256
    assert config.support_bins == 51
    assert config.value_support_type == "support"


def test_build_cartpole_with_efficientzero_v2_preset(cartpole_env: TrainingEnv) -> None:
    config = EfficientZeroConfig.for_dmc_state(require_implemented=False)
    model = build_efficient_zero_model_from_env(config, cartpole_env)
    params = init_efficient_zero_params_from_env(model, jax.random.PRNGKey(0), cartpole_env)
    obs = jnp.zeros((4,), dtype=jnp.float32)
    state, value, policy = model.initial_inference(params, obs, training=True)
    assert state.shape == (config.hidden_shape,)
    assert value.shape == (config.v_num, config.support_bins)
    assert policy.shape == (cartpole_env.action_dim * 2,)


def test_vector_networks_preserve_batch_dimension(cartpole_env: TrainingEnv) -> None:
    """Regression: batched learner inputs must produce per-sample outputs."""
    config = EfficientZeroConfig.for_dmc_state(require_implemented=False)
    model = build_efficient_zero_model_from_env(config, cartpole_env)
    params = init_efficient_zero_params_from_env(model, jax.random.PRNGKey(0), cartpole_env)

    batch_size = 3
    obs = jnp.asarray(
        jax.random.normal(jax.random.PRNGKey(1), (batch_size, 4)),
        dtype=jnp.float32,
    )
    actions = jnp.asarray(
        jax.random.normal(jax.random.PRNGKey(2), (batch_size, cartpole_env.action_dim)) * 0.1,
        dtype=jnp.float32,
    )

    with jax.default_matmul_precision("highest"):
        state = model.do_representation(params, obs)
        assert state.shape == (batch_size, config.hidden_shape)

        single = model.do_representation(params, obs[1])
        assert jnp.allclose(state[1], single, atol=1e-5)

        next_state = model.do_dynamics(params, state, actions)
        assert next_state.shape == (batch_size, config.hidden_shape)
        single_next = model.do_dynamics(params, state[1], actions[1])
        assert jnp.allclose(next_state[1], single_next, atol=1e-5)

        values, policy = model.do_value_policy_prediction(params, state)
        assert values.shape == (config.v_num, batch_size, config.support_bins)
        assert policy.shape == (batch_size, cartpole_env.action_dim * 2)
        single_values, single_policy = model.do_value_policy_prediction(params, state[1])
        assert jnp.allclose(values[:, 1], single_values, atol=1e-5)
        assert jnp.allclose(policy[1], single_policy, atol=1e-5)

        reward, _ = model.do_reward_prediction(params, next_state, None)
        assert reward.shape == (batch_size, config.support_bins)

        proj = model.do_projection(params, state, with_grad=True)
        assert proj.shape[0] == batch_size


def test_build_image_model_with_efficientzero_v2_preset() -> None:
    config = EfficientZeroConfig.for_dmc_image(require_implemented=False)
    model = build_efficient_zero_model(config, (3, 96, 96), num_actions=6)
    stacked_shape = (3 * config.n_stack, 96, 96)
    params = init_efficient_zero_params_from_model(
        model,
        jax.random.PRNGKey(0),
        (3, 96, 96),
        6,
    )
    obs = jnp.zeros(stacked_shape, dtype=jnp.float32)
    state, value, policy = model.initial_inference(params, obs, training=True)
    assert state.ndim == 3
    assert value.shape == (config.v_num, config.support_bins)
    assert policy.shape == (12,)


def test_build_atari_preset() -> None:
    config = EfficientZeroConfig.for_atari(require_implemented=False)
    model = build_efficient_zero_model(config, (4, 96, 96), num_actions=18)
    stacked_shape = (4 * config.n_stack, 96, 96)
    params = init_efficient_zero_params_from_model(
        model,
        jax.random.PRNGKey(0),
        (4, 96, 96),
        18,
    )
    obs = jnp.zeros(stacked_shape, dtype=jnp.float32)
    _, value, policy = model.initial_inference(params, obs, training=True)
    assert value.shape == (1, 51)
    assert policy.shape == (18,)


def test_recurrent_inference_cartpole(cartpole_env: TrainingEnv) -> None:
    config = EfficientZeroConfig.for_dmc_state(value_prefix=True, require_implemented=False)
    model = build_efficient_zero_model_from_env(config, cartpole_env)
    params = init_efficient_zero_params_from_env(model, jax.random.PRNGKey(0), cartpole_env)
    obs = jnp.zeros((4,), dtype=jnp.float32)
    state, _, _ = model.initial_inference(params, obs, training=True)
    next_state, prefix, values, policy, hidden = model.recurrent_inference(
        params,
        state,
        jnp.array(0, dtype=jnp.float32),
        None,
        training=True,
    )
    assert next_state.shape == state.shape
    assert prefix.shape == (config.support_bins,)
    assert values.shape == (config.v_num, config.support_bins)
    assert policy.shape == (2,)
    assert hidden is not None


def test_initial_inference_is_jittable(cartpole_env: TrainingEnv) -> None:
    config = EfficientZeroConfig.for_dmc_state(require_implemented=False)
    model = build_efficient_zero_model_from_env(config, cartpole_env)
    params = init_efficient_zero_params_from_env(model, jax.random.PRNGKey(0), cartpole_env)
    obs = jnp.zeros((4,), dtype=jnp.float32)
    rng = jax.random.PRNGKey(1)

    jit_infer = jax.jit(
        lambda p, o, k: model.initial_inference(p, o, training=False, rng=k),
    )
    state, value, policy = jit_infer(params, obs, rng)
    assert state.shape == (config.hidden_shape,)
    assert value.shape == ()
    assert policy.shape == (cartpole_env.action_dim * 2,)


def test_recurrent_inference_is_jittable(cartpole_env: TrainingEnv) -> None:
    config = EfficientZeroConfig.for_dmc_state(require_implemented=False)
    model = build_efficient_zero_model_from_env(config, cartpole_env)
    params = init_efficient_zero_params_from_env(model, jax.random.PRNGKey(0), cartpole_env)
    obs = jnp.zeros((4,), dtype=jnp.float32)
    rng = jax.random.PRNGKey(2)
    state, _, _ = model.initial_inference(params, obs, training=True)
    action = jnp.zeros((cartpole_env.action_dim,), dtype=jnp.float32)

    jit_recurrent = jax.jit(
        lambda p, s, a, k: model.recurrent_inference(
            p,
            s,
            a,
            None,
            training=False,
            rng=k,
        ),
    )
    next_state, reward, value, policy, hidden = jit_recurrent(params, state, action, rng)
    assert next_state.shape == state.shape
    assert reward.shape == ()
    assert value.shape == ()
    assert policy.shape == (cartpole_env.action_dim * 2,)
    assert hidden is None


def test_vector_to_scalar_is_jittable_with_flax_value_head(cartpole_env: TrainingEnv) -> None:
    config = EfficientZeroConfig.for_dmc_state(require_implemented=False)
    model = build_efficient_zero_model_from_env(config, cartpole_env)
    params = init_efficient_zero_params_from_env(model, jax.random.PRNGKey(0), cartpole_env)
    obs = jnp.zeros((4,), dtype=jnp.float32)

    def reduce_in_graph(p: dict, o: jnp.ndarray) -> jnp.ndarray:
        state = model.do_representation(p, o)
        values, _ = model.do_value_policy_prediction(p, state)
        return jnp.min(
            vector_to_scalar(
                values,
                support_type=config.value_support_type,
                support_bins=config.support_bins,
                support_range=config.value_support_range,
            ),
            axis=-1,
        )

    reduced = jax.jit(reduce_in_graph)(params, obs)
    assert reduced.shape == ()

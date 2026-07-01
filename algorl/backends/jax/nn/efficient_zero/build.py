"""Construct EfficientZero Flax models from config and environment shapes."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from algorl.agents.configs import EfficientZeroConfig
from algorl.envs.training_env import TrainingEnv
from algorl.backends.jax.nn.efficient_zero.blocks import (
    ConvDynamicsNetwork,
    ConvRepresentationNetwork,
    ConvSupportLSTMNetwork,
    ConvSupportNetwork,
    ConvValuePolicyNetwork,
    DenseProjectionHeadNetwork,
    DenseProjectionNetwork,
    VectorDynamicsNetwork,
    VectorProjectionHeadNetwork,
    VectorProjectionNetwork,
    VectorRepresentationNetwork,
    VectorRewardLSTMNetwork,
    VectorRewardNetwork,
    VectorValuePolicyNetwork,
    conv_state_shape,
    flatten_spatial_shape,
    support_output_size,
)
from algorl.backends.jax.nn.efficient_zero.model import EfficientZero


def infer_model_type(observation_shape: int | tuple[int, ...]) -> str:
    """Infer HyperCEZ model family from observation dimensionality."""
    if isinstance(observation_shape, int) or len(observation_shape) == 1:
        return "dmc_state"
    if len(observation_shape) == 3:
        return "dmc_image"
    raise ValueError(f"Unsupported observation shape {observation_shape!r}")


def resolve_observation_shape(
    observation_shape: int | tuple[int, ...],
    *,
    n_stack: int,
) -> tuple[int, ...]:
    if isinstance(observation_shape, int):
        return (observation_shape * n_stack,)
    shape = list(observation_shape)
    shape[0] *= n_stack
    return tuple(shape)


def _value_output_size(config: EfficientZeroConfig) -> int:
    if config.value_support_type == "symlog":
        return 1
    return support_output_size(config.value_support_type, config.support_bins)


def _reward_output_size(config: EfficientZeroConfig) -> int:
    if config.value_prefix and config.reward_support_type == "symlog":
        return 1
    if config.reward_support_type == "symlog":
        return 1
    return support_output_size(config.reward_support_type, config.support_bins)


def _policy_output_size(
    config: EfficientZeroConfig,
    *,
    num_actions: int,
    action_dim: int,
) -> int:
    if config.model_type == "atari":
        return num_actions
    if config.policy_distribution == "squashed_gaussian":
        return action_dim * 2
    return num_actions


def _build_dmc_state_model(
    config: EfficientZeroConfig,
    observation_shape: int | tuple[int, ...],
    num_actions: int,
    *,
    action_dim: int,
) -> EfficientZero:
    if isinstance(observation_shape, tuple) and len(observation_shape) == 1:
        obs_dim = observation_shape[0] // config.n_stack
    else:
        obs_dim = int(observation_shape) // config.n_stack

    value_output_size = _value_output_size(config)
    reward_output_size = _reward_output_size(config)
    policy_output_size = _policy_output_size(
        config,
        num_actions=num_actions,
        action_dim=action_dim,
    )

    representation_model = VectorRepresentationNetwork(
        obs_dim=obs_dim,
        n_stack=config.n_stack,
        num_blocks=config.num_blocks,
        rep_net_shape=config.rep_net_shape,
        hidden_shape=config.hidden_shape,
    )
    dynamics_model = VectorDynamicsNetwork(
        hidden_shape=config.hidden_shape,
        action_dim=action_dim,
        num_blocks=config.num_blocks,
        dyn_shape=config.dyn_shape,
        act_embed_shape=config.act_embed_shape,
    )
    value_policy_model = VectorValuePolicyNetwork(
        hidden_shape=config.hidden_shape,
        val_net_shape=config.val_net_shape,
        pi_net_shape=config.pi_net_shape,
        policy_output_size=policy_output_size,
        value_output_size=value_output_size,
        v_num=config.v_num,
        init_zero=config.init_zero,
        policy_distribution=config.policy_distribution,
        use_bn=config.use_bn,
    )
    if config.value_prefix:
        reward_prediction_model = VectorRewardLSTMNetwork(
            hidden_shape=config.hidden_shape,
            rew_net_shape=config.rew_net_shape,
            output_size=reward_output_size,
            lstm_hidden_size=config.lstm_hidden_size,
            init_zero=config.init_zero,
            use_bn=config.use_bn,
        )
    else:
        reward_prediction_model = VectorRewardNetwork(
            hidden_shape=config.hidden_shape,
            rew_net_shape=config.rew_net_shape,
            output_size=reward_output_size,
            init_zero=config.init_zero,
            use_bn=config.use_bn,
        )
    projection_model = VectorProjectionNetwork(
        hidden_shape=config.hidden_shape,
        proj_hid_shape=config.proj_hid_shape,
        proj_shape=config.proj_shape,
    )
    projection_head_model = VectorProjectionHeadNetwork(
        proj_shape=config.proj_shape,
        pred_hid_shape=config.pred_hid_shape,
        pred_shape=config.pred_shape,
    )
    return EfficientZero(
        representation_model=representation_model,
        dynamics_model=dynamics_model,
        reward_prediction_model=reward_prediction_model,
        value_policy_model=value_policy_model,
        projection_model=projection_model,
        projection_head_model=projection_head_model,
        config=config,
    )


def _build_conv_model(
    config: EfficientZeroConfig,
    observation_shape: tuple[int, int, int],
    num_actions: int,
    *,
    continuous: bool,
    policy_output_size: int,
    value_output_size: int,
    reward_output_size: int,
) -> EfficientZero:
    input_shape = resolve_observation_shape(observation_shape, n_stack=config.n_stack)
    if len(input_shape) != 3:
        raise ValueError(f"Expected CHW observation shape, got {input_shape!r}")

    obs_shape = (input_shape[0], input_shape[1], input_shape[2])
    state_shape = conv_state_shape(
        obs_shape,
        num_channels=config.num_channels,
        down_sample=config.down_sample,
    )
    state_dim, flatten_size = flatten_spatial_shape(
        state_shape,
        reduced_channels=config.reduced_channels,
    )

    representation_model = ConvRepresentationNetwork(
        input_shape=obs_shape,
        num_blocks=config.num_blocks,
        num_channels=config.num_channels,
        down_sample=config.down_sample,
    )
    dynamics_model = ConvDynamicsNetwork(
        num_blocks=config.num_blocks,
        num_channels=config.num_channels,
        num_actions=num_actions,
        action_embedding=config.action_embedding,
        action_embedding_dim=config.action_embedding_dim,
        continuous=continuous,
    )
    value_policy_model = ConvValuePolicyNetwork(
        num_blocks=config.num_blocks,
        num_channels=config.num_channels,
        reduced_channels=config.reduced_channels,
        flatten_size=flatten_size,
        fc_layers=config.fc_layers,
        value_output_size=value_output_size,
        policy_output_size=policy_output_size,
        v_num=config.v_num,
        init_zero=config.init_zero,
        continuous=continuous,
    )
    if config.value_prefix:
        reward_prediction_model = ConvSupportLSTMNetwork(
            num_channels=config.num_channels,
            reduced_channels=config.reduced_channels,
            flatten_size=flatten_size,
            fc_layers=config.fc_layers,
            output_size=reward_output_size,
            lstm_hidden_size=config.lstm_hidden_size,
            init_zero=config.init_zero,
        )
    else:
        reward_prediction_model = ConvSupportNetwork(
            num_channels=config.num_channels,
            reduced_channels=config.reduced_channels,
            flatten_size=flatten_size,
            fc_layers=config.fc_layers,
            output_size=reward_output_size,
            init_zero=config.init_zero,
        )

    assert config.projection_layers[1] == config.projection_head_layers[1]
    projection_model = DenseProjectionNetwork(
        state_dim=state_dim,
        hidden_dim=config.projection_layers[0],
        out_dim=config.projection_layers[1],
    )
    projection_head_model = DenseProjectionHeadNetwork(
        in_dim=config.projection_head_layers[1],
        hidden_dim=config.projection_head_layers[0],
        out_dim=config.projection_head_layers[1],
    )
    return EfficientZero(
        representation_model=representation_model,
        dynamics_model=dynamics_model,
        reward_prediction_model=reward_prediction_model,
        value_policy_model=value_policy_model,
        projection_model=projection_model,
        projection_head_model=projection_head_model,
        config=config,
    )


def build_efficient_zero_model(
    config: EfficientZeroConfig,
    observation_shape: int | tuple[int, ...],
    num_actions: int,
    *,
    action_dim: int | None = None,
) -> EfficientZero:
    """Build an EfficientZero model using the HyperCEZ ``EZAgent`` layout."""
    resolved_action_dim = action_dim if action_dim is not None else num_actions
    model_type = config.model_type
    if model_type == "auto":
        model_type = infer_model_type(observation_shape)

    if model_type == "dmc_state":
        return _build_dmc_state_model(
            config,
            observation_shape,
            num_actions,
            action_dim=resolved_action_dim,
        )

    if isinstance(observation_shape, int):
        raise ValueError(f"{model_type} models require a CHW observation shape.")
    obs_shape = resolve_observation_shape(observation_shape, n_stack=config.n_stack)
    if len(obs_shape) != 3:
        raise ValueError(f"Expected CHW observation shape, got {obs_shape!r}")

    value_output_size = _value_output_size(config)
    reward_output_size = _reward_output_size(config)

    if model_type == "atari":
        return _build_conv_model(
            config,
            obs_shape,
            num_actions,
            continuous=False,
            policy_output_size=num_actions,
            value_output_size=value_output_size,
            reward_output_size=reward_output_size,
        )
    if model_type == "dmc_image":
        return _build_conv_model(
            config,
            obs_shape,
            num_actions,
            continuous=True,
            policy_output_size=num_actions * 2,
            value_output_size=value_output_size,
            reward_output_size=reward_output_size,
        )

    raise ValueError(f"Unknown EfficientZero model_type {model_type!r}")


def build_efficient_zero_model_from_env(
    config: EfficientZeroConfig,
    env: TrainingEnv,
) -> EfficientZero:
    """Build a model from a :class:`~algorl.envs.training_env.TrainingEnv`."""
    return build_efficient_zero_model(
        config,
        env.observation_shape,
        env.num_actions,
        action_dim=env.action_dim,
    )


def init_efficient_zero_params_from_env(
    model: EfficientZero,
    rng: jax.Array,
    env: TrainingEnv,
) -> dict[str, Any]:
    """Initialize params using shapes from a :class:`~algorl.envs.training_env.TrainingEnv`."""
    return init_efficient_zero_params_from_model(
        model,
        rng,
        env.observation_shape,
        env.num_actions,
        action_dim=env.action_dim,
    )


def _dummy_observation(observation_shape: int | tuple[int, ...]) -> jnp.ndarray:
    if isinstance(observation_shape, int):
        return jnp.zeros((observation_shape,), dtype=jnp.float32)
    return jnp.zeros(tuple(observation_shape), dtype=jnp.float32)


def init_efficient_zero_params(
    model: EfficientZero,
    rng: jax.Array,
    observation_shape: int | tuple[int, ...],
    num_actions: int,
    *,
    action_dim: int | None = None,
) -> dict[str, Any]:
    """Initialize nested params for vector (dmc_state) EfficientZero submodules."""
    keys = jax.random.split(rng, 7)
    if isinstance(observation_shape, int):
        obs = jnp.zeros((observation_shape * model.config.n_stack,), dtype=jnp.float32)
    else:
        obs = _dummy_observation(resolve_observation_shape(observation_shape, n_stack=model.config.n_stack))

    action = jnp.array(0, dtype=jnp.float32).reshape(-1)
    if action_dim is not None and action_dim > 1:
        action = jnp.zeros((action_dim,), dtype=jnp.float32)
    state = jnp.zeros((model.config.hidden_shape,), dtype=jnp.float32)

    rep_vars = model.representation_model.init(keys[0], obs)
    dyn_vars = model.dynamics_model.init(keys[1], state, action)
    if model.config.value_prefix:
        reward_vars = model.reward_prediction_model.init(keys[2], state, None)
    else:
        reward_vars = model.reward_prediction_model.init(keys[2], state)
    value_policy_vars = model.value_policy_model.init(keys[3], state)
    projection_vars = model.projection_model.init(keys[4], state)
    projection_head_vars = model.projection_head_model.init(
        keys[5],
        jnp.zeros((model.config.proj_shape,), dtype=jnp.float32),
    )

    return {
        "representation_model": rep_vars["params"],
        "dynamics_model": dyn_vars["params"],
        "reward_prediction_model": reward_vars["params"],
        "value_policy_model": value_policy_vars["params"],
        "projection_model": projection_vars["params"],
        "projection_head_model": projection_head_vars["params"],
    }


def init_efficient_zero_params_from_model(
    model: EfficientZero,
    rng: jax.Array,
    observation_shape: int | tuple[int, ...],
    num_actions: int,
    *,
    action_dim: int | None = None,
) -> dict[str, Any]:
    """Initialize params using shapes inferred from the built model."""
    model_type = model.config.model_type
    if model_type == "auto":
        model_type = infer_model_type(observation_shape)
    if model_type == "dmc_state":
        return init_efficient_zero_params(
            model,
            rng,
            observation_shape,
            num_actions,
            action_dim=action_dim,
        )
    return _init_conv_params(model, rng, observation_shape, num_actions)


def _init_conv_params(
    model: EfficientZero,
    rng: jax.Array,
    observation_shape: int | tuple[int, ...],
    num_actions: int,
) -> dict[str, Any]:
    keys = jax.random.split(rng, 7)
    obs = _dummy_observation(resolve_observation_shape(observation_shape, n_stack=model.config.n_stack))
    action = jnp.array(0, dtype=jnp.int32)
    rep_vars = model.representation_model.init(keys[0], obs)
    state = model.representation_model.apply(rep_vars, obs)
    dyn_vars = model.dynamics_model.init(keys[1], state, action)
    if model.config.value_prefix:
        reward_vars = model.reward_prediction_model.init(keys[2], state, None)
    else:
        reward_vars = model.reward_prediction_model.init(keys[2], state)
    value_policy_vars = model.value_policy_model.init(keys[3], state)
    projection_vars = model.projection_model.init(keys[4], state)
    projection_head_vars = model.projection_head_model.init(
        keys[5],
        jnp.zeros((model.config.projection_layers[1],), dtype=jnp.float32),
    )

    return {
        "representation_model": rep_vars["params"],
        "dynamics_model": dyn_vars["params"],
        "reward_prediction_model": reward_vars["params"],
        "value_policy_model": value_policy_vars["params"],
        "projection_model": projection_vars["params"],
        "projection_head_model": projection_head_vars["params"],
    }

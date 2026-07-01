"""EfficientZero world model backed by the Flax latent-dynamics network."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import jax
import jax.numpy as jnp

from algorl.agents.configs import EfficientZeroConfig
from algorl.backends.jax.nn.efficient_zero.build import (
    build_efficient_zero_model_from_env,
    infer_model_type,
    init_efficient_zero_params_from_env,
)
from algorl.backends.jax.nn.efficient_zero.model import EfficientZero as EfficientZeroNetwork
from algorl.backends.jax.nn.efficient_zero.model import Params
from algorl.core.component_context import ComponentContext
from algorl.core.types import Action, LatentState, Observation
from algorl.core.world_model import WorldModel
from algorl.envs.training_env import TrainingEnv


@dataclass(frozen=True)
class EfficientZeroLatentState:
    """Latent state carried through MCTS unrolls and training."""

    state: jnp.ndarray
    reward_hidden: Any = None


class EfficientZeroWorldModel(WorldModel):
    """Wraps :class:`EfficientZeroNetwork` for search, rollout, and learning."""

    def __init__(self, context: ComponentContext) -> None:
        if not isinstance(context.config, EfficientZeroConfig):
            raise TypeError(
                "EfficientZeroWorldModel requires EfficientZeroConfig, "
                f"got {type(context.config)!r}."
            )
        if not isinstance(context.env, TrainingEnv):
            raise TypeError(
                "EfficientZeroWorldModel requires TrainingEnv, "
                f"got {type(context.env)!r}."
            )

        self.backend = context.backend
        self.config: EfficientZeroConfig = context.config
        self.env: TrainingEnv = context.env
        self.model: EfficientZeroNetwork = build_efficient_zero_model_from_env(
            self.config,
            self.env,
        )
        init_key = self.backend.random_key(self.config.seed)
        self.params: Params = init_efficient_zero_params_from_env(
            self.model,
            init_key,
            self.env,
        )
        self._rng_key = self.backend.random_key(self.config.seed + 1)

    @property
    def resolved_model_type(self) -> str:
        model_type = self.config.model_type
        if model_type == "auto":
            return infer_model_type(self.env.observation_shape)
        return model_type

    def _obs_array(self, observation: Observation) -> jnp.ndarray:
        if isinstance(observation, Mapping):
            raise TypeError(
                "EfficientZeroWorldModel does not support Dict observations yet."
            )
        return jnp.asarray(observation, dtype=jnp.float32)

    def _action_array(self, action: Action) -> jnp.ndarray:
        if self.resolved_model_type == "atari":
            return jnp.asarray(action, dtype=jnp.float32).reshape(())
        return jnp.asarray(action, dtype=jnp.float32).reshape(-1)

    def _split_rng(self) -> jax.Array:
        self._rng_key, subkey = jax.random.split(self._rng_key)
        return subkey

    def encode(self, observation: Observation) -> EfficientZeroLatentState:
        obs = self._obs_array(observation)
        state = self.model.do_representation(self.params, obs)
        return EfficientZeroLatentState(state=state)

    def transition(self, latent_state: LatentState, action: Action) -> EfficientZeroLatentState:
        latent = self._require_latent_state(latent_state)
        action_arr = self._action_array(action)
        next_state = self.model.do_dynamics(self.params, latent.state, action_arr)
        return EfficientZeroLatentState(
            state=next_state,
            reward_hidden=latent.reward_hidden,
        )

    def reward(self, latent_state: LatentState, action: Action) -> Any:
        latent = self._require_latent_state(latent_state)
        action_arr = self._action_array(action)
        next_state = self.model.do_dynamics(self.params, latent.state, action_arr)
        value_prefix, _ = self.model.do_reward_prediction(
            self.params,
            next_state,
            latent.reward_hidden,
        )
        return self.model._reduce_reward_prefix(value_prefix)

    def value(self, latent_state: LatentState) -> Any:
        latent = self._require_latent_state(latent_state)
        values, _ = self.model.do_value_policy_prediction(self.params, latent.state)
        return self.model._reduce_values(values, rng=self._split_rng())

    def initial_step(
        self,
        observation: Observation,
        *,
        training: bool = False,
    ) -> tuple[EfficientZeroLatentState, jnp.ndarray, jnp.ndarray]:
        """Root inference used by MCTS: latent state, value, policy head output."""
        obs = self._obs_array(observation)
        state, value, policy = self.model.initial_inference(
            self.params,
            obs,
            training=training,
            rng=self._split_rng(),
        )
        return EfficientZeroLatentState(state=state), value, policy

    def recurrent_step(
        self,
        latent_state: EfficientZeroLatentState,
        action: Action,
        *,
        training: bool = False,
    ) -> tuple[EfficientZeroLatentState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """One model unroll used by MCTS: next latent, reward, value, policy."""
        action_arr = self._action_array(action)
        next_state, reward, value, policy, reward_hidden = self.model.recurrent_inference(
            self.params,
            latent_state.state,
            action_arr,
            latent_state.reward_hidden,
            training=training,
            rng=self._split_rng(),
        )
        return (
            EfficientZeroLatentState(state=next_state, reward_hidden=reward_hidden),
            reward,
            value,
            policy,
        )

    @staticmethod
    def _require_latent_state(latent_state: LatentState) -> EfficientZeroLatentState:
        if not isinstance(latent_state, EfficientZeroLatentState):
            raise TypeError(
                "EfficientZeroWorldModel expects EfficientZeroLatentState, "
                f"got {type(latent_state)!r}."
            )
        return latent_state


def build_efficient_zero_world_model(context: ComponentContext) -> EfficientZeroWorldModel:
    return EfficientZeroWorldModel(context)

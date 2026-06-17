"""Gymnasium-backed search dynamics for discrete and continuous action spaces."""

from __future__ import annotations

import copy
import pickle
from dataclasses import dataclass
from typing import Any, NamedTuple

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from algorl.backends.jax.envs.core import SearchEnvironment
from algorl.core.types import Observation

# Pickled ``env.unwrapped`` snapshots are stored as fixed-size uint8 buffers so
# MCTS states remain JAX PyTrees compatible with ``jax.vmap``.
MAX_ENV_BLOB_BYTES = 8192


class GymSearchState(NamedTuple):
    """Internal search state carried through the MCTS tree."""

    observation: jnp.ndarray
    terminated: jnp.ndarray
    truncated: jnp.ndarray
    env_blob: jnp.ndarray
    blob_length: jnp.ndarray


@dataclass(frozen=True)
class _ActionSpaceInfo:
    is_discrete: bool
    num_actions: int
    action_shape: tuple[int, ...]
    action_low: np.ndarray
    action_high: np.ndarray


def _observation_to_array(observation: Observation) -> jnp.ndarray:
    if isinstance(observation, dict):
        raise TypeError(
            "GymnasiumSearchEnvironment does not support Dict observations yet. "
            "Wrap the env or provide a custom SearchEnvironment."
        )
    return jnp.asarray(observation, dtype=jnp.float32)


def _action_space_info(action_space: gym.Space) -> _ActionSpaceInfo:
    if isinstance(action_space, gym.spaces.Discrete):
        return _ActionSpaceInfo(
            is_discrete=True,
            num_actions=int(action_space.n),
            action_shape=(),
            action_low=np.zeros((), dtype=np.float32),
            action_high=np.zeros((), dtype=np.float32),
        )

    if isinstance(action_space, gym.spaces.Box):
        low = np.asarray(action_space.low, dtype=np.float32)
        high = np.asarray(action_space.high, dtype=np.float32)
        return _ActionSpaceInfo(
            is_discrete=False,
            num_actions=0,
            action_shape=tuple(int(dim) for dim in action_space.shape),
            action_low=low,
            action_high=high,
        )

    raise TypeError(
        "GymnasiumSearchEnvironment supports gym.spaces.Discrete and "
        f"gym.spaces.Box action spaces, got {type(action_space)!r}."
    )


def _pack_env_blob(env: gym.Env) -> tuple[np.ndarray, np.int32]:
    payload = pickle.dumps(copy.deepcopy(env.unwrapped))
    if len(payload) > MAX_ENV_BLOB_BYTES:
        raise ValueError(
            f"Pickled Gymnasium env state requires {len(payload)} bytes, "
            f"but MAX_ENV_BLOB_BYTES={MAX_ENV_BLOB_BYTES}."
        )
    blob = np.zeros(MAX_ENV_BLOB_BYTES, dtype=np.uint8)
    blob[: len(payload)] = np.frombuffer(payload, dtype=np.uint8)
    return blob, np.int32(len(payload))


def _unpack_env_blob(blob: np.ndarray, blob_length: np.int32) -> gym.Env:
    payload = bytes(np.asarray(blob[:blob_length], dtype=np.uint8))
    return pickle.loads(payload)


def _host_step_discrete(
    env_blob: np.ndarray,
    blob_length: np.int32,
    action: np.int32,
) -> tuple[np.ndarray, np.float32, np.bool_, np.bool_, np.ndarray, np.int32]:
    env = _unpack_env_blob(env_blob, blob_length)
    observation, reward, terminated, truncated, _ = env.step(int(action))
    next_blob, next_length = _pack_env_blob(env)
    return (
        np.asarray(observation, dtype=np.float32),
        np.float32(reward),
        np.bool_(terminated),
        np.bool_(truncated),
        next_blob,
        next_length,
    )


def _host_step_continuous(
    env_blob: np.ndarray,
    blob_length: np.int32,
    action: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> tuple[np.ndarray, np.float32, np.bool_, np.bool_, np.ndarray, np.int32]:
    env = _unpack_env_blob(env_blob, blob_length)
    clipped = np.clip(np.asarray(action, dtype=np.float32), action_low, action_high)
    observation, reward, terminated, truncated, _ = env.step(clipped)
    next_blob, next_length = _pack_env_blob(env)
    return (
        np.asarray(observation, dtype=np.float32),
        np.float32(reward),
        np.bool_(terminated),
        np.bool_(truncated),
        next_blob,
        next_length,
    )


class GymnasiumSearchEnvironment(SearchEnvironment):
    """Exact Gymnasium simulator for MCTS on discrete or continuous actions.

    Each search state stores a pickled snapshot of ``env.unwrapped`` so tree
    branches can fork without mutating the live training environment. Steps run
    on the host through ``jax.pure_callback`` and are safe to batch with
    ``jax.vmap`` when ``vmap_method='sequential'`` is set.

    Continuous ``Box`` actions are clipped to ``[low, high]`` before stepping.
    Discrete invalid-action masks are not defined for continuous spaces; planners
    that discretize actions (for example Gumbel MCTS) supply masks separately.
    """

    def __init__(self, env: gym.Env) -> None:
        self._env = env
        self._action_space = _action_space_info(env.action_space)

    @property
    def is_discrete(self) -> bool:
        return self._action_space.is_discrete

    @property
    def num_actions(self) -> int:
        return self._action_space.num_actions

    @property
    def action_shape(self) -> tuple[int, ...]:
        return self._action_space.action_shape

    def initial_state(self, observation: Observation) -> GymSearchState:
        env_blob, blob_length = _pack_env_blob(self._env)
        return GymSearchState(
            observation=_observation_to_array(observation),
            terminated=jnp.array(False),
            truncated=jnp.array(False),
            env_blob=jnp.asarray(env_blob),
            blob_length=jnp.asarray(blob_length, dtype=jnp.int32),
        )

    def step(self, state: GymSearchState, action: jnp.ndarray) -> tuple[GymSearchState, jnp.ndarray]:
        if self._action_space.is_discrete:
            return self._step_discrete(state, action)
        return self._step_continuous(state, action)

    def _step_discrete(
        self,
        state: GymSearchState,
        action: jnp.ndarray,
    ) -> tuple[GymSearchState, jnp.ndarray]:
        outputs = jax.pure_callback(
            _host_step_discrete,
            (
                jax.ShapeDtypeStruct(state.observation.shape, jnp.float32),
                jax.ShapeDtypeStruct((), jnp.float32),
                jax.ShapeDtypeStruct((), jnp.bool_),
                jax.ShapeDtypeStruct((), jnp.bool_),
                jax.ShapeDtypeStruct(state.env_blob.shape, jnp.uint8),
                jax.ShapeDtypeStruct((), jnp.int32),
            ),
            state.env_blob,
            state.blob_length,
            action.astype(jnp.int32),
            vmap_method="sequential",
        )
        return self._to_next_state(state, outputs)

    def _step_continuous(
        self,
        state: GymSearchState,
        action: jnp.ndarray,
    ) -> tuple[GymSearchState, jnp.ndarray]:
        action_vector = jnp.asarray(action, dtype=jnp.float32).reshape(self._action_space.action_shape)
        outputs = jax.pure_callback(
            _host_step_continuous,
            (
                jax.ShapeDtypeStruct(state.observation.shape, jnp.float32),
                jax.ShapeDtypeStruct((), jnp.float32),
                jax.ShapeDtypeStruct((), jnp.bool_),
                jax.ShapeDtypeStruct((), jnp.bool_),
                jax.ShapeDtypeStruct(state.env_blob.shape, jnp.uint8),
                jax.ShapeDtypeStruct((), jnp.int32),
            ),
            state.env_blob,
            state.blob_length,
            action_vector,
            jnp.asarray(self._action_space.action_low, dtype=jnp.float32),
            jnp.asarray(self._action_space.action_high, dtype=jnp.float32),
            vmap_method="sequential",
        )
        return self._to_next_state(state, outputs)

    def _to_next_state(
        self,
        state: GymSearchState,
        outputs: tuple[Any, ...],
    ) -> tuple[GymSearchState, jnp.ndarray]:
        observation, reward, terminated, truncated, env_blob, blob_length = outputs
        next_state = GymSearchState(
            observation=observation,
            terminated=terminated,
            truncated=truncated,
            env_blob=env_blob,
            blob_length=blob_length,
        )
        return next_state, reward

    def is_terminal(self, state: GymSearchState) -> jnp.ndarray:
        return jnp.asarray(state.terminated | state.truncated)

    def canonical_observation(self, state: GymSearchState) -> jnp.ndarray:
        return jnp.asarray(state.observation, dtype=jnp.float32)

    def invalid_actions(self, state: GymSearchState) -> jnp.ndarray | None:
        if not self._action_space.is_discrete:
            return None

        mask = jnp.zeros((self._action_space.num_actions,), dtype=bool)
        observation = jnp.asarray(state.observation)
        if observation.ndim > 1:
            batch_size = int(observation.shape[0])
            return jnp.broadcast_to(mask, (batch_size, self._action_space.num_actions))
        return mask


class GymnasiumSearchEnv(gym.Wrapper):
    """Optional Gymnasium wrapper exposing ``search_environment()``."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self._search_env = GymnasiumSearchEnvironment(self.env)

    def search_environment(self) -> SearchEnvironment:
        return self._search_env

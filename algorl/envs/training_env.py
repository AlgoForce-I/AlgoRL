"""Unified training environment facade for Gymnasium and JAX-native envs."""

from __future__ import annotations

from typing import Any, Callable

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from algorl.envs.jax_env import BatchedJaxEnv, JaxEnv, JaxRolloutBatch, JaxState, PolicyFn


def _box_space(shape: tuple[int, ...], *, low: float, high: float) -> gym.spaces.Box:
    return gym.spaces.Box(
        low=low,
        high=high,
        shape=shape,
        dtype=np.float32,
    )


def _observation_shape_from_space(space: gym.Space) -> int | tuple[int, ...]:
    if isinstance(space, gym.spaces.Box):
        shape = tuple(int(dim) for dim in space.shape)
        if len(shape) == 1:
            return shape[0]
        return shape
    if isinstance(space, gym.spaces.Discrete):
        return int(space.n)
    raise TypeError(f"Unsupported observation space type: {type(space)!r}")


def _num_actions_from_space(space: gym.Space) -> int:
    if isinstance(space, gym.spaces.Discrete):
        return int(space.n)
    if isinstance(space, gym.spaces.Box):
        return int(space.shape[0])
    raise TypeError(f"Unsupported action space type: {type(space)!r}")


def _action_dim_from_space(space: gym.Space) -> int:
    """HyperCEZ ``control_dim``: scalar discrete action or continuous vector size."""
    if isinstance(space, gym.spaces.Discrete):
        return 1
    if isinstance(space, gym.spaces.Box):
        return int(space.shape[0])
    raise TypeError(f"Unsupported action space type: {type(space)!r}")


class TrainingEnv:
    """Environment handle used by agents and :class:`~algorl.core.training_loop.TrainingLoop`."""

    def __init__(
        self,
        *,
        observation_space: gym.Space,
        action_space: gym.Space,
        raw: object,
        is_batched: bool = False,
        num_envs: int = 1,
        reset_fn: Callable[[int | None], tuple[np.ndarray, dict[str, Any]]] | None = None,
        step_fn: Callable[[Any], tuple[np.ndarray, float, bool, bool, dict[str, Any]]] | None = None,
        search_environment_fn: Callable[[], Any] | None = None,
        collect_rollout_fn: Callable[[PolicyFn, int, jnp.ndarray], JaxRolloutBatch] | None = None,
    ) -> None:
        self.observation_space = observation_space
        self.action_space = action_space
        self.raw = raw
        self.is_batched = is_batched
        self.num_envs = num_envs
        self._reset_fn = reset_fn
        self._step_fn = step_fn
        self._search_environment_fn = search_environment_fn
        self._collect_rollout_fn = collect_rollout_fn

    @classmethod
    def from_gymnasium(cls, env: gym.Env) -> TrainingEnv:
        return cls(
            observation_space=env.observation_space,
            action_space=env.action_space,
            raw=env,
            reset_fn=lambda seed: env.reset(seed=seed),
            step_fn=lambda action: env.step(action),
            search_environment_fn=lambda: env.search_environment() if hasattr(env, "search_environment") else None,
        )

    @classmethod
    def from_jax(cls, env: JaxEnv | BatchedJaxEnv, *, seed: int = 0) -> TrainingEnv:
        if isinstance(env, BatchedJaxEnv):
            return cls._from_batched_jax(env, seed=seed)
        return cls._from_single_jax(env, seed=seed)

    @classmethod
    def _from_single_jax(cls, env: JaxEnv, *, seed: int) -> TrainingEnv:
        key = jax.random.PRNGKey(seed)
        state_holder: dict[str, JaxState | None] = {"state": None}

        def reset(seed_value: int | None) -> tuple[np.ndarray, dict[str, Any]]:
            nonlocal key
            if seed_value is not None:
                key = jax.random.PRNGKey(seed_value)
            key, subkey = jax.random.split(key)
            if state_holder["state"] is not None:
                state_holder["state"] = env.reset_after_episode(subkey, state_holder["state"])
            else:
                state_holder["state"] = env.reset(subkey)
            obs = np.asarray(env.observation(state_holder["state"]), dtype=np.float32)
            return obs, env.info(state_holder["state"])

        def step(action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
            nonlocal key
            if state_holder["state"] is None:
                raise RuntimeError("TrainingEnv.step called before reset().")
            key, subkey = jax.random.split(key)
            del subkey
            action_array = jnp.asarray(action, dtype=jnp.float32)
            state_holder["state"] = env.step(state_holder["state"], action_array)
            obs = np.asarray(env.observation(state_holder["state"]), dtype=np.float32)
            reward = float(env.reward(state_holder["state"]))
            terminated = bool(env.terminated(state_holder["state"]))
            truncated = bool(env.truncated(state_holder["state"]))
            return obs, reward, terminated, truncated, env.info(state_holder["state"])

        search_fn = (
            (lambda: env.search_environment())
            if hasattr(env, "search_environment")
            else None
        )

        return cls(
            observation_space=_box_space(env.observation_shape, low=-np.inf, high=np.inf),
            action_space=_box_space(env.action_shape, low=-1.0, high=1.0),
            raw=env,
            reset_fn=reset,
            step_fn=step,
            search_environment_fn=search_fn,
        )

    @classmethod
    def _from_batched_jax(cls, env: BatchedJaxEnv, *, seed: int) -> TrainingEnv:
        key = jax.random.PRNGKey(seed)

        def collect_rollout(
            policy: PolicyFn,
            num_steps: int,
            rollout_key: jnp.ndarray,
        ) -> JaxRolloutBatch:
            return env.collect_rollout(policy, num_steps, key=rollout_key)

        def reset(seed_value: int | None) -> tuple[np.ndarray, dict[str, Any]]:
            nonlocal key
            if seed_value is not None:
                key = jax.random.PRNGKey(seed_value)
            key, subkey = jax.random.split(key)
            state = env.reset(subkey)
            obs = np.asarray(env.observation(state)[0], dtype=np.float32)
            return obs, {}

        def step(action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
            raise RuntimeError(
                "Batched JAX environments do not support sequential step(). "
                "TrainingLoop will use collect_rollout() instead."
            )

        search_fn = (
            (lambda: env.search_environment())
            if hasattr(env, "search_environment")
            else None
        )

        return cls(
            observation_space=_box_space(env.observation_shape, low=-np.inf, high=np.inf),
            action_space=_box_space(env.action_shape, low=-1.0, high=1.0),
            raw=env,
            is_batched=True,
            num_envs=env.num_envs,
            reset_fn=reset,
            step_fn=step,
            search_environment_fn=search_fn,
            collect_rollout_fn=collect_rollout,
        )

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if self._reset_fn is None:
            raise RuntimeError("TrainingEnv has no reset implementation.")
        return self._reset_fn(seed)

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._step_fn is None:
            raise RuntimeError("TrainingEnv has no step implementation.")
        return self._step_fn(action)

    def search_environment(self) -> Any | None:
        if self._search_environment_fn is None:
            return None
        return self._search_environment_fn()

    def collect_rollout(
        self,
        policy: PolicyFn,
        num_steps: int,
        *,
        key: jnp.ndarray | None = None,
    ) -> JaxRolloutBatch:
        if self._collect_rollout_fn is None:
            raise RuntimeError("TrainingEnv does not support batched rollouts.")
        rollout_key = key if key is not None else jax.random.PRNGKey(0)
        return self._collect_rollout_fn(policy, num_steps, rollout_key)

    @property
    def observation_shape(self) -> int | tuple[int, ...]:
        """Model input shape derived from :attr:`observation_space`."""
        return _observation_shape_from_space(self.observation_space)

    @property
    def num_actions(self) -> int:
        """Discrete branch count or continuous action dimension."""
        return _num_actions_from_space(self.action_space)

    @property
    def action_dim(self) -> int:
        """Control dimension fed to dynamics (1 for discrete scalar actions)."""
        return _action_dim_from_space(self.action_space)

    @property
    def unwrapped(self) -> object:
        raw = self.raw
        if isinstance(raw, gym.Env):
            return raw.unwrapped
        return raw

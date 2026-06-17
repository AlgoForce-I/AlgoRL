"""Pgx-backed search environment for AlphaZero."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import pgx
from pgx.core import Env as PgxEnv
from pgx.core import State as PgxState

from algorl.backends.jax.envs.core import SearchEnvironment
from algorl.core.types import Observation


class PgxSearchEnvironment(SearchEnvironment):
    """Wrap one ``pgx.Env`` for MCTX tree expansion."""

    def __init__(self, pgx_env: PgxEnv) -> None:
        self._env = pgx_env
        self._current_state: PgxState | None = None

    def bind_state(self, state: PgxState) -> None:
        """Called by the Gym wrapper on ``reset`` so search sees the live state."""
        self._current_state = state

    def initial_state(self, observation: Observation) -> PgxState:
        # v1: rely on bound state from last reset (obs alone is not invertible)
        if self._current_state is not None:
            return self._current_state
        raise RuntimeError(
            "PgxSearchEnvironment has no bound state. "
            "Reset a PgxGymEnv before calling planner.search()."
        )

    def step(self, state: PgxState, action: jnp.ndarray) -> tuple[PgxState, jnp.ndarray]:
        next_state = self._env.step(state, action)
        reward = self._player_reward(next_state)
        return next_state, reward

    def is_terminal(self, state: PgxState) -> jnp.ndarray:
        return state.terminated

    def canonical_observation(self, state: PgxState) -> jnp.ndarray:
        return jnp.asarray(state.observation, dtype=jnp.float32)

    def invalid_actions(self, state: PgxState) -> jnp.ndarray:
        legal = jnp.asarray(state.legal_action_mask, dtype=bool)
        return ~legal

    def _player_reward(self, state: PgxState) -> jnp.ndarray:
        """Terminal reward from the acting player's perspective; 0 otherwise."""
        rewards = jnp.asarray(state.rewards, dtype=jnp.float32)
        player = jnp.asarray(state.current_player, dtype=jnp.int32)
        terminal_reward = rewards[player]
        return jnp.where(state.terminated, terminal_reward, 0.0)


class PgxGymEnv(gym.Env):
    """Minimal Gymnasium facade over Pgx for AlgoRL training + search."""

    metadata = {"render_modes": []}

    def __init__(self, env_id: str, *, auto_reset: bool = True) -> None:
        super().__init__()
        self._pgx_env = pgx.make(env_id)
        self._search_env = PgxSearchEnvironment(self._pgx_env)
        self._state: PgxState | None = None
        self._auto_reset = auto_reset
        self._key = jax.random.PRNGKey(0)

        sample = self._pgx_env.init(self._key)
        self.action_space = gym.spaces.Discrete(int(sample.legal_action_mask.shape[-1]))
        obs_shape = np.asarray(sample.observation).shape
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=obs_shape, dtype=np.float32
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._key = jax.random.PRNGKey(seed)
        self._key, subkey = jax.random.split(self._key)
        self._state = self._pgx_env.init(subkey)
        self._search_env.bind_state(self._state)
        obs = np.asarray(self._state.observation, dtype=np.float32)
        return obs, {}

    def step(self, action: int):
        assert self._state is not None
        self._state = self._pgx_env.step(self._state, jnp.int32(action))
        self._search_env.bind_state(self._state)
        obs = np.asarray(self._state.observation, dtype=np.float32)
        reward = float(np.asarray(self._state.rewards)[int(self._state.current_player)])
        terminated = bool(np.asarray(self._state.terminated))
        truncated = False
        if terminated and self._auto_reset:
            obs, _ = self.reset()
        return obs, reward, terminated, truncated, {}

    def search_environment(self) -> SearchEnvironment:
        return self._search_env
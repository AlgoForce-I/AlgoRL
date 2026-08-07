"""Gymnasium vector environments as :class:`~algorl.envs.jax_env.BatchedJaxEnv` adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from algorl.envs.jax_env import BatchedJaxEnv, JaxRolloutBatch, JaxState, PolicyFn, RolloutStepCallback


def _emit_rollout_step(
    on_step: RolloutStepCallback | None,
    *,
    num_envs: int,
    reward: np.ndarray,
    done: np.ndarray,
    lane_infos: list[dict[str, Any]] | None = None,
    step_info: dict[str, Any] | None = None,
) -> None:
    if on_step is None:
        return
    reward_array = np.asarray(reward, dtype=np.float32).reshape(-1)
    done_array = np.asarray(done, dtype=bool).reshape(-1)
    infos = lane_infos if lane_infos is not None else [{} for _ in range(int(reward_array.shape[0]))]
    info: dict[str, Any] = {
        "train/reward": float(np.mean(reward_array)),
        "rewards": reward_array,
        "dones": done_array,
        "infos": [dict(item) for item in infos],
    }
    if step_info is not None:
        info.update(step_info)
    on_step(int(reward_array.shape[0]), info)


def _observation_shape_from_space(space: gym.Space) -> tuple[int, ...]:
    if isinstance(space, gym.spaces.Box):
        return tuple(int(dim) for dim in space.shape)
    if isinstance(space, gym.spaces.Discrete):
        return (int(space.n),)
    raise TypeError(f"Unsupported observation space type: {type(space)!r}")


def _action_shape_from_space(space: gym.Space) -> tuple[int, ...]:
    if isinstance(space, gym.spaces.Box):
        return tuple(int(dim) for dim in space.shape)
    if isinstance(space, gym.spaces.Discrete):
        return ()
    raise TypeError(f"Unsupported action space type: {type(space)!r}")


def _lane_infos_from_vector_infos(infos: dict[str, Any], num_envs: int) -> list[dict[str, Any]]:
    lane_infos: list[dict[str, Any]] = []
    for env_idx in range(num_envs):
        lane_info: dict[str, Any] = {}
        for key, value in infos.items():
            if key.startswith("_"):
                continue
            if isinstance(value, (list, tuple)):
                lane_info[key] = value[env_idx]
            elif isinstance(value, np.ndarray) and value.shape and value.shape[0] == num_envs:
                lane_info[key] = value[env_idx]
            elif not isinstance(value, (dict, list, tuple, np.ndarray)):
                lane_info[key] = value
        lane_infos.append(lane_info)
    return lane_infos


def _final_observation_batch(
    infos: dict[str, Any],
    num_envs: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if "final_observation" in infos or "_final_observation" in infos:
        mask = infos.get("_final_observation")
        finals = infos.get("final_observation")
    elif "final_obs" in infos or "_final_obs" in infos:
        mask = infos.get("_final_obs")
        finals = infos.get("final_obs")
    else:
        return None, None

    if mask is None or finals is None:
        return None, None

    mask_array = np.asarray(mask, dtype=bool).reshape(-1)
    if mask_array.shape[0] != num_envs:
        return None, None

    finals_raw = finals
    if isinstance(finals_raw, np.ndarray) and finals_raw.dtype == object:
        finals_array = np.stack(
            [np.asarray(item, dtype=np.float32) for item in finals_raw],
            axis=0,
        )
    else:
        finals_array = np.asarray(finals_raw, dtype=np.float32)
        if finals_array.ndim == 1 and num_envs == 1:
            finals_array = finals_array.reshape(1, -1)
    return mask_array, finals_array


def _replay_next_observations(
    next_observations: np.ndarray,
    terminations: np.ndarray,
    truncations: np.ndarray,
    infos: dict[str, Any],
) -> np.ndarray:
    """Use terminal observations for done lanes when the vector env auto-resets."""
    replay_next = np.asarray(next_observations, dtype=np.float32).copy()
    done = np.asarray(terminations | truncations, dtype=bool)
    if not done.any():
        return replay_next

    mask, finals = _final_observation_batch(infos, replay_next.shape[0])
    if mask is None or finals is None:
        return replay_next

    replay_next[mask] = finals[mask]
    return replay_next


def _format_policy_actions(
    actions: np.ndarray,
    *,
    action_space: gym.Space,
    num_envs: int,
) -> np.ndarray:
    array = np.asarray(actions)
    if isinstance(action_space, gym.spaces.Discrete):
        return array.astype(np.int64).reshape(num_envs)
    return array.astype(np.float32).reshape(num_envs, *_action_shape_from_space(action_space))


def _actions_for_vector_env(
    actions: np.ndarray,
    *,
    action_space: gym.Space,
) -> np.ndarray:
    array = np.asarray(actions)
    if isinstance(action_space, gym.spaces.Discrete):
        return array.astype(np.int64).reshape(-1)
    return array.astype(np.float32)


def _uses_next_step_autoreset(vector_env: gym.vector.VectorEnv) -> bool:
    """Whether the vector env resets done lanes on the *following* step call."""
    mode = getattr(vector_env, "metadata", {}).get("autoreset_mode")
    if mode is None:
        return False
    try:
        from gymnasium.vector import AutoresetMode
    except ImportError:
        return False
    return mode in (AutoresetMode.NEXT_STEP, AutoresetMode.NEXT_STEP.value)


@dataclass
class _GymVectorState:
    observation: np.ndarray


class GymnasiumVectorJaxEnv(BatchedJaxEnv):
    """Adapter for ``gymnasium.vector.VectorEnv`` with persisted rollout state."""

    def __init__(self, vector_env: gym.vector.VectorEnv, *, seed: int = 0) -> None:
        self._vector_env = vector_env
        self._seed = seed
        self._state: _GymVectorState | None = None
        self._next_step_autoreset = _uses_next_step_autoreset(vector_env)
        # Lanes whose previous step ended an episode; under NEXT_STEP autoreset
        # their next step() call is a reset, not a real transition.
        self._autoreset_lanes = np.zeros((int(vector_env.num_envs),), dtype=bool)

    @property
    def vector_env(self) -> gym.vector.VectorEnv:
        return self._vector_env

    @property
    def num_envs(self) -> int:
        return int(self._vector_env.num_envs)

    @property
    def observation_shape(self) -> tuple[int, ...]:
        return _observation_shape_from_space(self._vector_env.single_observation_space)

    @property
    def action_shape(self) -> tuple[int, ...]:
        return _action_shape_from_space(self._vector_env.single_action_space)

    def reset(self, key: jnp.ndarray) -> JaxState:
        del key
        observation, _ = self._vector_env.reset(seed=self._seed)
        self._state = _GymVectorState(observation=np.asarray(observation, dtype=np.float32))
        self._autoreset_lanes[:] = False
        return self._state

    def observation(self, state: JaxState) -> jnp.ndarray:
        if not isinstance(state, _GymVectorState):
            raise TypeError(f"Expected _GymVectorState, got {type(state)!r}.")
        return jnp.asarray(state.observation, dtype=jnp.float32)

    def step(self, state: JaxState, actions: jnp.ndarray) -> JaxState:
        if not isinstance(state, _GymVectorState):
            raise TypeError(f"Expected _GymVectorState, got {type(state)!r}.")
        vector_actions = _actions_for_vector_env(
            np.asarray(actions),
            action_space=self._vector_env.single_action_space,
        )
        next_observation, _, terminations, truncations, _ = self._vector_env.step(vector_actions)
        self._autoreset_lanes = np.asarray(terminations | truncations, dtype=bool)
        next_state = _GymVectorState(observation=np.asarray(next_observation, dtype=np.float32))
        self._state = next_state
        return next_state

    def collect_rollout(
        self,
        policy: PolicyFn,
        num_steps: int,
        *,
        key: jnp.ndarray,
        on_step: RolloutStepCallback | None = None,
    ) -> JaxRolloutBatch:
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")

        if self._state is None:
            key, reset_key = jax.random.split(key)
            del reset_key
            self.reset(key)

        assert self._state is not None
        _, rollout_key = jax.random.split(key)
        step_keys = jax.random.split(rollout_key, num_steps)
        state = self._state

        observations: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        rewards: list[np.ndarray] = []
        next_observations: list[np.ndarray] = []
        dones: list[np.ndarray] = []
        step_infos: list[list[dict[str, Any]]] = []

        action_space = self._vector_env.single_action_space
        for step_idx in range(num_steps):
            obs = np.asarray(state.observation, dtype=np.float32)
            raw_action = policy(obs, step_keys[step_idx])
            action = _format_policy_actions(
                np.asarray(raw_action),
                action_space=action_space,
                num_envs=self.num_envs,
            )
            vector_actions = _actions_for_vector_env(action, action_space=action_space)
            resetting_lanes = (
                self._autoreset_lanes.copy()
                if self._next_step_autoreset
                else np.zeros((self.num_envs,), dtype=bool)
            )
            next_observation, reward, terminations, truncations, infos = self._vector_env.step(
                vector_actions
            )
            replay_next = _replay_next_observations(
                np.asarray(next_observation, dtype=np.float32),
                np.asarray(terminations, dtype=bool),
                np.asarray(truncations, dtype=bool),
                infos,
            )
            done = np.asarray(terminations | truncations, dtype=bool)
            self._autoreset_lanes = done.copy()
            lane_infos = _lane_infos_from_vector_infos(infos, self.num_envs)
            for env_idx in np.flatnonzero(resetting_lanes):
                # NEXT_STEP autoreset: this lane's step was a reset (action
                # ignored, reward forced to 0); flag it so replay storage
                # skips the fabricated transition.
                lane_infos[int(env_idx)]["replay_skip"] = True

            observations.append(obs)
            actions.append(action)
            rewards.append(np.asarray(reward, dtype=np.float32))
            next_observations.append(replay_next)
            dones.append(done)
            step_infos.append(lane_infos)
            state = _GymVectorState(observation=np.asarray(next_observation, dtype=np.float32))
            _emit_rollout_step(
                on_step,
                num_envs=self.num_envs,
                reward=rewards[-1],
                done=done,
                lane_infos=lane_infos,
            )

        self._state = state
        return JaxRolloutBatch(
            observation=np.stack(observations, axis=0),
            action=np.stack(actions, axis=0),
            reward=np.stack(rewards, axis=0),
            next_observation=np.stack(next_observations, axis=0),
            done=np.stack(dones, axis=0),
            step_info=step_infos,
        )


def make_gymnasium_vector_env(
    make_env: Callable[[], gym.Env],
    num_envs: int,
    *,
    vector_cls: type[gym.vector.VectorEnv] | None = None,
    vector_kwargs: dict[str, Any] | None = None,
    seed: int = 0,
) -> GymnasiumVectorJaxEnv:
    """Build a batched adapter from ``num_envs`` independent Gymnasium env factories."""
    if num_envs < 1:
        raise ValueError("num_envs must be >= 1")
    if vector_cls is None:
        vector_cls = gym.vector.SyncVectorEnv
    kwargs = dict(vector_kwargs or {})
    if vector_cls is gym.vector.AsyncVectorEnv and "context" not in kwargs:
        # JAX uses threads; fork after import can deadlock worker processes.
        kwargs["context"] = "spawn"
    vector_env = vector_cls([make_env for _ in range(num_envs)], **kwargs)
    return GymnasiumVectorJaxEnv(vector_env, seed=seed)

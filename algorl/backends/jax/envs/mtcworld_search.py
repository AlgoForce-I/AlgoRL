"""MTCWorldMJX search dynamics for MCTS planners."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from algorl.backends.jax.envs.core import SearchEnvironment
from algorl.core.types import Observation

if TYPE_CHECKING:
    from MTCWorldMJX import mjx_env
    from MTCWorldMJX.cw_env import ContinualLearningEnv, CWTaskEnv
    from MTCWorldMJX.envs.sawyer_xyz import SawyerXYZEnv


class _MtcworldMJXSearchEnvironment(SearchEnvironment):
    """Shared MJX state search plumbing for continuous Sawyer tasks."""

    def __init__(self, action_size: int, *, name: str) -> None:
        self._action_size = action_size
        self._name = name
        self._current_state: mjx_env.State | None = None

    @property
    def is_discrete(self) -> bool:
        return False

    @property
    def action_shape(self) -> tuple[int, ...]:
        return (self._action_size,)

    def bind_state(self, state: mjx_env.State) -> None:
        self._current_state = state

    def initial_state(self, observation: Observation) -> mjx_env.State:
        del observation
        if self._current_state is not None:
            return self._current_state
        raise RuntimeError(
            f"{self._name} has no bound state. Reset the environment before calling planner.search()."
        )

    def is_terminal(self, state: mjx_env.State) -> jnp.ndarray:
        return jnp.asarray((state.terminated + state.truncated) > 0)

    def canonical_observation(self, state: mjx_env.State) -> jnp.ndarray:
        return jnp.asarray(state.obs, dtype=jnp.float32)

    def invalid_actions(self, state: mjx_env.State) -> jnp.ndarray | None:
        del state
        return None

    def _reshape_action(self, action: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(action, dtype=jnp.float32).reshape(self.action_shape)


class MtcworldSearchEnvironment(_MtcworldMJXSearchEnvironment):
    """Wrap a MTCWorldMJX ``SawyerXYZEnv`` for MCTX tree expansion."""

    def __init__(self, env: SawyerXYZEnv) -> None:
        super().__init__(env.action_size, name="MtcworldSearchEnvironment")
        self._env = env
        self._jit_step = jax.jit(env.step)

    def step(
        self,
        state: mjx_env.State,
        action: jnp.ndarray,
    ) -> tuple[mjx_env.State, jnp.ndarray]:
        next_state = self._jit_step(state, self._reshape_action(action))
        return next_state, jnp.asarray(next_state.reward, dtype=jnp.float32)


class MtcworldCWSearchEnvironment(_MtcworldMJXSearchEnvironment):
    """Search dynamics for one active Continual World task."""

    def __init__(self, cl_env: ContinualLearningEnv) -> None:
        super().__init__(cl_env.task_envs[0].env.action_size, name="MtcworldCWSearchEnvironment")
        self._cl_env = cl_env

    def _active_task_env(self, state: mjx_env.State) -> CWTaskEnv:
        seq_idx = int(state.info["seq_idx"])
        return self._cl_env.task_envs[seq_idx]

    def step(
        self,
        state: mjx_env.State,
        action: jnp.ndarray,
    ) -> tuple[mjx_env.State, jnp.ndarray]:
        task_env = self._active_task_env(state)
        next_state = task_env.step(state, self._reshape_action(action))
        info = dict(next_state.info)
        info["seq_idx"] = state.info["seq_idx"]
        info["global_step"] = state.info["global_step"]
        info.pop("forced_task_change", None)
        info.pop("task_changed", None)
        next_state = next_state.replace(info=info)
        return next_state, jnp.asarray(next_state.reward, dtype=jnp.float32)


class MtcworldCWTaskSearchEnvironment(_MtcworldMJXSearchEnvironment):
    """Search dynamics for a fixed ``CWTaskEnv`` evaluation task."""

    def __init__(self, task_env: CWTaskEnv) -> None:
        super().__init__(task_env.env.action_size, name="MtcworldCWTaskSearchEnvironment")
        self._task_env = task_env
        self._jit_step = jax.jit(task_env.step)

    def step(
        self,
        state: mjx_env.State,
        action: jnp.ndarray,
    ) -> tuple[mjx_env.State, jnp.ndarray]:
        next_state = self._jit_step(state, self._reshape_action(action))
        return next_state, jnp.asarray(next_state.reward, dtype=jnp.float32)

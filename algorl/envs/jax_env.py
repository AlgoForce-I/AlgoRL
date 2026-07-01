"""JAX-native environment protocols (Brax-style)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

import jax.numpy as jnp
import numpy as np

from algorl.core.types import Transition

JaxState = Any
PolicyFn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


@dataclass(frozen=True)
class JaxRolloutBatch:
    """On-device rollout tensors with shape ``(num_steps, num_envs, ...)``."""

    observation: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    next_observation: np.ndarray
    done: np.ndarray

    @property
    def num_steps(self) -> int:
        return int(self.observation.shape[0])

    @property
    def num_envs(self) -> int:
        return int(self.observation.shape[1])

    def to_transitions(self) -> list[Transition]:
        transitions: list[Transition] = []
        for step_idx in range(self.num_steps):
            for env_idx in range(self.num_envs):
                transitions.append(
                    Transition(
                        observation=self.observation[step_idx, env_idx],
                        action=self.action[step_idx, env_idx],
                        reward=float(self.reward[step_idx, env_idx]),
                        next_observation=self.next_observation[step_idx, env_idx],
                        done=bool(self.done[step_idx, env_idx]),
                        info={},
                    )
                )
        return transitions


@runtime_checkable
class JaxEnv(Protocol):
    """Single-lane JAX environment with functional ``reset`` / ``step``."""

    @property
    def observation_shape(self) -> tuple[int, ...]: ...

    @property
    def action_shape(self) -> tuple[int, ...]: ...

    def reset(self, key: jnp.ndarray) -> JaxState: ...

    def step(self, state: JaxState, action: jnp.ndarray) -> JaxState: ...

    def observation(self, state: JaxState) -> jnp.ndarray: ...

    def reward(self, state: JaxState) -> jnp.ndarray | float: ...

    def terminated(self, state: JaxState) -> jnp.ndarray | bool: ...

    def truncated(self, state: JaxState) -> jnp.ndarray | bool: ...

    def info(self, state: JaxState) -> dict[str, Any]: ...

    def reset_after_episode(self, key: jnp.ndarray, state: JaxState) -> JaxState:
        """Reset after an episode boundary while preserving protocol state."""


@runtime_checkable
class BatchedJaxEnv(Protocol):
    """Vectorized JAX environment with on-device rollouts."""

    @property
    def num_envs(self) -> int: ...

    @property
    def observation_shape(self) -> tuple[int, ...]: ...

    @property
    def action_shape(self) -> tuple[int, ...]: ...

    def reset(self, key: jnp.ndarray) -> JaxState: ...

    def step(self, state: JaxState, actions: jnp.ndarray) -> JaxState: ...

    def observation(self, state: JaxState) -> jnp.ndarray: ...

    def collect_rollout(
        self,
        policy: PolicyFn,
        num_steps: int,
        *,
        key: jnp.ndarray,
    ) -> JaxRolloutBatch: ...

"""JAX search dynamics for MCTS expansion (not Gymnasium stepping)."""

from __future__ import annotations

from abc import ABC
from typing import Any

import jax.numpy as jnp

from algorl.core.types import Observation


class SearchEnvironment(ABC):
    """Batched-friendly game rules used inside MCTX ``recurrent_fn``."""

    @property
    def is_discrete(self) -> bool:
        """Whether search actions are discrete indices."""
        return True

    def initial_state(self, observation: Observation) -> Any:
        """Map a Gymnasium/root observation to an internal search state."""

    def step(self, state: Any, action: jnp.ndarray) -> tuple[Any, jnp.ndarray]:
        """Apply one action to a single search state.

        For discrete spaces ``action`` is a scalar index. For continuous
        ``Box`` spaces it is a vector with shape ``action_shape``. The planner
        batches this with ``jax.vmap(..., in_axes=(0, 0))``.
        """

    def is_terminal(self, state: Any) -> jnp.ndarray:
        """Bool tensor with shape ``[B]`` when ``state`` is batched."""

    def canonical_observation(self, state: Any) -> jnp.ndarray:
        """Network input from search state, shape ``[B, ...]`` when batched."""

    def invalid_actions(self, state: Any) -> jnp.ndarray | None:
        """Bool mask ``[B, num_actions]`` where ``True`` = illegal.

        Return ``None`` when the action space is continuous and no discrete
        invalid-action mask applies.
        """

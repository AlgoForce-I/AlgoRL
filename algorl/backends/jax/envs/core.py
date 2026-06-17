"""JAX search dynamics for MCTS expansion (not Gymnasium stepping)."""

from __future__ import annotations

from abc import ABC
from typing import Any

import jax.numpy as jnp

from algorl.core.types import Observation


class SearchEnvironment(ABC):
    """Batched-friendly game rules used inside MCTX ``recurrent_fn``."""

    def initial_state(self, observation: Observation) -> Any:
        """Map a Gymnasium/root observation to an internal search state."""

    def step(self, state: Any, action: jnp.ndarray) -> tuple[Any, jnp.ndarray]:
        """Apply one action to a single search state.

        The planner batches this with ``jax.vmap(..., in_axes=(0, 0))``; do not
        pass Python objects (such as ``self``) as extra vmap arguments.
        """

    def is_terminal(self, state: Any) -> jnp.ndarray:
        """Bool tensor with shape ``[B]`` when ``state`` is batched."""

    def canonical_observation(self, state: Any) -> jnp.ndarray:
        """Network input from search state, shape ``[B, ...]`` when batched."""

    def invalid_actions(self, state: Any) -> jnp.ndarray:
        """Bool mask ``[B, num_actions]``; ``True`` = illegal."""
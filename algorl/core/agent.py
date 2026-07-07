"""Base agent interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from algorl.core.types import Action, Observation
from algorl.envs.training_env import TrainingEnv


class BaseAgent(ABC):
    """Common interface for all AlgoRL agents.

    Every agent expects a Gymnasium environment, a JAX-native environment, or a
    :class:`~algorl.envs.training_env.TrainingEnv` wrapper.
    """

    env: TrainingEnv

    @abstractmethod
    def learn(self, total_timesteps: int, **kwargs: Any) -> None:
        """Train the agent for ``total_timesteps`` environment steps.

        Implement in each agent under ``agents/``:
        - interact with ``self.env`` via ``reset()`` and ``step()``
        - use ``self.planner`` to choose actions
        - store data in a replay buffer
        - call ``self.learner.train_step(...)`` on a schedule
        """

    @abstractmethod
    def predict(
        self,
        observation: Observation,
        deterministic: bool = True,
    ) -> Action:
        """Select an action for ``observation``."""

    def save(self, path: str) -> None:
        """Persist agent state to ``path``.

        Implement: save model parameters, optimizer state, replay buffer, and step count.
        """
        raise NotImplementedError

    def load(self, path: str) -> None:
        """Restore agent state from ``path``.

        Implement: load the data written by ``save()``.
        """
        raise NotImplementedError

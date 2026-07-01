"""Shared context passed to component builders during agent composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from algorl.agents.configs import BaseAgentConfig
    from algorl.core.backend import Backend
    from algorl.core.learner import Learner
    from algorl.core.planner import Planner
    from algorl.core.replay_buffer import ReplayBuffer
    from algorl.core.world_model import WorldModel
    from algorl.envs.training_env import TrainingEnv


@dataclass
class ComponentContext:
    """Mutable build context used to wire dependent components together."""

    backend: Backend
    config: BaseAgentConfig
    env: TrainingEnv
    world_model: WorldModel | None = None
    planner: Planner | None = None
    learner: Learner | None = None
    replay_buffer: ReplayBuffer | None = None

"""Built agent components."""

from __future__ import annotations

from dataclasses import dataclass

from algorl.core.backend import Backend
from algorl.core.learner import Learner
from algorl.core.planner import Planner
from algorl.core.world_model import WorldModel


@dataclass
class AgentComponents:
    """Concrete components wired together for one agent."""

    backend: Backend
    world_model: WorldModel
    planner: Planner
    learner: Learner

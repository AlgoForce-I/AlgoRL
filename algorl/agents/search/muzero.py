"""MuZero agent."""

from __future__ import annotations

from algorl.agents._composed import ComposedAgent
from algorl.agents.configs import MuZeroConfig


class MuZero(ComposedAgent):
    """MuZero search-based agent."""

    composition_name = "muzero"
    config_class = MuZeroConfig

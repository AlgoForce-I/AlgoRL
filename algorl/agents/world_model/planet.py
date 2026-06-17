"""PlaNet agent."""

from __future__ import annotations

from algorl.agents._composed import ComposedAgent
from algorl.agents.configs import PlaNetConfig


class PlaNet(ComposedAgent):
    """PlaNet world-model agent."""

    composition_name = "planet"
    config_class = PlaNetConfig

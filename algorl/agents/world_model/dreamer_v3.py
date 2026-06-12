"""DreamerV3 agent."""

from __future__ import annotations

from algorl.agents._composed import ComposedAgent
from algorl.agents.configs import DreamerV3Config


class DreamerV3(ComposedAgent):
    """DreamerV3 world-model agent."""

    composition_name = "dreamer_v3"
    config_class = DreamerV3Config

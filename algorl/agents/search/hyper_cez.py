"""HyperCEZ agent (EfficientZero + task-conditioned hypernetworks)."""

from __future__ import annotations

from algorl.agents._composed import ComposedAgent
from algorl.agents.configs import HyperCEZConfig


class HyperCEZ(ComposedAgent):
    """HyperCEZDelta continual-learning agent built on EfficientZero."""

    composition_name = "hyper_cez"
    config_class = HyperCEZConfig

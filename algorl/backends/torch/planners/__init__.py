"""PyTorch planner registry (stub).

Register torch planners here once the PyTorch backend is implemented.
"""

from __future__ import annotations

from algorl.core.planner import Planner
from algorl.core.registry import KindRegistry

registry: KindRegistry[Planner] = KindRegistry("PyTorch planner")

# Register torch implementations with @registry.register("kind") when ready.

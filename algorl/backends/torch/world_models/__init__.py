"""PyTorch world model registry (stub).

Register torch world models here once the PyTorch backend is implemented.
"""

from __future__ import annotations

from algorl.core.registry import KindRegistry
from algorl.core.world_model import WorldModel

registry: KindRegistry[WorldModel] = KindRegistry("PyTorch world model")

# Register torch implementations with @registry.register("kind") when ready.

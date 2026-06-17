"""Replay buffer registry."""

from __future__ import annotations

from algorl.buffers.efficient_zero import EfficientZeroReplayBuffer
from algorl.buffers.episode import EpisodeReplayBuffer
from algorl.buffers.replay import UniformReplayBuffer
from algorl.buffers.search import SearchReplayBuffer
from algorl.core.component_context import ComponentContext
from algorl.core.registry import KindRegistry
from algorl.core.replay_buffer import ReplayBuffer

registry: KindRegistry[ReplayBuffer] = KindRegistry("replay buffer")


def _build_uniform(context: ComponentContext) -> ReplayBuffer:
    return UniformReplayBuffer(capacity=context.config.buffer_capacity)


def _build_efficient_zero(context: ComponentContext) -> ReplayBuffer:
    return EfficientZeroReplayBuffer(capacity=context.config.buffer_capacity)


def _build_search(context: ComponentContext) -> ReplayBuffer:
    return SearchReplayBuffer(capacity=context.config.buffer_capacity)


def _build_episode(context: ComponentContext) -> ReplayBuffer:
    return EpisodeReplayBuffer(capacity=context.config.buffer_capacity)


registry.register("uniform", _build_uniform)
registry.register("search", _build_search, stub=True)
registry.register("episode", _build_episode, stub=True)
registry.register("efficient_zero", _build_efficient_zero, stub=True)

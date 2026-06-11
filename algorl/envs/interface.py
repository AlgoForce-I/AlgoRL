"""Gymnasium environment adapter."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EnvLike(Protocol):
    """Minimal environment interface used by agents."""

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]: ...

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]: ...


class EnvAdapter:
    """Thin wrapper around a Gymnasium-compatible environment."""

    def __init__(self, env: EnvLike) -> None:
        self.env = env

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        return self.env.reset(**kwargs)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        return self.env.step(action)

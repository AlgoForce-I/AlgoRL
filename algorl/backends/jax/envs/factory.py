from __future__ import annotations

import gymnasium as gym
import pgx

from algorl.backends.jax.envs.core import SearchEnvironment
from algorl.backends.jax.envs.gymnasium_env import GymnasiumSearchEnvironment
from algorl.backends.jax.envs.pgx import PgxSearchEnvironment
from algorl.core.component_context import ComponentContext
from algorl.envs.resolve import resolve_env
from algorl.envs.training_env import TrainingEnv

_PGX_ENV_IDS = frozenset(pgx.available_envs())


def _supports_gymnasium_search(action_space: gym.Space) -> bool:
    return isinstance(action_space, (gym.spaces.Discrete, gym.spaces.Box))


def search_env_from_context(context: ComponentContext) -> SearchEnvironment:
    """Resolve a :class:`SearchEnvironment` from the agent build context."""
    training_env = (
        context.env if isinstance(context.env, TrainingEnv) else resolve_env(context.env)
    )

    if training_env.is_batched:
        raise NotImplementedError(
            "Batched environments do not support search-based planners. "
            "Use a single-lane environment instead."
        )

    search_env = training_env.search_environment()
    if search_env is not None:
        return search_env

    raw = training_env.raw
    if isinstance(raw, gym.Env):
        env_id = getattr(getattr(raw, "spec", None), "id", None)
        if env_id in _PGX_ENV_IDS:
            return PgxSearchEnvironment(pgx.make(env_id))
        if _supports_gymnasium_search(raw.action_space):
            return GymnasiumSearchEnvironment(raw)

    raise NotImplementedError(
        f"No JAX SearchEnvironment adapter for {type(raw)!r}. "
        "Use a Gymnasium env, JAX-native MTCWorld env, or implement search_environment()."
    )

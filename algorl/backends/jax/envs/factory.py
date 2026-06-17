from __future__ import annotations

import gymnasium as gym
import pgx

from algorl.backends.jax.envs.gymnasium_env import GymnasiumSearchEnvironment, GymnasiumSearchEnv
from algorl.backends.jax.envs.pgx import PgxGymEnv, PgxSearchEnvironment
from algorl.backends.jax.envs.core import SearchEnvironment
from algorl.core.component_context import ComponentContext

_PGX_ENV_IDS = frozenset(pgx.available_envs())


def _supports_gymnasium_search(action_space: gym.Space) -> bool:
    return isinstance(action_space, (gym.spaces.Discrete, gym.spaces.Box))


def search_env_from_context(context: ComponentContext) -> SearchEnvironment:
    """Resolve a :class:`SearchEnvironment` from the agent build context."""
    env = context.env

    if hasattr(env, "search_environment"):
        return env.search_environment()

    unwrapped = env.unwrapped
    if isinstance(unwrapped, PgxSearchEnvironment):
        return unwrapped

    if isinstance(unwrapped, PgxGymEnv):
        return unwrapped.search_environment()

    if isinstance(unwrapped, GymnasiumSearchEnv):
        return unwrapped.search_environment()

    env_id = env.spec.id if env.spec is not None else None
    if env_id in _PGX_ENV_IDS:
        pgx_env = pgx.make(env_id)
        return PgxSearchEnvironment(pgx_env)

    if _supports_gymnasium_search(env.action_space):
        return GymnasiumSearchEnvironment(env)

    raise NotImplementedError(
        f"No JAX SearchEnvironment adapter for {type(unwrapped)!r} "
        f"(spec={env_id!r}, action_space={env.action_space!r}). "
        "Use a discrete or Box Gymnasium env, PgxGymEnv, or implement search_environment()."
    )

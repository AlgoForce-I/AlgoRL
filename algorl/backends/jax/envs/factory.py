from __future__ import annotations

import pgx

from algorl.backends.jax.envs.pgx import PgxGymEnv, PgxSearchEnvironment
from algorl.backends.jax.envs.core import SearchEnvironment
from algorl.core.component_context import ComponentContext

_PGX_ENV_IDS = frozenset(pgx.available_envs())  # or a smaller allowlist for v1


def search_env_from_context(context: ComponentContext) -> SearchEnvironment:
    env = context.env

    if hasattr(env, "search_environment"):
        return env.search_environment()

    unwrapped = env.unwrapped
    if isinstance(unwrapped, PgxSearchEnvironment):
        return unwrapped

    if isinstance(unwrapped, PgxGymEnv):
        return unwrapped.search_environment()

    env_id = env.spec.id if env.spec is not None else None
    if env_id in _PGX_ENV_IDS:
        pgx_env = pgx.make(env_id)
        return PgxSearchEnvironment(pgx_env)

    raise NotImplementedError(
        f"No JAX SearchEnvironment adapter for {type(unwrapped)!r} "
        f"(spec={env_id!r}). Use PgxGymEnv or implement search_environment()."
    )
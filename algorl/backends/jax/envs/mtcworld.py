"""Gymnasium facades and factories for MTCWorldMJX."""

from __future__ import annotations

from typing import TYPE_CHECKING

from algorl.backends.jax.envs.mtcworld_jax import (
    ContinualLearningJaxEnv,
    CWTaskJaxEnv,
    SawyerJaxEnv,
    require_mtcworld,
)
from algorl.backends.jax.envs.mtcworld_search import (
    MtcworldCWSearchEnvironment,
    MtcworldCWTaskSearchEnvironment,
    MtcworldSearchEnvironment,
)
from algorl.envs.gym_jax import JaxGymEnv

if TYPE_CHECKING:
    from MTCWorldMJX.cw_benchmarks import CWConfig
    from MTCWorldMJX.cw_env import RandomizationKind
    from MTCWorldMJX.envs.sawyer_xyz import SawyerXYZConfig

CW_BENCHMARK_NAMES = ("CW10", "CW20")


class MtcworldGymEnv(JaxGymEnv):
    """Gymnasium facade over one MTCWorldMJX task."""

    def __init__(
        self,
        env_id: str,
        *,
        config: SawyerXYZConfig | None = None,
        seed: int = 0,
    ) -> None:
        mtc = require_mtcworld()
        if env_id not in mtc.ENV_CLS_MAP:
            raise ValueError(
                f"Unknown MTCWorldMJX environment {env_id!r}. "
                f"Expected one of {sorted(mtc.ENV_CLS_MAP)}."
            )
        self._env_id = env_id
        super().__init__(SawyerJaxEnv(mtc.make(env_id, config=config)), seed=seed)

    @property
    def env_id(self) -> str:
        return self._env_id

    def search_environment(self) -> MtcworldSearchEnvironment:
        return super().search_environment()


class MtcworldContinualGymEnv(JaxGymEnv):
    """Gymnasium facade over ``ContinualLearningEnv``."""

    def __init__(
        self,
        benchmark: str = "CW10",
        *,
        config: CWConfig | None = None,
        seed: int = 0,
        steps_per_task: int | None = None,
        randomization: RandomizationKind | None = None,
    ) -> None:
        if benchmark not in CW_BENCHMARK_NAMES:
            raise ValueError(f"Unknown Continual World benchmark {benchmark!r}.")
        mtc = require_mtcworld()
        cl_env = mtc.make_cl_train_env(
            benchmark,
            config=config,
            seed=seed,
            steps_per_task=steps_per_task,
            randomization=randomization,
        )
        self.benchmark_name = benchmark
        super().__init__(ContinualLearningJaxEnv(cl_env), seed=seed)

    @property
    def num_tasks(self) -> int:
        return self.jax_env.continual_env.num_tasks

    @property
    def steps_per_task(self) -> int:
        return self.jax_env.continual_env.steps_per_task

    @property
    def steps_limit(self) -> int:
        return self.jax_env.continual_env.steps_limit

    @property
    def task_names(self) -> tuple[str, ...]:
        return self.jax_env.task_names

    def search_environment(self) -> MtcworldCWSearchEnvironment:
        return super().search_environment()


class MtcworldCWEvalGymEnv(JaxGymEnv):
    """Gymnasium facade over one CW eval task."""

    def __init__(
        self,
        benchmark: str,
        task_index: int,
        *,
        config: CWConfig | None = None,
        seed: int = 0,
        randomization: RandomizationKind = "deterministic",
    ) -> None:
        mtc = require_mtcworld()
        task_env = mtc.make_cl_test_envs(
            benchmark,
            config=config,
            seed=seed,
            randomization=randomization,
        )[task_index]
        self.benchmark_name = benchmark
        self.task_index = task_index
        super().__init__(CWTaskJaxEnv(task_env), seed=seed)

    @property
    def env_name(self) -> str:
        return self.jax_env.task_env.env_name

    def search_environment(self) -> MtcworldCWTaskSearchEnvironment:
        return super().search_environment()

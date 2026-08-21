"""Periodic evaluation schedule, isolation, and aggregation tests."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from algorl.common.logger import Logger
from algorl.core.evaluation import (
    EVAL_EPISODES,
    EvalTask,
    PeriodicEvaluator,
    discover_eval_tasks,
    should_evaluate,
)
from algorl.core.learner import Learner
from algorl.core.planner import BatchedPlanner, Planner
from algorl.core.replay_buffer import ReplayBuffer
from algorl.core.types import Action, Observation
from algorl.envs.training_env import TrainingEnv


class _FakeEvalEnv:
    def __init__(
        self,
        *,
        num_envs: int = EVAL_EPISODES,
        episode_len: int = 2,
        reward: float = 1.0,
        success: float | None = None,
    ) -> None:
        self.num_envs = num_envs
        self.episode_len = episode_len
        self.reward = reward
        self.success = success
        self.reset_calls = 0
        self.step_calls = 0
        self.action_widths: list[int] = []
        self._t = 0

    def reset(self, *, seed: int | None = None) -> np.ndarray:
        del seed
        self.reset_calls += 1
        self._t = 0
        return np.zeros((self.num_envs, 3), dtype=np.float32)

    def step(self, actions: np.ndarray):
        self.step_calls += 1
        width = int(np.asarray(actions).shape[0])
        self.action_widths.append(width)
        self._t += 1
        rewards = np.full((self.num_envs,), self.reward, dtype=np.float32)
        dones = np.full((self.num_envs,), self._t >= self.episode_len)
        infos: list[dict[str, object]] = [{} for _ in range(self.num_envs)]
        if self.success is not None:
            for info in infos:
                info["success"] = float(self.success)
        obs = np.zeros((self.num_envs, 3), dtype=np.float32)
        return obs, rewards, dones, infos

    def close(self) -> None:
        return


class _FakePlanner(BatchedPlanner):
    def __init__(self) -> None:
        self.search_batch_calls = 0
        self.search_batch_sizes: list[int] = []
        self.kwargs: list[dict[str, object]] = []
        self.last_result = None

    def search(self, observation: Observation, **kwargs) -> Action:
        del observation, kwargs
        return np.zeros((1,), dtype=np.float32)

    def search_batch(self, observations, **kwargs):
        self.search_batch_calls += 1
        self.search_batch_sizes.append(len(observations))
        self.kwargs.append(dict(kwargs))
        batch_size = len(observations)
        result = type(
            "Result",
            (),
            {"actions": np.zeros((batch_size, 1), dtype=np.float32)},
        )()
        self.last_result = result
        return result


class _BoundaryLearner(Learner):
    def __init__(self) -> None:
        self.task_id = 0
        self.materialized: list[int] = []

    def train_step(self, replay_buffer: ReplayBuffer) -> dict[str, float]:
        del replay_buffer
        return {}

    def on_task_boundary(self, new_task_id: int) -> None:
        raise AssertionError(f"eval must not call on_task_boundary({new_task_id})")

    def materialize_task(self, task_id: int):
        self.materialized.append(int(task_id))
        return {"task": int(task_id)}


class _NoOpLearner(Learner):
    def train_step(self, replay_buffer: ReplayBuffer) -> dict[str, float]:
        del replay_buffer
        return {}


def _placeholder_train_env() -> TrainingEnv:
    return TrainingEnv(
        observation_space=gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        action_space=gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        raw=object(),
    )


def test_should_evaluate_skips_before_period_and_step_zero() -> None:
    kwargs = dict(
        period=100,
        last_eval_bucket=0,
        last_eval_step=0,
        total_timesteps=1000,
    )
    assert should_evaluate(0, **kwargs) is False
    assert should_evaluate(99, **kwargs) is False
    assert should_evaluate(100, **kwargs) is True
    assert should_evaluate(150, last_eval_bucket=1, last_eval_step=100, period=100, total_timesteps=1000) is False
    assert should_evaluate(200, last_eval_bucket=1, last_eval_step=100, period=100, total_timesteps=1000) is True
    assert should_evaluate(
        250,
        period=100,
        last_eval_bucket=2,
        last_eval_step=200,
        total_timesteps=250,
        at_end=True,
    ) is True
    assert should_evaluate(
        80,
        period=100,
        last_eval_bucket=0,
        last_eval_step=0,
        total_timesteps=80,
        at_end=True,
    ) is False


def test_discover_eval_tasks_from_cw_train_env() -> None:
    class _CWRaw:
        benchmark_name = "CW10"
        num_tasks = 2
        task_names = ("hammer-v3", "push-v3")

    env = TrainingEnv(
        observation_space=gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        action_space=gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        raw=_CWRaw(),
    )
    tasks = discover_eval_tasks(env)
    assert [task.name for task in tasks] == ["hammer-v3", "push-v3"]
    assert [task.task_index for task in tasks] == [0, 1]
    assert all(task.kind == "cw" for task in tasks)


def test_discover_eval_tasks_from_gymnasium_env() -> None:
    env = TrainingEnv.from_gymnasium(gym.make("CartPole-v1"))
    tasks = discover_eval_tasks(env)
    assert len(tasks) == 1
    assert tasks[0].kind == "gym"
    assert tasks[0].env_id == "CartPole-v1"
    assert tasks[0].max_steps == 500


def test_periodic_evaluator_runs_three_parallel_episodes() -> None:
    fake_env = _FakeEvalEnv(reward=1.5, episode_len=2, success=1.0)
    planner = _FakePlanner()
    logger = Logger()
    evaluator = PeriodicEvaluator(
        period=10,
        train_env=_placeholder_train_env(),
        planner=planner,
        learner=_NoOpLearner(),
        logger=logger,
        seed=0,
        env_factory=lambda task: fake_env,
    )
    metrics = evaluator.maybe_run(10, total_timesteps=40)
    assert metrics is not None
    assert fake_env.reset_calls == 1
    assert fake_env.step_calls == 2
    assert fake_env.action_widths == [EVAL_EPISODES, EVAL_EPISODES]
    assert planner.search_batch_sizes == [EVAL_EPISODES, EVAL_EPISODES]
    assert metrics["eval/mean_return"] == 3.0
    assert metrics["eval/mean_success"] == 1.0
    assert logger.history[-1]["step"] == 9
    assert logger.history[-1]["eval/mean_return"] == 3.0


def test_periodic_evaluator_period_none_path_is_not_constructed_by_default() -> None:
    assert should_evaluate(100, period=0, last_eval_bucket=0, last_eval_step=0, total_timesteps=100) is False


def test_periodic_evaluator_evaluates_each_cl_task_and_skips_task_boundary() -> None:
    created: list[EvalTask] = []
    envs: list[_FakeEvalEnv] = []

    def factory(task: EvalTask) -> _FakeEvalEnv:
        created.append(task)
        env = _FakeEvalEnv(reward=float(task.task_index or 0) + 1.0, episode_len=1)
        envs.append(env)
        return env

    planner = _FakePlanner()
    learner = _BoundaryLearner()
    evaluator = PeriodicEvaluator(
        period=5,
        train_env=_placeholder_train_env(),
        planner=planner,
        learner=learner,
        logger=Logger(),
        seed=0,
        env_factory=factory,
        tasks=(
            EvalTask(name="hammer-v3", task_index=0, kind="cw", max_steps=4),
            EvalTask(name="push-v3", task_index=1, kind="cw", max_steps=4),
        ),
    )
    marker = object()
    planner.last_result = marker
    metrics = evaluator.maybe_run(5, total_timesteps=20)
    assert metrics is not None
    assert [task.name for task in created] == ["hammer-v3", "push-v3"]
    assert learner.materialized == [0, 1]
    assert learner.task_id == 0
    assert planner.last_result is marker
    assert metrics["eval/task/hammer-v3/mean_return"] == 1.0
    assert metrics["eval/task/push-v3/mean_return"] == 2.0
    assert metrics["eval/mean_return"] == 1.5
    assert all(width == EVAL_EPISODES for env in envs for width in env.action_widths)
    assert any(kwargs.get("deterministic") is True for kwargs in planner.kwargs)
    assert any(kwargs.get("params") == {"task": 0} for kwargs in planner.kwargs)


def test_periodic_evaluator_pulses_progress_bar() -> None:
    class _Bar:
        def __init__(self) -> None:
            self.pulses: list[dict[str, object]] = []

        def pulse(self, step_info=None) -> None:
            if step_info is not None:
                self.pulses.append(dict(step_info))

    bar = _Bar()
    evaluator = PeriodicEvaluator(
        period=3,
        train_env=_placeholder_train_env(),
        planner=_FakePlanner(),
        learner=_NoOpLearner(),
        logger=Logger(),
        seed=0,
        env_factory=lambda task: _FakeEvalEnv(episode_len=1),
    )
    evaluator.maybe_run(3, total_timesteps=9, progress_bar=bar)
    phases = [pulse.get("phase") for pulse in bar.pulses]
    assert "eval" in phases
    assert any("eval_step" in pulse for pulse in bar.pulses)


class _SequentialPlanner(Planner):
    def __init__(self) -> None:
        self.calls = 0
        self.deterministics: list[bool | None] = []

    def search(self, observation: Observation, **kwargs) -> Action:
        del observation
        self.calls += 1
        self.deterministics.append(kwargs.get("deterministic"))
        return 0


def test_periodic_evaluator_uses_sequential_planner_across_parallel_lanes() -> None:
    fake_env = _FakeEvalEnv(episode_len=1)
    planner = _SequentialPlanner()
    evaluator = PeriodicEvaluator(
        period=1,
        train_env=_placeholder_train_env(),
        planner=planner,
        learner=_NoOpLearner(),
        logger=Logger(),
        seed=0,
        env_factory=lambda task: fake_env,
    )
    evaluator.maybe_run(1, total_timesteps=1)
    assert planner.calls == EVAL_EPISODES
    assert planner.deterministics == [True] * EVAL_EPISODES
    assert fake_env.action_widths == [EVAL_EPISODES]

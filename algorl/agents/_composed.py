"""Shared composed-agent base class."""

from __future__ import annotations

from typing import Any, ClassVar

from algorl.agents._base import Agent
from algorl.agents._compose import compose_agent
from algorl.agents.configs import BaseAgentConfig, EfficientZeroConfig
from algorl.common.checkpoints import (
    load_run_checkpoint,
    save_run_checkpoint,
    split_resume_overrides,
    write_resume_overrides,
)
from algorl.core.training_loop import TrainingLoop
from algorl.core.types import Action, Observation
from algorl.envs.training_env import TrainingEnv


class ComposedAgent(Agent):
    """Agent that wires world model, planner, learner, and buffer from a composition."""

    composition_name: ClassVar[str]
    config_class: ClassVar[type[BaseAgentConfig]]

    def __init__(self, env: TrainingEnv | object, config: BaseAgentConfig | None = None) -> None:
        self.config = config or self.config_class()
        super().__init__(env, config=self.config)
        components = compose_agent(
            self.composition_name,
            env=self.env,
            config=self.config,
        )
        self.backend = components.backend
        self.world_model = components.world_model
        self.planner = components.planner
        self.learner = components.learner
        self.replay_buffer = components.replay_buffer

    def learn(self, total_timesteps: int, **kwargs: Any) -> None:
        callbacks = kwargs.pop("callbacks", None)
        logger = kwargs.pop("logger", None)
        checkpoint_path = kwargs.pop("checkpoint_path", None)
        checkpoint_dir = kwargs.pop("checkpoint_dir", None)
        resume_from = kwargs.pop("resume_from", None)
        config_overrides = kwargs.pop("config_overrides", None)
        tensorboard = bool(kwargs.pop("tensorboard", False))
        tensorboard_log_dir = kwargs.pop("tensorboard_log_dir", None)
        progress_bar = kwargs.pop("progress_bar", None)
        progress_bar_kwargs = kwargs.pop("progress_bar_kwargs", None)

        if config_overrides:
            allowed, rejected = split_resume_overrides(dict(config_overrides))
            if rejected:
                raise ValueError(
                    "Unsupported or structural config_overrides on resume: "
                    + ", ".join(sorted(rejected))
                )
            self.config = self.config.with_overrides(**allowed)
            self._sync_component_configs(self.config)

        if logger is None and (tensorboard or tensorboard_log_dir is not None):
            log_dir = tensorboard_log_dir or f"runs/{self.composition_name}"
            from algorl.common.tensorboard_logger import TensorboardLogger

            logger = TensorboardLogger(log_dir)

        if progress_bar is True:
            from algorl.common.progress_bar import TqdmProgressBar

            bar_kwargs = progress_bar_kwargs or {}
            progress_bar = TqdmProgressBar(**bar_kwargs)
        elif progress_bar is False:
            progress_bar = None

        run_config = self._resolve_run_config(total_timesteps)
        if checkpoint_dir is None:
            checkpoint_dir = getattr(run_config, "checkpoint_dir", None)
        if checkpoint_dir is not None:
            run_config = run_config.with_overrides(checkpoint_dir=checkpoint_dir)
            self.config = run_config
            self._sync_component_configs(run_config)
            if config_overrides:
                allowed, _ = split_resume_overrides(dict(config_overrides))
                write_resume_overrides(checkpoint_dir, allowed)

        loop = TrainingLoop(
            env=self.env,
            planner=self.planner,
            learner=self.learner,
            replay_buffer=self.replay_buffer,
            config=run_config,
            callbacks=callbacks,
            logger=logger,
            agent_name=self.composition_name,
        )
        start_step = 0
        if resume_from is not None:
            payload = loop.restore_run_checkpoint(resume_from)
            start_step = int(payload["loop"]["step"]) + 1
            # Rebuild optimizers if allowlisted LR/clip knobs changed.
            if config_overrides and hasattr(self.learner, "_recompile_train_kernels"):
                if hasattr(self.learner, "_optimizer"):
                    rebuild = getattr(self.learner, "_build_task_optimizer", None)
                    if callable(rebuild):
                        self.learner._optimizer = rebuild()
                self.learner._recompile_train_kernels()

        loop.run(
            total_timesteps,
            checkpoint_path=checkpoint_path,
            checkpoint_dir=checkpoint_dir,
            start_step=start_step,
            extra_step_info=kwargs or None,
            progress_bar=progress_bar,
        )

    def save(self, path: str) -> None:
        """Persist a full multi-file run checkpoint to ``path`` (directory)."""
        task_id = getattr(self.learner, "task_id", None)
        step = int(getattr(self.learner, "_train_steps", 0))
        save_run_checkpoint(
            directory=path,
            agent_name=self.composition_name,
            step=step,
            config=self.config,
            learner=self.learner,
            replay_buffer=self.replay_buffer,
            env=self.env,
            loop_state={
                "step": step,
                "min_train_step": 0,
                "last_memory_log_step": -1,
                "task_id": None if task_id is None else int(task_id),
                "best_score": float("-inf"),
                "recent_returns": [],
                "rng_key": [self.config.seed, 0],
            },
            tag="manual",
        )

    def load(self, path: str) -> None:
        """Restore learner/buffer/env from a multi-file checkpoint directory."""
        load_run_checkpoint(
            directory=path,
            learner=self.learner,
            replay_buffer=self.replay_buffer,
            env=self.env,
        )

    def predict(
        self,
        observation: Observation,
        deterministic: bool = True,
    ) -> Action:
        return self.planner.search(observation, deterministic=deterministic)

    def _resolve_run_config(self, total_timesteps: int) -> BaseAgentConfig:
        """Resolve per-run schedule horizons and sync live components."""
        if not isinstance(self.config, EfficientZeroConfig):
            return self.config

        run_config = self.config.with_schedule_for_run(
            total_timesteps,
            num_envs=self.env.num_envs if self.env.is_batched else 1,
        )
        if run_config is self.config:
            return run_config

        self.config = run_config
        self._sync_component_configs(run_config)
        return run_config

    def _sync_component_configs(self, config: BaseAgentConfig) -> None:
        if hasattr(self.learner, "config"):
            self.learner.config = config  # type: ignore[assignment]
        if hasattr(self.replay_buffer, "config"):
            self.replay_buffer.config = config  # type: ignore[assignment]
        if hasattr(self.planner, "config"):
            self.planner.config = config  # type: ignore[assignment]

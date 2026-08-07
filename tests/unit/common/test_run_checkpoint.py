"""Integration tests for multi-file run checkpoints + resume overrides."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from algorl.agents.configs import EfficientZeroConfig, HyperCEZConfig
from algorl.agents.search.efficient_zero import EfficientZero
from algorl.agents.search.hyper_cez import HyperCEZ
from algorl.common.checkpoints import (
    read_manifest,
    split_resume_overrides,
)
from algorl.core.training_loop import TrainingLoop
from algorl.envs.training_env import TrainingEnv


def test_split_resume_overrides_allowlist() -> None:
    allowed, rejected = split_resume_overrides(
        {"beta": 2.0, "hnet_arch": (64, 64), "learning_rate": 1e-4}
    )
    assert allowed == {"beta": 2.0, "learning_rate": 1e-4}
    assert "hnet_arch" in rejected


def test_agent_save_load_roundtrip(tmp_path: Path) -> None:
    env = TrainingEnv.from_gymnasium(gym.make("CartPole-v1"))
    config = EfficientZeroConfig.for_dmc_state(
        batch_size=2,
        unroll_steps=2,
        trajectory_size=4,
        learning_starts=0,
        mcts_simulations=2,
        reanalyze_ratio=0.0,
        use_priority=False,
        burst_compile_steps=1,
        buffer_capacity=200,
    )
    agent = EfficientZero(env, config=config)
    # Seed a few transitions so buffer is non-empty after save/load identity.
    from algorl.buffers.efficientzero import (
        BEST_ACTION_INFO_KEY,
        POLICY_TARGET_INFO_KEY,
        ROOT_CANDIDATES_INFO_KEY,
        SEARCH_VALUE_INFO_KEY,
    )
    from algorl.core.types import Transition

    for index in range(12):
        policy = np.full((4,), 0.25, dtype=np.float32)
        candidates = np.stack(
            [np.asarray([0.1], dtype=np.float32) for _ in range(4)],
            axis=0,
        )
        agent.replay_buffer.add(
            Transition(
                observation=np.full((4,), float(index), dtype=np.float32),
                action=np.asarray([0.1], dtype=np.float32),
                reward=0.1,
                next_observation=np.full((4,), float(index + 1), dtype=np.float32),
                done=index % 4 == 3,
                info={
                    POLICY_TARGET_INFO_KEY: policy,
                    SEARCH_VALUE_INFO_KEY: 0.05,
                    ROOT_CANDIDATES_INFO_KEY: candidates,
                    BEST_ACTION_INFO_KEY: np.asarray([0.1], dtype=np.float32),
                },
            )
        )
    agent.learner.train_step(agent.replay_buffer, skip_reanalyze=True)
    steps = agent.learner._train_steps
    ckpt = tmp_path / "manual_ckpt"
    agent.save(str(ckpt))
    assert (ckpt / "manifest.json").is_file()
    assert (ckpt / "learner").is_dir()
    assert (ckpt / "buffer").is_dir()

    agent2 = EfficientZero(env, config=config)
    agent2.load(str(ckpt))
    assert agent2.learner._train_steps == steps
    assert len(agent2.replay_buffer) == len(agent.replay_buffer)


def test_training_loop_autosave_best(tmp_path: Path) -> None:
    env = TrainingEnv.from_gymnasium(gym.make("CartPole-v1"))
    config = EfficientZeroConfig.for_dmc_state(
        batch_size=2,
        unroll_steps=2,
        trajectory_size=4,
        learning_starts=10_000,
        mcts_simulations=2,
        reanalyze_ratio=0.0,
        use_priority=False,
        buffer_capacity=50,
        checkpoint_dir=str(tmp_path / "ckpts"),
        autosave_best=True,
        autosave_best_window=1,
        autosave_best_min_step=0,
    )
    agent = EfficientZero(env, config=config)
    loop = TrainingLoop(
        env=agent.env,
        planner=agent.planner,
        learner=agent.learner,
        replay_buffer=agent.replay_buffer,
        config=config,
        agent_name="efficient_zero",
    )
    loop._maybe_autosave_best(10, 1.0, {})
    loop._maybe_autosave_best(20, 5.0, {})
    loop._maybe_autosave_best(30, 3.0, {})
    best = tmp_path / "ckpts" / "best"
    assert best.is_dir()
    manifest = read_manifest(best)
    assert manifest["tag"] == "best"
    score = (tmp_path / "ckpts" / "best_score.json").read_text()
    assert "5.0" in score or "5" in score


def test_learn_rejects_structural_overrides() -> None:
    env = TrainingEnv.from_gymnasium(gym.make("CartPole-v1"))
    config = HyperCEZConfig.for_dmc_state(
        batch_size=2,
        unroll_steps=2,
        trajectory_size=4,
        num_tasks=2,
        emb_size=8,
        hnet_arch=(16, 16),
        learning_starts=10_000,
        mcts_simulations=2,
        reanalyze_ratio=0.0,
        head_init_std=1e-3,
    )
    agent = HyperCEZ(env, config=config)
    with pytest.raises(ValueError, match="hnet_arch"):
        agent.learn(total_timesteps=1, config_overrides={"hnet_arch": (8, 8)})

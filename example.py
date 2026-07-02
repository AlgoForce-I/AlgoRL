"""Minimal AlgoRL usage example.

Run from the project root::

    python example.py

Install the package and JAX backend first::

    pip install -e ".[jax]"
"""

from __future__ import annotations

import gymnasium as gym

import algorl as arl
from algorl.agents.configs import EfficientZeroConfig
from MTCWorldMJX import CWConfig, make_cl_train_env


def main() -> None:
    cw_config = CWConfig(seed=42, steps_per_task=1_000_000)
    env = make_cl_train_env("CW10", config=cw_config)

    ez_config = EfficientZeroConfig.for_dmc_state(
        seed=0,
        buffer_capacity=10_000,
        learning_starts=1_000,
    )

    agent = arl.EfficientZero(env, config=ez_config)

    # The agent composes its world model, planner, learner, and replay buffer.
    print(f"AlgoRL {arl.__version__}")
    print(f"Environment: {env}")
    print(f"Backend: {agent.backend.name}")
    print(f"World model: {type(agent.world_model).__name__}")
    print(f"Planner: {type(agent.planner).__name__}")
    print(f"Learner: {type(agent.learner).__name__}")
    print(f"Replay buffer: {type(agent.replay_buffer).__name__}")

    agent.learn(total_timesteps=10_000_000, tensorboard_log_dir="runs/cw10_ez")


if __name__ == "__main__":
    main()

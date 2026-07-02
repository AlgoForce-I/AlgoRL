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
    config = CWConfig(seed=42, steps_per_task=1_000_000)
    env = make_cl_train_env("CW10", config=config)

    config = EfficientZeroConfig(
        backend="jax",
        seed=0,
        buffer_capacity=10_000,
        learning_starts=1_000,
        require_implemented=False,  # allow stub components during development
    )

    agent = arl.EfficientZero(env, config=config)

    # The agent composes its world model, planner, learner, and replay buffer.
    print(f"AlgoRL {arl.__version__}")
    print(f"Environment: {env}")
    print(f"Backend: {agent.backend.name}")
    print(f"World model: {type(agent.world_model).__name__}")
    print(f"Planner: {type(agent.planner).__name__}")
    print(f"Learner: {type(agent.learner).__name__}")
    print(f"Replay buffer: {type(agent.replay_buffer).__name__}")

    agent.learn(total_timesteps=10_000_000)

    # Typical usage once implementations are complete:
    #
    #   observation, _ = env.reset()
    #   action = agent.predict(observation)
    #   agent.learn(total_timesteps=10_000)
    #
    # Other agents follow the same pattern, e.g.:
    #
    #   from algorl.agents.configs import DreamerV3Config
    #   agent = arl.DreamerV3(env, config=DreamerV3Config(require_implemented=False))



if __name__ == "__main__":
    main()

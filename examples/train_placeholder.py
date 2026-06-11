"""Example training entrypoint placeholder."""

from __future__ import annotations

import gymnasium as gym

import algorl as arl


def main() -> None:
    env = gym.make("CartPole-v1")
    agent = arl.EfficientZero(env, backend="jax")
    print(f"AlgoRL {arl.__version__} on {env.spec.id if env.spec else 'unknown env'}")
    print("Call agent.learn() once training is implemented.")
    env.close()


if __name__ == "__main__":
    main()

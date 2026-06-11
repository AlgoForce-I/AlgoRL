# AlgoRL Project Context

## Vision

AlgoRL aims to become the model-based reinforcement learning counterpart to Stable-Baselines3.

Stable-Baselines3 solved the model-free RL ecosystem by providing:

- Consistent APIs
- Reliable implementations
- Strong documentation
- Reproducibility
- Easy experimentation

AlgoRL aims to provide the same experience for model-based and planning-based reinforcement learning.

The goal is not to chase every new paper immediately.

The goal is to become the trusted engineering platform for world-model RL research.

---

## Scope

### World-Model Algorithms

- DreamerV2
- DreamerV3
- PlaNet
- TD-MPC
- TD-MPC2
- Future world-model methods

### Search-Based Algorithms

- AlphaZero
- MuZero
- EfficientZero
- EfficientZero-V2
- Gumbel MuZero
- Sampled EfficientZero
- Future MCTS-based methods

### Explicit Non-Goals (Initially)

The project is not intended to compete with SB3 for:

- PPO
- SAC
- TD3
- DQN
- A2C

---

## Core Thesis

Dreamer and MuZero are more similar than they appear.

Both:

1. Learn latent dynamics.
2. Learn a world model.
3. Perform reasoning inside latent space.
4. Produce actions based on imagined future trajectories.

They differ primarily in the reasoning mechanism.

### Dreamer

Latent dynamics → imagination rollout → gradient optimization

### MuZero

Latent dynamics → tree search → action selection

### EfficientZero

Latent dynamics → tree search → improved sample efficiency

Therefore AlgoRL should be built around world models rather than around search.

---

## Technical Direction

Primary backend:

- JAX

Primary libraries:

- JAX
- Optax
- Flax NNX
- MCTX
- Gymnasium
- NumPy

---

## Backend Philosophy

Public APIs must not expose JAX-specific concepts.

```python
agent = EfficientZero(...)
```

Backend implementation details should remain internal.

---

## Future Backend Support

- PyTorch backend
- Custom C++ MCTS backend
- Torch-native search backend

Example:

```python
agent = EfficientZero(backend="jax")
```

```python
agent = EfficientZero(backend="torch")
```

---

## High-Level Architecture

```text
Agent
├── World Model
├── Planner
├── Replay Buffer
├── Learner
├── Environment Interface
└── Backend
```

---

## World Model Interface

```python
class WorldModel:
    def encode(...)
    def transition(...)
    def reward(...)
    def value(...)
```

Implementations:

- RSSM (Dreamer)
- MuZero latent dynamics
- EfficientZero latent dynamics
- Future transformer world models

---

## Planner Interface

```python
class Planner:
    def search(...)
```

Implementations:

- MCTS
- Gumbel MCTS
- Sampled MCTS
- CEM
- MPC
- Future planners

---

## Agent Interface

```python
agent = DreamerV3(...)
agent.learn()
```

```python
agent = EfficientZero(...)
agent.learn()
```

```python
agent = MuZero(...)
agent.learn()
```

---

## Project Philosophy

Priorities:

1. Correctness
2. Reproducibility
3. Clean APIs
4. Documentation
5. Testing
6. Performance

---

## Long-Term Goal

Create the default open-source framework for:

- World-model RL
- Planning-based RL
- Search-based RL
- Future agent architectures

---

## Naming

Organization: AlgoForce

Library: AlgoRL

```python
import algorl as arl

agent = arl.EfficientZero(...)
agent.learn()
```

Tagline:

> Unified Model-Based Reinforcement Learning.

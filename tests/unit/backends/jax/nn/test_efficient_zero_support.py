"""DMC support encode/decode tests."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from algorl.backends.jax.nn.efficientzero.support import scalar_to_support, vector_to_scalar


@pytest.mark.parametrize(
    ("support_range", "values"),
    [
        ((-299.0, 299.0), jnp.array([-50.0, 0.0, 12.5, 200.0])),
        ((-2.0, 2.0), jnp.array([-1.0, -0.05, 0.0, 0.8, 1.5])),
    ],
)
def test_support_round_trip_is_close(
    support_range: tuple[float, float],
    values: jnp.ndarray,
) -> None:
  support_bins = 51
  target = scalar_to_support(
      values,
      support_range=support_range,
      support_bins=support_bins,
  )
  logits = jnp.log(target + 1e-8)
  decoded = vector_to_scalar(
      logits,
      support_type="support",
      support_bins=support_bins,
      support_range=support_range,
  )
  np.testing.assert_allclose(np.asarray(decoded), np.asarray(values), rtol=0.05, atol=0.15)


def test_buffer_priority_zeroing_matches_hypercez_window() -> None:
    from algorl.agents.configs import EfficientZeroConfig
    from algorl.buffers.efficientzero import EfficientZeroReplayBuffer
    from algorl.core.types import Transition

    def transition(index: int, *, done: bool = False) -> Transition:
        policy = np.full((4,), 0.25, dtype=np.float32)
        return Transition(
            observation=np.full((3,), float(index), dtype=np.float32),
            action=np.asarray([0.25], dtype=np.float32),
            reward=float(index),
            next_observation=np.full((3,), float(index + 1), dtype=np.float32),
            done=done,
            info={},
        )

    config = EfficientZeroConfig(
        unroll_steps=1,
        trajectory_size=2,
        use_priority=True,
        top_transitions=3.0,
    )
    buffer = EfficientZeroReplayBuffer(
        capacity=100,
        config=config,
        unroll_steps=1,
        trajectory_size=2,
    )
    for index in range(8):
        buffer.add(transition(index, done=index % 2 == 1))

    assert len(buffer) >= 4
    buffer._sample_indices(1, beta=1.0)
    assert sum(1 for priority in buffer._priorities if priority == 0.0) >= max(
        0, len(buffer._priorities) - int(config.top_transitions)
    )

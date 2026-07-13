"""EfficientZero-V2 config preset tests."""

from algorl.agents.configs import EfficientZeroConfig


def test_for_sequential_defaults() -> None:
    config = EfficientZeroConfig.for_sequential()
    assert config.search_batch_size == 1
    assert config.jax_rollout_chunk == 10
    assert config.batch_size == 256
    assert config.reanalyze_ratio == 1.0
    assert config.reanalyze_search_batch_size == 10_240
    assert config.seed == 42
    assert config.learning_starts == 2_000
    assert config.gradient_steps_per_rollout == 1


def test_for_batched_defaults() -> None:
    config = EfficientZeroConfig.for_batched(num_envs=32)
    assert config.search_batch_size == 32
    assert config.jax_rollout_chunk == 10
    assert config.batch_size == 256
    assert config.reanalyze_ratio == 1.0
    assert config.reanalyze_search_batch_size == 10_240
    assert config.seed == 42
    assert config.buffer_capacity == 100_000
    assert config.learning_starts == 2_000
    assert config.gradient_steps_per_rollout == 32 * 10
    assert config.burst_compile_steps is None


def test_legacy_aliases_match_new_presets() -> None:
    sequential = EfficientZeroConfig.for_sequential()
    assert EfficientZeroConfig.for_dmc_state_sequential_gpu() == sequential

    batched = EfficientZeroConfig.for_batched(num_envs=8)
    assert EfficientZeroConfig.for_dmc_state_batched_cl_gpu(num_envs=8) == batched

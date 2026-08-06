"""Tests for multi-file pytree checkpoint store + manifest."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from algorl.common.checkpoints import (
    SCHEMA_VERSION,
    build_manifest,
    commit_checkpoint,
    load_pytree,
    load_pytree_as_jax,
    open_checkpoint_write,
    read_manifest,
    save_pytree,
    write_json,
    write_manifest,
)


def test_pytree_roundtrip_nested_dict_and_int_keys(tmp_path: Path) -> None:
    tree = {
        "hnets": {"kernel": np.arange(6, dtype=np.float32).reshape(2, 3)},
        "alphas": {
            0: {"dynamics_model": np.asarray(1.5, dtype=np.float32)},
            1: {"dynamics_model": np.asarray(2.5, dtype=np.float32)},
        },
        "flag": True,
        "name": "task0",
        "none_leaf": None,
        "tuple_leaf": (1, 2.0, "x"),
        "list_leaf": [np.ones((2,), dtype=np.float32), 3],
    }
    save_pytree(tmp_path / "learner", tree)
    loaded = load_pytree(tmp_path / "learner")

    assert loaded["flag"] is True
    assert loaded["name"] == "task0"
    assert loaded["none_leaf"] is None
    assert loaded["tuple_leaf"] == (1, 2.0, "x")
    assert set(loaded["alphas"]) == {0, 1}
    np.testing.assert_array_equal(loaded["hnets"]["kernel"], tree["hnets"]["kernel"])
    np.testing.assert_allclose(
        loaded["alphas"][0]["dynamics_model"],
        tree["alphas"][0]["dynamics_model"],
    )
    np.testing.assert_array_equal(loaded["list_leaf"][0], tree["list_leaf"][0])
    assert loaded["list_leaf"][1] == 3


def test_namedtuple_roundtrip_optax_state(tmp_path: Path) -> None:
    optax = pytest.importorskip("optax")

    state = optax.ScaleByAdamState(
        count=np.asarray(4, dtype=np.int32),
        mu=np.zeros((3,), dtype=np.float32),
        nu=np.ones((3,), dtype=np.float32),
    )
    save_pytree(tmp_path / "opt", {"opt": state})
    loaded = load_pytree(tmp_path / "opt")
    assert type(loaded["opt"]) is type(state)
    np.testing.assert_array_equal(loaded["opt"].mu, state.mu)
    assert int(loaded["opt"].count) == 4


def test_load_pytree_as_jax(tmp_path: Path) -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    tree = {"w": np.arange(4, dtype=np.float32)}
    save_pytree(tmp_path / "jax_tree", tree)
    loaded = load_pytree_as_jax(tmp_path / "jax_tree")
    assert isinstance(loaded["w"], jax.Array)
    np.testing.assert_array_equal(np.asarray(loaded["w"]), tree["w"])
    assert loaded["w"].dtype == jnp.float32


def test_atomic_checkpoint_dir_with_manifest(tmp_path: Path) -> None:
    final = tmp_path / "step_1000_task_0"
    staging = open_checkpoint_write(final)
    write_json(staging / "config.json", {"beta": 0.5, "num_tasks": 10})
    save_pytree(staging / "learner", {"w": np.asarray([1.0, 2.0], dtype=np.float32)})
    manifest = build_manifest(
        agent="hyper_cez",
        step=1000,
        task_id=0,
        artifacts={
            "config": "config.json",
            "learner": "learner",
        },
        tag="boundary_task_0",
    )
    write_manifest(staging, manifest)
    commit_checkpoint(staging, final)

    assert not staging.exists()
    assert final.is_dir()
    loaded_manifest = read_manifest(final)
    assert loaded_manifest["schema_version"] == SCHEMA_VERSION
    assert loaded_manifest["step"] == 1000
    assert loaded_manifest["tag"] == "boundary_task_0"
    loaded = load_pytree(final / "learner")
    np.testing.assert_array_equal(loaded["w"], np.asarray([1.0, 2.0], dtype=np.float32))


def test_commit_replaces_existing_checkpoint(tmp_path: Path) -> None:
    final = tmp_path / "best"
    staging = open_checkpoint_write(final)
    save_pytree(staging / "learner", {"w": np.asarray([1.0], dtype=np.float32)})
    write_manifest(
        staging,
        build_manifest(agent="efficient_zero", step=1, artifacts={"learner": "learner"}),
    )
    commit_checkpoint(staging, final)

    staging2 = open_checkpoint_write(final)
    save_pytree(staging2 / "learner", {"w": np.asarray([9.0], dtype=np.float32)})
    write_manifest(
        staging2,
        build_manifest(agent="efficient_zero", step=2, artifacts={"learner": "learner"}),
    )
    commit_checkpoint(staging2, final)

    assert read_manifest(final)["step"] == 2
    np.testing.assert_array_equal(
        load_pytree(final / "learner")["w"],
        np.asarray([9.0], dtype=np.float32),
    )


def test_manifest_rejects_bad_schema(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="schema_version"):
        write_manifest(
            tmp_path,
            {
                "schema_version": 999,
                "agent": "x",
                "step": 0,
                "task_id": None,
                "created_at": "now",
                "artifacts": {},
            },
        )

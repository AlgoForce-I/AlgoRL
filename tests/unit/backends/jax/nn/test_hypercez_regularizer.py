"""HyperCEZ continual-learning regularizer tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from algorl.backends.jax.nn.hypercez.hyper_model import (
    build_hypernetwork_for_component,
    init_hypernetwork_params,
)
from algorl.backends.jax.nn.hypercez.regularizer import (
    calc_fix_target_reg,
    snapshot_reg_targets,
)


@pytest.fixture
def tiny_hnet():
    component_params = {
        "layer": {"kernel": jnp.ones((4, 2)), "bias": jnp.zeros((2,))},
    }
    module = build_hypernetwork_for_component(
        component_params,
        hidden_dims=(8,),
        emb_size=4,
        num_tasks=3,
    )
    rng = jax.random.PRNGKey(0)
    params = init_hypernetwork_params(module, rng)
    return module, params


def test_snapshot_reg_targets_empty_for_task_zero(tiny_hnet) -> None:
    module, params = tiny_hnet
    targets = snapshot_reg_targets(
        {"comp": params},
        {"comp": module},
        ("comp",),
        task_id=0,
    )
    assert targets["comp"] == []


def test_snapshot_and_zero_reg_at_same_params(tiny_hnet) -> None:
    module, params = tiny_hnet
    modules = {"comp": module}
    hnet_params = {"comp": params}
    targets = snapshot_reg_targets(
        hnet_params,
        modules,
        ("comp",),
        task_id=1,
    )
    assert len(targets["comp"]) == 1
    reg = calc_fix_target_reg(
        params,
        hnet_module=module,
        task_id=1,
        targets=targets["comp"],
    )
    assert np.isclose(float(reg), 0.0, atol=1e-6)


def test_reg_increases_when_theta_moves(tiny_hnet) -> None:
    module, params = tiny_hnet
    targets = snapshot_reg_targets(
        {"comp": params},
        {"comp": module},
        ("comp",),
        task_id=1,
    )
    perturbed = jax.tree.map(
        lambda leaf: leaf + 0.05,
        params,
    )
    reg = calc_fix_target_reg(
        perturbed,
        hnet_module=module,
        task_id=1,
        targets=targets["comp"],
    )
    assert float(reg) > 0.0


def test_lookahead_dtheta_changes_reg(tiny_hnet) -> None:
    module, params = tiny_hnet
    targets = snapshot_reg_targets(
        {"comp": params},
        {"comp": module},
        ("comp",),
        task_id=1,
    )
    dtheta = jax.tree.map(lambda leaf: jnp.full_like(leaf, 0.02), params)
    base = calc_fix_target_reg(
        params,
        hnet_module=module,
        task_id=1,
        targets=targets["comp"],
        dtheta=None,
    )
    with_dt = calc_fix_target_reg(
        params,
        hnet_module=module,
        task_id=1,
        targets=targets["comp"],
        dtheta=dtheta,
    )
    assert float(base) == 0.0
    assert float(with_dt) > 0.0

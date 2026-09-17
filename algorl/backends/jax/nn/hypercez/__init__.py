"""HyperCEZ Flax hypernetwork helpers (task-conditioned weight generation)."""

from algorl.agents.configs import DEFAULT_HYPERCEZ_HNET_COMPONENTS
from algorl.backends.jax.nn.hypercez.hyper_model import (
    ChunkedHyperNetwork,
    HyperNetwork,
    apply_hypernetwork,
    build_hypernetwork_for_component,
    component_outputs_to_tree,
    hypernetwork_layer_inputs,
    init_hypernetwork_params,
    outputs_to_tree,
)
from algorl.backends.jax.nn.hypercez.materialize import (
    materialize_delta_component,
    materialize_ez_params,
)
from algorl.backends.jax.nn.hypercez.nullspace import (
    NullspaceBases,
    build_nullspace_bases,
    free_direction_fraction,
    mask_alpha_updates,
    project_hnet_updates,
)
from algorl.backends.jax.nn.hypercez.regularizer import (
    RegTargets,
    calc_component_reg_loss,
    calc_fix_target_reg,
    reg_scaling_from_ema,
    snapshot_reg_targets,
    update_per_task_reg_ema,
)
from algorl.backends.jax.nn.hypercez.shapes import (
    generated_treedef,
    is_shared_leaf,
    merge_params,
    partition_params,
    path_strings,
    target_shapes,
)

__all__ = [
    "DEFAULT_HYPERCEZ_HNET_COMPONENTS",
    "ChunkedHyperNetwork",
    "HyperNetwork",
    "apply_hypernetwork",
    "build_hypernetwork_for_component",
    "build_nullspace_bases",
    "component_outputs_to_tree",
    "free_direction_fraction",
    "generated_treedef",
    "hypernetwork_layer_inputs",
    "init_hypernetwork_params",
    "is_shared_leaf",
    "mask_alpha_updates",
    "materialize_delta_component",
    "materialize_ez_params",
    "merge_params",
    "NullspaceBases",
    "outputs_to_tree",
    "project_hnet_updates",
    "RegTargets",
    "calc_component_reg_loss",
    "calc_fix_target_reg",
    "partition_params",
    "path_strings",
    "reg_scaling_from_ema",
    "snapshot_reg_targets",
    "target_shapes",
    "update_per_task_reg_ema",
]

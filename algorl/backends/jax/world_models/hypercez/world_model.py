"""HyperCEZ world model: EfficientZero inference with task-conditioned deltas."""

from __future__ import annotations

import copy
from typing import Any

import jax
import jax.numpy as jnp

from algorl.agents.configs import HyperCEZConfig
from algorl.backends.jax.nn.hypercez.hyper_model import (
    apply_hypernetwork,
    build_hypernetwork_for_component,
    component_outputs_to_tree,
    init_hypernetwork_params,
)
from algorl.backends.jax.nn.hypercez.materialize import materialize_ez_params
from algorl.backends.jax.nn.efficientzero.model import Params
from algorl.backends.jax.world_models.efficientzero.world_model import (
    EfficientZeroWorldModel,
)
from algorl.core.component_context import ComponentContext


class HyperCEZWorldModel(EfficientZeroWorldModel):
    """EfficientZero world model whose weights are materialized per task.

    Owns:
    - ``frozen_ez``: init snapshot W0 for hypernet-wrapped components
    - ``live_ez``: trainable shared leaves (LayerNorm, obs-norm) + projections
    - ``hnet_modules`` / ``hnet_params`` / ``alphas``: HyperCEZDelta generators

    ``self.params`` is always a fully materialized EfficientZero params tree so
    :class:`~algorl.backends.jax.planners.efficientzero.planner.EfficientZeroPlanner`
    and the parent inference API keep working unchanged.
    """

    def __init__(self, context: ComponentContext) -> None:
        if not isinstance(context.config, HyperCEZConfig):
            raise TypeError(
                "HyperCEZWorldModel requires HyperCEZConfig, "
                f"got {type(context.config)!r}."
            )
        super().__init__(context)
        self.config: HyperCEZConfig = context.config
        if self.config.hnet_type not in ("unchunked", "chunked"):
            raise ValueError(
                f"hnet_type must be 'unchunked' or 'chunked', "
                f"got {self.config.hnet_type!r}."
            )
        self.task_id = 0
        self.live_ez: Params = copy.deepcopy(self.params)
        self.frozen_ez: Params = copy.deepcopy(self.params)
        self.hnet_modules: dict[str, Any] = {}
        self.hnet_params: dict[str, Any] = {}
        self.alphas: dict[int, dict[str, jnp.ndarray]] = {}

        key = self.backend.random_key(self.config.seed + 17)
        for component_name in self.config.hnet_components:
            if component_name not in self.live_ez:
                raise KeyError(
                    f"hnet_components entry {component_name!r} is not in EfficientZero params"
                )
            module = build_hypernetwork_for_component(
                self.live_ez[component_name],
                hidden_dims=self.config.hnet_arch,
                emb_size=self.config.emb_size,
                num_tasks=self.config.num_tasks,
                emb_init_std=self.config.emb_init_std,
                head_init_std=self.config.head_init_std,
                hnet_type=self.config.hnet_type,  # type: ignore[arg-type]
                chunk_dim=self.config.chunk_dim,
                cemb_size=self.config.cemb_size,
                cemb_init_std=self.config.cemb_init_std,
            )
            key, subkey = jax.random.split(key)
            self.hnet_modules[component_name] = module
            self.hnet_params[component_name] = init_hypernetwork_params(module, subkey)

        # One distinct array per (task, component) so donate_argnums on the
        # train-state pytree does not see aliased leaves.
        for task_id in range(self.config.num_tasks):
            self.alphas[task_id] = {
                component_name: jnp.asarray(
                    self.config.alpha_init, dtype=jnp.float32
                )
                for component_name in self.config.hnet_components
            }

        self.refresh_params()

    def materialize(
        self,
        task_id: int,
        *,
        hnet_params: dict[str, Any] | None = None,
        live_ez: Params | None = None,
        alphas: dict[str, jnp.ndarray] | None = None,
    ) -> Params:
        """Build EfficientZero params for ``task_id`` via HyperCEZDelta."""
        if task_id < 0 or task_id >= self.config.num_tasks:
            raise ValueError(
                f"task_id {task_id} out of range for num_tasks={self.config.num_tasks}"
            )
        hnets = self.hnet_params if hnet_params is None else hnet_params
        live = self.live_ez if live_ez is None else live_ez
        task_alphas = self.alphas[task_id] if alphas is None else alphas

        component_deltas: dict[str, Any] = {}
        for component_name in self.config.hnet_components:
            outputs = apply_hypernetwork(
                self.hnet_modules[component_name],
                hnets[component_name],
                task_id,
            )
            component_deltas[component_name] = component_outputs_to_tree(
                outputs,
                self.frozen_ez[component_name],
            )

        return materialize_ez_params(
            component_deltas=component_deltas,
            frozen_ez=self.frozen_ez,
            live_ez=live,
            alphas=task_alphas,
            hnet_components=self.config.hnet_components,
            alpha_max=self.config.alpha_max,
        )

    def set_task_id(self, task_id: int) -> Params:
        """Select the active task embedding and refresh ``self.params``."""
        self.task_id = int(task_id)
        return self.refresh_params()

    def refresh_params(self) -> Params:
        """Rematerialize ``self.params`` from current live / hnet / alpha state."""
        self.params = self.materialize(self.task_id)
        return self.params


def build_hyper_cez_world_model(context: ComponentContext) -> HyperCEZWorldModel:
    if not isinstance(context.config, HyperCEZConfig):
        if getattr(context.config, "require_implemented", True) is False:
            from algorl.backends.jax.world_models import _StubWorldModel

            return _StubWorldModel("hyper_cez", context)  # type: ignore[return-value]
        raise TypeError(
            "HyperCEZWorldModel requires HyperCEZConfig, "
            f"got {type(context.config)!r}."
        )
    return HyperCEZWorldModel(context)

"""Formal ``tsl`` STCNModel adapter using the shared physical graph resource."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from models.base import DataInfoView, ForecastModel, ModelInput
from resources.graph import build_graph_resource, validate_edge_tensors


_CONFIG_FIELDS = {
    "hidden_size",
    "ff_size",
    "n_layers",
    "temporal_kernel_size",
    "spatial_kernel_size",
    "temporal_convs_layer",
    "spatial_convs_layer",
    "dilation",
    "norm",
    "gated",
    "activation",
    "dropout",
}


class STCN(ForecastModel):
    """Normalize the installed TSL STCN output to ``(B, N, H)``."""

    def __init__(
        self,
        *,
        stcn_model: type,
        num_nodes: int,
        input_dim: int,
        lookback: int,
        horizon: int,
        model_config: dict[str, Any],
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.input_dim = int(input_dim)
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.model_config = dict(model_config)
        validate_edge_tensors(edge_index, edge_weight, num_nodes=self.num_nodes)
        self.register_buffer("edge_index", edge_index.detach().clone(), persistent=True)
        self.register_buffer("edge_weight", edge_weight.detach().clone(), persistent=True)
        self.upstream = stcn_model(
            input_size=self.input_dim,
            exog_size=0,
            output_size=1,
            horizon=self.horizon,
            **self.model_config,
        )

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        if not isinstance(inputs, ModelInput):
            raise TypeError("STCN expects ModelInput")
        if any(
            value is not None
            for value in (
                inputs.time_features,
                inputs.node_features,
                inputs.adjacency,
                inputs.static_features,
            )
        ):
            raise ValueError("STCN receives its shared graph at construction and accepts history x only")
        x = inputs.x
        if x.ndim != 4:
            raise ValueError("STCN expects x with shape (B, L, N, C)")
        batch, steps, nodes, channels = x.shape
        if (steps, nodes, channels) != (self.lookback, self.num_nodes, self.input_dim):
            raise ValueError(
                "unexpected STCN input shape: "
                f"{tuple(x.shape)}; expected (*, {self.lookback}, {self.num_nodes}, {self.input_dim})"
            )
        if not torch.isfinite(x).all():
            raise FloatingPointError("STCN input contains NaN or Inf")
        # The installed STCNModel source returns [batch, horizon, nodes, output].
        upstream_output = self.upstream(x, self.edge_index, self.edge_weight)
        expected = (batch, self.horizon, self.num_nodes, 1)
        if tuple(upstream_output.shape) != expected:
            raise ValueError(
                f"installed STCNModel output must have shape {expected}, got {tuple(upstream_output.shape)}"
            )
        if not torch.isfinite(upstream_output).all():
            raise FloatingPointError("installed STCNModel output contains NaN or Inf")
        output = upstream_output[..., 0].permute(0, 2, 1).contiguous()
        return self.validate_output(output, batch=batch, nodes=nodes, horizon=self.horizon)

    def canonical_model_config(self) -> dict[str, Any]:
        return {
            "num_nodes": self.num_nodes,
            "input_dim": self.input_dim,
            "lookback": self.lookback,
            "horizon": self.horizon,
            **self.model_config,
        }


def _validate_config(model_config: dict[str, Any]) -> None:
    unknown = sorted(set(model_config) - _CONFIG_FIELDS)
    missing = sorted(_CONFIG_FIELDS - set(model_config))
    if unknown:
        raise ValueError(f"STCN model config has unknown field: {unknown[0]}")
    if missing:
        raise ValueError(f"STCN model config is missing field: {missing[0]}")
    for name in (
        "hidden_size",
        "ff_size",
        "n_layers",
        "temporal_kernel_size",
        "spatial_kernel_size",
        "temporal_convs_layer",
        "spatial_convs_layer",
        "dilation",
    ):
        if int(model_config[name]) < 1:
            raise ValueError(f"STCN {name} must be positive")
    if str(model_config["norm"]) not in {"none", "batch", "layer"}:
        raise ValueError("STCN norm must be none, batch or layer")
    if str(model_config["activation"]) not in {"relu", "elu", "gelu", "silu"}:
        raise ValueError("STCN activation must be a supported TSL activation")
    if not isinstance(model_config["gated"], bool):
        raise ValueError("STCN gated must be a boolean")
    if not 0.0 <= float(model_config["dropout"]) < 1.0:
        raise ValueError("STCN dropout must be in [0, 1)")


def build_model(model_config: dict[str, Any], data_info: DataInfoView) -> STCN:
    _validate_config(model_config)
    if not data_info.node_ids or len(data_info.node_ids) != data_info.num_nodes:
        raise ValueError("STCN requires public data node_ids aligned to every graph node")
    if data_info.project_root is None:
        raise ValueError("STCN requires project_root metadata for the shared graph resource")
    try:
        from tsl.nn.models.stgn import STCNModel
    except ImportError as exc:
        raise ImportError(
            "STCN requires the formal tsl.nn.models.stgn.STCNModel in the tsl runtime environment"
        ) from exc
    graph = build_graph_resource(
        data_info.graph_config,
        node_ids=data_info.node_ids,
        num_nodes=data_info.num_nodes,
        project_root=Path(data_info.project_root),
    )
    return STCN(
        stcn_model=STCNModel,
        num_nodes=data_info.num_nodes,
        input_dim=data_info.num_features,
        lookback=data_info.lookback,
        horizon=data_info.max_pred_len,
        model_config=model_config,
        edge_index=graph.edge_index,
        edge_weight=graph.edge_weight,
    )

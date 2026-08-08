"""Formal ``tsl`` DCRNN adapter using the shared physical graph resource."""

from __future__ import annotations

from typing import Any

import torch

from models.base import DataInfoView, ForecastModel, ModelInput
from resources.graph import build_graph_resource, validate_edge_tensors


_CONFIG_FIELDS = {
    "hidden_size",
    "kernel_size",
    "ff_size",
    "n_layers",
    "cache_support",
    "dropout",
    "activation",
}


def _validate_config(model_config: dict[str, Any]) -> None:
    if not isinstance(model_config, dict):
        raise ValueError("DCRNN model config must be a mapping")
    unknown = sorted(set(model_config) - _CONFIG_FIELDS)
    missing = sorted(_CONFIG_FIELDS - set(model_config))
    if unknown:
        raise ValueError(f"DCRNN model config has unknown field: {unknown[0]}")
    if missing:
        raise ValueError(f"DCRNN model config is missing field: {missing[0]}")
    for name in ("hidden_size", "kernel_size", "ff_size", "n_layers"):
        value = model_config[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"DCRNN {name} must be a positive integer")
    if not isinstance(model_config["cache_support"], bool):
        raise ValueError("DCRNN cache_support must be a boolean")
    dropout = model_config["dropout"]
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0.0 <= float(dropout) < 1.0:
        raise ValueError("DCRNN dropout must be in [0, 1)")
    if model_config["activation"] not in {
        "relu",
        "elu",
        "gelu",
        "silu",
        "softplus",
        "tanh",
        "sigmoid",
        "linear",
    }:
        raise ValueError("DCRNN activation is not supported by TSL")


def _validate_node_ids(data_info: DataInfoView) -> None:
    node_ids = tuple(data_info.node_ids)
    if not node_ids or len(node_ids) != int(data_info.num_nodes):
        raise ValueError("DCRNN requires public data node_ids aligned to every graph node")
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("DCRNN node_ids must not contain duplicates")


class DCRNN(ForecastModel):
    """Normalize the installed TSL DCRNN output to ``(B, N, H)``."""

    execution_mode = "full_spatiotemporal"

    def __init__(
        self,
        *,
        upstream_model: type,
        data_info: DataInfoView,
        model_config: dict[str, Any],
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> None:
        super().__init__()
        self.num_nodes = int(data_info.num_nodes)
        self.input_dim = int(data_info.num_features)
        self.lookback = int(data_info.lookback)
        self.horizon = int(data_info.max_pred_len)
        self.model_config = dict(model_config)
        validate_edge_tensors(edge_index, edge_weight, num_nodes=self.num_nodes)
        self.register_buffer("edge_index", edge_index.detach().clone(), persistent=True)
        self.register_buffer("edge_weight", edge_weight.detach().clone(), persistent=True)
        self.upstream = upstream_model(
            input_size=self.input_dim,
            output_size=1,
            horizon=self.horizon,
            exog_size=0,
            hidden_size=int(model_config["hidden_size"]),
            kernel_size=int(model_config["kernel_size"]),
            ff_size=int(model_config["ff_size"]),
            n_layers=int(model_config["n_layers"]),
            cache_support=bool(model_config["cache_support"]),
            dropout=float(model_config["dropout"]),
            activation=str(model_config["activation"]),
        )

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        if not isinstance(inputs, ModelInput):
            raise TypeError("DCRNN expects ModelInput")
        if any(
            value is not None
            for value in (
                inputs.time_features,
                inputs.node_features,
                inputs.adjacency,
                inputs.static_features,
            )
        ):
            raise ValueError("DCRNN receives its shared graph at construction and accepts history x only")
        x = inputs.x
        if x.ndim != 4:
            raise ValueError("DCRNN expects x with shape (B, L, N, C)")
        batch, steps, nodes, channels = x.shape
        expected_input = (self.lookback, self.num_nodes, self.input_dim)
        if (steps, nodes, channels) != expected_input:
            raise ValueError(
                "unexpected DCRNN input shape: "
                f"{tuple(x.shape)}; expected (*, {expected_input[0]}, {expected_input[1]}, {expected_input[2]})"
            )
        if not torch.isfinite(x).all():
            raise FloatingPointError("DCRNN input contains NaN or Inf")
        upstream_output = self.upstream(x, self.edge_index, self.edge_weight)
        expected_output = (batch, self.horizon, self.num_nodes, 1)
        if tuple(upstream_output.shape) != expected_output:
            raise ValueError(
                f"DCRNNModel output must have shape {expected_output}, got {tuple(upstream_output.shape)}"
            )
        if not torch.isfinite(upstream_output).all():
            raise FloatingPointError("DCRNNModel output contains NaN or Inf")
        output = upstream_output[..., 0].permute(0, 2, 1).contiguous()
        return self.validate_output(output, batch=int(batch), nodes=int(nodes), horizon=self.horizon)

    def canonical_model_config(self) -> dict[str, Any]:
        return {
            "num_nodes": self.num_nodes,
            "input_dim": self.input_dim,
            "lookback": self.lookback,
            "horizon": self.horizon,
            **self.model_config,
        }


def build_model(model_config: dict[str, Any], data_info: DataInfoView) -> DCRNN:
    _validate_config(model_config)
    _validate_node_ids(data_info)
    if data_info.project_root is None:
        raise ValueError("DCRNN requires project_root metadata for the shared graph resource")
    try:
        from tsl.nn.models.stgn import DCRNNModel
    except ImportError as exc:
        raise ImportError(
            "DCRNN requires the formal tsl.nn.models.stgn.DCRNNModel in the tsl runtime environment"
        ) from exc
    graph = build_graph_resource(
        data_info.graph_config,
        node_ids=data_info.node_ids,
        num_nodes=data_info.num_nodes,
        project_root=data_info.project_root,
    )
    return DCRNN(
        upstream_model=DCRNNModel,
        data_info=data_info,
        model_config=model_config,
        edge_index=graph.edge_index,
        edge_weight=graph.edge_weight,
    )

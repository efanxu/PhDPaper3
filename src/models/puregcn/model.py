"""Spatial-only PureGCN built from the official TSL GraphConv layer."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from models.base import DataInfoView, ForecastModel, ModelInput
from resources.graph import build_graph_resource, validate_edge_tensors


_CONFIG_FIELDS = {
    "hidden_size",
    "n_layers",
    "norm",
    "root_weight",
    "dropout",
    "activation",
    "cached",
}
_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "linear": nn.Identity,
}


def _validate_config(model_config: dict[str, Any]) -> None:
    if not isinstance(model_config, dict):
        raise ValueError("PureGCN model config must be a mapping")
    unknown = sorted(set(model_config) - _CONFIG_FIELDS)
    missing = sorted(_CONFIG_FIELDS - set(model_config))
    if unknown:
        raise ValueError(f"PureGCN model config has unknown field: {unknown[0]}")
    if missing:
        raise ValueError(f"PureGCN model config is missing field: {missing[0]}")
    for name in ("hidden_size", "n_layers"):
        value = model_config[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"PureGCN {name} must be a positive integer")
    if model_config["n_layers"] != 2:
        raise ValueError("PureGCN n_layers must be exactly 2")
    if model_config["norm"] not in {"mean", "gcn", "asym", "none"}:
        raise ValueError("PureGCN norm is not supported by TSL GraphConv")
    for name in ("root_weight", "cached"):
        if not isinstance(model_config[name], bool):
            raise ValueError(f"PureGCN {name} must be a boolean")
    dropout = model_config["dropout"]
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0.0 <= float(dropout) < 1.0:
        raise ValueError("PureGCN dropout must be in [0, 1)")
    if model_config["activation"] not in _ACTIVATIONS:
        raise ValueError("PureGCN activation is not supported")


def _validate_node_ids(data_info: DataInfoView) -> None:
    node_ids = tuple(data_info.node_ids)
    if not node_ids or len(node_ids) != int(data_info.num_nodes):
        raise ValueError("PureGCN requires public data node_ids aligned to every graph node")
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("PureGCN node_ids must not contain duplicates")


class PureGCN(ForecastModel):
    """Spatial-only GCN over flattened historical node features."""

    uses_public_graph_resource = True
    execution_mode = "full_spatiotemporal"

    def __init__(
        self,
        *,
        graph_conv: type,
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

        flattened_input = self.lookback * self.input_dim
        hidden_size = int(model_config["hidden_size"])
        norm = str(model_config["norm"])
        root_weight = bool(model_config["root_weight"])
        cached = bool(model_config["cached"])
        self.graph_conv1 = graph_conv(
            input_size=flattened_input,
            output_size=hidden_size,
            norm=norm,
            root_weight=root_weight,
            activation=None,
            cached=cached,
        )
        self.graph_conv2 = graph_conv(
            input_size=hidden_size,
            output_size=hidden_size,
            norm=norm,
            root_weight=root_weight,
            activation=None,
            cached=cached,
        )
        self.activation = _ACTIVATIONS[str(model_config["activation"])]()
        self.dropout = nn.Dropout(float(model_config["dropout"]))
        self.readout = nn.Linear(hidden_size, self.horizon)

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        if not isinstance(inputs, ModelInput):
            raise TypeError("PureGCN expects ModelInput")
        if any(
            value is not None
            for value in (
                inputs.time_features,
                inputs.node_features,
                inputs.adjacency,
                inputs.static_features,
            )
        ):
            raise ValueError("PureGCN receives its shared graph at construction and accepts history x only")
        x = inputs.x
        if x.ndim != 4:
            raise ValueError("PureGCN expects x with shape (B, L, N, C)")
        batch, steps, nodes, channels = x.shape
        expected_input = (self.lookback, self.num_nodes, self.input_dim)
        if (steps, nodes, channels) != expected_input:
            raise ValueError(
                "unexpected PureGCN input shape: "
                f"{tuple(x.shape)}; expected (*, {expected_input[0]}, {expected_input[1]}, {expected_input[2]})"
            )
        if not torch.isfinite(x).all():
            raise FloatingPointError("PureGCN input contains NaN or Inf")

        node_history = x.permute(0, 2, 1, 3).contiguous().reshape(
            batch, nodes, steps * channels
        )
        hidden = self.graph_conv1(node_history, self.edge_index, self.edge_weight)
        hidden = self.dropout(self.activation(hidden))
        hidden = self.graph_conv2(hidden, self.edge_index, self.edge_weight)
        hidden = self.dropout(self.activation(hidden))
        output = self.readout(hidden)
        return self.validate_output(output, batch=int(batch), nodes=int(nodes), horizon=self.horizon)

    def canonical_model_config(self) -> dict[str, Any]:
        return {
            "num_nodes": self.num_nodes,
            "input_dim": self.input_dim,
            "lookback": self.lookback,
            "horizon": self.horizon,
            **self.model_config,
        }


def build_model(model_config: dict[str, Any], data_info: DataInfoView) -> PureGCN:
    _validate_config(model_config)
    _validate_node_ids(data_info)
    if data_info.project_root is None:
        raise ValueError("PureGCN requires project_root metadata for the shared graph resource")
    try:
        from tsl.nn.layers.graph_convs import GraphConv
    except ImportError as exc:
        raise ImportError(
            "PureGCN requires the formal tsl.nn.layers.graph_convs.GraphConv in the tsl runtime environment"
        ) from exc
    graph = build_graph_resource(
        data_info.graph_config,
        node_ids=data_info.node_ids,
        num_nodes=data_info.num_nodes,
        project_root=data_info.project_root,
    )
    return PureGCN(
        graph_conv=GraphConv,
        data_info=data_info,
        model_config=model_config,
        edge_index=graph.edge_index,
        edge_weight=graph.edge_weight,
    )

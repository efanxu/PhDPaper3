"""Dense GCN migrated from the legacy PyTorch PureGCN baseline."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from models.base import DataInfoView, ForecastModel, ModelInput
from resources.graph import build_graph_resource, validate_edge_tensors


_CONFIG_FIELDS = {"hidden_dim", "num_layers", "dropout"}


def _validate_config(model_config: dict[str, Any]) -> None:
    if not isinstance(model_config, dict):
        raise ValueError("DenseGCN model config must be a mapping")
    unknown = sorted(set(model_config) - _CONFIG_FIELDS)
    missing = sorted(_CONFIG_FIELDS - set(model_config))
    if unknown:
        raise ValueError(f"DenseGCN model config has unknown field: {unknown[0]}")
    if missing:
        raise ValueError(f"DenseGCN model config is missing field: {missing[0]}")
    for name in ("hidden_dim", "num_layers"):
        value = model_config[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"DenseGCN {name} must be a positive integer")
    dropout = model_config["dropout"]
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0.0 <= float(dropout) < 1.0:
        raise ValueError("DenseGCN dropout must be in [0, 1)")


def _validate_node_ids(data_info: DataInfoView) -> None:
    node_ids = tuple(data_info.node_ids)
    if not node_ids or len(node_ids) != int(data_info.num_nodes):
        raise ValueError("DenseGCN requires public data node_ids aligned to every graph node")
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("DenseGCN node_ids must not contain duplicates")


def dense_adjacency_from_edges(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    *,
    num_nodes: int,
) -> torch.Tensor:
    """Reconstruct the public graph once in its existing deterministic node order."""

    validate_edge_tensors(edge_index, edge_weight, num_nodes=num_nodes)
    adjacency = edge_weight.new_zeros((num_nodes, num_nodes))
    adjacency[edge_index[0], edge_index[1]] = edge_weight
    if not torch.isfinite(adjacency).all():
        raise ValueError("dense adjacency contains NaN or Inf")
    return adjacency


class WeightedGraphConv(nn.Module):
    """Legacy dense propagation followed by its learned linear projection."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        propagated = torch.einsum("ij,bjd->bid", adjacency, x)
        return self.linear(propagated)


class DenseGCN(ForecastModel):
    """Flatten node histories, then forecast with dense residual GCN layers."""

    uses_public_graph_resource = True
    execution_mode = "full_spatiotemporal"

    def __init__(
        self,
        *,
        data_info: DataInfoView,
        model_config: dict[str, Any],
        adjacency: torch.Tensor,
    ) -> None:
        super().__init__()
        self.num_nodes = int(data_info.num_nodes)
        self.input_dim = int(data_info.num_features)
        self.lookback = int(data_info.lookback)
        self.horizon = int(data_info.max_pred_len)
        self.model_config = dict(model_config)
        if tuple(adjacency.shape) != (self.num_nodes, self.num_nodes):
            raise ValueError(
                f"DenseGCN adjacency must have shape ({self.num_nodes}, {self.num_nodes})"
            )
        if not torch.isfinite(adjacency).all():
            raise ValueError("DenseGCN adjacency contains NaN or Inf")
        self.register_buffer("adjacency", adjacency.detach().clone(), persistent=True)

        hidden_dim = int(model_config["hidden_dim"])
        num_layers = int(model_config["num_layers"])
        self.input_proj = nn.Linear(self.lookback * self.input_dim, hidden_dim)
        self.gcn_layers = nn.ModuleList(
            [WeightedGraphConv(hidden_dim) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(float(model_config["dropout"]))
        self.readout = nn.Linear(hidden_dim, self.horizon)

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        if not isinstance(inputs, ModelInput):
            raise TypeError("DenseGCN expects ModelInput")
        if any(
            value is not None
            for value in (
                inputs.time_features,
                inputs.node_features,
                inputs.adjacency,
                inputs.static_features,
            )
        ):
            raise ValueError("DenseGCN receives its shared graph at construction and accepts history x only")
        x = inputs.x
        if x.ndim != 4:
            raise ValueError("DenseGCN expects x with shape (B, L, N, C)")
        batch, steps, nodes, channels = x.shape
        expected_input = (self.lookback, self.num_nodes, self.input_dim)
        if (steps, nodes, channels) != expected_input:
            raise ValueError(
                "unexpected DenseGCN input shape: "
                f"{tuple(x.shape)}; expected (*, {expected_input[0]}, {expected_input[1]}, {expected_input[2]})"
            )
        if not torch.isfinite(x).all():
            raise FloatingPointError("DenseGCN input contains NaN or Inf")

        hidden = x.permute(0, 2, 1, 3).contiguous().reshape(
            batch, nodes, steps * channels
        )
        hidden = self.activation(self.input_proj(hidden))
        adjacency = self.adjacency.to(dtype=hidden.dtype)
        for graph_conv, norm in zip(self.gcn_layers, self.norms):
            update = self.dropout(self.activation(graph_conv(hidden, adjacency)))
            hidden = norm(hidden + update)
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


def build_model(model_config: dict[str, Any], data_info: DataInfoView) -> DenseGCN:
    _validate_config(model_config)
    _validate_node_ids(data_info)
    if data_info.project_root is None:
        raise ValueError("DenseGCN requires project_root metadata for the shared graph resource")
    graph = build_graph_resource(
        data_info.graph_config,
        node_ids=data_info.node_ids,
        num_nodes=data_info.num_nodes,
        project_root=data_info.project_root,
    )
    adjacency = dense_adjacency_from_edges(
        graph.edge_index,
        graph.edge_weight,
        num_nodes=int(data_info.num_nodes),
    )
    return DenseGCN(
        data_info=data_info,
        model_config=model_config,
        adjacency=adjacency,
    )

"""Validated shared-parameter LSTM adapter."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from models.base import DataInfoView, ModelInput, NodeSharedForecastModel


class LSTM(NodeSharedForecastModel):
    def __init__(
        self,
        *,
        num_nodes: int,
        input_dim: int,
        lookback: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        horizon: int,
    ) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.input_dim = int(input_dim)
        self.lookback = int(lookback)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.horizon = int(horizon)
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )
        self.prediction_head = nn.Linear(self.hidden_dim, self.horizon)

    def forward_node_chunk(
        self,
        inputs: ModelInput,
        node_start: int,
        node_end: int,
    ) -> torch.Tensor:
        x = self._node_chunk_x(
            inputs,
            node_start,
            node_end,
            model_name="LSTM",
        )
        batch, steps, nodes, channels = x.shape
        node_history = x.permute(0, 2, 1, 3).contiguous().view(
            batch * nodes, steps, channels
        )
        _, (hidden, _) = self.lstm(node_history)
        output = self.prediction_head(hidden[-1]).view(batch, nodes, self.horizon)
        return self.validate_output(
            output,
            batch=batch,
            nodes=nodes,
            horizon=self.horizon,
        )

    def canonical_model_config(self) -> dict[str, Any]:
        return {
            "num_nodes": self.num_nodes,
            "input_dim": self.input_dim,
            "lookback": self.lookback,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "horizon": self.horizon,
        }


def build_model(model_config: dict[str, Any], data_info: DataInfoView) -> LSTM:
    expected = {"hidden_dim", "num_layers", "dropout"}
    unknown = sorted(set(model_config) - expected)
    missing = sorted(expected - set(model_config))
    if unknown:
        raise ValueError(f"LSTM model config has unknown field: {unknown[0]}")
    if missing:
        raise ValueError(f"LSTM model config is missing field: {missing[0]}")
    return LSTM(
        num_nodes=data_info.num_nodes,
        input_dim=data_info.num_features,
        lookback=data_info.lookback,
        hidden_dim=int(model_config["hidden_dim"]),
        num_layers=int(model_config["num_layers"]),
        dropout=float(model_config["dropout"]),
        horizon=data_info.max_pred_len,
    )

"""The small model boundary shared by every experiment model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class DataInfoView:
    num_nodes: int
    num_features: int
    lookback: int
    max_pred_len: int

    @classmethod
    def from_object(cls, value: Any) -> "DataInfoView":
        if isinstance(value, cls):
            return value
        if hasattr(value, "num_nodes"):
            return cls(
                int(value.num_nodes),
                int(value.num_features),
                int(value.lookback),
                int(value.max_pred_len),
            )
        if isinstance(value, dict):
            return cls(
                int(value["num_nodes"]),
                int(value.get("num_features", value.get("input_dim"))),
                int(value["lookback"]),
                int(value.get("max_pred_len", value.get("horizon"))),
            )
        raise TypeError("data_info must expose num_nodes, num_features, lookback and max_pred_len")


@dataclass(frozen=True)
class ModelInput:
    """Only approved model inputs; labels are intentionally absent."""

    x: torch.Tensor
    time_features: torch.Tensor | None = None
    node_features: torch.Tensor | None = None
    adjacency: torch.Tensor | None = None
    static_features: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> "ModelInput":
        def move(value: torch.Tensor | None) -> torch.Tensor | None:
            return value.to(device) if value is not None else None

        return ModelInput(
            x=self.x.to(device),
            time_features=move(self.time_features),
            node_features=move(self.node_features),
            adjacency=move(self.adjacency),
            static_features=move(self.static_features),
        )


class ForecastModel(nn.Module):
    """Marker base class for models that return ``(B, N, H)`` forecasts."""

    output_layout = "batch_nodes_horizon"

    def validate_output(self, value: torch.Tensor, *, batch: int, nodes: int, horizon: int) -> torch.Tensor:
        if tuple(value.shape) != (batch, nodes, horizon):
            raise ValueError(
                f"model output must have shape {(batch, nodes, horizon)}, got {tuple(value.shape)}"
            )
        if not torch.isfinite(value).all():
            raise FloatingPointError("model output contains NaN or Inf")
        return value

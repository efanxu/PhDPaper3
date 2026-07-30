"""The small model boundary shared by every experiment model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class DataInfoView:
    num_nodes: int
    num_features: int
    lookback: int
    max_pred_len: int
    feature_columns: tuple[str, ...] = ()
    input_power_column: str = ""
    input_power_index: int = -1
    node_ids: tuple[int, ...] = ()
    graph_config: dict[str, Any] | None = None
    project_root: Path | None = None

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
                tuple(getattr(value, "feature_columns", ())),
                str(getattr(value, "input_power_column", "")),
                int(getattr(value, "input_power_index", -1)),
                tuple(int(item) for item in getattr(value, "node_ids", ())),
                dict(getattr(value, "graph_config", {}) or {}) or None,
                Path(getattr(value, "project_root")).resolve()
                if getattr(value, "project_root", None) is not None
                else None,
            )
        if isinstance(value, dict):
            feature_columns = tuple(str(item) for item in value.get("feature_columns", ()))
            input_power_column = str(value.get("input_power_column", ""))
            input_power_index = int(
                value.get(
                    "input_power_index",
                    feature_columns.index(input_power_column)
                    if input_power_column in feature_columns
                    else -1,
                )
            )
            return cls(
                int(value["num_nodes"]),
                int(value.get("num_features", value.get("input_dim"))),
                int(value["lookback"]),
                int(value.get("max_pred_len", value.get("horizon"))),
                feature_columns,
                input_power_column,
                input_power_index,
                tuple(int(item) for item in value.get("node_ids", ())),
                dict(value.get("graph_config", {}) or {}) or None,
                Path(value["project_root"]).resolve() if value.get("project_root") else None,
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

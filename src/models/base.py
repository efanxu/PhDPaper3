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

    uses_public_graph_resource: bool = False
    output_layout = "batch_nodes_horizon"

    def validate_output(self, value: torch.Tensor, *, batch: int, nodes: int, horizon: int) -> torch.Tensor:
        if tuple(value.shape) != (batch, nodes, horizon):
            raise ValueError(
                f"model output must have shape {(batch, nodes, horizon)}, got {tuple(value.shape)}"
            )
        if not torch.isfinite(value).all():
            raise FloatingPointError("model output contains NaN or Inf")
        return value


class NodeSharedForecastModel(ForecastModel):
    """Shared-parameter temporal model with an explicit node-chunk seam.

    A NodeShared model applies one temporal forecasting function independently
    to every node.  The public ``forward`` contract remains the complete
    ``(B, N, H)`` forecast; the shared execution layer may call
    ``forward_node_chunk`` with a contiguous node range when the model is safe
    to micro-batch.  Concrete adapters implement only that one-node-range
    operation and never own the chunk loop or optimizer logic.
    """

    execution_mode = "full_nodes"

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        if not isinstance(inputs, ModelInput):
            raise TypeError(f"{type(self).__name__} expects ModelInput")
        if inputs.x.ndim != 4:
            raise ValueError("NodeSharedForecastModel expects x with shape (B, L, N, C)")
        return self.forward_node_chunk(inputs, 0, int(inputs.x.shape[2]))

    def forward_node_chunk(
        self,
        inputs: ModelInput,
        node_start: int,
        node_end: int,
    ) -> torch.Tensor:
        """Return ``(B, node_end-node_start, H)`` for one contiguous node range."""

        raise NotImplementedError(
            f"{type(self).__name__} must implement forward_node_chunk(inputs, node_start, node_end)"
        )

    def _node_chunk_x(
        self,
        inputs: ModelInput,
        node_start: int,
        node_end: int,
        *,
        model_name: str,
    ) -> torch.Tensor:
        """Validate shared temporal input and return one node slice.

        This helper keeps shape and node-range invariants at the model seam;
        model adapters only contain their original per-node forward call.
        """

        if not isinstance(inputs, ModelInput):
            raise TypeError(f"{model_name} expects ModelInput")
        if any(
            value is not None
            for value in (
                inputs.time_features,
                inputs.node_features,
                inputs.adjacency,
                inputs.static_features,
            )
        ):
            raise ValueError(f"{model_name} accepts history x only")
        x = inputs.x
        if x.ndim != 4:
            raise ValueError(f"{model_name} expects x with shape (B, L, N, C)")
        batch, steps, nodes, channels = x.shape
        expected = (
            getattr(self, "lookback", steps),
            getattr(self, "num_nodes", nodes),
            getattr(self, "input_dim", channels),
        )
        if (steps, nodes, channels) != expected:
            raise ValueError(
                f"unexpected {model_name} input shape: {tuple(x.shape)}; "
                f"expected (*, {expected[0]}, {expected[1]}, {expected[2]})"
            )
        if not 0 <= int(node_start) < int(node_end) <= nodes:
            raise ValueError(
                f"{model_name} node range must satisfy 0 <= start < end <= {nodes}; "
                f"got ({node_start}, {node_end})"
            )
        if not torch.isfinite(x).all():
            raise FloatingPointError(f"{model_name} input contains NaN or Inf")
        del batch
        return x[:, :, int(node_start) : int(node_end), :]

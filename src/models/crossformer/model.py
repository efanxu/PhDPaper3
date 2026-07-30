"""Node-shared adapter around the local upstream Crossformer implementation."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from integrations.time_series_library import load_time_series_library_model_class
from models.base import DataInfoView, ForecastModel, ModelInput


_CONFIG_FIELDS = {"d_model", "n_heads", "d_ff", "e_layers", "dropout", "factor"}


class Crossformer(ForecastModel):
    """Apply one upstream Crossformer instance independently to each node."""

    def __init__(
        self,
        *,
        upstream_model: type,
        num_nodes: int,
        input_dim: int,
        lookback: int,
        horizon: int,
        input_power_index: int,
        model_config: dict[str, Any],
        source_root: Path,
    ) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.input_dim = int(input_dim)
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.input_power_index = int(input_power_index)
        self.model_config = dict(model_config)
        self.source_root = Path(source_root).resolve()
        upstream_config = SimpleNamespace(
            enc_in=self.input_dim,
            seq_len=self.lookback,
            pred_len=self.horizon,
            task_name="long_term_forecast",
            d_model=int(model_config["d_model"]),
            n_heads=int(model_config["n_heads"]),
            d_ff=int(model_config["d_ff"]),
            e_layers=int(model_config["e_layers"]),
            dropout=float(model_config["dropout"]),
            factor=int(model_config["factor"]),
        )
        self.upstream = upstream_model(upstream_config)

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        if not isinstance(inputs, ModelInput):
            raise TypeError("Crossformer expects ModelInput")
        if any(
            value is not None
            for value in (
                inputs.time_features,
                inputs.node_features,
                inputs.adjacency,
                inputs.static_features,
            )
        ):
            raise ValueError("Crossformer accepts history x only")
        x = inputs.x
        if x.ndim != 4:
            raise ValueError("Crossformer expects x with shape (B, L, N, C)")
        batch, steps, nodes, channels = x.shape
        if (steps, nodes, channels) != (self.lookback, self.num_nodes, self.input_dim):
            raise ValueError(
                "unexpected Crossformer input shape: "
                f"{tuple(x.shape)}; expected (*, {self.lookback}, {self.num_nodes}, {self.input_dim})"
            )
        if not torch.isfinite(x).all():
            raise FloatingPointError("Crossformer input contains NaN or Inf")
        node_history = x.permute(0, 2, 1, 3).reshape(batch * nodes, steps, channels)
        upstream_output = self.upstream(node_history, None, None, None)
        expected = (batch * nodes, self.horizon, self.input_dim)
        if tuple(upstream_output.shape) != expected:
            raise ValueError(
                f"upstream Crossformer output must have shape {expected}, got {tuple(upstream_output.shape)}"
            )
        if not torch.isfinite(upstream_output).all():
            raise FloatingPointError("upstream Crossformer output contains NaN or Inf")
        output = upstream_output[..., self.input_power_index].reshape(batch, nodes, self.horizon)
        return self.validate_output(output, batch=batch, nodes=nodes, horizon=self.horizon)

    def canonical_model_config(self) -> dict[str, Any]:
        return {
            "num_nodes": self.num_nodes,
            "input_dim": self.input_dim,
            "lookback": self.lookback,
            "horizon": self.horizon,
            "input_power_index": self.input_power_index,
            **self.model_config,
        }


def _validate_config(model_config: dict[str, Any]) -> None:
    unknown = sorted(set(model_config) - _CONFIG_FIELDS)
    missing = sorted(_CONFIG_FIELDS - set(model_config))
    if unknown:
        raise ValueError(f"Crossformer model config has unknown field: {unknown[0]}")
    if missing:
        raise ValueError(f"Crossformer model config is missing field: {missing[0]}")
    if int(model_config["d_model"]) < 1 or int(model_config["n_heads"]) < 1:
        raise ValueError("Crossformer d_model and n_heads must be positive")
    if int(model_config["d_model"]) % int(model_config["n_heads"]):
        raise ValueError("Crossformer d_model must be divisible by n_heads")
    if int(model_config["d_ff"]) < 1 or int(model_config["e_layers"]) < 1 or int(model_config["factor"]) < 1:
        raise ValueError("Crossformer d_ff, e_layers and factor must be positive")
    if not 0.0 <= float(model_config["dropout"]) < 1.0:
        raise ValueError("Crossformer dropout must be in [0, 1)")


def build_model(model_config: dict[str, Any], data_info: DataInfoView) -> Crossformer:
    _validate_config(model_config)
    if not data_info.feature_columns or not data_info.input_power_column:
        raise ValueError("Crossformer requires data feature_columns and input_power_column metadata")
    if not 0 <= data_info.input_power_index < data_info.num_features:
        raise ValueError("Crossformer requires a valid input_power_index")
    if data_info.feature_columns[data_info.input_power_index] != data_info.input_power_column:
        raise ValueError("Crossformer input_power_index does not match input_power_column")
    project_root = (
        Path(data_info.project_root).resolve()
        if data_info.project_root is not None
        else Path(__file__).resolve().parents[3]
    )
    source_root = project_root / "Time-Series-Library"
    if not source_root.is_dir():
        raise FileNotFoundError(f"Crossformer requires Time-Series-Library source root: {source_root}")
    upstream_class = load_time_series_library_model_class(
        "Crossformer",
        source_root=source_root,
    )
    return Crossformer(
        upstream_model=upstream_class,
        num_nodes=data_info.num_nodes,
        input_dim=data_info.num_features,
        lookback=data_info.lookback,
        horizon=data_info.max_pred_len,
        input_power_index=data_info.input_power_index,
        model_config=model_config,
        source_root=source_root,
    )

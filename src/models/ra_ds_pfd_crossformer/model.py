"""Node-shared RA-DS-PFD P1 adapter around the local canonical backbone."""

from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Any

import torch

from models.base import DataInfoView, ForecastModel, ModelInput

from .backbone import CanonicalBackbone, CanonicalTrace


_CONFIG_FIELDS = {
    "d_model",
    "n_heads",
    "d_ff",
    "e_layers",
    "dropout",
    "factor",
    "seg_len",
    "win_size",
    "spatial_disabled",
}
_POSITIVE_INTEGER_FIELDS = {"d_model", "n_heads", "d_ff", "e_layers", "factor", "seg_len", "win_size"}


class RADSPFDCrossformer(ForecastModel):
    """Apply one local canonical Crossformer independently to every node."""

    def __init__(
        self,
        *,
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
        self.backbone = CanonicalBackbone(
            source_root=self.source_root,
            enc_in=self.input_dim,
            seq_len=self.lookback,
            pred_len=self.horizon,
            model_config=self.model_config,
        )

    def _node_history(self, inputs: ModelInput) -> tuple[torch.Tensor, int, int]:
        if not isinstance(inputs, ModelInput):
            raise TypeError("RA-DS-PFD Crossformer expects ModelInput")
        if any(
            value is not None
            for value in (
                inputs.time_features,
                inputs.node_features,
                inputs.adjacency,
                inputs.static_features,
            )
        ):
            raise ValueError("RA-DS-PFD Crossformer accepts history x only")
        x = inputs.x
        if not isinstance(x, torch.Tensor) or x.ndim != 4:
            raise ValueError("RA-DS-PFD Crossformer expects x with shape (B, L, N, C)")
        batch, steps, nodes, channels = x.shape
        expected = (self.lookback, self.num_nodes, self.input_dim)
        if (steps, nodes, channels) != expected:
            raise ValueError(
                "unexpected RA-DS-PFD Crossformer input shape: "
                f"{tuple(x.shape)}; expected (*, {expected[0]}, {expected[1]}, {expected[2]})"
            )
        if not torch.isfinite(x).all():
            raise FloatingPointError("RA-DS-PFD Crossformer input contains NaN or Inf")
        node_history = x.permute(0, 2, 1, 3).reshape(batch * nodes, steps, channels)
        return node_history, batch, nodes

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        node_history, batch, nodes = self._node_history(inputs)
        full_output = self.backbone(node_history)
        expected = (batch * nodes, self.horizon, self.input_dim)
        if tuple(full_output.shape) != expected:
            raise ValueError(
                "local canonical Crossformer output must have shape "
                f"{expected}, got {tuple(full_output.shape)}"
            )
        if not torch.isfinite(full_output).all():
            raise FloatingPointError("local canonical Crossformer output contains NaN or Inf")
        output = full_output[..., self.input_power_index].reshape(batch, nodes, self.horizon)
        return self.validate_output(output, batch=batch, nodes=nodes, horizon=self.horizon)

    def forward_canonical_trace(self, inputs: ModelInput) -> CanonicalTrace:
        """Return flattened node-wise canonical stages for focused unit tests."""

        node_history, _, _ = self._node_history(inputs)
        trace = self.backbone.forward_backbone(node_history, return_trace=True)
        assert isinstance(trace, CanonicalTrace)
        return trace

    def load_upstream_state_dict(self, upstream_state_dict: dict[str, torch.Tensor]) -> None:
        """Strictly transfer the upstream canonical state to this adapter."""

        self.backbone.load_upstream_state_dict(upstream_state_dict)

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
    if not isinstance(model_config, dict):
        raise TypeError("RA-DS-PFD Crossformer model config must be a mapping")
    unknown = sorted(set(model_config) - _CONFIG_FIELDS)
    missing = sorted(_CONFIG_FIELDS - set(model_config))
    if unknown:
        raise ValueError(f"RA-DS-PFD Crossformer model config has unknown field: {unknown[0]}")
    if missing:
        raise ValueError(f"RA-DS-PFD Crossformer model config is missing field: {missing[0]}")
    for name in sorted(_POSITIVE_INTEGER_FIELDS):
        value = model_config[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"RA-DS-PFD Crossformer {name} must be a positive integer")
    if model_config["d_model"] % model_config["n_heads"]:
        raise ValueError("RA-DS-PFD Crossformer d_model must be divisible by n_heads")
    if model_config["d_model"] % 2:
        raise ValueError("RA-DS-PFD Crossformer d_model must be even")
    dropout = model_config["dropout"]
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not isfinite(float(dropout)):
        raise ValueError("RA-DS-PFD Crossformer dropout must be a finite number")
    if not 0.0 <= float(dropout) < 1.0:
        raise ValueError("RA-DS-PFD Crossformer dropout must be in [0, 1)")
    if model_config["seg_len"] != 12:
        raise ValueError("RA-DS-PFD Crossformer P1 only supports seg_len=12")
    if model_config["win_size"] != 2:
        raise ValueError("RA-DS-PFD Crossformer P1 only supports win_size=2")
    if model_config["e_layers"] != 2:
        raise ValueError("RA-DS-PFD Crossformer P1 only supports e_layers=2")
    if not isinstance(model_config["spatial_disabled"], bool):
        raise ValueError("RA-DS-PFD Crossformer spatial_disabled must be a boolean")
    if not model_config["spatial_disabled"]:
        raise ValueError(
            "RA-DS-PFD Crossformer P1 spatial_disabled=false is fail-closed: "
            "spatial implementation is not available"
        )


def _validate_data_info(data_info: DataInfoView) -> None:
    if data_info.num_nodes < 1 or data_info.num_features < 1:
        raise ValueError("RA-DS-PFD Crossformer requires positive node and feature counts")
    if data_info.lookback < 1 or data_info.max_pred_len < 1:
        raise ValueError("RA-DS-PFD Crossformer requires positive lookback and horizon")
    if len(data_info.feature_columns) != data_info.num_features:
        raise ValueError("RA-DS-PFD Crossformer feature_columns must match num_features")
    if not data_info.feature_columns or not data_info.input_power_column:
        raise ValueError("RA-DS-PFD Crossformer requires feature and input power metadata")
    if not 0 <= data_info.input_power_index < data_info.num_features:
        raise ValueError("RA-DS-PFD Crossformer requires a valid input_power_index")
    if data_info.feature_columns[data_info.input_power_index] != data_info.input_power_column:
        raise ValueError("RA-DS-PFD Crossformer input_power_index does not match input_power_column")


def build_model(model_config: dict[str, Any], data_info: DataInfoView) -> RADSPFDCrossformer:
    _validate_config(model_config)
    _validate_data_info(data_info)
    project_root = (
        Path(data_info.project_root).resolve()
        if data_info.project_root is not None
        else Path(__file__).resolve().parents[3]
    )
    source_root = project_root / "Time-Series-Library"
    if not source_root.is_dir():
        raise FileNotFoundError(
            f"RA-DS-PFD Crossformer requires Time-Series-Library source root: {source_root}"
        )
    return RADSPFDCrossformer(
        num_nodes=data_info.num_nodes,
        input_dim=data_info.num_features,
        lookback=data_info.lookback,
        horizon=data_info.max_pred_len,
        input_power_index=data_info.input_power_index,
        model_config=model_config,
        source_root=source_root,
    )

"""Adapter around the official TorchSpatiotemporal STIDModel."""

from __future__ import annotations

from typing import Any

import torch

from models.base import DataInfoView, ForecastModel, ModelInput


_CONFIG_FIELDS = {"hidden_size", "n_layers", "dropout"}


class STID(ForecastModel):
    """Preserve STID's node identity embedding and full-node input layout."""

    execution_mode = "full_spatiotemporal"

    def __init__(
        self,
        *,
        upstream_model: type,
        data_info: DataInfoView,
        model_config: dict[str, Any],
    ) -> None:
        super().__init__()
        self.num_nodes = int(data_info.num_nodes)
        self.input_dim = int(data_info.num_features)
        self.lookback = int(data_info.lookback)
        self.horizon = int(data_info.max_pred_len)
        self.model_config = dict(model_config)
        self.upstream = upstream_model(
            input_size=self.input_dim,
            n_nodes=self.num_nodes,
            window=self.lookback,
            horizon=self.horizon,
            n_exog_emb=[],
            output_size=1,
            hidden_size=int(model_config["hidden_size"]),
            n_layers=int(model_config["n_layers"]),
            dropout=float(model_config["dropout"]),
        )

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        if not isinstance(inputs, ModelInput):
            raise TypeError("STID expects ModelInput")
        if any(
            value is not None
            for value in (
                inputs.time_features,
                inputs.node_features,
                inputs.adjacency,
                inputs.static_features,
            )
        ):
            raise ValueError("STID accepts history x only")
        x = inputs.x
        if x.ndim != 4:
            raise ValueError("STID expects x with shape (B, L, N, C)")
        batch, steps, nodes, channels = x.shape
        expected_input = (self.lookback, self.num_nodes, self.input_dim)
        if (steps, nodes, channels) != expected_input:
            raise ValueError(
                "unexpected STID input shape: "
                f"{tuple(x.shape)}; expected (*, {expected_input[0]}, {expected_input[1]}, {expected_input[2]})"
            )
        if not torch.isfinite(x).all():
            raise FloatingPointError("STID input contains NaN or Inf")
        upstream_output = self.upstream(x)
        expected_output = (batch, self.horizon, self.num_nodes, 1)
        if tuple(upstream_output.shape) != expected_output:
            raise ValueError(
                f"STIDModel output must have shape {expected_output}, got {tuple(upstream_output.shape)}"
            )
        if not torch.isfinite(upstream_output).all():
            raise FloatingPointError("STIDModel output contains NaN or Inf")
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
    if not isinstance(model_config, dict):
        raise ValueError("STID model config must be a mapping")
    unknown = sorted(set(model_config) - _CONFIG_FIELDS)
    missing = sorted(_CONFIG_FIELDS - set(model_config))
    if unknown:
        raise ValueError(f"STID model config has unknown field: {unknown[0]}")
    if missing:
        raise ValueError(f"STID model config is missing field: {missing[0]}")
    for name in ("hidden_size", "n_layers"):
        value = model_config[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"STID {name} must be a positive integer")
    dropout = model_config["dropout"]
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0.0 <= float(dropout) < 1.0:
        raise ValueError("STID dropout must be in [0, 1)")


def build_model(model_config: dict[str, Any], data_info: DataInfoView) -> STID:
    _validate_config(model_config)
    node_ids = tuple(data_info.node_ids)
    if not node_ids or len(node_ids) != data_info.num_nodes:
        raise ValueError("STID requires node_ids aligned to every input node")
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("STID node_ids must not contain duplicates")
    try:
        from tsl.nn.models.temporal import STIDModel
    except ImportError as exc:
        raise ImportError(
            "STID requires the formal tsl.nn.models.temporal.STIDModel in the tsl runtime environment"
        ) from exc
    return STID(
        upstream_model=STIDModel,
        data_info=data_info,
        model_config=model_config,
    )

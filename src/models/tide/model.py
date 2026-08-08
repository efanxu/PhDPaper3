"""Node-shared adapter around the official Time-Series-Library TiDE."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from integrations.time_series_library import (
    load_time_series_library_model_class,
    resolve_time_series_library_source_root,
    run_time_series_library_forecast,
    validate_time_series_library_config_fields,
    validate_time_series_library_data_info,
)
from models.base import DataInfoView, ModelInput, NodeSharedForecastModel


_CONFIG_FIELDS = {
    "d_model",
    "d_ff",
    "e_layers",
    "d_layers",
    "dropout",
    "label_len",
    "freq",
    "bias",
    "feature_encode_dim",
}


class TiDE(NodeSharedForecastModel):
    """Apply one shared official TiDE instance independently per node."""

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
        self.input_power_index = int(data_info.input_power_index)
        self.label_len = int(model_config["label_len"])
        self.model_config = dict(model_config)
        upstream_config = SimpleNamespace(
            task_name="long_term_forecast",
            seq_len=self.lookback,
            label_len=self.label_len,
            pred_len=self.horizon,
            enc_in=self.input_dim,
            c_out=self.input_dim,
            d_model=int(model_config["d_model"]),
            d_ff=int(model_config["d_ff"]),
            e_layers=int(model_config["e_layers"]),
            d_layers=int(model_config["d_layers"]),
            dropout=float(model_config["dropout"]),
            freq=str(model_config["freq"]),
        )
        self.upstream = upstream_model(
            upstream_config,
            bias=bool(model_config["bias"]),
            feature_encode_dim=int(model_config["feature_encode_dim"]),
        )

    def forward_node_chunk(
        self,
        inputs: ModelInput,
        node_start: int,
        node_end: int,
    ) -> torch.Tensor:
        x = self._node_chunk_x(inputs, node_start, node_end, model_name="TiDE")
        # The official forecast fallback creates zero dynamic features when
        # both mark arguments are None.  No future temporal or power values
        # are synthesized by this adapter.
        output, batch, nodes = run_time_series_library_forecast(
            x,
            self.upstream,
            horizon=self.horizon,
            input_power_index=self.input_power_index,
            model_name="TiDE",
        )
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


def _validate_config(model_config: dict[str, Any], *, lookback: int) -> None:
    validate_time_series_library_config_fields(
        model_config, model_name="TiDE", fields=_CONFIG_FIELDS
    )
    for name in ("d_model", "d_ff", "e_layers", "d_layers", "label_len", "feature_encode_dim"):
        value = model_config[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"TiDE {name} must be a positive integer")
    if model_config["label_len"] > lookback:
        raise ValueError("TiDE label_len must not exceed lookback")
    dropout = model_config["dropout"]
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0.0 <= float(dropout) < 1.0:
        raise ValueError("TiDE dropout must be in [0, 1)")
    if model_config["freq"] not in {"h", "t", "s", "m", "a", "w", "d", "b"}:
        raise ValueError("TiDE freq is not supported by Time-Series-Library")
    if not isinstance(model_config["bias"], bool):
        raise ValueError("TiDE bias must be a boolean")


def build_model(model_config: dict[str, Any], data_info: DataInfoView) -> TiDE:
    validate_time_series_library_data_info(data_info, model_name="TiDE")
    _validate_config(model_config, lookback=int(data_info.lookback))
    source_root = resolve_time_series_library_source_root(
        data_info.project_root, model_name="TiDE"
    )
    upstream_class = load_time_series_library_model_class(
        "TiDE", source_root=source_root
    )
    return TiDE(
        upstream_model=upstream_class,
        data_info=data_info,
        model_config=model_config,
    )

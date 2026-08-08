"""Node-shared adapter around Time-Series-Library SegRNN."""

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


_CONFIG_FIELDS = {"d_model", "dropout", "seg_len"}


class SegRNN(NodeSharedForecastModel):
    """Apply one shared upstream SegRNN instance independently per node."""

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
        self.model_config = dict(model_config)
        upstream_config = SimpleNamespace(
            task_name="long_term_forecast",
            seq_len=self.lookback,
            pred_len=self.horizon,
            enc_in=self.input_dim,
            d_model=int(model_config["d_model"]),
            dropout=float(model_config["dropout"]),
            seg_len=int(model_config["seg_len"]),
        )
        self.upstream = upstream_model(upstream_config)

    def forward_node_chunk(
        self,
        inputs: ModelInput,
        node_start: int,
        node_end: int,
    ) -> torch.Tensor:
        x = self._node_chunk_x(inputs, node_start, node_end, model_name="SegRNN")
        output, batch, nodes = run_time_series_library_forecast(
            x,
            self.upstream,
            horizon=self.horizon,
            input_power_index=self.input_power_index,
            model_name="SegRNN",
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


def _validate_config(model_config: dict[str, Any], *, lookback: int, horizon: int) -> None:
    validate_time_series_library_config_fields(
        model_config, model_name="SegRNN", fields=_CONFIG_FIELDS
    )
    d_model = model_config["d_model"]
    dropout = model_config["dropout"]
    seg_len = model_config["seg_len"]
    if not isinstance(d_model, int) or isinstance(d_model, bool) or d_model < 2 or d_model % 2:
        raise ValueError("SegRNN d_model must be a positive even integer")
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0.0 <= float(dropout) < 1.0:
        raise ValueError("SegRNN dropout must be in [0, 1)")
    if not isinstance(seg_len, int) or isinstance(seg_len, bool) or seg_len < 1:
        raise ValueError("SegRNN seg_len must be a positive integer")
    if lookback % seg_len or horizon % seg_len:
        raise ValueError("SegRNN lookback and max_pred_len must be divisible by seg_len")


def build_model(model_config: dict[str, Any], data_info: DataInfoView) -> SegRNN:
    validate_time_series_library_data_info(data_info, model_name="SegRNN")
    _validate_config(
        model_config,
        lookback=int(data_info.lookback),
        horizon=int(data_info.max_pred_len),
    )
    source_root = resolve_time_series_library_source_root(
        data_info.project_root, model_name="SegRNN"
    )
    upstream_class = load_time_series_library_model_class(
        "SegRNN", source_root=source_root
    )
    return SegRNN(
        upstream_model=upstream_class,
        data_info=data_info,
        model_config=model_config,
    )

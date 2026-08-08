"""Node-shared adapter around the official Time-Series-Library FreTS."""

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


_CONFIG_FIELDS = {"channel_independence"}


class FreTS(NodeSharedForecastModel):
    """Apply one shared official FreTS instance independently per node."""

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
            channel_independence=int(model_config["channel_independence"]),
        )
        self.upstream = upstream_model(upstream_config)

    def forward_node_chunk(
        self,
        inputs: ModelInput,
        node_start: int,
        node_end: int,
    ) -> torch.Tensor:
        x = self._node_chunk_x(inputs, node_start, node_end, model_name="FreTS")
        output, batch, nodes = run_time_series_library_forecast(
            x,
            self.upstream,
            horizon=self.horizon,
            input_power_index=self.input_power_index,
            model_name="FreTS",
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


def _validate_config(model_config: dict[str, Any]) -> None:
    validate_time_series_library_config_fields(
        model_config, model_name="FreTS", fields=_CONFIG_FIELDS
    )
    value = model_config["channel_independence"]
    if not isinstance(value, int) or isinstance(value, bool) or value != 1:
        raise ValueError("FreTS channel_independence must be 1")


def build_model(model_config: dict[str, Any], data_info: DataInfoView) -> FreTS:
    _validate_config(model_config)
    validate_time_series_library_data_info(data_info, model_name="FreTS")
    source_root = resolve_time_series_library_source_root(
        data_info.project_root, model_name="FreTS"
    )
    upstream_class = load_time_series_library_model_class(
        "FreTS", source_root=source_root
    )
    return FreTS(
        upstream_model=upstream_class,
        data_info=data_info,
        model_config=model_config,
    )

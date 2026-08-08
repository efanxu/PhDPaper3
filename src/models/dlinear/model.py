"""Node-shared adapter around Time-Series-Library DLinear."""

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


_CONFIG_FIELDS = {"moving_avg"}


class DLinear(NodeSharedForecastModel):
    """Apply one shared upstream DLinear instance independently per node."""

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
            moving_avg=int(model_config["moving_avg"]),
        )
        self.upstream = upstream_model(upstream_config, individual=False)

    def forward_node_chunk(
        self,
        inputs: ModelInput,
        node_start: int,
        node_end: int,
    ) -> torch.Tensor:
        x = self._node_chunk_x(inputs, node_start, node_end, model_name="DLinear")
        output, batch, nodes = run_time_series_library_forecast(
            x,
            self.upstream,
            horizon=self.horizon,
            input_power_index=self.input_power_index,
            model_name="DLinear",
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
        model_config, model_name="DLinear", fields=_CONFIG_FIELDS
    )
    value = model_config["moving_avg"]
    if not isinstance(value, int) or isinstance(value, bool) or value < 1 or value % 2 == 0:
        raise ValueError("DLinear moving_avg must be a positive odd integer")


def build_model(model_config: dict[str, Any], data_info: DataInfoView) -> DLinear:
    _validate_config(model_config)
    validate_time_series_library_data_info(data_info, model_name="DLinear")
    source_root = resolve_time_series_library_source_root(
        data_info.project_root, model_name="DLinear"
    )
    upstream_class = load_time_series_library_model_class(
        "DLinear", source_root=source_root
    )
    return DLinear(
        upstream_model=upstream_class,
        data_info=data_info,
        model_config=model_config,
    )

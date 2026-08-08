"""Node-shared adapter around Time-Series-Library TimeMixer."""

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
    "down_sampling_layers",
    "down_sampling_window",
    "channel_independence",
    "moving_avg",
    "decomp_method",
    "top_k",
    "use_norm",
    "down_sampling_method",
    "dropout",
    "label_len",
    "embed",
    "freq",
}


class TimeMixer(NodeSharedForecastModel):
    """Apply one shared upstream TimeMixer instance independently per node."""

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
            label_len=int(model_config["label_len"]),
            pred_len=self.horizon,
            enc_in=self.input_dim,
            c_out=self.input_dim,
            d_model=int(model_config["d_model"]),
            d_ff=int(model_config["d_ff"]),
            e_layers=int(model_config["e_layers"]),
            down_sampling_layers=int(model_config["down_sampling_layers"]),
            down_sampling_window=int(model_config["down_sampling_window"]),
            channel_independence=int(model_config["channel_independence"]),
            moving_avg=int(model_config["moving_avg"]),
            decomp_method=str(model_config["decomp_method"]),
            top_k=int(model_config["top_k"]),
            use_norm=int(model_config["use_norm"]),
            down_sampling_method=str(model_config["down_sampling_method"]),
            dropout=float(model_config["dropout"]),
            embed=str(model_config["embed"]),
            freq=str(model_config["freq"]),
        )
        self.upstream = upstream_model(upstream_config)

    def forward_node_chunk(
        self,
        inputs: ModelInput,
        node_start: int,
        node_end: int,
    ) -> torch.Tensor:
        x = self._node_chunk_x(inputs, node_start, node_end, model_name="TimeMixer")
        output, batch, nodes = run_time_series_library_forecast(
            x,
            self.upstream,
            horizon=self.horizon,
            input_power_index=self.input_power_index,
            model_name="TimeMixer",
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


def _validate_config(model_config: dict[str, Any], *, sequence_length: int) -> None:
    validate_time_series_library_config_fields(
        model_config, model_name="TimeMixer", fields=_CONFIG_FIELDS
    )
    positive_fields = (
        "d_model",
        "d_ff",
        "e_layers",
        "down_sampling_layers",
        "down_sampling_window",
        "top_k",
        "label_len",
    )
    if not all(
        isinstance(model_config[name], int)
        and not isinstance(model_config[name], bool)
        and model_config[name] > 0
        for name in positive_fields
    ):
        raise ValueError("TimeMixer integer dimensions and counts must be positive")
    if model_config["down_sampling_window"] <= 1:
        raise ValueError("TimeMixer down_sampling_window must be greater than 1")
    if sequence_length % (model_config["down_sampling_window"] ** model_config["down_sampling_layers"]):
        raise ValueError("TimeMixer lookback must support every down-sampling scale")
    if model_config["channel_independence"] not in {0, 1}:
        raise ValueError("TimeMixer channel_independence must be 0 or 1")
    if model_config["use_norm"] not in {0, 1}:
        raise ValueError("TimeMixer use_norm must be 0 or 1")
    if model_config["moving_avg"] < 1 or model_config["moving_avg"] % 2 == 0:
        raise ValueError("TimeMixer moving_avg must be a positive odd integer")
    if model_config["decomp_method"] not in {"moving_avg", "dft_decomp"}:
        raise ValueError("TimeMixer decomp_method must be moving_avg or dft_decomp")
    if model_config["down_sampling_method"] not in {"avg", "max", "conv"}:
        raise ValueError("TimeMixer down_sampling_method must be avg, max or conv")
    dropout = model_config["dropout"]
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0.0 <= float(dropout) < 1.0:
        raise ValueError("TimeMixer dropout must be in [0, 1)")
    if model_config["embed"] not in {"timeF", "fixed", "learned"}:
        raise ValueError("TimeMixer embed must be timeF, fixed or learned")
    if model_config["freq"] not in {"h", "t", "s", "m", "a", "w", "d", "b"}:
        raise ValueError("TimeMixer freq is not supported by Time-Series-Library")


def build_model(model_config: dict[str, Any], data_info: DataInfoView) -> TimeMixer:
    validate_time_series_library_data_info(data_info, model_name="TimeMixer")
    _validate_config(
        model_config,
        sequence_length=int(data_info.lookback),
    )
    source_root = resolve_time_series_library_source_root(
        data_info.project_root, model_name="TimeMixer"
    )
    upstream_class = load_time_series_library_model_class(
        "TimeMixer", source_root=source_root
    )
    return TimeMixer(
        upstream_model=upstream_class,
        data_info=data_info,
        model_config=model_config,
    )

"""Node-shared adapter around Time-Series-Library TimesNet."""

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
    "top_k",
    "num_kernels",
    "dropout",
    "label_len",
    "embed",
    "freq",
}


class TimesNet(NodeSharedForecastModel):
    """Apply one shared upstream TimesNet instance independently per node."""

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
            top_k=int(model_config["top_k"]),
            num_kernels=int(model_config["num_kernels"]),
            embed=str(model_config["embed"]),
            freq=str(model_config["freq"]),
            dropout=float(model_config["dropout"]),
        )
        self.upstream = upstream_model(upstream_config)

    def forward_node_chunk(
        self,
        inputs: ModelInput,
        node_start: int,
        node_end: int,
    ) -> torch.Tensor:
        x = self._node_chunk_x(inputs, node_start, node_end, model_name="TimesNet")
        # The upstream FFT_for_Period aggregates over its whole batch when it
        # selects periods.  Keep each turbine's history as its own upstream
        # batch so that a node chunk cannot change another node's periods.
        outputs = []
        for node_offset in range(x.shape[2]):
            output, batch, _ = run_time_series_library_forecast(
                x[:, :, node_offset : node_offset + 1, :],
                self.upstream,
                horizon=self.horizon,
                input_power_index=self.input_power_index,
                model_name="TimesNet",
            )
            outputs.append(output)
        output = torch.cat(outputs, dim=1)
        batch = int(x.shape[0])
        nodes = int(x.shape[2])
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
        model_config, model_name="TimesNet", fields=_CONFIG_FIELDS
    )
    integer_fields = (
        "d_model",
        "d_ff",
        "e_layers",
        "top_k",
        "num_kernels",
        "label_len",
    )
    if not all(
        isinstance(model_config[name], int)
        and not isinstance(model_config[name], bool)
        and model_config[name] > 0
        for name in integer_fields
    ):
        raise ValueError("TimesNet integer dimensions and counts must be positive")
    if model_config["top_k"] > sequence_length // 2:
        raise ValueError("TimesNet top_k is too large for the upstream FFT")
    dropout = model_config["dropout"]
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0.0 <= float(dropout) < 1.0:
        raise ValueError("TimesNet dropout must be in [0, 1)")
    if model_config["embed"] not in {"timeF", "fixed", "learned"}:
        raise ValueError("TimesNet embed must be timeF, fixed or learned")
    if model_config["freq"] not in {"h", "t", "s", "m", "a", "w", "d", "b"}:
        raise ValueError("TimesNet freq is not supported by Time-Series-Library")


def build_model(model_config: dict[str, Any], data_info: DataInfoView) -> TimesNet:
    validate_time_series_library_data_info(data_info, model_name="TimesNet")
    _validate_config(
        model_config,
        sequence_length=int(data_info.lookback) + int(data_info.max_pred_len),
    )
    source_root = resolve_time_series_library_source_root(
        data_info.project_root, model_name="TimesNet"
    )
    upstream_class = load_time_series_library_model_class(
        "TimesNet", source_root=source_root
    )
    return TimesNet(
        upstream_model=upstream_class,
        data_info=data_info,
        model_config=model_config,
    )

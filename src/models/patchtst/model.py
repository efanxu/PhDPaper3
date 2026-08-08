"""Node-shared adapter around the official Time-Series-Library PatchTST."""

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
    "n_heads",
    "d_ff",
    "e_layers",
    "factor",
    "dropout",
    "activation",
    "patch_len",
    "stride",
}


class PatchTST(NodeSharedForecastModel):
    """Apply one shared official PatchTST instance independently per node."""

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
            n_heads=int(model_config["n_heads"]),
            d_ff=int(model_config["d_ff"]),
            e_layers=int(model_config["e_layers"]),
            factor=int(model_config["factor"]),
            dropout=float(model_config["dropout"]),
            activation=str(model_config["activation"]),
        )
        self.upstream = upstream_model(
            upstream_config,
            patch_len=int(model_config["patch_len"]),
            stride=int(model_config["stride"]),
        )

    def forward_node_chunk(
        self,
        inputs: ModelInput,
        node_start: int,
        node_end: int,
    ) -> torch.Tensor:
        x = self._node_chunk_x(inputs, node_start, node_end, model_name="PatchTST")
        output, batch, nodes = run_time_series_library_forecast(
            x,
            self.upstream,
            horizon=self.horizon,
            input_power_index=self.input_power_index,
            model_name="PatchTST",
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
        model_config, model_name="PatchTST", fields=_CONFIG_FIELDS
    )
    for name in ("d_model", "n_heads", "d_ff", "e_layers", "factor", "patch_len", "stride"):
        value = model_config[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"PatchTST {name} must be a positive integer")
    if model_config["d_model"] % model_config["n_heads"]:
        raise ValueError("PatchTST d_model must be divisible by n_heads")
    if lookback < model_config["patch_len"]:
        raise ValueError("PatchTST lookback must be at least patch_len")
    dropout = model_config["dropout"]
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0.0 <= float(dropout) < 1.0:
        raise ValueError("PatchTST dropout must be in [0, 1)")
    if model_config["activation"] not in {"relu", "gelu"}:
        raise ValueError("PatchTST activation must be relu or gelu")


def build_model(model_config: dict[str, Any], data_info: DataInfoView) -> PatchTST:
    validate_time_series_library_data_info(data_info, model_name="PatchTST")
    _validate_config(model_config, lookback=int(data_info.lookback))
    source_root = resolve_time_series_library_source_root(
        data_info.project_root, model_name="PatchTST"
    )
    upstream_class = load_time_series_library_model_class(
        "PatchTST", source_root=source_root
    )
    return PatchTST(
        upstream_model=upstream_class,
        data_info=data_info,
        model_config=model_config,
    )

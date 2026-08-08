"""Node-shared adapter around the official Fourier FEDformer model."""

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
    "d_layers",
    "moving_avg",
    "dropout",
    "activation",
    "embed",
    "freq",
    "version",
    "mode_select",
    "modes",
}


class FEDformer(NodeSharedForecastModel):
    """Apply one shared official Fourier FEDformer independently per node."""

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
        self.label_len = self.lookback // 2
        self.model_config = dict(model_config)
        upstream_config = SimpleNamespace(
            task_name="long_term_forecast",
            seq_len=self.lookback,
            label_len=self.label_len,
            pred_len=self.horizon,
            enc_in=self.input_dim,
            dec_in=self.input_dim,
            c_out=self.input_dim,
            d_model=int(model_config["d_model"]),
            n_heads=int(model_config["n_heads"]),
            d_ff=int(model_config["d_ff"]),
            e_layers=int(model_config["e_layers"]),
            d_layers=int(model_config["d_layers"]),
            moving_avg=int(model_config["moving_avg"]),
            dropout=float(model_config["dropout"]),
            activation=str(model_config["activation"]),
            embed=str(model_config["embed"]),
            freq=str(model_config["freq"]),
        )
        self.upstream = upstream_model(
            upstream_config,
            version=str(model_config["version"]),
            mode_select=str(model_config["mode_select"]),
            modes=int(model_config["modes"]),
        )

    def forward_node_chunk(
        self,
        inputs: ModelInput,
        node_start: int,
        node_end: int,
    ) -> torch.Tensor:
        x = self._node_chunk_x(
            inputs,
            node_start,
            node_end,
            model_name="FEDformer",
        )
        node_history = x.permute(0, 2, 1, 3).contiguous().reshape(
            x.shape[0] * x.shape[2], x.shape[1], x.shape[3]
        )
        history = node_history[:, -self.label_len :, :]
        zeros = torch.zeros(
            node_history.shape[0],
            self.horizon,
            node_history.shape[2],
            dtype=node_history.dtype,
            device=node_history.device,
        )
        decoder_input = torch.cat([history, zeros], dim=1)
        output, batch, nodes = run_time_series_library_forecast(
            x,
            self.upstream,
            horizon=self.horizon,
            input_power_index=self.input_power_index,
            model_name="FEDformer",
            decoder_input=decoder_input,
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
            "label_len": self.label_len,
        }


def _validate_config(model_config: dict[str, Any], *, lookback: int) -> None:
    validate_time_series_library_config_fields(
        model_config, model_name="FEDformer", fields=_CONFIG_FIELDS
    )
    for name in ("d_model", "n_heads", "d_ff", "e_layers", "d_layers", "moving_avg", "modes"):
        value = model_config[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"FEDformer {name} must be a positive integer")
    if model_config["d_model"] % model_config["n_heads"]:
        raise ValueError("FEDformer d_model must be divisible by n_heads")
    if model_config["moving_avg"] % 2 == 0:
        raise ValueError("FEDformer moving_avg must be a positive odd integer")
    if lookback < 2:
        raise ValueError("FEDformer lookback must be at least 2")
    dropout = model_config["dropout"]
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0.0 <= float(dropout) < 1.0:
        raise ValueError("FEDformer dropout must be in [0, 1)")
    if model_config["activation"] not in {"relu", "gelu"}:
        raise ValueError("FEDformer activation must be relu or gelu")
    if model_config["embed"] not in {"timeF", "fixed", "learned"}:
        raise ValueError("FEDformer embed is not supported by Time-Series-Library")
    if model_config["freq"] not in {"h", "t", "s", "m", "a", "w", "d", "b"}:
        raise ValueError("FEDformer freq is not supported by Time-Series-Library")
    if model_config["version"] != "fourier":
        raise ValueError("FEDformer version must be fourier")
    if model_config["mode_select"] not in {"random", "low"}:
        raise ValueError("FEDformer mode_select must be random or low")


def build_model(model_config: dict[str, Any], data_info: DataInfoView) -> FEDformer:
    validate_time_series_library_data_info(data_info, model_name="FEDformer")
    _validate_config(model_config, lookback=int(data_info.lookback))
    source_root = resolve_time_series_library_source_root(
        data_info.project_root, model_name="FEDformer"
    )
    upstream_class = load_time_series_library_model_class(
        "FEDformer", source_root=source_root
    )
    return FEDformer(
        upstream_model=upstream_class,
        data_info=data_info,
        model_config=model_config,
    )

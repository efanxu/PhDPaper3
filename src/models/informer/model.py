"""Node-shared adapter around the official Time-Series-Library Informer."""

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
    "factor",
    "dropout",
    "activation",
    "embed",
    "freq",
    "label_len",
    "distil",
}


class Informer(NodeSharedForecastModel):
    """Apply one shared official Informer instance independently per node."""

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
            dec_in=self.input_dim,
            c_out=self.input_dim,
            d_model=int(model_config["d_model"]),
            n_heads=int(model_config["n_heads"]),
            d_ff=int(model_config["d_ff"]),
            e_layers=int(model_config["e_layers"]),
            d_layers=int(model_config["d_layers"]),
            factor=int(model_config["factor"]),
            dropout=float(model_config["dropout"]),
            activation=str(model_config["activation"]),
            embed=str(model_config["embed"]),
            freq=str(model_config["freq"]),
            distil=bool(model_config["distil"]),
        )
        self.upstream = upstream_model(upstream_config)

    def forward_node_chunk(
        self,
        inputs: ModelInput,
        node_start: int,
        node_end: int,
    ) -> torch.Tensor:
        x = self._node_chunk_x(inputs, node_start, node_end, model_name="Informer")
        node_history = x.permute(0, 2, 1, 3).contiguous().view(
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
            model_name="Informer",
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
        }


def _validate_config(model_config: dict[str, Any], *, lookback: int) -> None:
    validate_time_series_library_config_fields(
        model_config, model_name="Informer", fields=_CONFIG_FIELDS
    )
    positive_fields = ("d_model", "n_heads", "d_ff", "e_layers", "d_layers", "factor", "label_len")
    if not all(
        isinstance(model_config[name], int)
        and not isinstance(model_config[name], bool)
        and model_config[name] > 0
        for name in positive_fields
    ):
        raise ValueError("Informer integer dimensions and counts must be positive")
    if model_config["d_model"] % model_config["n_heads"]:
        raise ValueError("Informer d_model must be divisible by n_heads")
    if model_config["label_len"] > lookback:
        raise ValueError("Informer label_len must not exceed lookback")
    dropout = model_config["dropout"]
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0.0 <= float(dropout) < 1.0:
        raise ValueError("Informer dropout must be in [0, 1)")
    if model_config["activation"] not in {"relu", "gelu"}:
        raise ValueError("Informer activation must be relu or gelu")
    if model_config["embed"] not in {"timeF", "fixed", "learned"}:
        raise ValueError("Informer embed must be timeF, fixed or learned")
    if model_config["freq"] not in {"h", "t", "s", "m", "a", "w", "d", "b"}:
        raise ValueError("Informer freq is not supported by Time-Series-Library")
    if not isinstance(model_config["distil"], bool):
        raise ValueError("Informer distil must be a boolean")


def build_model(model_config: dict[str, Any], data_info: DataInfoView) -> Informer:
    validate_time_series_library_data_info(data_info, model_name="Informer")
    _validate_config(model_config, lookback=int(data_info.lookback))
    source_root = resolve_time_series_library_source_root(
        data_info.project_root, model_name="Informer"
    )
    upstream_class = load_time_series_library_model_class(
        "Informer", source_root=source_root
    )
    return Informer(
        upstream_model=upstream_class,
        data_info=data_info,
        model_config=model_config,
    )

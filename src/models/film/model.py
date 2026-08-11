"""Node-shared adapter around the official Time-Series-Library FiLM."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
import sys

import torch

from integrations.time_series_library import (
    load_time_series_library_model_class,
    resolve_time_series_library_source_root,
    run_time_series_library_forecast,
    validate_time_series_library_config_fields,
    validate_time_series_library_data_info,
)
from models.base import DataInfoView, ModelInput, NodeSharedForecastModel


_CONFIG_FIELDS = {"e_layers", "label_len"}


class FiLM(NodeSharedForecastModel):
    """Apply one shared official FiLM instance independently per node."""

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
        upstream_module = _upstream_module(upstream_model)
        # The upstream module chooses cuda:0 at import time.  Construction is
        # forced to CPU so the public runtime device remains authoritative.
        upstream_module.device = torch.device("cpu")
        upstream_config = SimpleNamespace(
            task_name="long_term_forecast",
            seq_len=self.lookback,
            label_len=int(model_config["label_len"]),
            pred_len=self.horizon,
            enc_in=self.input_dim,
            e_layers=int(model_config["e_layers"]),
        )
        self.upstream = upstream_model(upstream_config)

    def forward_node_chunk(
        self,
        inputs: ModelInput,
        node_start: int,
        node_end: int,
    ) -> torch.Tensor:
        x = self._node_chunk_x(inputs, node_start, node_end, model_name="FiLM")
        _upstream_module(type(self.upstream)).device = x.device
        output, batch, nodes = run_time_series_library_forecast(
            x,
            self.upstream,
            horizon=self.horizon,
            input_power_index=self.input_power_index,
            model_name="FiLM",
            force_cuda_fp32_forecast=True,
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


def _upstream_module(upstream_model: type):
    module = sys.modules.get(upstream_model.__module__)
    if module is None:
        raise RuntimeError(
            f"FiLM upstream module is not loaded: {upstream_model.__module__}"
        )
    return module


def _validate_config(model_config: dict[str, Any], *, lookback: int) -> None:
    validate_time_series_library_config_fields(
        model_config, model_name="FiLM", fields=_CONFIG_FIELDS
    )
    for name in ("e_layers", "label_len"):
        value = model_config[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"FiLM {name} must be a positive integer")
    if model_config["label_len"] > lookback:
        raise ValueError("FiLM label_len must not exceed lookback")


def build_model(model_config: dict[str, Any], data_info: DataInfoView) -> FiLM:
    validate_time_series_library_data_info(data_info, model_name="FiLM")
    _validate_config(model_config, lookback=int(data_info.lookback))
    source_root = resolve_time_series_library_source_root(
        data_info.project_root, model_name="FiLM"
    )
    upstream_class = load_time_series_library_model_class(
        "FiLM", source_root=source_root
    )
    return FiLM(
        upstream_model=upstream_class,
        data_info=data_info,
        model_config=model_config,
    )

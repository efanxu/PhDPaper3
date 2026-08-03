"""Node-shared RA-DS-PFD adapter around the local canonical backbone."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from pathlib import Path
from typing import Any

import torch

from models.base import DataInfoView, ForecastModel, ModelInput

from .backbone import CanonicalBackbone, CanonicalTrace
from .pfd0 import PFD0Propagation
from .relation_resource import RelationResource, load_relation_resource
from .relation_spatial import RelationBiasProvider, RelationSpatialInsertion


_CONFIG_FIELDS = {
    "d_model",
    "n_heads",
    "d_ff",
    "e_layers",
    "dropout",
    "factor",
    "seg_len",
    "win_size",
    "spatial_disabled",
    "pfd_mode",
    "spatial_heads",
    "spatial_d_ff",
    "relation_dim",
    "spatial_dropout",
    "gamma_init",
    "relation_resource",
    "spatial_edge_chunk_size",
}
_CANONICAL_CONFIG_FIELDS = {
    "d_model",
    "n_heads",
    "d_ff",
    "e_layers",
    "dropout",
    "factor",
    "seg_len",
    "win_size",
    "spatial_disabled",
}
_POSITIVE_INTEGER_FIELDS = {"d_model", "n_heads", "d_ff", "e_layers", "factor", "seg_len", "win_size"}
_P2_POSITIVE_INTEGER_FIELDS = {"spatial_heads", "spatial_d_ff", "relation_dim"}


class RADSPFDCrossformer(ForecastModel):
    """Apply one local canonical Crossformer independently to every node."""

    def __init__(
        self,
        *,
        num_nodes: int,
        input_dim: int,
        lookback: int,
        horizon: int,
        input_power_index: int,
        model_config: dict[str, Any],
        source_root: Path,
        wspd_index: int | None = None,
        relation_resource: RelationResource | None = None,
    ) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.input_dim = int(input_dim)
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.input_power_index = int(input_power_index)
        self.model_config = dict(model_config)
        self.source_root = Path(source_root).resolve()
        self.spatial_disabled = bool(model_config["spatial_disabled"])
        self.wspd_index = None if wspd_index is None else int(wspd_index)
        self.relation_resource = relation_resource

        if self.spatial_disabled:
            # These are deliberately None rather than empty modules/buffers:
            # the legacy P1 state_dict remains exactly upstream-canonical.
            self.pfd0: PFD0Propagation | None = None
            self.relation_bias_provider: RelationBiasProvider | None = None
            spatial_modules = None
        else:
            if self.wspd_index is None:
                raise ValueError("enabled PFD0 requires Wspd resolved from DataInfoView.feature_columns")
            if relation_resource is None:
                raise ValueError("enabled relation spatial path requires a validated relation resource")
            d_model = int(model_config["d_model"])
            self.relation_bias_provider = RelationBiasProvider(
                edge_index=relation_resource.edge_index,
                edge_static_features=relation_resource.edge_static_features,
                num_nodes=self.num_nodes,
                spatial_heads=int(model_config["spatial_heads"]),
                spatial_d_ff=int(model_config["spatial_d_ff"]),
                relation_dim=int(model_config["relation_dim"]),
            )
            self.pfd0 = PFD0Propagation(
                lookback=self.lookback,
                seg_len=int(model_config["seg_len"]),
                win_size=int(model_config["win_size"]),
                d_model=d_model,
                dropout=float(model_config["spatial_dropout"]),
                wspd_index=self.wspd_index,
            )
            spatial_modules = (
                RelationSpatialInsertion(
                    d_model=d_model,
                    spatial_heads=int(model_config["spatial_heads"]),
                    spatial_dropout=float(model_config["spatial_dropout"]),
                    gamma_init=float(model_config["gamma_init"]),
                    bias_provider=self.relation_bias_provider,
                    edge_chunk_size=model_config.get("spatial_edge_chunk_size", 128),
                ),
                RelationSpatialInsertion(
                    d_model=d_model,
                    spatial_heads=int(model_config["spatial_heads"]),
                    spatial_dropout=float(model_config["spatial_dropout"]),
                    gamma_init=float(model_config["gamma_init"]),
                    bias_provider=self.relation_bias_provider,
                    edge_chunk_size=model_config.get("spatial_edge_chunk_size", 128),
                ),
            )
        self.backbone = CanonicalBackbone(
            source_root=self.source_root,
            enc_in=self.input_dim,
            seq_len=self.lookback,
            pred_len=self.horizon,
            model_config=self.model_config,
            num_nodes=self.num_nodes,
            spatial_modules=spatial_modules,
        )

    def _node_history(self, inputs: ModelInput) -> tuple[torch.Tensor, int, int]:
        if not isinstance(inputs, ModelInput):
            raise TypeError("RA-DS-PFD Crossformer expects ModelInput")
        if any(
            value is not None
            for value in (
                inputs.time_features,
                inputs.node_features,
                inputs.adjacency,
                inputs.static_features,
            )
        ):
            raise ValueError("RA-DS-PFD Crossformer accepts history x only")
        x = inputs.x
        if not isinstance(x, torch.Tensor) or x.ndim != 4:
            raise ValueError("RA-DS-PFD Crossformer expects x with shape (B, L, N, C)")
        batch, steps, nodes, channels = x.shape
        expected = (self.lookback, self.num_nodes, self.input_dim)
        if (steps, nodes, channels) != expected:
            raise ValueError(
                "unexpected RA-DS-PFD Crossformer input shape: "
                f"{tuple(x.shape)}; expected (*, {expected[0]}, {expected[1]}, {expected[2]})"
            )
        if not torch.isfinite(x).all():
            raise FloatingPointError("RA-DS-PFD Crossformer input contains NaN or Inf")
        node_history = x.permute(0, 2, 1, 3).reshape(batch * nodes, steps, channels)
        return node_history, batch, nodes

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        node_history, batch, nodes = self._node_history(inputs)
        if self.pfd0 is None:
            full_output = self.backbone(node_history)
        else:
            full_output = self.backbone(
                node_history,
                propagation_tokens=self.pfd0(inputs.x),
            )
        expected = (batch * nodes, self.horizon, self.input_dim)
        if tuple(full_output.shape) != expected:
            raise ValueError(
                "local canonical Crossformer output must have shape "
                f"{expected}, got {tuple(full_output.shape)}"
            )
        if not torch.isfinite(full_output).all():
            raise FloatingPointError("local canonical Crossformer output contains NaN or Inf")
        output = full_output[..., self.input_power_index].reshape(batch, nodes, self.horizon)
        return self.validate_output(output, batch=batch, nodes=nodes, horizon=self.horizon)

    def forward_canonical_trace(self, inputs: ModelInput) -> CanonicalTrace:
        """Return canonical stages, using local ``[B,N,C,S,D]`` in P2 mode."""

        node_history, _, _ = self._node_history(inputs)
        propagation_tokens = self.pfd0(inputs.x) if self.pfd0 is not None else None
        trace = self.backbone.forward_backbone(
            node_history,
            return_trace=True,
            propagation_tokens=propagation_tokens,
        )
        assert isinstance(trace, CanonicalTrace)
        return trace

    def load_upstream_state_dict(self, upstream_state_dict: dict[str, torch.Tensor]) -> None:
        """Strictly transfer the upstream canonical state to this adapter."""

        self.backbone.load_upstream_state_dict(upstream_state_dict)

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
    if not isinstance(model_config, dict):
        raise TypeError("RA-DS-PFD Crossformer model config must be a mapping")
    unknown = sorted(set(model_config) - _CONFIG_FIELDS)
    missing = sorted(_CANONICAL_CONFIG_FIELDS - set(model_config))
    if unknown:
        raise ValueError(f"RA-DS-PFD Crossformer model config has unknown field: {unknown[0]}")
    if missing:
        raise ValueError(f"RA-DS-PFD Crossformer model config is missing field: {missing[0]}")
    for name in sorted(_POSITIVE_INTEGER_FIELDS):
        value = model_config[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"RA-DS-PFD Crossformer {name} must be a positive integer")
    if model_config["d_model"] % model_config["n_heads"]:
        raise ValueError("RA-DS-PFD Crossformer d_model must be divisible by n_heads")
    if model_config["d_model"] % 2:
        raise ValueError("RA-DS-PFD Crossformer d_model must be even")
    dropout = model_config["dropout"]
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not isfinite(float(dropout)):
        raise ValueError("RA-DS-PFD Crossformer dropout must be a finite number")
    if not 0.0 <= float(dropout) < 1.0:
        raise ValueError("RA-DS-PFD Crossformer dropout must be in [0, 1)")
    if model_config["seg_len"] != 12:
        raise ValueError("RA-DS-PFD Crossformer canonical path only supports seg_len=12")
    if model_config["win_size"] != 2:
        raise ValueError("RA-DS-PFD Crossformer P1 only supports win_size=2")
    if model_config["e_layers"] != 2:
        raise ValueError("RA-DS-PFD Crossformer P1 only supports e_layers=2")
    if not isinstance(model_config["spatial_disabled"], bool):
        raise ValueError("RA-DS-PFD Crossformer spatial_disabled must be a boolean")
    if "pfd_mode" in model_config and model_config["pfd_mode"] != "pfd0":
        raise ValueError("RA-DS-PFD Crossformer only supports pfd_mode=pfd0 in P2")
    if model_config["spatial_disabled"]:
        return

    required_p2 = {
        "pfd_mode",
        "spatial_heads",
        "spatial_d_ff",
        "relation_dim",
        "spatial_dropout",
        "gamma_init",
        "relation_resource",
    }
    missing_p2 = sorted(required_p2 - set(model_config))
    if missing_p2:
        raise ValueError(f"RA-DS-PFD Crossformer P2 config is missing field: {missing_p2[0]}")
    if model_config["pfd_mode"] != "pfd0":
        raise ValueError("RA-DS-PFD Crossformer only supports pfd_mode=pfd0 in P2")
    for name in sorted(_P2_POSITIVE_INTEGER_FIELDS):
        value = model_config[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"RA-DS-PFD Crossformer {name} must be a positive integer")
    if model_config["d_model"] % model_config["spatial_heads"]:
        raise ValueError("RA-DS-PFD Crossformer d_model must be divisible by spatial_heads")
    spatial_dropout = model_config["spatial_dropout"]
    if (
        isinstance(spatial_dropout, bool)
        or not isinstance(spatial_dropout, (int, float))
        or not isfinite(float(spatial_dropout))
    ):
        raise ValueError("RA-DS-PFD Crossformer spatial_dropout must be a finite number")
    if not 0.0 <= float(spatial_dropout) < 1.0:
        raise ValueError("RA-DS-PFD Crossformer spatial_dropout must be in [0, 1)")
    gamma_init = model_config["gamma_init"]
    if (
        isinstance(gamma_init, bool)
        or not isinstance(gamma_init, (int, float))
        or not isfinite(float(gamma_init))
    ):
        raise ValueError("RA-DS-PFD Crossformer gamma_init must be a finite number")
    if abs(float(gamma_init) - 0.1) > 1e-12:
        raise ValueError("RA-DS-PFD Crossformer P2 requires gamma_init=0.1")
    if "spatial_edge_chunk_size" in model_config:
        chunk_size = model_config["spatial_edge_chunk_size"]
        if chunk_size is not None and (
            not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1
        ):
            raise ValueError("RA-DS-PFD Crossformer spatial_edge_chunk_size must be positive or null")
    if not isinstance(model_config["relation_resource"], Mapping):
        raise ValueError("RA-DS-PFD Crossformer relation_resource must be a mapping")


def _validate_data_info(data_info: DataInfoView) -> None:
    if data_info.num_nodes < 1 or data_info.num_features < 1:
        raise ValueError("RA-DS-PFD Crossformer requires positive node and feature counts")
    if data_info.lookback < 1 or data_info.max_pred_len < 1:
        raise ValueError("RA-DS-PFD Crossformer requires positive lookback and horizon")
    if len(data_info.feature_columns) != data_info.num_features:
        raise ValueError("RA-DS-PFD Crossformer feature_columns must match num_features")
    if not data_info.feature_columns or not data_info.input_power_column:
        raise ValueError("RA-DS-PFD Crossformer requires feature and input power metadata")
    if not 0 <= data_info.input_power_index < data_info.num_features:
        raise ValueError("RA-DS-PFD Crossformer requires a valid input_power_index")
    if data_info.feature_columns[data_info.input_power_index] != data_info.input_power_column:
        raise ValueError("RA-DS-PFD Crossformer input_power_index does not match input_power_column")


def build_model(model_config: dict[str, Any], data_info: DataInfoView) -> RADSPFDCrossformer:
    _validate_config(model_config)
    _validate_data_info(data_info)
    project_root = (
        Path(data_info.project_root).resolve()
        if data_info.project_root is not None
        else Path(__file__).resolve().parents[3]
    )
    source_root = project_root / "Time-Series-Library"
    if not source_root.is_dir():
        raise FileNotFoundError(
            f"RA-DS-PFD Crossformer requires Time-Series-Library source root: {source_root}"
        )
    relation_resource = None
    wspd_index = None
    if not model_config["spatial_disabled"]:
        if len(data_info.node_ids) != data_info.num_nodes:
            raise ValueError("enabled RA-DS-PFD relation spatial requires public data node_ids")
        if len(set(data_info.node_ids)) != len(data_info.node_ids):
            raise ValueError("enabled RA-DS-PFD relation spatial requires unique public node_ids")
        if "Wspd" not in data_info.feature_columns:
            raise ValueError("enabled PFD0 requires a Wspd feature column")
        wspd_index = data_info.feature_columns.index("Wspd")
        relation_resource = load_relation_resource(
            model_config["relation_resource"],
            project_root=project_root,
            node_ids=data_info.node_ids,
        )
    return RADSPFDCrossformer(
        num_nodes=data_info.num_nodes,
        input_dim=data_info.num_features,
        lookback=data_info.lookback,
        horizon=data_info.max_pred_len,
        input_power_index=data_info.input_power_index,
        model_config=model_config,
        source_root=source_root,
        wspd_index=wspd_index,
        relation_resource=relation_resource,
    )

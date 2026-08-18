"""Node-shared RA-DS-PFD adapter around the local canonical backbone."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from pathlib import Path
from typing import Any

import torch

from models.base import DataInfoView, ForecastModel, ModelInput, NodeSharedForecastModel

from .backbone import CanonicalBackbone, CanonicalTrace
from .pfd0 import build_pfd0_propagation
from .p3_feature_bank import validate_p3_model_config
from .p3_propagation import P3GlobalTopKPropagation
from .relation_resource import RelationResource, load_relation_resource
from .relation_spatial import (
    RelationBiasProvider,
    RelationSpatialInsertion,
    TurbineIdentityEmbedding,
)


CONFIG_FIELDS = frozenset(
    {
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
        "spatial_query_mode",
        "propagation_encoder_mode",
        "turbine_embedding_mode",
        "bias_scaling_mode",
        "base_turbine_dim",
        "p3",
    }
)
CANONICAL_CONFIG_FIELDS = frozenset(
    {
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
)
_POSITIVE_INTEGER_FIELDS = {"d_model", "n_heads", "d_ff", "e_layers", "factor", "seg_len", "win_size"}
_P2_POSITIVE_INTEGER_FIELDS = {"spatial_heads", "spatial_d_ff", "relation_dim", "base_turbine_dim"}
SPATIAL_QUERY_MODES = frozenset({"per_variable", "node_pooled"})
PROPAGATION_ENCODER_MODES = frozenset({"segment_fusion", "cross_time_then_fusion"})
TURBINE_EMBEDDING_MODES = frozenset({"relation_only", "temporal_and_relation"})
BIAS_SCALING_MODES = frozenset({"direct", "learnable_per_scale"})

# Keep the existing private names local to this module while exposing the
# formal architecture domains to the R0-R7 suite resolver as one source.
_CONFIG_FIELDS = CONFIG_FIELDS
_CANONICAL_CONFIG_FIELDS = CANONICAL_CONFIG_FIELDS
_SPATIAL_QUERY_MODES = SPATIAL_QUERY_MODES
_PROPAGATION_ENCODER_MODES = PROPAGATION_ENCODER_MODES
_TURBINE_EMBEDDING_MODES = TURBINE_EMBEDDING_MODES
_BIAS_SCALING_MODES = BIAS_SCALING_MODES


class _RADSPFDCrossformerImplementation:
    """Shared RA-DS-PFD construction and canonical forward implementation.

    The concrete adapters below own only the execution seam: P1 exposes one
    node range through ``NodeSharedForecastModel`` while P2 keeps one complete
    spatiotemporal forward.  Modules stay directly on the adapter so existing
    ``backbone.*`` checkpoint keys remain unchanged.
    """

    def _init_shared(
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
        feature_columns: tuple[str, ...] | None = None,
        relation_resource: RelationResource | None = None,
    ) -> None:
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
            self.pfd0 = None
            self.p3_propagation = None
            self.relation_bias_provider: RelationBiasProvider | None = None
            self.turbine_identity = None
            spatial_modules = None
        else:
            if self.wspd_index is None:
                raise ValueError("enabled PFD0 requires Wspd resolved from DataInfoView.feature_columns")
            if relation_resource is None:
                raise ValueError("enabled relation spatial path requires a validated relation resource")
            d_model = int(model_config["d_model"])
            spatial_query_mode = str(model_config["spatial_query_mode"])
            propagation_encoder_mode = str(model_config["propagation_encoder_mode"])
            turbine_embedding_mode = str(model_config["turbine_embedding_mode"])
            bias_scaling_mode = str(model_config["bias_scaling_mode"])
            if turbine_embedding_mode == "temporal_and_relation":
                self.turbine_identity = TurbineIdentityEmbedding(
                    self.num_nodes,
                    int(model_config.get("base_turbine_dim", 16)),
                    d_model,
                    int(model_config["relation_dim"]),
                )
            else:
                self.turbine_identity = None
            self.relation_bias_provider = RelationBiasProvider(
                edge_index=relation_resource.edge_index,
                edge_static_features=relation_resource.edge_static_features,
                num_nodes=self.num_nodes,
                spatial_heads=int(model_config["spatial_heads"]),
                spatial_d_ff=int(model_config["spatial_d_ff"]),
                relation_dim=int(model_config["relation_dim"]),
                turbine_embedding_mode=turbine_embedding_mode,
                bias_scaling_mode=bias_scaling_mode,
                turbine_identity=self.turbine_identity,
            )
            pfd_mode = str(model_config.get("pfd_mode", "pfd0"))
            if pfd_mode == "pfd0":
                self.pfd0 = build_pfd0_propagation(
                    propagation_encoder_mode,
                    lookback=self.lookback,
                    seg_len=int(model_config["seg_len"]),
                    win_size=int(model_config["win_size"]),
                    d_model=d_model,
                    dropout=float(model_config["spatial_dropout"]),
                    wspd_index=self.wspd_index,
                    n_heads=int(model_config["n_heads"]),
                    d_ff=int(model_config["d_ff"]),
                    factor=int(model_config["factor"]),
                    source_root=self.source_root,
                )
                self.p3_propagation = None
            elif pfd_mode == "pfd3_global_topk":
                if feature_columns is None:
                    raise ValueError(
                        "P3 propagation requires DataInfoView.feature_columns"
                    )
                self.pfd0 = None
                p3_config = model_config["p3"]
                self.p3_propagation = P3GlobalTopKPropagation(
                    feature_columns=feature_columns,
                    candidate_features=p3_config["candidate_features"],
                    candidate_transforms=p3_config["candidate_transforms"],
                    top_k=int(p3_config["top_k"]),
                    lookback=self.lookback,
                    seg_len=int(model_config["seg_len"]),
                    win_size=int(model_config["win_size"]),
                    d_model=d_model,
                    n_heads=int(model_config["n_heads"]),
                    d_ff=int(model_config["d_ff"]),
                    factor=int(model_config["factor"]),
                    dropout=float(model_config["dropout"]),
                    source_root=self.source_root,
                )
            else:
                raise ValueError(f"unsupported RA-DS-PFD propagation mode: {pfd_mode}")
            spatial_modules = (
                RelationSpatialInsertion(
                    d_model=d_model,
                    spatial_heads=int(model_config["spatial_heads"]),
                    spatial_dropout=float(model_config["spatial_dropout"]),
                    gamma_init=float(model_config["gamma_init"]),
                    bias_provider=self.relation_bias_provider,
                    scale_id=0,
                    spatial_query_mode=spatial_query_mode,
                    edge_chunk_size=model_config.get("spatial_edge_chunk_size", 128),
                ),
                RelationSpatialInsertion(
                    d_model=d_model,
                    spatial_heads=int(model_config["spatial_heads"]),
                    spatial_dropout=float(model_config["spatial_dropout"]),
                    gamma_init=float(model_config["gamma_init"]),
                    bias_provider=self.relation_bias_provider,
                    scale_id=1,
                    spatial_query_mode=spatial_query_mode,
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
            turbine_identity=self.turbine_identity,
        )

    def _node_history(
        self,
        inputs: ModelInput,
        node_start: int,
        node_end: int,
    ) -> tuple[torch.Tensor, int, int]:
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
        if not 0 <= int(node_start) < int(node_end) <= nodes:
            raise ValueError(
                "RA-DS-PFD Crossformer node range must satisfy "
                f"0 <= start < end <= {nodes}; got ({node_start}, {node_end})"
            )
        if not torch.isfinite(x).all():
            raise FloatingPointError("RA-DS-PFD Crossformer input contains NaN or Inf")
        node_x = x[:, :, int(node_start) : int(node_end), :]
        local_nodes = int(node_end) - int(node_start)
        node_history = node_x.permute(0, 2, 1, 3).reshape(
            batch * local_nodes, steps, channels
        )
        return node_history, batch, local_nodes

    def _forward_canonical_range(
        self,
        inputs: ModelInput,
        node_start: int,
        node_end: int,
        *,
        propagation_tokens: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        node_history, batch, nodes = self._node_history(inputs, node_start, node_end)
        if propagation_tokens is None:
            full_output = self.backbone(node_history)
        else:
            full_output = self.backbone(
                node_history,
                propagation_tokens=propagation_tokens,
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

    def _propagation_tokens(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.pfd0 is not None:
            return self.pfd0(x)
        if self.p3_propagation is not None:
            return self.p3_propagation(x)
        return None

    def forward_canonical_trace(self, inputs: ModelInput) -> CanonicalTrace:
        """Return canonical stages, using local ``[B,N,C,S,D]`` in P2 mode."""

        node_history, _, _ = self._node_history(inputs, 0, self.num_nodes)
        propagation_tokens = self._propagation_tokens(inputs.x)
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

    def selection_report(self) -> list[dict[str, Any]]:
        """Return the read-only P3 ranking report, or an empty legacy report."""

        if self.p3_propagation is None:
            return []
        return self.p3_propagation.selection_report()

    def propagation_selection_report(self) -> list[dict[str, Any]]:
        """Explicit alias for callers interested only in propagation content."""

        return self.selection_report()


class RADSPFDCrossformerP1(_RADSPFDCrossformerImplementation, NodeSharedForecastModel):
    """P1 canonical Crossformer adapter with public node micro-batching."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self._init_shared(**kwargs)

    def forward_node_chunk(
        self,
        inputs: ModelInput,
        node_start: int,
        node_end: int,
    ) -> torch.Tensor:
        return self._forward_canonical_range(inputs, node_start, node_end)


class RADSPFDCrossformerP2(_RADSPFDCrossformerImplementation, ForecastModel):
    """P2 relation-spatial adapter with one complete-node forward."""

    execution_mode = "full_spatiotemporal"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self._init_shared(**kwargs)

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        if self.pfd0 is None:
            raise RuntimeError("RA-DS-PFD P2 requires the enabled spatial path")
        return self._forward_canonical_range(
            inputs,
            0,
            self.num_nodes,
            propagation_tokens=self._propagation_tokens(inputs.x),
        )


class RADSPFDCrossformerP3(_RADSPFDCrossformerImplementation, ForecastModel):
    """P3-A full spatiotemporal adapter with propagation-only selection."""

    execution_mode = "full_spatiotemporal"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self._init_shared(**kwargs)

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        if self.p3_propagation is None or self.pfd0 is not None:
            raise RuntimeError("RA-DS-PFD P3 requires the P3 propagation path")
        return self._forward_canonical_range(
            inputs,
            0,
            self.num_nodes,
            propagation_tokens=self._propagation_tokens(inputs.x),
        )


# Keep the historical P1 class name importable for callers that used the
# concrete adapter directly; build_model selects the explicit P1/P2 adapter.
RADSPFDCrossformer = RADSPFDCrossformerP1


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
    if "pfd_mode" in model_config and model_config["pfd_mode"] not in {
        "pfd0",
        "pfd3_global_topk",
    }:
        raise ValueError(
            "unsupported RA-DS-PFD Crossformer pfd_mode; "
            "expected pfd_mode=pfd0 or pfd_mode=pfd3_global_topk"
        )
    if model_config["spatial_disabled"]:
        if model_config.get("pfd_mode") == "pfd3_global_topk":
            raise ValueError("P3 propagation requires spatial_disabled=false")
        if "p3" in model_config:
            raise ValueError("P3 model config cannot be attached to the legacy P1 path")
        for field, allowed in (
            ("spatial_query_mode", _SPATIAL_QUERY_MODES),
            ("propagation_encoder_mode", _PROPAGATION_ENCODER_MODES),
            ("turbine_embedding_mode", _TURBINE_EMBEDDING_MODES),
            ("bias_scaling_mode", _BIAS_SCALING_MODES),
        ):
            if field in model_config and model_config[field] not in allowed:
                raise ValueError(f"RA-DS-PFD Crossformer {field} has unknown value")
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
    if model_config["pfd_mode"] == "pfd0" and "p3" in model_config:
        raise ValueError("pfd0 model config must not define P3 propagation fields")
    if model_config["pfd_mode"] == "pfd3_global_topk":
        if "p3" not in model_config:
            raise ValueError("P3 model config is missing field: p3")
        validate_p3_model_config(model_config["p3"])
    mode_fields = {
        "spatial_query_mode": _SPATIAL_QUERY_MODES,
        "propagation_encoder_mode": _PROPAGATION_ENCODER_MODES,
        "turbine_embedding_mode": _TURBINE_EMBEDDING_MODES,
        "bias_scaling_mode": _BIAS_SCALING_MODES,
    }
    provided_modes = set(model_config).intersection(mode_fields)
    if provided_modes != set(mode_fields):
        missing_modes = sorted(set(mode_fields) - provided_modes)
        raise ValueError(
            "RA-DS-PFD Crossformer P2 config is missing architecture mode field: "
            f"{missing_modes[0]}"
        )
    for field, allowed in mode_fields.items():
        if field in model_config and model_config[field] not in allowed:
            raise ValueError(f"RA-DS-PFD Crossformer {field} has unknown value")
    for name in sorted(_P2_POSITIVE_INTEGER_FIELDS):
        if name == "base_turbine_dim" and name not in model_config:
            continue
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


def build_model(
    model_config: dict[str, Any], data_info: DataInfoView
) -> RADSPFDCrossformerP1 | RADSPFDCrossformerP2 | RADSPFDCrossformerP3:
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
    adapter = (
        RADSPFDCrossformerP1
        if model_config["spatial_disabled"]
        else (
            RADSPFDCrossformerP3
            if model_config.get("pfd_mode") == "pfd3_global_topk"
            else RADSPFDCrossformerP2
        )
    )
    return adapter(
        num_nodes=data_info.num_nodes,
        input_dim=data_info.num_features,
        lookback=data_info.lookback,
        horizon=data_info.max_pred_len,
        input_power_index=data_info.input_power_index,
        model_config=model_config,
        source_root=source_root,
        wspd_index=wspd_index,
        feature_columns=tuple(data_info.feature_columns),
        relation_resource=relation_resource,
    )

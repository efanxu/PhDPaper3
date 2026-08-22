"""IA-1.1 selected-only temporal closure modules.

IA-1.1 keeps the canonical candidate bank and the fixed two-candidate seam,
but makes the temporal experiment explicit.  The independent arm mirrors the
frozen R2 temporal path for the selected pair.  The adapter arm keeps one
shared Cross-Time module per scale and adds only operator-specific residual
adapters for the two registered history operators.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import ceil, isfinite
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .p3_feature_bank import (
    P3_BASE_FEATURES,
    P3_CANDIDATE_TRANSFORMS,
    P3CandidateBank,
)
from .p3_ia_propagation import (
    parse_candidate_name,
    resolve_fixed_candidate_indices,
    validate_selected_candidates,
)
from .pfd0 import (
    CanonicalCrossTime,
    PFD0SegmentMerging,
)


IA11_SELECTION_MODE = "fixed"
IA11_SELECTED_CANDIDATES = ("Wspd.level", "Wspd.diff1")
IA11_TEMPORAL_ENCODER_MODES = frozenset(
    {"independent_cross_time", "operator_adapter_shared_cross_time"}
)
IA11_OPERATOR_TYPES = tuple(P3_CANDIDATE_TRANSFORMS)
P3_IA11_MODEL_CONFIG_FIELDS = frozenset(
    {"selection_mode", "selected_candidates", "temporal_encoder_mode"}
)


def validate_p3_ia_temporal_model_config(value: Any) -> dict[str, Any]:
    """Validate the model-owned IA-1.1 fixed temporal configuration."""

    if not isinstance(value, Mapping):
        raise ValueError("RA-DS-PFD IA-1.1 p3_ia_temporal must be a mapping")
    unknown = sorted(set(value) - P3_IA11_MODEL_CONFIG_FIELDS)
    missing = sorted(P3_IA11_MODEL_CONFIG_FIELDS - set(value))
    if unknown:
        raise ValueError(
            "RA-DS-PFD IA-1.1 p3_ia_temporal has unsupported field: "
            f"{unknown[0]}"
        )
    if missing:
        raise ValueError(
            "RA-DS-PFD IA-1.1 p3_ia_temporal is missing field: " f"{missing[0]}"
        )
    if value["selection_mode"] != IA11_SELECTION_MODE:
        raise ValueError("RA-DS-PFD IA-1.1 selection_mode must be fixed")
    selected = validate_selected_candidates(value["selected_candidates"])
    if selected != IA11_SELECTED_CANDIDATES:
        raise ValueError(
            "RA-DS-PFD IA-1.1 must use the fixed Wspd.level/Wspd.diff1 candidate pair"
        )
    temporal_mode = value["temporal_encoder_mode"]
    if temporal_mode not in IA11_TEMPORAL_ENCODER_MODES:
        raise ValueError(
            "RA-DS-PFD IA-1.1 temporal_encoder_mode has an unsupported value: "
            f"{temporal_mode}"
        )
    return dict(value)


class SemanticCandidateIdentity(nn.Module):
    """Encode canonical base-variable and operator identity, never slot order."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = int(d_model)
        if self.d_model < 1:
            raise ValueError("semantic candidate identity d_model must be positive")
        self.base_variable_names = tuple(P3_BASE_FEATURES)
        self.operator_names = tuple(IA11_OPERATOR_TYPES)
        self._base_indices = {
            name: index for index, name in enumerate(self.base_variable_names)
        }
        self._operator_indices = {
            name: index for index, name in enumerate(self.operator_names)
        }
        self.base_variable_embedding = nn.Embedding(
            len(self.base_variable_names), self.d_model
        )
        self.operator_embedding = nn.Embedding(len(self.operator_names), self.d_model)

    def forward(self, candidate_names: Sequence[str]) -> torch.Tensor:
        names = tuple(candidate_names)
        if isinstance(candidate_names, (str, bytes)) or not names:
            raise ValueError("semantic candidate identity requires candidate names")
        base_indices: list[int] = []
        operator_indices: list[int] = []
        for name in names:
            feature, operator = parse_candidate_name(name)
            try:
                base_indices.append(self._base_indices[feature])
                operator_indices.append(self._operator_indices[operator])
            except KeyError as exc:
                raise ValueError(f"unsupported semantic candidate identity: {name}") from exc
        device = self.base_variable_embedding.weight.device
        base = self.base_variable_embedding(
            torch.tensor(base_indices, dtype=torch.long, device=device)
        )
        operator = self.operator_embedding(
            torch.tensor(operator_indices, dtype=torch.long, device=device)
        )
        result = base + operator
        if not torch.isfinite(result).all():
            raise FloatingPointError("semantic candidate identity contains NaN or Inf")
        return result

    def identity_for(self, candidate_name: str) -> torch.Tensor:
        """Return one identity vector through the same semantic lookup path."""

        return self((candidate_name,))[0]


class OperatorResidualAdapter(nn.Module):
    """Residual bottleneck adapter used by one semantic temporal operator."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = int(d_model)
        if self.d_model < 1:
            raise ValueError("operator adapter d_model must be positive")
        self.adapter_dim = max(1, self.d_model // 4)
        self.norm = nn.LayerNorm(self.d_model)
        self.down = nn.Linear(self.d_model, self.adapter_dim)
        self.up = nn.Linear(self.adapter_dim, self.d_model)
        self.gamma = nn.Parameter(torch.zeros(()))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if not isinstance(h, torch.Tensor) or h.shape[-1] != self.d_model:
            raise ValueError(
                "operator residual adapter expects a tensor whose last dimension "
                f"is {self.d_model}"
            )
        correction = self.up(torch.nn.functional.gelu(self.down(self.norm(h))))
        result = h + self.gamma * correction
        if not torch.isfinite(result).all():
            raise FloatingPointError("operator residual adapter contains NaN or Inf")
        return result


class _IA11SelectedOnlyBase(nn.Module):
    """Shared selected-only candidate boundary for the two IA-1.1 arms."""

    def __init__(
        self,
        *,
        feature_columns: Sequence[str] | Any,
        selected_candidates: Sequence[str],
        lookback: int,
        seg_len: int,
        win_size: int,
        d_model: int,
        spatial_dropout: float | None,
        dropout: float | None,
    ) -> None:
        super().__init__()
        self.lookback = int(lookback)
        self.seg_len = int(seg_len)
        self.win_size = int(win_size)
        self.d_model = int(d_model)
        if self.lookback < 1 or self.seg_len < 1 or self.win_size < 1 or self.d_model < 1:
            raise ValueError("IA-1.1 temporal dimensions must be positive")
        self.scale0_segments = ceil(self.lookback / self.seg_len)
        self.scale1_segments = ceil(self.scale0_segments / self.win_size)
        if spatial_dropout is None:
            if dropout is None:
                raise ValueError("IA-1.1 temporal propagation requires spatial_dropout")
            spatial_dropout = dropout
        elif dropout is not None and float(spatial_dropout) != float(dropout):
            raise ValueError("IA-1.1 temporal propagation dropout aliases disagree")
        self.spatial_dropout = float(spatial_dropout)
        if not isfinite(self.spatial_dropout) or not 0.0 <= self.spatial_dropout < 1.0:
            raise ValueError("IA-1.1 temporal spatial_dropout must be finite and in [0, 1)")

        self.candidate_bank = P3CandidateBank(
            feature_columns,
            candidate_features=P3_BASE_FEATURES,
            candidate_transforms=P3_CANDIDATE_TRANSFORMS,
        )
        self.candidate_names = self.candidate_bank.candidate_names
        self.candidate_count = self.candidate_bank.candidate_count
        self.selected_candidate_names = validate_selected_candidates(
            selected_candidates,
            candidate_names=self.candidate_names,
        )
        self.selected_candidate_indices = resolve_fixed_candidate_indices(
            self.selected_candidate_names,
            candidate_names=self.candidate_names,
        )
        self.effective_candidate_count = len(self.selected_candidate_names)
        self.selected_candidate_count = self.effective_candidate_count
        self.candidate_identity = SemanticCandidateIdentity(self.d_model)
        self.dropout = nn.Dropout(self.spatial_dropout)
        self.cross_time_candidate_counts: tuple[int, ...] = ()
        self.execution_trace: dict[str, Any] = {}

    def candidate_history(self, x: torch.Tensor) -> torch.Tensor:
        """Construct the full cheap bank, then return only the selected K streams."""

        full_history = self.candidate_bank(x)
        indices = torch.tensor(
            self.selected_candidate_indices,
            dtype=torch.long,
            device=full_history.device,
        )
        selected = full_history.index_select(-1, indices)
        if selected.shape[-1] != self.effective_candidate_count:
            raise AssertionError("IA-1.1 selected candidate history contract drifted")
        return selected

    def _selected_segments(self, candidates: torch.Tensor) -> torch.Tensor:
        if candidates.ndim != 4:
            raise ValueError("IA-1.1 candidate projection expects (B,L,N,K)")
        batch, length, nodes, candidate_count = candidates.shape
        if candidate_count != self.effective_candidate_count:
            raise ValueError("IA-1.1 candidate projection received an unexpected selected count")
        if ceil(length / self.seg_len) != self.scale0_segments:
            raise ValueError("IA-1.1 candidate segment count changed from configured lookback")
        pad = self.scale0_segments * self.seg_len - length
        if pad:
            candidates = torch.cat(
                (candidates.new_zeros(batch, pad, nodes, candidate_count), candidates),
                dim=1,
            )
        return candidates.permute(0, 2, 3, 1).reshape(
            batch,
            nodes,
            candidate_count,
            self.scale0_segments,
            self.seg_len,
        )

    def _build_fusion(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(self.effective_candidate_count * self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )

    def _fuse(self, tokens: torch.Tensor, fusion: nn.Module) -> torch.Tensor:
        batch, nodes, candidates, segments, d_model = tokens.shape
        if candidates != self.effective_candidate_count or d_model != self.d_model:
            raise ValueError("IA-1.1 fusion received an unexpected selected candidate shape")
        inputs = tokens.permute(0, 1, 3, 2, 4).reshape(
            batch,
            nodes,
            segments,
            candidates * d_model,
        )
        return fusion(inputs)

    def _finish_forward(
        self,
        x: torch.Tensor,
        scale0_candidates: torch.Tensor,
        scale1_candidates: torch.Tensor,
        *,
        temporal_path_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale0 = self._fuse(scale0_candidates, self.scale0_fusion)
        scale1 = self._fuse(scale1_candidates, self.scale1_fusion)
        self.execution_trace.update(
            {
                "candidate_bank_count": self.candidate_count,
                "effective_candidate_count": self.effective_candidate_count,
                "temporal_path_candidate_count": int(temporal_path_count),
                "scale0_cross_time_candidate_count": self.effective_candidate_count,
                "scale1_cross_time_candidate_count": self.effective_candidate_count,
            }
        )
        if tuple(scale0.shape) != (
            x.shape[0],
            x.shape[2],
            self.scale0_segments,
            self.d_model,
        ):
            raise AssertionError("IA-1.1 Scale0 propagation contract drifted")
        if tuple(scale1.shape) != (
            x.shape[0],
            x.shape[2],
            self.scale1_segments,
            self.d_model,
        ):
            raise AssertionError("IA-1.1 Scale1 propagation contract drifted")
        if not torch.isfinite(scale0).all() or not torch.isfinite(scale1).all():
            raise FloatingPointError("IA-1.1 propagation output contains NaN or Inf")
        return scale0, scale1

    def fixed_selection_report(self) -> list[dict[str, Any]]:
        selected = set(self.selected_candidate_indices)
        return [
            {
                "candidate_name": name,
                "bank_index": index,
                "selected": index in selected,
            }
            for index, name in enumerate(self.candidate_names)
        ]


class IAIndependentTemporalPropagation(_IA11SelectedOnlyBase):
    """IA-1.1 diagnostic arm with independent Cross-Time per selected stream."""

    def __init__(
        self,
        *,
        feature_columns: Sequence[str] | Any,
        selected_candidates: Sequence[str],
        lookback: int,
        seg_len: int,
        win_size: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        factor: int,
        spatial_dropout: float | None = None,
        source_root: Path | None = None,
        dropout: float | None = None,
    ) -> None:
        super().__init__(
            feature_columns=feature_columns,
            selected_candidates=selected_candidates,
            lookback=lookback,
            seg_len=seg_len,
            win_size=win_size,
            d_model=d_model,
            spatial_dropout=spatial_dropout,
            dropout=dropout,
        )
        root = (
            Path(source_root).resolve()
            if source_root is not None
            else Path(__file__).resolve().parents[3] / "Time-Series-Library"
        )
        # Keep the R2 candidate-specific value and position path explicit.
        # The selected-only seam owns exactly K projections and K temporal
        # streams; no unused candidate projection is constructed.
        self.candidate_projections = nn.ModuleList(
            [
                nn.Linear(self.seg_len, self.d_model, bias=False)
                for _ in range(self.effective_candidate_count)
            ]
        )
        self.candidate_position_embeddings = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(1, 1, self.scale0_segments, self.d_model) * 0.02
                )
                for _ in range(self.effective_candidate_count)
            ]
        )
        self.scale0_cross_time = nn.ModuleList(
            [
                CanonicalCrossTime(
                    source_root=root,
                    d_model=self.d_model,
                    n_heads=int(n_heads),
                    d_ff=int(d_ff),
                    factor=int(factor),
                    dropout=self.spatial_dropout,
                )
                for _ in range(self.effective_candidate_count)
            ]
        )
        self.scale1_merging = PFD0SegmentMerging(self.d_model, self.win_size)
        self.scale1_cross_time = nn.ModuleList(
            [
                CanonicalCrossTime(
                    source_root=root,
                    d_model=self.d_model,
                    n_heads=int(n_heads),
                    d_ff=int(d_ff),
                    factor=int(factor),
                    dropout=self.spatial_dropout,
                )
                for _ in range(self.effective_candidate_count)
            ]
        )
        self.scale0_fusion = self._build_fusion()
        self.scale1_fusion = self._build_fusion()
        self.execution_trace["temporal_encoder_mode"] = "independent_cross_time"

    def _embed_selected(self, candidates: torch.Tensor) -> torch.Tensor:
        segments = self._selected_segments(candidates)
        identities = self.candidate_identity(self.selected_candidate_names).to(
            device=candidates.device,
            dtype=candidates.dtype,
        )
        embedded: list[torch.Tensor] = []
        for index, projection in enumerate(self.candidate_projections):
            stream = projection(segments[:, :, index])
            stream = stream + self.candidate_position_embeddings[index]
            stream = self.dropout(stream)
            stream = stream + identities[index].view(1, 1, 1, self.d_model)
            embedded.append(stream)
        return torch.stack(embedded, dim=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        candidates = self.candidate_history(x)
        embedded = self._embed_selected(candidates)
        scale0_streams: list[torch.Tensor] = []
        scale0_names: list[str] = []
        for index, encoder in enumerate(self.scale0_cross_time):
            name = self.selected_candidate_names[index]
            scale0_streams.append(encoder(embedded[:, :, index]))
            scale0_names.append(name)
        scale0_candidates = torch.stack(scale0_streams, dim=2)

        scale1_streams: list[torch.Tensor] = []
        scale1_names: list[str] = []
        for index, encoder in enumerate(self.scale1_cross_time):
            merged = self.scale1_merging(scale0_candidates[:, :, index])
            scale1_streams.append(encoder(merged))
            scale1_names.append(self.selected_candidate_names[index])
        scale1_candidates = torch.stack(scale1_streams, dim=2)
        self.cross_time_candidate_counts = (
            self.effective_candidate_count,
            self.effective_candidate_count,
        )
        self.execution_trace.update(
            {
                "scale0_cross_time_candidate_names": tuple(scale0_names),
                "scale1_cross_time_candidate_names": tuple(scale1_names),
                "scale0_cross_time_module_count": len(self.scale0_cross_time),
                "scale1_cross_time_module_count": len(self.scale1_cross_time),
            }
        )
        return self._finish_forward(
            x,
            scale0_candidates,
            scale1_candidates,
            temporal_path_count=self.effective_candidate_count,
        )


class IAOperatorAdapterPropagation(_IA11SelectedOnlyBase):
    """IA-1.1 arm with two operator adapters and shared Cross-Time backbones."""

    def __init__(
        self,
        *,
        feature_columns: Sequence[str] | Any,
        selected_candidates: Sequence[str],
        lookback: int,
        seg_len: int,
        win_size: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        factor: int,
        spatial_dropout: float | None = None,
        source_root: Path | None = None,
        dropout: float | None = None,
    ) -> None:
        super().__init__(
            feature_columns=feature_columns,
            selected_candidates=selected_candidates,
            lookback=lookback,
            seg_len=seg_len,
            win_size=win_size,
            d_model=d_model,
            spatial_dropout=spatial_dropout,
            dropout=dropout,
        )
        root = (
            Path(source_root).resolve()
            if source_root is not None
            else Path(__file__).resolve().parents[3] / "Time-Series-Library"
        )
        self.candidate_projections = nn.ModuleList(
            [nn.Linear(self.seg_len, self.d_model, bias=False) for _ in range(self.effective_candidate_count)]
        )
        self.position_embedding = nn.Parameter(
            torch.randn(1, 1, 1, self.scale0_segments, self.d_model) * 0.02
        )
        self.scale0_operator_adapters = nn.ModuleDict(
            {operator: OperatorResidualAdapter(self.d_model) for operator in IA11_OPERATOR_TYPES}
        )
        self.scale1_operator_adapters = nn.ModuleDict(
            {operator: OperatorResidualAdapter(self.d_model) for operator in IA11_OPERATOR_TYPES}
        )
        self.scale0_cross_time = CanonicalCrossTime(
            source_root=root,
            d_model=self.d_model,
            n_heads=int(n_heads),
            d_ff=int(d_ff),
            factor=int(factor),
            dropout=self.spatial_dropout,
        )
        self.scale1_merging = PFD0SegmentMerging(self.d_model, self.win_size)
        self.scale1_cross_time = CanonicalCrossTime(
            source_root=root,
            d_model=self.d_model,
            n_heads=int(n_heads),
            d_ff=int(d_ff),
            factor=int(factor),
            dropout=self.spatial_dropout,
        )
        self.scale0_fusion = self._build_fusion()
        self.scale1_fusion = self._build_fusion()
        self.execution_trace["temporal_encoder_mode"] = "operator_adapter_shared_cross_time"

    def _project_selected(self, candidates: torch.Tensor) -> torch.Tensor:
        segments = self._selected_segments(candidates)
        projected = torch.stack(
            [
                projection(segments[:, :, index])
                for index, projection in enumerate(self.candidate_projections)
            ],
            dim=2,
        )
        identities = self.candidate_identity(self.selected_candidate_names).to(
            device=projected.device,
            dtype=projected.dtype,
        )
        identity = identities.view(1, 1, self.effective_candidate_count, 1, self.d_model)
        return self.dropout(projected + self.position_embedding + identity)

    def _apply_operator_adapters(
        self,
        tokens: torch.Tensor,
        adapters: nn.ModuleDict,
    ) -> torch.Tensor:
        if tokens.ndim != 5:
            raise ValueError("IA-1.1 operator adapters expect (B,N,K,S,D)")
        outputs: list[torch.Tensor] = []
        for index, name in enumerate(self.selected_candidate_names):
            _, operator = parse_candidate_name(name)
            outputs.append(adapters[operator](tokens[:, :, index]))
        return torch.stack(outputs, dim=2)

    @staticmethod
    def _shared_cross_time(
        encoder: CanonicalCrossTime,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        if tokens.ndim != 5:
            raise ValueError("IA-1.1 shared Cross-Time expects (B,N,K,S,D)")
        batch, nodes, candidates, segments, d_model = tokens.shape
        folded = tokens.permute(0, 2, 1, 3, 4).reshape(
            batch * candidates,
            nodes,
            segments,
            d_model,
        )
        encoded = encoder(folded)
        return encoded.reshape(batch, candidates, nodes, segments, d_model).permute(
            0, 2, 1, 3, 4
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        candidates = self.candidate_history(x)
        projected = self._project_selected(candidates)
        scale0_adapted = self._apply_operator_adapters(
            projected,
            self.scale0_operator_adapters,
        )
        scale0_candidates = self._shared_cross_time(self.scale0_cross_time, scale0_adapted)
        scale1_merged = self.scale1_merging(
            scale0_candidates.reshape(
                x.shape[0],
                x.shape[2] * self.effective_candidate_count,
                self.scale0_segments,
                self.d_model,
            )
        ).reshape(
            x.shape[0],
            x.shape[2],
            self.effective_candidate_count,
            self.scale1_segments,
            self.d_model,
        )
        scale1_adapted = self._apply_operator_adapters(
            scale1_merged,
            self.scale1_operator_adapters,
        )
        scale1_candidates = self._shared_cross_time(self.scale1_cross_time, scale1_adapted)
        self.cross_time_candidate_counts = (
            self.effective_candidate_count,
            self.effective_candidate_count,
        )
        self.execution_trace.update(
            {
                "scale0_cross_time_candidate_names": self.selected_candidate_names,
                "scale1_cross_time_candidate_names": self.selected_candidate_names,
                "scale0_cross_time_module_count": 1,
                "scale1_cross_time_module_count": 1,
                "operator_types": IA11_OPERATOR_TYPES,
            }
        )
        return self._finish_forward(
            x,
            scale0_candidates,
            scale1_candidates,
            temporal_path_count=self.effective_candidate_count,
        )


__all__ = [
    "IA11_OPERATOR_TYPES",
    "IA11_SELECTED_CANDIDATES",
    "IA11_SELECTION_MODE",
    "IA11_TEMPORAL_ENCODER_MODES",
    "IAIndependentTemporalPropagation",
    "IAOperatorAdapterPropagation",
    "OperatorResidualAdapter",
    "P3_IA11_MODEL_CONFIG_FIELDS",
    "SemanticCandidateIdentity",
    "validate_p3_ia_temporal_model_config",
]

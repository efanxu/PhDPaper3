"""Fixed-candidate, selected-only propagation primitives for IA-1."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import ceil, isfinite
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .pfd0 import CanonicalCrossTime, PFD0SegmentMerging
from .p3_feature_bank import (
    P3_BASE_FEATURES,
    P3_CANDIDATE_TRANSFORMS,
    P3CandidateBank,
    TEMPORAL_OPERATORS,
)


IA_SELECTION_MODE = "fixed"
P3_IA_MODEL_CONFIG_FIELDS = frozenset({"selection_mode", "selected_candidates"})


def canonical_candidate_names() -> tuple[str, ...]:
    """Return the candidate names in the existing P3 bank order."""

    return tuple(
        f"{feature}.{operator}"
        for feature in P3_BASE_FEATURES
        for operator in P3_CANDIDATE_TRANSFORMS
    )


def parse_candidate_name(value: Any) -> tuple[str, str]:
    """Parse one canonical ``<base_feature>.<operator>`` candidate name."""

    if not isinstance(value, str) or not value:
        raise ValueError("IA-1 candidate name must be a non-empty string")
    if value.count(".") != 1:
        raise ValueError(
            "IA-1 candidate name must use '<base_feature>.<operator>'"
        )
    feature, operator = value.split(".")
    if feature not in P3_BASE_FEATURES:
        raise ValueError(f"IA-1 candidate has unknown base feature: {feature}")
    if operator not in TEMPORAL_OPERATORS:
        raise ValueError(f"IA-1 candidate has unknown operator: {operator}")
    return feature, operator


def _ordered_strings(value: Any, *, field: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"IA-1 {field} must be a non-empty ordered list")
    result = tuple(value)
    if not result or any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"IA-1 {field} must contain non-empty strings")
    return result


def _duplicates(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return tuple(duplicates)


def validate_selected_candidates(
    value: Any,
    *,
    candidate_names: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Validate a fixed candidate set against the canonical P3 candidate bank."""

    selected = _ordered_strings(value, field="selected_candidates")
    bank_names = (
        canonical_candidate_names()
        if candidate_names is None
        else tuple(candidate_names)
    )
    if len(selected) > len(bank_names):
        raise ValueError(
            "IA-1 selected_candidates count exceeds the candidate bank: "
            f"{len(selected)} > {len(bank_names)}"
        )
    duplicate = _duplicates(selected)
    if duplicate:
        raise ValueError(f"IA-1 selected_candidates contains duplicate: {duplicate[0]}")
    for name in selected:
        parse_candidate_name(name)
        if name not in bank_names:
            raise ValueError(f"IA-1 selected candidate is not in the candidate bank: {name}")
    return selected


def resolve_fixed_candidate_indices(
    selected_candidates: Sequence[str],
    *,
    candidate_names: Sequence[str] | None = None,
) -> tuple[int, ...]:
    """Resolve fixed candidate names to stable indices in the P3 bank."""

    bank_names = (
        canonical_candidate_names()
        if candidate_names is None
        else tuple(candidate_names)
    )
    selected = validate_selected_candidates(selected_candidates, candidate_names=bank_names)
    index_by_name = {name: index for index, name in enumerate(bank_names)}
    return tuple(index_by_name[name] for name in selected)


def validate_p3_ia_model_config(value: Any) -> dict[str, Any]:
    """Validate the model-owned IA-1 fixed-selection mapping."""

    if not isinstance(value, Mapping):
        raise ValueError("RA-DS-PFD IA-1 p3_ia must be a mapping")
    unknown = sorted(set(value) - P3_IA_MODEL_CONFIG_FIELDS)
    missing = sorted(P3_IA_MODEL_CONFIG_FIELDS - set(value))
    if unknown:
        raise ValueError(f"RA-DS-PFD IA-1 p3_ia has unsupported field: {unknown[0]}")
    if missing:
        raise ValueError(f"RA-DS-PFD IA-1 p3_ia is missing field: {missing[0]}")
    if value["selection_mode"] != IA_SELECTION_MODE:
        raise ValueError("RA-DS-PFD IA-1 selection_mode must be fixed")
    validate_selected_candidates(value["selected_candidates"])
    return dict(value)


class IAFixedPropagation(nn.Module):
    """Encode only a fixed selected candidate set at both temporal scales.

    The complete P3 candidate bank is a cheap history construction boundary.
    It is indexed immediately afterward; projection, Scale0 Cross-Time,
    Scale1 merging, and Scale1 Cross-Time all operate on the resulting K-axis.
    """

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
        selection_mode: str = IA_SELECTION_MODE,
        dropout: float | None = None,
    ) -> None:
        super().__init__()
        if selection_mode != IA_SELECTION_MODE:
            raise ValueError("IA-1 propagation selection_mode must be fixed")
        self.selection_mode = selection_mode
        self.lookback = int(lookback)
        self.seg_len = int(seg_len)
        self.win_size = int(win_size)
        self.d_model = int(d_model)
        self.scale0_segments = ceil(self.lookback / self.seg_len)
        self.scale1_segments = ceil(self.scale0_segments / self.win_size)
        if self.lookback < 1 or self.seg_len < 1 or self.win_size < 1:
            raise ValueError("IA-1 propagation lookback, seg_len and win_size must be positive")
        if self.scale0_segments < 1 or self.scale1_segments < 1:
            raise ValueError("IA-1 propagation requires positive segment counts")
        if spatial_dropout is None:
            if dropout is None:
                raise ValueError("IA-1 propagation requires spatial_dropout")
            spatial_dropout = dropout
        elif dropout is not None and float(spatial_dropout) != float(dropout):
            raise ValueError("IA-1 propagation dropout aliases disagree")
        self.spatial_dropout = float(spatial_dropout)
        if not isfinite(self.spatial_dropout) or not 0.0 <= self.spatial_dropout < 1.0:
            raise ValueError("IA-1 propagation spatial_dropout must be finite and in [0, 1)")

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

        self.candidate_projections = nn.ModuleList(
            [
                nn.Linear(self.seg_len, self.d_model, bias=False)
                for _ in range(self.effective_candidate_count)
            ]
        )
        self.position_embedding = nn.Parameter(
            torch.randn(1, 1, 1, self.scale0_segments, self.d_model) * 0.02
        )
        self.candidate_identity = nn.Embedding(self.effective_candidate_count, self.d_model)
        self.dropout = nn.Dropout(self.spatial_dropout)

        root = (
            Path(source_root).resolve()
            if source_root is not None
            else Path(__file__).resolve().parents[3] / "Time-Series-Library"
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

        self.cross_time_candidate_counts: tuple[int, ...] = ()
        self.execution_trace: dict[str, Any] = {}
        self._current_cross_time_candidate_counts: list[int] = []

    def _build_fusion(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(self.effective_candidate_count * self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )

    def candidate_history(self, x: torch.Tensor) -> torch.Tensor:
        """Return only the selected ``[B,L,N,K]`` history streams."""

        full_history = self.candidate_bank(x)
        indices = torch.tensor(
            self.selected_candidate_indices,
            dtype=torch.long,
            device=full_history.device,
        )
        selected = full_history.index_select(-1, indices)
        if selected.shape[-1] != self.effective_candidate_count:
            raise AssertionError("IA-1 selected candidate history contract drifted")
        return selected

    def _project_selected(self, candidates: torch.Tensor) -> torch.Tensor:
        if candidates.ndim != 4:
            raise ValueError("IA-1 candidate projection expects (B,L,N,K)")
        batch, length, nodes, candidate_count = candidates.shape
        if candidate_count != self.effective_candidate_count:
            raise ValueError("IA-1 candidate projection received an unexpected selected count")
        if ceil(length / self.seg_len) != self.scale0_segments:
            raise ValueError("IA-1 candidate segment count changed from configured lookback")
        pad = self.scale0_segments * self.seg_len - length
        if pad:
            candidates = torch.cat(
                (candidates.new_zeros(batch, pad, nodes, candidate_count), candidates),
                dim=1,
            )
        segments = candidates.permute(0, 2, 3, 1).reshape(
            batch,
            nodes,
            candidate_count,
            self.scale0_segments,
            self.seg_len,
        )
        projected = torch.stack(
            [
                projection(segments[:, :, index])
                for index, projection in enumerate(self.candidate_projections)
            ],
            dim=2,
        )
        identity = self.candidate_identity.weight.view(
            1, 1, candidate_count, 1, self.d_model
        )
        return self.dropout(projected + self.position_embedding + identity)

    def _shared_cross_time(self, encoder: CanonicalCrossTime, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 5:
            raise ValueError("IA-1 shared Cross-Time expects (B,N,K,S,D)")
        batch, nodes, candidates, segments, d_model = tokens.shape
        self._current_cross_time_candidate_counts.append(int(candidates))
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

    def _shared_scale1_merge(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, nodes, candidates, segments, d_model = tokens.shape
        merged = self.scale1_merging(
            tokens.reshape(batch, nodes * candidates, segments, d_model)
        )
        return merged.reshape(batch, nodes, candidates, self.scale1_segments, d_model)

    def encode_selected_candidates(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        selected = self.candidate_history(x)
        projected = self._project_selected(selected)
        scale0 = self._shared_cross_time(self.scale0_cross_time, projected)
        scale1 = self._shared_scale1_merge(scale0)
        scale1 = self._shared_cross_time(self.scale1_cross_time, scale1)
        if tuple(scale0.shape[2:]) != (
            self.effective_candidate_count,
            self.scale0_segments,
            self.d_model,
        ):
            raise AssertionError("IA-1 Scale0 selected candidate contract drifted")
        if tuple(scale1.shape[2:]) != (
            self.effective_candidate_count,
            self.scale1_segments,
            self.d_model,
        ):
            raise AssertionError("IA-1 Scale1 selected candidate contract drifted")
        if not torch.isfinite(scale0).all() or not torch.isfinite(scale1).all():
            raise FloatingPointError("IA-1 selected-only propagation contains NaN or Inf")
        return scale0, scale1

    def _fuse(self, tokens: torch.Tensor, fusion: nn.Module) -> torch.Tensor:
        batch, nodes, candidates, segments, d_model = tokens.shape
        if candidates != self.effective_candidate_count or d_model != self.d_model:
            raise ValueError("IA-1 fusion received an unexpected selected candidate shape")
        inputs = tokens.permute(0, 1, 3, 2, 4).reshape(
            batch,
            nodes,
            segments,
            candidates * d_model,
        )
        return fusion(inputs)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._current_cross_time_candidate_counts = []
        scale0_candidates, scale1_candidates = self.encode_selected_candidates(x)
        scale0 = self._fuse(scale0_candidates, self.scale0_fusion)
        scale1 = self._fuse(scale1_candidates, self.scale1_fusion)
        self.cross_time_candidate_counts = tuple(self._current_cross_time_candidate_counts)
        if self.cross_time_candidate_counts != (self.effective_candidate_count,) * 2:
            raise AssertionError("IA-1 Cross-Time candidate axis was not selected-only")
        if tuple(scale0.shape) != (x.shape[0], x.shape[2], self.scale0_segments, self.d_model):
            raise AssertionError("IA-1 Scale0 propagation contract drifted")
        if tuple(scale1.shape) != (x.shape[0], x.shape[2], self.scale1_segments, self.d_model):
            raise AssertionError("IA-1 Scale1 propagation contract drifted")
        if not torch.isfinite(scale0).all() or not torch.isfinite(scale1).all():
            raise FloatingPointError("IA-1 propagation output contains NaN or Inf")
        self.execution_trace = {
            "candidate_bank_count": self.candidate_count,
            "effective_candidate_count": self.effective_candidate_count,
            "scale0_cross_time_candidate_count": self.cross_time_candidate_counts[0],
            "scale1_cross_time_candidate_count": self.cross_time_candidate_counts[1],
        }
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


SelectedOnlyPropagation = IAFixedPropagation


__all__ = [
    "IA_SELECTION_MODE",
    "IAFixedPropagation",
    "P3_IA_MODEL_CONFIG_FIELDS",
    "SelectedOnlyPropagation",
    "canonical_candidate_names",
    "parse_candidate_name",
    "resolve_fixed_candidate_indices",
    "validate_p3_ia_model_config",
    "validate_selected_candidates",
]

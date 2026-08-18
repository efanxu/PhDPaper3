"""History-only candidate construction for the P3-A propagation bank.

The bank is deliberately a small, deterministic data boundary. It resolves
feature locations from DataInfoView.feature_columns and never receives
anything other than the model-visible history tensor.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


P3_BASE_FEATURES = (
    "Wspd",
    "Etmp",
    "Itmp",
    "Pab1",
    "Pab2",
    "Pab3",
    "Prtv",
    "T2m",
    "Sp",
    "RelH",
    "Wspd_w",
    "Tp",
    "Patv_clean_for_input",
)
P3_CANDIDATE_TRANSFORMS = ("level", "diff1")
P3_CIRCULAR_DIRECTION_FEATURES = frozenset({"Wdir", "Ndir", "Wdir_w"})
P3_MODEL_CONFIG_FIELDS = frozenset(
    {"mode", "top_k", "candidate_features", "candidate_transforms"}
)


@dataclass(frozen=True)
class P3Candidate:
    """One stable logical candidate and its resolved input channel."""

    name: str
    feature: str
    transform: str
    feature_index: int


def _as_ordered_strings(value: Any, *, field: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"P3 {field} must be a non-empty ordered list")
    result = tuple(value)
    if not result or any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"P3 {field} must contain non-empty strings")
    return result


def _duplicates(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return tuple(duplicates)


def validate_p3_model_config(value: Any) -> dict[str, Any]:
    """Validate the frozen P3-A model-owned selection document.

    The model seam intentionally accepts only the complete first candidate
    bank. A smaller or reordered bank would be a different experiment and
    belongs in a later P3 phase.
    """

    if not isinstance(value, Mapping):
        raise ValueError("P3 model config p3 must be a mapping")
    unknown = sorted(set(value) - P3_MODEL_CONFIG_FIELDS)
    missing = sorted(P3_MODEL_CONFIG_FIELDS - set(value))
    if unknown:
        raise ValueError(f"P3 model config p3 has unsupported field: {unknown[0]}")
    if missing:
        raise ValueError(f"P3 model config p3 is missing field: {missing[0]}")
    if value["mode"] != "global_topk":
        raise ValueError("P3 model config p3.mode must be global_topk")
    top_k = value["top_k"]
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k != 2:
        raise ValueError("P3 model config p3.top_k must equal 2")

    features = _as_ordered_strings(value["candidate_features"], field="candidate_features")
    duplicate = _duplicates(features)
    if duplicate:
        raise ValueError(f"P3 candidate_features contains duplicate feature: {duplicate[0]}")
    circular = sorted(set(features).intersection(P3_CIRCULAR_DIRECTION_FEATURES))
    if circular:
        raise ValueError(
            "P3-A candidate bank excludes standardized circular direction feature: "
            f"{circular[0]}"
        )
    unknown_features = sorted(set(features) - set(P3_BASE_FEATURES))
    if unknown_features:
        raise ValueError(f"P3 candidate_features contains illegal feature: {unknown_features[0]}")
    if features != P3_BASE_FEATURES:
        missing_features = [feature for feature in P3_BASE_FEATURES if feature not in features]
        if missing_features:
            raise ValueError(
                "P3-A candidate_features must contain the frozen 13-feature bank; "
                f"missing {missing_features[0]}"
            )
        raise ValueError("P3-A candidate_features must use the canonical feature order")

    transforms = _as_ordered_strings(value["candidate_transforms"], field="candidate_transforms")
    if transforms != P3_CANDIDATE_TRANSFORMS:
        raise ValueError(
            "P3 candidate_transforms must be exactly [level, diff1] in that order"
        )
    if top_k > len(features) * len(transforms):
        raise ValueError("P3 top_k cannot exceed candidate count")
    return dict(value)


def _feature_columns(value: Any) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        columns = value.get("feature_columns")
    else:
        columns = getattr(value, "feature_columns", value)
    if isinstance(columns, (str, bytes)) or not isinstance(columns, Sequence):
        raise ValueError("P3 Candidate Bank requires DataInfoView.feature_columns")
    result = tuple(columns)
    if not result or any(not isinstance(item, str) or not item for item in result):
        raise ValueError("P3 DataInfoView.feature_columns must contain non-empty strings")
    duplicate = _duplicates(result)
    if duplicate:
        raise ValueError(
            "P3 DataInfoView.feature_columns contains duplicate feature name: "
            f"{duplicate[0]}"
        )
    return result


class P3CandidateBank(nn.Module):
    """Resolve and construct stable level/diff1 candidates from model history."""

    def __init__(
        self,
        data_info_or_feature_columns: Any | None = None,
        *,
        feature_columns: Any | None = None,
        candidate_features: Sequence[str] = P3_BASE_FEATURES,
        candidate_transforms: Sequence[str] = P3_CANDIDATE_TRANSFORMS,
    ) -> None:
        super().__init__()
        if feature_columns is not None:
            if data_info_or_feature_columns is not None:
                raise ValueError(
                    "P3 Candidate Bank received both data_info and feature_columns"
                )
            data_info_or_feature_columns = feature_columns
        if data_info_or_feature_columns is None:
            raise ValueError("P3 Candidate Bank requires DataInfoView.feature_columns")
        self.feature_columns = _feature_columns(data_info_or_feature_columns)
        features = _as_ordered_strings(candidate_features, field="candidate_features")
        duplicate = _duplicates(features)
        if duplicate:
            raise ValueError(f"P3 candidate_features contains duplicate feature: {duplicate[0]}")
        circular = sorted(set(features).intersection(P3_CIRCULAR_DIRECTION_FEATURES))
        if circular:
            raise ValueError(
                "P3-A candidate bank excludes standardized circular direction feature: "
                f"{circular[0]}"
            )
        illegal = sorted(set(features) - set(P3_BASE_FEATURES))
        if illegal:
            raise ValueError(f"P3 candidate_features contains illegal feature: {illegal[0]}")
        absent = [feature for feature in features if feature not in self.feature_columns]
        if absent:
            raise ValueError(
                "P3 Candidate Bank feature is missing from DataInfoView.feature_columns: "
                f"{absent[0]}"
            )
        transforms = _as_ordered_strings(candidate_transforms, field="candidate_transforms")
        if transforms != P3_CANDIDATE_TRANSFORMS:
            raise ValueError(
                "P3 candidate_transforms must be exactly [level, diff1] in that order"
            )

        # Resolve by the frozen base-feature order, independent of YAML list
        # presentation, so candidate indices remain canonical and reproducible.
        ordered_features = tuple(feature for feature in P3_BASE_FEATURES if feature in features)
        feature_indices = {name: self.feature_columns.index(name) for name in self.feature_columns}
        candidates: list[P3Candidate] = []
        for feature in ordered_features:
            for transform in transforms:
                candidates.append(
                    P3Candidate(
                        name=f"{feature}.{transform}",
                        feature=feature,
                        transform=transform,
                        feature_index=feature_indices[feature],
                    )
                )
        if not candidates:
            raise ValueError("P3 Candidate Bank must contain at least one candidate")
        self.candidates = tuple(candidates)
        self.candidate_names = tuple(candidate.name for candidate in candidates)
        self.candidate_count = len(self.candidates)

    def candidate_history(self, x: torch.Tensor) -> torch.Tensor:
        """Return [B,L,N,M] candidates using history only."""

        if not isinstance(x, torch.Tensor) or x.ndim != 4:
            raise ValueError("P3 Candidate Bank expects x with shape (B, L, N, C)")
        if x.shape[-1] != len(self.feature_columns):
            raise ValueError(
                "P3 Candidate Bank input channels do not match "
                "DataInfoView.feature_columns"
            )
        if not torch.isfinite(x).all():
            raise FloatingPointError("P3 Candidate Bank input contains NaN or Inf")

        outputs: list[torch.Tensor] = []
        for candidate in self.candidates:
            level = x[..., candidate.feature_index]
            if candidate.transform == "level":
                outputs.append(level)
            elif candidate.transform == "diff1":
                first = torch.zeros_like(level[:, :1])
                outputs.append(torch.cat((first, level[:, 1:] - level[:, :-1]), dim=1))
            else:  # The constructor rejects this; keep the boundary defensive.
                raise AssertionError(f"unsupported P3 candidate transform: {candidate.transform}")
        result = torch.stack(outputs, dim=-1)
        if not torch.isfinite(result).all():
            raise FloatingPointError("P3 Candidate Bank output contains NaN or Inf")
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.candidate_history(x)

    def report_metadata(self) -> list[dict[str, Any]]:
        """Return serializable candidate metadata without tensor values."""

        return [
            {
                "candidate_name": candidate.name,
                "feature": candidate.feature,
                "transform": candidate.transform,
                "feature_index": candidate.feature_index,
            }
            for candidate in self.candidates
        ]


def build_p3_candidate_history(
    x: torch.Tensor,
    data_info_or_feature_columns: Any,
    *,
    candidate_features: Sequence[str] = P3_BASE_FEATURES,
    candidate_transforms: Sequence[str] = P3_CANDIDATE_TRANSFORMS,
) -> torch.Tensor:
    """Functional convenience wrapper around P3CandidateBank."""

    return P3CandidateBank(
        data_info_or_feature_columns,
        candidate_features=candidate_features,
        candidate_transforms=candidate_transforms,
    )(x)


P3FeatureBank = P3CandidateBank
CandidateFeatureBank = P3CandidateBank
build_candidate_history = build_p3_candidate_history


__all__ = [
    "P3_BASE_FEATURES",
    "P3_CANDIDATE_TRANSFORMS",
    "P3_CIRCULAR_DIRECTION_FEATURES",
    "P3Candidate",
    "P3CandidateBank",
    "P3FeatureBank",
    "CandidateFeatureBank",
    "P3_MODEL_CONFIG_FIELDS",
    "build_candidate_history",
    "build_p3_candidate_history",
    "validate_p3_model_config",
]

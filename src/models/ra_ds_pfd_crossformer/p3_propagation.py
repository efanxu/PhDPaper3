"""P3-A candidate-specific projection with shared two-scale temporal encoding."""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .pfd0 import CanonicalCrossTime, PFD0SegmentMerging
from .p3_feature_bank import P3CandidateBank
from .p3_selector import (
    DEFAULT_SELECTOR_BISECTION_ITERATIONS,
    DEFAULT_SELECTOR_TEMPERATURE,
    GlobalTopKSelector,
)


class P3GlobalTopKPropagation(nn.Module):
    """Produce the same two-scale propagation contract as frozen R2.

    Candidate-specific value projections are followed by candidate-shared
    Cross-Time modules. The differentiable fixed-cardinality gate is converted
    to normalized mixture weights only after both scales have been encoded, so
    every candidate remains differentiable in P3-A2.
    """

    def __init__(
        self,
        *,
        feature_columns: Sequence[str] | Any,
        candidate_features: Sequence[str],
        candidate_transforms: Sequence[str],
        top_k: int,
        lookback: int,
        seg_len: int,
        win_size: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        factor: int,
        spatial_dropout: float | None = None,
        source_root: Path | None = None,
        selector_temperature: float = DEFAULT_SELECTOR_TEMPERATURE,
        selector_bisection_iterations: int = DEFAULT_SELECTOR_BISECTION_ITERATIONS,
        # Compatibility alias for direct callers of the P3-A foundation. The
        # production model passes the frozen R2 spatial_dropout explicitly.
        dropout: float | None = None,
    ) -> None:
        super().__init__()
        self.lookback = int(lookback)
        self.seg_len = int(seg_len)
        self.win_size = int(win_size)
        self.d_model = int(d_model)
        self.scale0_segments = ceil(self.lookback / self.seg_len)
        self.scale1_segments = ceil(self.scale0_segments / self.win_size)
        if self.lookback < 1 or self.seg_len < 1 or self.win_size < 1:
            raise ValueError("P3 propagation lookback, seg_len and win_size must be positive")
        if self.scale0_segments < 1 or self.scale1_segments < 1:
            raise ValueError("P3 propagation requires positive segment counts")
        if spatial_dropout is None:
            if dropout is None:
                raise ValueError("P3 propagation requires spatial_dropout")
            spatial_dropout = dropout
        elif dropout is not None and float(spatial_dropout) != float(dropout):
            raise ValueError("P3 propagation dropout aliases disagree")
        self.spatial_dropout = float(spatial_dropout)

        self.candidate_bank = P3CandidateBank(
            feature_columns,
            candidate_features=candidate_features,
            candidate_transforms=candidate_transforms,
        )
        self.candidate_names = self.candidate_bank.candidate_names
        self.candidate_count = self.candidate_bank.candidate_count
        self.selector = GlobalTopKSelector(
            self.candidate_names,
            top_k=top_k,
            temperature=selector_temperature,
            bisection_iterations=selector_bisection_iterations,
        )

        self.candidate_projections = nn.ModuleList(
            [nn.Linear(self.seg_len, self.d_model, bias=False) for _ in range(self.candidate_count)]
        )
        # One position embedding is broadcast over the candidate axis.
        self.position_embedding = nn.Parameter(
            torch.randn(1, 1, 1, self.scale0_segments, self.d_model) * 0.02
        )
        self.candidate_identity = nn.Embedding(self.candidate_count, self.d_model)
        self.dropout = nn.Dropout(self.spatial_dropout)

        root = (
            Path(source_root).resolve()
            if source_root is not None
            else Path(__file__).resolve().parents[3] / "Time-Series-Library"
        )
        # Exactly one temporal network is constructed at each scale. The
        # candidate axis is folded into the batch axis during each call.
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

    def _project_candidates(self, candidates: torch.Tensor) -> torch.Tensor:
        if candidates.ndim != 4:
            raise ValueError("P3 candidate projection expects (B,L,N,M)")
        batch, length, nodes, candidate_count = candidates.shape
        if candidate_count != self.candidate_count:
            raise ValueError("P3 candidate projection received an unexpected candidate count")
        if ceil(length / self.seg_len) != self.scale0_segments:
            raise ValueError("P3 candidate segment count changed from configured lookback")
        pad = self.scale0_segments * self.seg_len - length
        if pad:
            candidates = torch.cat(
                (candidates.new_zeros(batch, pad, nodes, candidate_count), candidates),
                dim=1,
            )
        segments = candidates.permute(0, 2, 3, 1).reshape(
            batch, nodes, candidate_count, self.scale0_segments, self.seg_len
        )
        projected = torch.stack(
            [
                projection(segments[:, :, index])
                for index, projection in enumerate(self.candidate_projections)
            ],
            dim=2,
        )
        identity = self.candidate_identity.weight.view(1, 1, candidate_count, 1, self.d_model)
        encoded = projected + self.position_embedding + identity
        return self.dropout(encoded)

    @staticmethod
    def _shared_cross_time(encoder: CanonicalCrossTime, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 5:
            raise ValueError("P3 shared Cross-Time expects (B,N,M,S,D)")
        batch, nodes, candidates, segments, d_model = tokens.shape
        folded = tokens.permute(0, 2, 1, 3, 4).reshape(
            batch * candidates, nodes, segments, d_model
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

    def encode_candidates(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        candidates = self.candidate_bank(x)
        projected = self._project_candidates(candidates)
        scale0 = self._shared_cross_time(self.scale0_cross_time, projected)
        scale1 = self._shared_scale1_merge(scale0)
        scale1 = self._shared_cross_time(self.scale1_cross_time, scale1)
        if tuple(scale0.shape[2:]) != (
            self.candidate_count,
            self.scale0_segments,
            self.d_model,
        ):
            raise AssertionError("P3 Scale0 candidate contract drifted")
        if tuple(scale1.shape[2:]) != (
            self.candidate_count,
            self.scale1_segments,
            self.d_model,
        ):
            raise AssertionError("P3 Scale1 candidate contract drifted")
        if not torch.isfinite(scale0).all() or not torch.isfinite(scale1).all():
            raise FloatingPointError("P3 candidate propagation contains NaN or Inf")
        return scale0, scale1

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale0_candidates, scale1_candidates = self.encode_candidates(x)
        relaxed_gate = self.selector()
        mixture_weights = self.selector.mixture_weights(relaxed_gate)
        weights = mixture_weights.view(1, 1, self.candidate_count, 1, 1)
        scale0 = (scale0_candidates * weights).sum(dim=2)
        scale1 = (scale1_candidates * weights).sum(dim=2)
        if tuple(scale0.shape) != (x.shape[0], x.shape[2], self.scale0_segments, self.d_model):
            raise AssertionError("P3 Scale0 propagation contract drifted")
        if tuple(scale1.shape) != (x.shape[0], x.shape[2], self.scale1_segments, self.d_model):
            raise AssertionError("P3 Scale1 propagation contract drifted")
        if not torch.isfinite(scale0).all() or not torch.isfinite(scale1).all():
            raise FloatingPointError("P3 propagation output contains NaN or Inf")
        return scale0, scale1

    def selection_report(self) -> list[dict[str, Any]]:
        return self.selector.selection_report()


# Short aliases make the model-owned seam easy to discover without adding a
# second model name or a second execution path.
P3Propagation = P3GlobalTopKPropagation
GlobalTopKPropagation = P3GlobalTopKPropagation
P3PropagationModule = P3GlobalTopKPropagation


__all__ = [
    "GlobalTopKPropagation",
    "P3GlobalTopKPropagation",
    "P3Propagation",
    "P3PropagationModule",
]

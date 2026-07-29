"""The one public training loss, migrated from the old formal engine."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ScoreAlignedHybridTerms:
    absolute_error_sum: torch.Tensor
    squared_error_sum: torch.Tensor
    valid_count: int

    def __add__(self, other: "ScoreAlignedHybridTerms") -> "ScoreAlignedHybridTerms":
        return ScoreAlignedHybridTerms(
            self.absolute_error_sum + other.absolute_error_sum,
            self.squared_error_sum + other.squared_error_sum,
            self.valid_count + other.valid_count,
        )

    def loss(self) -> torch.Tensor:
        if self.valid_count <= 0:
            raise ValueError("loss batch contains no valid targets")
        mae = self.absolute_error_sum / self.valid_count
        rmse = torch.sqrt(self.squared_error_sum / self.valid_count)
        return 0.5 * mae + 0.5 * rmse


def _valid(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError("prediction, target and mask shapes must match")
    valid = mask.bool()
    if not bool(valid.any()):
        raise ValueError("loss or metric batch contains no valid targets")
    return prediction[valid], target[valid]


def score_aligned_hybrid_terms(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> ScoreAlignedHybridTerms:
    pred, true = _valid(prediction, target, mask)
    error = pred - true
    return ScoreAlignedHybridTerms(
        absolute_error_sum=error.abs().sum(),
        squared_error_sum=error.square().sum(),
        valid_count=int(error.numel()),
    )


def masked_score_aligned_hybrid(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    return score_aligned_hybrid_terms(prediction, target, mask).loss()

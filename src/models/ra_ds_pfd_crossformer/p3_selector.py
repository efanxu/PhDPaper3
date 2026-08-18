"""Differentiable global candidate ranking for P3-A."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from typing import Any

import torch
from torch import nn


class GlobalTopKSelector(nn.Module):
    """One global learnable score vector shared by every P3 tensor axis."""

    def __init__(self, candidate_names: Sequence[str], *, top_k: int = 2) -> None:
        super().__init__()
        if isinstance(candidate_names, (str, bytes)):
            raise ValueError("P3 selector candidate_names must be an ordered sequence")
        names = tuple(candidate_names)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("P3 selector candidate_names must contain non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("P3 selector candidate_names must be unique")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k != 2:
            raise ValueError("P3 selector top_k must equal 2")
        if top_k > len(names):
            raise ValueError("P3 selector top_k cannot exceed candidate count")
        self.candidate_names = names
        self.candidate_count = len(names)
        self.top_k = int(top_k)
        self.logits = nn.Parameter(torch.zeros(self.candidate_count))

    def scores(self) -> torch.Tensor:
        if not torch.isfinite(self.logits).all():
            raise FloatingPointError("P3 selector logits contain NaN or Inf")
        values = torch.softmax(self.logits, dim=0)
        if not torch.isfinite(values).all():
            raise FloatingPointError("P3 selector scores contain NaN or Inf")
        return values

    def forward(self) -> torch.Tensor:
        return self.scores()

    def ranking(self, scores: torch.Tensor | None = None) -> tuple[int, ...]:
        values = self.scores() if scores is None else scores
        if tuple(values.shape) != (self.candidate_count,):
            raise ValueError("P3 selector ranking received an unexpected score shape")
        if not torch.isfinite(values).all():
            raise FloatingPointError("P3 selector ranking scores contain NaN or Inf")
        # Convert only the global readout to Python values. The forward path
        # stays fully differentiable and never uses this hard ranking.
        scores_cpu = values.detach().cpu().tolist()
        return tuple(
            sorted(
                range(self.candidate_count),
                key=lambda index: (-float(scores_cpu[index]), index),
            )
        )

    def selection_report(self) -> list[dict[str, Any]]:
        """Return a JSON-serializable, deterministic ranking readout."""

        values = self.scores()
        order = self.ranking(values)
        rank_by_index = {index: rank for rank, index in enumerate(order, start=1)}
        selected_indices = set(order[: self.top_k])
        report: list[dict[str, Any]] = []
        score_values = values.detach().cpu().tolist()
        for index, name in enumerate(self.candidate_names):
            score = float(score_values[index])
            if not isfinite(score):
                raise FloatingPointError("P3 selector report contains a non-finite score")
            report.append(
                {
                    "candidate_name": name,
                    "score": score,
                    "rank": int(rank_by_index[index]),
                    "selected": bool(index in selected_indices),
                }
            )
        return report

    # Keep the read-only report discoverable under a short name as well.
    def report(self) -> list[dict[str, Any]]:
        return self.selection_report()


P3GlobalSelector = GlobalTopKSelector
GlobalPropagationSelector = GlobalTopKSelector


__all__ = [
    "GlobalPropagationSelector",
    "GlobalTopKSelector",
    "P3GlobalSelector",
]

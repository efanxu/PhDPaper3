"""One dataset shape used by all models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .loader import DataArrays
from .normalization import NormalizationStats


@dataclass(frozen=True)
class ForecastBatch:
    """Trainer-owned batch; target and mask never cross into ``ModelInput``."""

    x: torch.Tensor
    target: torch.Tensor
    target_mask: torch.Tensor
    starts: torch.Tensor

    def to(self, device: torch.device | str) -> "ForecastBatch":
        return ForecastBatch(
            x=self.x.to(device),
            target=self.target.to(device),
            target_mask=self.target_mask.to(device),
            starts=self.starts,
        )

    def model_input(self):
        from models.base import ModelInput

        return ModelInput(x=self.x)


class ForecastDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]):
    def __init__(
        self,
        arrays: DataArrays,
        starts: np.ndarray,
        normalization: NormalizationStats,
        *,
        lookback: int,
        horizon: int,
    ) -> None:
        self.arrays = arrays
        self.starts = np.asarray(starts, dtype=np.int64)
        self.normalization = normalization
        self.lookback = int(lookback)
        self.horizon = int(horizon)

    def __len__(self) -> int:
        return int(len(self.starts))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        start = int(self.starts[index])
        if start < self.lookback or start + self.horizon > len(self.arrays.x):
            raise IndexError(f"window start is outside arrays: {start}")
        x = self.arrays.x[start - self.lookback : start]
        target = self.arrays.target[start : start + self.horizon].T
        mask = self.arrays.target_mask[start : start + self.horizon].T
        normalized_x = self.normalization.normalize_input(x)
        normalized_target = self.normalization.normalize_target(target, mask)
        return (
            torch.from_numpy(np.ascontiguousarray(normalized_x)),
            torch.from_numpy(np.ascontiguousarray(normalized_target)),
            torch.from_numpy(np.ascontiguousarray(mask, dtype=np.bool_)),
            start,
        )


def collate_forecast_batch(
    items: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]],
) -> ForecastBatch:
    if not items:
        raise ValueError("cannot collate an empty forecast batch")
    x, target, mask, starts = zip(*items)
    return ForecastBatch(
        x=torch.stack(x),
        target=torch.stack(target),
        target_mask=torch.stack(mask).bool(),
        starts=torch.tensor(starts, dtype=torch.int64),
    )

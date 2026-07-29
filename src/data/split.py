"""Chronological split semantics shared by every model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from runtime.config import ExperimentConfig


@dataclass(frozen=True)
class SplitRange:
    name: str
    start: int
    end: int

    @property
    def count(self) -> int:
        return self.end - self.start

    def as_dict(self, timestamps: np.ndarray) -> dict[str, Any]:
        return {
            "start_index_inclusive": self.start,
            "end_index_exclusive": self.end,
            "start_timestamp": str(timestamps[self.start]),
            "end_timestamp": str(timestamps[self.end - 1]),
            "timestamp_count": self.count,
        }


@dataclass(frozen=True)
class SplitBoundaries:
    train: SplitRange
    validation: SplitRange
    test: SplitRange
    ratios: tuple[float, float, float]

    def as_dict(self, timestamps: np.ndarray) -> dict[str, Any]:
        return {
            "method": "chronological_ratio",
            "ratios": list(self.ratios),
            "time_count": int(len(timestamps)),
            "boundaries": {
                "train": self.train.as_dict(timestamps),
                "validation": self.validation.as_dict(timestamps),
                "test": self.test.as_dict(timestamps),
            },
        }


def chronological_split(time_count: int, config: ExperimentConfig) -> SplitBoundaries:
    """Match the old floor-based 0.8/0.1/0.1 boundary calculation."""

    if time_count < 3:
        raise ValueError("at least three timestamps are required")
    split = config.split
    train_end = int(time_count * float(split["train_ratio"]))
    validation_end = int(
        time_count * (float(split["train_ratio"]) + float(split["val_ratio"]))
    )
    if not 0 < train_end < validation_end < time_count:
        raise ValueError("configured split creates an empty range")
    ratios = (
        float(split["train_ratio"]),
        float(split["val_ratio"]),
        float(split["test_ratio"]),
    )
    return SplitBoundaries(
        train=SplitRange("train", 0, train_end),
        validation=SplitRange("validation", train_end, validation_end),
        test=SplitRange("test", validation_end, time_count),
        ratios=ratios,
    )

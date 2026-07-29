"""Sliding-window index generation with explicit split-local boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .split import SplitBoundaries


@dataclass(frozen=True)
class WindowIndex:
    train: np.ndarray
    train_valid: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    invalid_train_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "lookback": self.lookback,
            "max_pred_len": self.horizon,
            "strides": {
                "train": self.strides["train"],
                "val": self.strides["validation"],
                "test": self.strides["test"],
            },
            "counts": {
                "train": int(len(self.train)),
                "train_valid": int(len(self.train_valid)),
                "validation": int(len(self.validation)),
                "test": int(len(self.test)),
            },
            "invalid_train_count": self.invalid_train_count,
        }

    lookback: int = 0
    horizon: int = 0
    strides: dict[str, int] | None = None


def _starts(start: int, end: int, lookback: int, horizon: int, stride: int) -> np.ndarray:
    first = start + lookback
    last_exclusive = end - horizon + 1
    if first >= last_exclusive:
        return np.empty(0, dtype=np.int64)
    return np.arange(first, last_exclusive, stride, dtype=np.int64)


def build_window_index(
    splits: SplitBoundaries,
    *,
    lookback: int,
    horizon: int,
    strides: dict[str, int],
    target_mask: np.ndarray | None = None,
) -> WindowIndex:
    """Generate forecast-start indices; train validity means any valid target.

    The old artifacts contain both all train candidates and the filtered
    ``train_window_starts_valid`` array. Validation and test retain all
    split-local windows so their metrics use the complete evaluation range.
    """

    for name in ("train", "validation", "test"):
        if name not in strides or int(strides[name]) < 1:
            raise ValueError(f"missing or invalid stride for {name}")
    train = _starts(splits.train.start, splits.train.end, lookback, horizon, int(strides["train"]))
    validation = _starts(
        splits.validation.start,
        splits.validation.end,
        lookback,
        horizon,
        int(strides["validation"]),
    )
    test = _starts(splits.test.start, splits.test.end, lookback, horizon, int(strides["test"]))
    if target_mask is None:
        train_valid = train.copy()
        invalid = 0
    else:
        if target_mask.ndim != 2:
            raise ValueError("target_mask must have shape (time, nodes)")
        valid_flags = np.asarray(
            [target_mask[start : start + horizon].any() for start in train],
            dtype=bool,
        )
        train_valid = train[valid_flags]
        invalid = int((~valid_flags).sum())
    return WindowIndex(
        train=train,
        train_valid=train_valid,
        validation=validation,
        test=test,
        invalid_train_count=invalid,
        lookback=int(lookback),
        horizon=int(horizon),
        strides={name: int(value) for name, value in strides.items()},
    )

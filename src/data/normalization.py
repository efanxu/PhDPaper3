"""Train-only population standardization and its portable saved form."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .loader import DataArrays
from .split import SplitRange


@dataclass(frozen=True)
class NormalizationStats:
    input_mean: np.ndarray
    input_scale: np.ndarray
    target_mean: float
    target_scale: float

    def normalize_input(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.input_mean) / self.input_scale).astype(np.float32, copy=False)

    def normalize_target(self, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
        result = np.zeros_like(target, dtype=np.float32)
        result[mask] = ((target[mask] - self.target_mean) / self.target_scale).astype(np.float32, copy=False)
        return result

    def denormalize_target(self, target: np.ndarray) -> np.ndarray:
        return target * self.target_scale + self.target_mean

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_mean": self.input_mean.astype(float).tolist(),
            "input_scale": self.input_scale.astype(float).tolist(),
            "target_mean": float(self.target_mean),
            "target_scale": float(self.target_scale),
            "input_scope": "train_timestamps_all_turbines",
            "target_scope": "train_valid_targets_only",
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            input_mean=self.input_mean,
            input_scale=self.input_scale,
            target_mean=np.asarray(self.target_mean, dtype=np.float64),
            target_scale=np.asarray(self.target_scale, dtype=np.float64),
        )

    @classmethod
    def load(cls, path: Path) -> "NormalizationStats":
        with np.load(path, allow_pickle=False) as value:
            return cls(
                value["input_mean"].astype(np.float32),
                value["input_scale"].astype(np.float32),
                float(value["target_mean"]),
                float(value["target_scale"]),
            )


def _safe_scale(value: np.ndarray | float) -> np.ndarray | float:
    if np.isscalar(value):
        return 1.0 if float(value) == 0.0 else float(value)
    return np.where(value == 0.0, 1.0, value)


def fit_normalization(arrays: DataArrays, train: SplitRange) -> NormalizationStats:
    """Fit input statistics on all train rows and target statistics on valid train targets."""

    train_x = arrays.x[train.start : train.end].reshape(-1, arrays.x.shape[-1]).astype(np.float64)
    input_mean = train_x.mean(axis=0)
    input_scale = _safe_scale(train_x.std(axis=0, ddof=0))
    train_target = arrays.target[train.start : train.end]
    train_mask = arrays.target_mask[train.start : train.end]
    valid_target = train_target[train_mask].astype(np.float64)
    if valid_target.size == 0:
        raise ValueError("train split contains no valid targets for normalization")
    target_mean = float(valid_target.mean())
    target_scale = float(_safe_scale(valid_target.std(ddof=0)))
    return NormalizationStats(
        input_mean=np.asarray(input_mean, dtype=np.float32),
        input_scale=np.asarray(input_scale, dtype=np.float32),
        target_mean=target_mean,
        target_scale=target_scale,
    )

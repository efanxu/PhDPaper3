"""One evaluation path used for validation, test and evaluate-only runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import ForecastBatch
from data.normalization import NormalizationStats
from models.base import ForecastModel

from .metrics import compute_metrics


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, Any]
    normalized_prediction: np.ndarray
    prediction_kw: np.ndarray
    target_kw: np.ndarray
    target_mask: np.ndarray
    starts: np.ndarray


def evaluate(
    model: ForecastModel,
    loader: Iterable[ForecastBatch],
    *,
    device: torch.device,
    normalization: NormalizationStats,
    horizons: tuple[int, ...],
    total_nodes: int,
    physical_clip: bool = False,
    physical_min_kw: float | None = None,
    physical_max_kw: float | None = None,
    max_batches: int | None = None,
) -> EvaluationResult:
    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    starts: list[torch.Tensor] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            device_batch = batch.to(device)
            prediction = model(device_batch.model_input())
            if prediction.ndim != 3 or prediction.shape[:2] != device_batch.target.shape[:2]:
                raise ValueError("model output shape is incompatible with target shape")
            predictions.append(prediction.detach().float().cpu())
            targets.append(batch.target.detach().float().cpu())
            masks.append(batch.target_mask.detach().bool().cpu())
            starts.append(batch.starts.detach().cpu())
    if not predictions:
        raise ValueError("evaluation loader produced no batches")
    normalized_prediction = torch.cat(predictions, dim=0)
    normalized_target = torch.cat(targets, dim=0)
    target_mask = torch.cat(masks, dim=0)
    prediction_kw = normalization.denormalize_target(normalized_prediction.numpy())
    target_kw = normalization.denormalize_target(normalized_target.numpy())
    mask_np = target_mask.numpy()
    by_horizon: dict[str, dict[str, Any]] = {}
    for horizon in horizons:
        if horizon < 1 or horizon > prediction_kw.shape[-1]:
            raise ValueError(f"evaluation horizon {horizon} is outside model output")
        by_horizon[str(horizon)] = compute_metrics(
            torch.from_numpy(prediction_kw[..., :horizon]),
            torch.from_numpy(target_kw[..., :horizon]),
            target_mask[..., :horizon],
            total_nodes=total_nodes,
            physical_clip=physical_clip,
            physical_min_kw=physical_min_kw,
            physical_max_kw=physical_max_kw,
        )
    official_scores = [value["SDWPF Official Score"] for value in by_horizon.values()]
    return EvaluationResult(
        metrics={
            "by_horizon": by_horizon,
            "monitor": float(sum(official_scores) / len(official_scores)),
        },
        normalized_prediction=normalized_prediction.numpy(),
        prediction_kw=prediction_kw,
        target_kw=target_kw,
        target_mask=mask_np,
        starts=torch.cat(starts, dim=0).numpy(),
    )

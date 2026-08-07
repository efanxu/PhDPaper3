"""One evaluation path used for validation, test and evaluate-only runs."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import ForecastBatch
from data.normalization import NormalizationStats
from models.base import ForecastModel

from .metrics import compute_metrics
from .model_execution import (
    DEFAULT_NODE_SHARED_CHUNK_SIZE,
    ExecutionPlan,
    build_execution_plan,
    forward_with_execution_plan,
)
from runtime.performance import summarize_evaluation


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, Any]
    normalized_prediction: np.ndarray
    prediction_kw: np.ndarray
    target_kw: np.ndarray
    target_mask: np.ndarray
    starts: np.ndarray
    performance: dict[str, Any]


def evaluate(
    model: ForecastModel,
    loader: Iterable[ForecastBatch],
    *,
    device: torch.device,
    normalization: NormalizationStats,
    horizons: tuple[int, ...],
    total_nodes: int,
    execution_plan: ExecutionPlan | None = None,
    node_shared_chunk_size: int = DEFAULT_NODE_SHARED_CHUNK_SIZE,
    physical_clip: bool = False,
    physical_min_kw: float | None = None,
    physical_max_kw: float | None = None,
    max_batches: int | None = None,
) -> EvaluationResult:
    plan = execution_plan or build_execution_plan(
        model,
        total_nodes=total_nodes,
        node_shared_chunk_size=node_shared_chunk_size,
    )
    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    starts: list[torch.Tensor] = []
    end_to_end_seconds: list[float] = []
    forward_seconds: list[float] = []
    batch_sizes: list[int] = []

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    iterator = iter(loader)
    with torch.inference_mode():
        batch_index = 0
        while max_batches is None or batch_index < max_batches:
            end_to_end_started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            device_batch = batch.to(device)
            synchronize()
            forward_started = time.perf_counter()
            prediction = forward_with_execution_plan(
                model,
                device_batch.model_input(),
                plan,
            )
            synchronize()
            forward_seconds.append(time.perf_counter() - forward_started)
            end_to_end_seconds.append(time.perf_counter() - end_to_end_started)
            batch_sizes.append(int(batch.x.shape[0]))
            if prediction.ndim != 3 or prediction.shape[:2] != device_batch.target.shape[:2]:
                raise ValueError("model output shape is incompatible with target shape")
            predictions.append(prediction.detach().float().cpu())
            targets.append(batch.target.detach().float().cpu())
            masks.append(batch.target_mask.detach().bool().cpu())
            starts.append(batch.starts.detach().cpu())
            batch_index += 1
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
    performance = summarize_evaluation(
        end_to_end_seconds,
        forward_seconds,
        batch_sizes,
        nodes=int(prediction_kw.shape[1]),
        horizon=int(prediction_kw.shape[2]),
    )
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
        performance=performance,
    )

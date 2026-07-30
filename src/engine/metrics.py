"""MAE, RMSE, R2 and the SDWPF official score."""

from __future__ import annotations

import math
from typing import Any

import torch

from .losses import _valid


def display_metric(value: float | torch.Tensor) -> float | None:
    """Return a JSON-safe, three-decimal display value without changing raw metrics."""

    parsed = float(value)
    return round(parsed, 3) if math.isfinite(parsed) else None


def compute_metrics(
    prediction_kw: torch.Tensor,
    target_kw: torch.Tensor,
    mask: torch.Tensor,
    *,
    total_nodes: int,
    physical_clip: bool = False,
    physical_min_kw: float | None = None,
    physical_max_kw: float | None = None,
) -> dict[str, Any]:
    if physical_clip:
        if physical_min_kw is None or physical_max_kw is None:
            raise ValueError("physical clipping requires both bounds")
        prediction_kw = prediction_kw.clamp(min=physical_min_kw, max=physical_max_kw)
    pred, true = _valid(prediction_kw, target_kw, mask)
    pred = pred.double()
    true = true.double()
    error = pred - true
    mae = error.abs().mean()
    rmse = error.square().mean().sqrt()
    denominator = (true - true.mean()).square().sum()
    r2 = (
        1.0 - error.square().sum() / denominator
        if denominator > 0
        else torch.tensor(float("nan"), dtype=torch.float64)
    )
    epsilon = torch.finfo(torch.float64).eps
    smape = (2.0 * error.abs() / (pred.abs() + true.abs()).clamp_min(epsilon)).mean()
    nonzero = true.abs() > epsilon
    mape = (
        (error[nonzero].abs() / true[nonzero].abs()).mean()
        if bool(nonzero.any())
        else torch.tensor(float("nan"), dtype=torch.float64)
    )

    # The official definition is computed per sample and turbine in MW,
    # averaged over available turbines, then scaled by the total node count.
    error_mw = (prediction_kw.double() - target_kw.double()) / 1000.0
    sample_scores: list[torch.Tensor] = []
    for sample_index in range(prediction_kw.shape[0]):
        node_scores: list[torch.Tensor] = []
        for node_index in range(prediction_kw.shape[1]):
            node_mask = mask[sample_index, node_index].bool()
            if bool(node_mask.any()):
                node_error = error_mw[sample_index, node_index, node_mask]
                node_scores.append(0.5 * node_error.abs().mean() + 0.5 * node_error.square().mean().sqrt())
        if node_scores:
            sample_scores.append(torch.stack(node_scores).mean() * total_nodes)
    if not sample_scores:
        raise ValueError("official score has no valid sample-node pairs")
    official = float(torch.stack(sample_scores).mean())
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "SMAPE": float(smape),
        "MAPE": float(mape),
        "SDWPF Official Score": official,
        "official_align_score": official,
        "score": official,
        "valid_target_count": int(mask.sum().item()),
        "valid_target_ratio": int(mask.sum().item()) / mask.numel(),
        "lower_is_better": True,
        "display": {
            "MAE": display_metric(mae),
            "RMSE": display_metric(rmse),
            "R2": display_metric(r2),
            "MAPE": display_metric(mape),
            "SMAPE": display_metric(smape),
            "SDWPF Official Score": display_metric(official),
        },
    }


metrics = compute_metrics

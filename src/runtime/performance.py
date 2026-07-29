"""Shared paper-oriented timing and throughput calculations."""

from __future__ import annotations

from statistics import median
from typing import Iterable


def _finite_nonnegative(value: float) -> float:
    return max(0.0, float(value))


def summarize_evaluation(
    end_to_end_seconds: Iterable[float],
    forward_seconds: Iterable[float],
    batch_sizes: Iterable[int],
    *,
    nodes: int,
    horizon: int,
) -> dict[str, float | int | bool | None]:
    """Summarize one existing evaluation pass without running it again.

    The first batch is reported separately and excluded from steady-state
    aggregates when more than one batch exists.  A one-batch evaluation is
    explicitly marked as having no independent warmup observation.
    """

    end_to_end = [_finite_nonnegative(value) for value in end_to_end_seconds]
    forward = [_finite_nonnegative(value) for value in forward_seconds]
    sizes = [int(value) for value in batch_sizes]
    if len(end_to_end) != len(forward) or len(end_to_end) != len(sizes) or not end_to_end:
        raise ValueError("evaluation timing arrays must be non-empty and have equal length")
    if any(value < 1 for value in sizes):
        raise ValueError("evaluation batch sizes must be positive")
    if nodes < 1 or horizon < 1:
        raise ValueError("nodes and horizon must be positive")
    steady_start = 1 if len(end_to_end) > 1 else 0
    steady_end_to_end = end_to_end[steady_start:]
    steady_sizes = sizes[steady_start:]
    sample_count = sum(sizes)
    forecast_values = sample_count * int(nodes) * int(horizon)
    steady_seconds = sum(steady_end_to_end)
    steady_sample_count = sum(steady_sizes)
    steady_forecast_values = steady_sample_count * int(nodes) * int(horizon)
    mean_batch_seconds = sum(steady_end_to_end) / len(steady_end_to_end)
    mean_batch_size = steady_sample_count / len(steady_sizes)
    return {
        "evaluation_end_to_end_seconds": sum(end_to_end),
        "model_forward_seconds": sum(forward),
        "batch_count": len(sizes),
        "sample_count": sample_count,
        "first_batch_latency_ms": end_to_end[0] * 1000.0,
        "mean_batch_latency_ms": mean_batch_seconds * 1000.0,
        "median_batch_latency_ms": float(median(steady_end_to_end)) * 1000.0,
        "p95_batch_latency_ms": _percentile(steady_end_to_end, 95.0) * 1000.0,
        "mean_sample_latency_ms": mean_batch_seconds * 1000.0 / mean_batch_size,
        "samples_per_second": steady_sample_count / steady_seconds if steady_seconds > 0 else None,
        "forecast_values_per_second": steady_forecast_values / steady_seconds if steady_seconds > 0 else None,
        "forecast_values": forecast_values,
        "steady_sample_count": steady_sample_count,
        "steady_forecast_values": steady_forecast_values,
        "steady_seconds": steady_seconds,
        "warmup_excluded": len(end_to_end) > 1,
        "has_independent_warmup": len(end_to_end) > 1,
    }


def _percentile(values: list[float], percentile: float) -> float:
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction

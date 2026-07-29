from __future__ import annotations

from runtime.performance import summarize_evaluation
from runtime.run_info import write_json
import pytest


def test_evaluation_performance_excludes_first_batch_from_steady_state() -> None:
    result = summarize_evaluation(
        [0.010, 0.004, 0.006],
        [0.008, 0.002, 0.003],
        [2, 2, 1],
        nodes=3,
        horizon=4,
    )
    assert result["evaluation_end_to_end_seconds"] == pytest.approx(0.02)
    assert result["model_forward_seconds"] == pytest.approx(0.013)
    assert result["batch_count"] == 3
    assert result["sample_count"] == 5
    assert result["forecast_values"] == 60
    assert result["steady_sample_count"] == 3
    assert result["steady_forecast_values"] == 36
    assert result["forecast_values_per_second"] == pytest.approx(3600.0)
    assert result["first_batch_latency_ms"] == 10.0
    assert result["median_batch_latency_ms"] == 5.0
    assert result["p95_batch_latency_ms"] == pytest.approx(5.9)
    assert result["has_independent_warmup"] is True
    assert result["samples_per_second"] > 0
    assert result["forecast_values_per_second"] > 0


def test_single_batch_performance_marks_missing_independent_warmup() -> None:
    result = summarize_evaluation([0.005], [0.004], [2], nodes=2, horizon=3)
    assert result["has_independent_warmup"] is False
    assert result["warmup_excluded"] is False
    assert result["mean_batch_latency_ms"] == 5.0
    assert result["mean_sample_latency_ms"] == 2.5
    assert result["forecast_values_per_second"] == 2400.0


def test_zero_steady_time_serializes_rates_as_null() -> None:
    result = summarize_evaluation([0.0, 0.0], [0.0, 0.0], [2, 3], nodes=2, horizon=3)
    assert result["steady_sample_count"] == 3
    assert result["steady_forecast_values"] == 18
    assert result["steady_seconds"] == 0.0
    assert result["samples_per_second"] is None
    assert result["forecast_values_per_second"] is None


def test_json_metrics_convert_nonfinite_values_to_null(tmp_path) -> None:
    import json
    import math

    path = tmp_path / "metrics.json"
    write_json(path, {"r2": math.nan, "nested": {"mape": math.inf}})
    assert json.loads(path.read_text(encoding="utf-8")) == {"r2": None, "nested": {"mape": None}}

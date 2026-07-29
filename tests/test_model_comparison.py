from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import pytest
import yaml

from cli import orchestrator


MODEL_COMPARISON_FIELDS = [
    "model",
    "status",
    "parameter_count",
    "best_epoch",
    "H3_MAE",
    "H3_RMSE",
    "H3_R2",
    "H3_SMAPE",
    "H3_MAPE",
    "H3_Official_Score",
    "H6_MAE",
    "H6_RMSE",
    "H6_R2",
    "H6_SMAPE",
    "H6_MAPE",
    "H6_Official_Score",
    "H10_MAE",
    "H10_RMSE",
    "H10_R2",
    "H10_SMAPE",
    "H10_MAPE",
    "H10_Official_Score",
    "mean_official_score",
    "training_wall_seconds",
    "test_model_forward_seconds",
    "mean_sample_latency_ms",
    "samples_per_second",
    "peak_gpu_allocated_mb",
    "runtime_environment",
    "python_executable",
    "result_dir",
]


def _write_metrics(path: Path, values: dict[str, float]) -> None:
    path.write_text(json.dumps(values) + "\n", encoding="utf-8")


def test_model_comparison_contains_complete_horizon_and_failure_rows(tmp_path: Path) -> None:
    result_dir = tmp_path / "model_a" / "paper-run"
    result_dir.mkdir(parents=True)
    (result_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump({"data": {"eval_horizons": [3, 6, 10]}}),
        encoding="utf-8",
    )
    metric_values = {
        3: {"mae": 1.0, "rmse": 2.0, "r2": 0.3, "smape": 0.4, "mape": 0.5, "score": 0.6},
        6: {"MAE": 2.0, "RMSE": 3.0, "R2": 0.4, "SMAPE": 0.5, "MAPE": 0.6, "official_score": 0.7},
        10: {"MAE": 3.0, "RMSE": 4.0, "R2": 0.5, "SMAPE": 0.6, "MAPE": 0.7, "Official_Score": 0.8},
    }
    for horizon, values in metric_values.items():
        _write_metrics(result_dir / f"metrics_test_h{horizon}.json", values)
    (result_dir / "performance.json").write_text(
        json.dumps(
            {
                "parameter_count": 123,
                "training_wall_seconds": 12.5,
                "test": {
                    "model_forward_seconds": 1.5,
                    "mean_sample_latency_ms": 2.5,
                    "samples_per_second": 4.5,
                },
                "peak_gpu_allocated_mb": 99.0,
            }
        ),
        encoding="utf-8",
    )
    (result_dir / "run_info.json").write_text(
        json.dumps({"best_epoch": 7}), encoding="utf-8"
    )

    failed_dir = tmp_path / "model_b" / "paper-run"
    records = [
        {
            "model": "model_a",
            "status": "COMPLETED",
            "result_dir": str(result_dir),
            "runtime_environment": "tslib",
            "python_executable": sys.executable,
        },
        {
            "model": "model_b",
            "status": "FAILED",
            "result_dir": str(failed_dir),
            "runtime_environment": "tsl",
            "python_executable": "target-python",
        },
    ]

    orchestrator._write_summaries(tmp_path / "_runs" / "paper-run", records)

    with (tmp_path / "_runs" / "paper-run" / "model_comparison.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
        assert handle is not None
    assert list(rows[0]) == MODEL_COMPARISON_FIELDS
    assert [row["model"] for row in rows] == ["model_a", "model_b"]
    assert rows[0]["H3_Official_Score"] == "0.6"
    assert rows[0]["H6_Official_Score"] == "0.7"
    assert rows[0]["H10_Official_Score"] == "0.8"
    assert float(rows[0]["mean_official_score"]) == pytest.approx(0.7)
    assert rows[0]["runtime_environment"] == "tslib"
    assert rows[0]["python_executable"] == sys.executable
    assert rows[0]["result_dir"] == str(result_dir)
    assert rows[1]["status"] == "FAILED"
    assert rows[1]["H3_MAE"] == ""
    assert rows[1]["H10_Official_Score"] == ""

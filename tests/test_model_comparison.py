from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import yaml

from cli import orchestrator
from runtime.status import FAILED, PASS


def _write_metrics(path: Path, values: dict[str, float]) -> None:
    path.write_text(json.dumps(values) + "\n", encoding="utf-8")


def _result_dir(tmp_path: Path) -> Path:
    result_dir = tmp_path / "node_shared_lstm" / "paper-run"
    result_dir.mkdir(parents=True)
    (result_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump({"data": {"eval_horizons": [3, 6, 10]}}),
        encoding="utf-8",
    )
    _write_metrics(
        result_dir / "metrics_test_h3.json",
        {
            "MAE": 1.0,
            "RMSE": 2.0,
            "R2": 0.0,
            "MAPE": 0.0,
            "SMAPE": 0.4,
            "SDWPF Official Score": 0.6,
            "score": 99.0,
        },
    )
    _write_metrics(
        result_dir / "metrics_test_h6.json",
        {
            "MAE": 2.0,
            "RMSE": 3.0,
            "R2": -1.25,
            "MAPE": 0.6,
            "SMAPE": 0.5,
            "SDWPF Official Score": 0.7,
        },
    )
    _write_metrics(
        result_dir / "metrics_test_h10.json",
        {
            "MAE": 3.0,
            "RMSE": 4.0,
            "R2": math.nan,
            "MAPE": math.nan,
            "SMAPE": 0.6,
            "SDWPF Official Score": 0.8,
        },
    )
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
    return result_dir


def test_model_comparison_writes_paper_and_flat_csv_from_same_ordered_rows(tmp_path: Path) -> None:
    result_dir = _result_dir(tmp_path)
    records = [
        {
            "model": "node_shared_lstm",
            "status": PASS,
            "result_dir": str(result_dir),
            "runtime_environment": "tslib",
            "python_executable": sys.executable,
        },
        {
            "model": "crossformer",
            "status": PASS,
            "result_dir": str(result_dir),
            "runtime_environment": "tslib",
            "python_executable": sys.executable,
        },
        {
            "model": "stcn",
            "status": FAILED,
            "result_dir": str(tmp_path / "stcn" / "paper-run"),
            "runtime_environment": "tsl",
            "python_executable": "target-python",
        },
    ]
    run_root = tmp_path / "_runs" / "paper-run"
    orchestrator._write_summaries(run_root, records)

    with (run_root / "model_comparison.csv").open(encoding="utf-8", newline="") as handle:
        paper_rows = list(csv.reader(handle))
    assert paper_rows[0] == orchestrator._PAPER_GROUP_HEADER
    assert paper_rows[1] == orchestrator._PAPER_FIELD_HEADER
    assert [row[0] for row in paper_rows[2:]] == ["node_shared_lstm", "crossformer", "stcn"]
    assert paper_rows[2][4:9] == ["1.000", "2.000", "0.000", "0.600", ""]
    assert paper_rows[2][8] == ""
    assert paper_rows[2][13] == ""
    assert paper_rows[2][11] == "-1.250"
    assert paper_rows[2][16] == ""
    assert paper_rows[4][4] == ""

    with (
            run_root / "model_comparison_flat.csv"
    ).open(
        encoding="utf-8",
        newline="",
    ) as handle:
        flat_rows = list(csv.DictReader(handle))

    assert list(flat_rows[0]) == orchestrator.MODEL_COMPARISON_FIELDS

    assert flat_rows[0]["H3_Official_Score"] == "0.600"
    assert flat_rows[0]["H6_R2"] == "-1.250"
    assert flat_rows[0]["H10_R2"] == ""

    assert "H3_MAPE" not in flat_rows[0]
    assert "H3_SMAPE" not in flat_rows[0]

    assert flat_rows[0]["metric_note"] == "R2_UNDEFINED"


def test_summarize_rebuilds_legacy_csv_from_existing_top_level_metrics(tmp_path: Path) -> None:
    result_dir = _result_dir(tmp_path)
    run_root = tmp_path / "results" / "_runs" / "legacy-run"
    run_root.mkdir(parents=True)
    (run_root / "status.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "legacy-run",
                "operation": "train",
                "models": [
                    {
                        "model": "node_shared_lstm",
                        "status": PASS,
                        "classification": "PASS_SMOKE",
                        "result_dir": str(result_dir),
                        "runtime_environment": "tslib",
                        "python_executable": sys.executable,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = orchestrator.summarize_existing_run(
        run_id="legacy-run", project_root=tmp_path
    )
    assert result["model_count"] == 1
    with Path(result["model_comparison_flat_csv"]).open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["H3_R2"] == "0.000"
    assert "H3_MAPE" not in row
    assert "H3_SMAPE" not in row
    saved = json.loads((run_root / "status.json").read_text(encoding="utf-8"))
    assert saved["schema_version"] == 2
    assert saved["models"][0]["metrics_complete"] is True

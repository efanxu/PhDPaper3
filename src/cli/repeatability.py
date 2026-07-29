"""Repeatability orchestration using two fully independent worker processes."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .orchestrator import run_training_models
from runtime.paths import archive_directory, effective_run_id, project_root_from_config, resolve_output_root
from runtime.run_info import write_json


def _remove_run_id(value: dict[str, Any]) -> dict[str, Any]:
    copied = json.loads(json.dumps(value))
    copied.get("resolved", {}).pop("run_id", None)
    return copied


def _history(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            {key: row[key] for key in ("epoch", "train_loss", "monitor", "learning_rate", "checkpoint_selected", "train_updates")}
            for row in csv.DictReader(handle)
        ]


def _values_close(left: Any, right: Any, *, atol: float) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_values_close(left[key], right[key], atol=atol) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_values_close(a, b, atol=atol) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=atol)
    return left == right


def _compare_one(
    *,
    model: str,
    first_dir: Path,
    second_dir: Path,
    prediction_atol: float,
    metric_atol: float,
) -> dict[str, Any]:
    first_config = yaml.safe_load((first_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    second_config = yaml.safe_load((second_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    first_info = json.loads((first_dir / "run_info.json").read_text(encoding="utf-8"))
    second_info = json.loads((second_dir / "run_info.json").read_text(encoding="utf-8"))
    first_npz = np.load(first_dir / "predictions.npz", allow_pickle=False)
    second_npz = np.load(second_dir / "predictions.npz", allow_pickle=False)
    checks: list[tuple[str, bool]] = []
    checks.append(("resolved_config", _remove_run_id(first_config) == _remove_run_id(second_config)))
    checks.append(("initial_weights", first_info.get("initial_weight_hash") == second_info.get("initial_weight_hash")))
    checks.append(("first_step_loss", first_info.get("first_step_loss") == second_info.get("first_step_loss")))
    checks.append(("short_training_loss_curve", _history(first_dir / "train_history.csv") == _history(second_dir / "train_history.csv")))
    checks.append(("best_epoch", first_info.get("best_epoch") == second_info.get("best_epoch")))
    first_validation = json.loads((first_dir / "metrics_validation.json").read_text(encoding="utf-8"))
    second_validation = json.loads((second_dir / "metrics_validation.json").read_text(encoding="utf-8"))
    checks.append(("validation_metrics", _values_close(first_validation, second_validation, atol=metric_atol)))
    max_prediction_diff = 0.0
    for key in first_npz.files:
        if key not in second_npz.files:
            checks.append((f"predictions:{key}", False))
            continue
        left = first_npz[key]
        right = second_npz[key]
        diff = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))) if left.size else 0.0
        max_prediction_diff = max(max_prediction_diff, diff)
    checks.append(("predictions", max_prediction_diff <= prediction_atol))
    first_test = json.loads((first_dir / "metrics_test_h10.json").read_text(encoding="utf-8")) if (first_dir / "metrics_test_h10.json").is_file() else json.loads((first_dir / "metrics_test_h3.json").read_text(encoding="utf-8"))
    second_test = json.loads((second_dir / "metrics_test_h10.json").read_text(encoding="utf-8")) if (second_dir / "metrics_test_h10.json").is_file() else json.loads((second_dir / "metrics_test_h3.json").read_text(encoding="utf-8"))
    checks.append(("final_metrics", _values_close(first_test, second_test, atol=metric_atol)))
    first_failure = next((name for name, passed in checks if not passed), None)
    return {
        "passed": first_failure is None,
        "model": model,
        "checks": {name: passed for name, passed in checks},
        "first_failure": first_failure,
        "max_prediction_difference": max_prediction_diff,
        "prediction_atol": prediction_atol,
        "metric_atol": metric_atol,
    }


def compare_repeated_runs(
    *,
    models: list[str],
    config_path: str | Path,
    model_config_path: str | Path | None,
    run_id: str | None = None,
    device: str = "auto",
    output_root: str | Path | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
    prediction_atol: float = 1e-6,
    metric_atol: float = 0.0,
    overwrite: bool = False,
    id_suffix: str | None = None,
    command_argv: list[str] | None = None,
) -> dict[str, Any]:
    if prediction_atol < 0 or metric_atol < 0:
        raise ValueError("repeatability tolerances must be non-negative")
    if not models:
        raise ValueError("at least one model is required")
    config_file = Path(config_path).resolve()
    root = project_root_from_config(config_file)
    results_root = resolve_output_root(root, output_root)
    base_id = effective_run_id(run_id, id_suffix)
    repeat_root = results_root / "_repeatability" / base_id
    repeat_root.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    for model in models:
        report_dir = repeat_root / model
        if report_dir.exists() and any(report_dir.iterdir()):
            if not overwrite:
                raise ValueError(f"repeatability report already exists: {report_dir}; choose --overwrite or --id-suffix")
            archive_directory(report_dir, results_root / "_archive" / "_repeatability" / model, label=base_id)
        report_dir.mkdir(parents=True, exist_ok=True)
        run_a_id = f"{base_id}__repeat_a"
        run_b_id = f"{base_id}__repeat_b"
        first = run_training_models(models=[model], config_path=config_file, model_config_path=model_config_path, run_id=run_a_id, device=device, output_root=results_root, resume=False, overwrite=overwrite, id_suffix=None, fail_fast=True, smoke=True, smoke_epochs=1, smoke_max_train_updates=2, smoke_max_eval_batches=2, cli_overrides=cli_overrides, command_argv=command_argv)
        first_dir = Path(first["models"][0]["result_dir"])
        write_json(report_dir / "run_a.json", first)
        second = run_training_models(models=[model], config_path=config_file, model_config_path=model_config_path, run_id=run_b_id, device=device, output_root=results_root, resume=False, overwrite=overwrite, id_suffix=None, fail_fast=True, smoke=True, smoke_epochs=1, smoke_max_train_updates=2, smoke_max_eval_batches=2, cli_overrides=cli_overrides, command_argv=command_argv)
        second_dir = Path(second["models"][0]["result_dir"])
        write_json(report_dir / "run_b.json", second)
        if first["passed"] and second["passed"]:
            comparison = _compare_one(model=model, first_dir=first_dir, second_dir=second_dir, prediction_atol=prediction_atol, metric_atol=metric_atol)
        else:
            comparison = {"passed": False, "model": model, "checks": {}, "first_failure": "worker_failure", "max_prediction_difference": None, "prediction_atol": prediction_atol, "metric_atol": metric_atol}
        comparison.update({"run_a": str(first_dir), "run_b": str(second_dir), "subprocess_pids": [first["models"][0].get("pid"), second["models"][0].get("pid")]})
        write_json(report_dir / "repeatability_report.json", comparison)
        reports.append(comparison)
    return {"passed": all(item["passed"] for item in reports), "run_id": base_id, "models": reports, "repeatability_root": str(repeat_root)}

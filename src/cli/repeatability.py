"""Repeatability orchestration using two fully independent worker processes."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping
from pathlib import Path
import re
from typing import Any

import numpy as np
import yaml

from .orchestrator import (
    _prepare_batch_environments,
    _validate_model_configs,
    extract_metric_value,
    run_training_models,
)
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


_HORIZON_FILE = re.compile(r"^metrics_test_h(?P<horizon>[1-9][0-9]*)\.json$")
_REPEATABILITY_METRICS = ("MAE", "RMSE", "R2", "SMAPE", "MAPE", "Official_Score")


def _test_metric_files(path: Path) -> dict[int, Path]:
    return {
        int(match.group("horizon")): item
        for item in sorted(path.glob("metrics_test_h*.json"), key=lambda value: value.name)
        if (match := _HORIZON_FILE.fullmatch(item.name)) is not None
    }


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
    first_pid: int | None = None,
    second_pid: int | None = None,
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
    for field in (
        "runtime_environment",
        "conda_env",
        "python_executable",
        "environment_resolution_source",
        "python_version",
        "pytorch_version",
        "cuda_version",
    ):
        checks.append((f"environment:{field}", first_info.get(field) == second_info.get(field)))
    first_validation = json.loads((first_dir / "metrics_validation.json").read_text(encoding="utf-8"))
    second_validation = json.loads((second_dir / "metrics_validation.json").read_text(encoding="utf-8"))
    checks.append(("validation_metrics", _values_close(first_validation, second_validation, atol=metric_atol)))
    max_prediction_diff = 0.0
    checks.append(("prediction_keys", set(first_npz.files) == set(second_npz.files)))
    for key in first_npz.files:
        if key not in second_npz.files:
            checks.append((f"predictions:{key}", False))
            continue
        left = first_npz[key]
        right = second_npz[key]
        diff = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))) if left.size else 0.0
        max_prediction_diff = max(max_prediction_diff, diff)
    checks.append(("predictions", max_prediction_diff <= prediction_atol))
    first_test_files = _test_metric_files(first_dir)
    second_test_files = _test_metric_files(second_dir)
    first_horizons = set(first_test_files)
    second_horizons = set(second_test_files)
    horizon_set_matches = first_horizons == second_horizons
    checks.append(("test_horizon_set", horizon_set_matches))
    horizon_reports: dict[str, dict[str, Any]] = {}
    for horizon in sorted(first_horizons | second_horizons):
        first_path = first_test_files.get(horizon)
        second_path = second_test_files.get(horizon)
        if first_path is None or second_path is None:
            missing_in = "run_a" if first_path is None else "run_b"
            horizon_report = {
                "match": False,
                "metric_differences": {"horizon_missing_in": missing_in},
            }
            checks.append((f"test_metrics:metrics_test_h{horizon}.json", False))
            horizon_reports[str(horizon)] = horizon_report
            continue
        first_test = json.loads(first_path.read_text(encoding="utf-8"))
        second_test = json.loads(second_path.read_text(encoding="utf-8"))
        metric_differences: dict[str, Any] = {}
        for metric_name in _REPEATABILITY_METRICS:
            first_value = extract_metric_value(first_test, metric_name)
            second_value = extract_metric_value(second_test, metric_name)
            if first_value is None and second_value is None:
                continue
            if not _values_close(first_value, second_value, atol=metric_atol):
                metric_differences[metric_name] = {
                    "run_a": first_value,
                    "run_b": second_value,
                }
        horizon_match = not metric_differences
        checks.append((f"test_metrics:metrics_test_h{horizon}.json", horizon_match))
        horizon_reports[str(horizon)] = {
            "match": horizon_match,
            "metric_differences": metric_differences,
        }
    checks.append(("different_worker_pid", first_pid is not None and second_pid is not None and first_pid != second_pid))
    first_failure = next((name for name, passed in checks if not passed), None)
    return {
        "passed": first_failure is None,
        "model": model,
        "checks": {name: passed for name, passed in checks},
        "first_failure": first_failure,
        "max_prediction_difference": max_prediction_diff,
        "prediction_atol": prediction_atol,
        "metric_atol": metric_atol,
        "horizons": horizon_reports,
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
    environment_preflight_only: bool = False,
) -> dict[str, Any]:
    if prediction_atol < 0 or metric_atol < 0:
        raise ValueError("repeatability tolerances must be non-negative")
    if not models:
        raise ValueError("at least one model is required")
    if len(set(models)) != len(models):
        raise ValueError("repeatability requires unique model names")
    if model_config_path is not None and len(models) > 1:
        raise ValueError("--model-config is only valid with exactly one --model; multi-model repeatability loads configs/models/<model>.yaml separately")
    config_file = Path(config_path).resolve()
    root = project_root_from_config(config_file)
    results_root = resolve_output_root(root, output_root)
    base_id = effective_run_id(run_id, id_suffix)
    if environment_preflight_only:
        model_configs = _validate_model_configs(models, config_file, model_config_path)
        model_environments, preflight_results = _prepare_batch_environments(
            models=list(models),
            model_configs=model_configs,
            project_root=root,
            device=device,
            config_path=config_file,
        )
        results = [
            {
                "model": model,
                "status": "PREFLIGHTED",
                "runtime_environment": model_environments[model].environment_id,
                "conda_env": model_environments[model].conda_env,
                "python_executable": str(model_environments[model].python_executable),
                "python_version": preflight_results[
                    model_environments[model].environment_id
                ].get("python_version"),
                "environment_resolution_source": model_environments[model].resolution_source,
                "exit_code": 0,
            }
            for model in models
        ]
        return {
            "passed": True,
            "run_id": base_id,
            "models": results,
            "environment_preflight_only": True,
        }
    repeat_root = results_root / "_repeatability" / base_id
    if repeat_root.exists() and any(repeat_root.iterdir()):
        if not overwrite:
            raise ValueError(
                f"repeatability report already exists: {repeat_root}; choose --overwrite or --id-suffix"
            )
        archive_directory(
            repeat_root,
            results_root / "_archive" / "_repeatability",
            label=base_id,
        )
    repeat_root.mkdir(parents=True, exist_ok=True)

    run_a_id = f"{base_id}__repeat_a"
    run_b_id = f"{base_id}__repeat_b"
    environment_context_holder: dict[str, Any] = {}
    first = run_training_models(
        models=list(models),
        config_path=config_file,
        model_config_path=model_config_path,
        run_id=run_a_id,
        device=device,
        output_root=results_root,
        resume=False,
        overwrite=overwrite,
        id_suffix=None,
        fail_fast=False,
        smoke=True,
        smoke_epochs=1,
        smoke_max_train_updates=2,
        smoke_max_eval_batches=2,
        cli_overrides=cli_overrides,
        command_argv=command_argv,
        environment_context_holder=environment_context_holder,
    )
    second = run_training_models(
        models=list(models),
        config_path=config_file,
        model_config_path=model_config_path,
        run_id=run_b_id,
        device=device,
        output_root=results_root,
        resume=False,
        overwrite=overwrite,
        id_suffix=None,
        fail_fast=False,
        smoke=True,
        smoke_epochs=1,
        smoke_max_train_updates=2,
        smoke_max_eval_batches=2,
        cli_overrides=cli_overrides,
        command_argv=command_argv,
        environment_context_holder=environment_context_holder,
    )
    write_json(repeat_root / "run_a.json", first)
    write_json(repeat_root / "run_b.json", second)

    first_by_model = {item["model"]: item for item in first["models"]}
    second_by_model = {item["model"]: item for item in second["models"]}
    reports: list[dict[str, Any]] = []

    def worker_succeeded(record: Mapping[str, Any] | None) -> bool:
        if record is None:
            return False
        status = record.get("status")
        if status is None:
            # Keep the seam usable for lightweight test doubles that predate
            # the scheduler's explicit status field.
            return True
        return status in {"COMPLETED", "RESUMED", "OVERWRITTEN", "SKIPPED_COMPLETED"}

    for model in models:
        first_record = first_by_model.get(model)
        second_record = second_by_model.get(model)
        first_dir = Path(first_record["result_dir"]) if first_record and first_record.get("result_dir") else Path()
        second_dir = Path(second_record["result_dir"]) if second_record and second_record.get("result_dir") else Path()
        report_dir = repeat_root / model
        report_dir.mkdir(parents=True, exist_ok=True)
        if worker_succeeded(first_record) and worker_succeeded(second_record):
            comparison = _compare_one(
                model=model,
                first_dir=first_dir,
                second_dir=second_dir,
                prediction_atol=prediction_atol,
                metric_atol=metric_atol,
                first_pid=first_record.get("pid") if first_record else None,
                second_pid=second_record.get("pid") if second_record else None,
            )
        else:
            comparison = {
                "passed": False,
                "model": model,
                "checks": {
                    "worker_a_succeeded": worker_succeeded(first_record),
                    "worker_b_succeeded": worker_succeeded(second_record),
                    "different_worker_pid": first_record is not None
                    and second_record is not None
                    and first_record.get("pid") is not None
                    and second_record.get("pid") is not None
                    and first_record.get("pid") != second_record.get("pid"),
                },
                "first_failure": "worker_failure",
                "max_prediction_difference": None,
                "prediction_atol": prediction_atol,
                "metric_atol": metric_atol,
                "horizons": {},
            }
        comparison.update(
            {
                "run_a": str(first_dir),
                "run_b": str(second_dir),
                "subprocess_pids": [
                    first_record.get("pid") if first_record else None,
                    second_record.get("pid") if second_record else None,
                ],
                "batch_run_a": str(first.get("run_root")),
                "batch_run_b": str(second.get("run_root")),
            }
        )
        write_json(report_dir / "repeatability_report.json", comparison)
        reports.append(comparison)
    return {
        "passed": bool(reports) and all(item["passed"] for item in reports),
        "run_id": base_id,
        "models": reports,
        "repeatability_root": str(repeat_root),
        "run_a": first,
        "run_b": second,
    }

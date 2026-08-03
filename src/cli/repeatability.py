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
    _prepare_batch_environments_with_seed,
    _resolved_training_seed,
    _validate_model_configs,
    extract_metric_value,
    run_training_models,
    validate_unique_models,
)
from runtime.paths import archive_directory, effective_run_id, project_root_from_config, resolve_output_root
from runtime.status import (
    FAILED,
    PASS,
    REPEATABILITY,
    write_status,
)


def _remove_run_id(value: dict[str, Any]) -> dict[str, Any]:
    copied = json.loads(json.dumps(value))
    copied.get("resolved", {}).pop("run_id", None)
    return copied


def _history(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows: list[dict[str, Any]] = []
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "train_loss": float(row["train_loss"]),
                    "monitor": float(row["monitor"]),
                    "learning_rate": float(row["learning_rate"]),
                    "checkpoint_selected": row["checkpoint_selected"].casefold() == "true",
                    "train_updates": int(row["train_updates"]),
                }
            )
        return rows


_HORIZON_FILE = re.compile(r"^metrics_test_h(?P<horizon>[1-9][0-9]*)\.json$")
_REPEATABILITY_METRICS = ("MAE", "RMSE", "R2", "SMAPE", "MAPE", "Official_Score")


def _test_metric_files(path: Path) -> dict[int, Path]:
    return {
        int(match.group("horizon")): item
        for item in sorted(path.glob("metrics_test_h*.json"), key=lambda value: value.name)
        if (match := _HORIZON_FILE.fullmatch(item.name)) is not None
    }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _plain_number(value: Any) -> int | float:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _numeric_compare(
    left: Any,
    right: Any,
    *,
    atol: float,
    rtol: float,
    include_equal: bool = False,
    path: str = "",
) -> dict[str, Any]:
    """Compare nested values with one fixed absolute/relative tolerance."""

    differences: dict[str, Any] = {}
    max_absolute = 0.0
    max_relative = 0.0
    passed = True

    def visit(first: Any, second: Any, location: str) -> None:
        nonlocal max_absolute, max_relative, passed
        if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
            if not isinstance(first, np.ndarray) or not isinstance(second, np.ndarray):
                passed = False
                max_absolute = float("inf")
                max_relative = float("inf")
                differences[location or "$"] = {"run_a": first, "run_b": second}
                return
            result = _array_numeric_compare(first, second, atol=atol, rtol=rtol)
            passed = passed and bool(result["passed"])
            max_absolute = max(max_absolute, float(result["max_absolute_difference"]))
            max_relative = max(max_relative, float(result["max_relative_difference"]))
            if result["difference"] is not None:
                differences[location or "$"] = result["difference"]
            return
        if isinstance(first, Mapping) or isinstance(second, Mapping):
            if not isinstance(first, Mapping) or not isinstance(second, Mapping) or set(first) != set(second):
                passed = False
                max_absolute = float("inf")
                max_relative = float("inf")
                differences[location or "$"] = {"run_a": first, "run_b": second}
                return
            for key in first:
                child = f"{location}.{key}" if location else str(key)
                visit(first[key], second[key], child)
            return
        if isinstance(first, (list, tuple)) or isinstance(second, (list, tuple)):
            if not isinstance(first, (list, tuple)) or not isinstance(second, (list, tuple)) or len(first) != len(second):
                passed = False
                max_absolute = float("inf")
                max_relative = float("inf")
                differences[location or "$"] = {"run_a": first, "run_b": second}
                return
            for index, (first_item, second_item) in enumerate(zip(first, second)):
                visit(first_item, second_item, f"{location}[{index}]")
            return
        if _is_number(first) or _is_number(second):
            if not _is_number(first) or not _is_number(second):
                passed = False
                max_absolute = float("inf")
                max_relative = float("inf")
                differences[location or "$"] = {"run_a": first, "run_b": second}
                return
            first_number = float(first)
            second_number = float(second)
            finite = math.isfinite(first_number) and math.isfinite(second_number)
            absolute = abs(first_number - second_number) if finite else float("inf")
            relative = (
                absolute / max(abs(first_number), abs(second_number), 1.0)
                if finite
                else float("inf")
            )
            threshold = max(atol, rtol * max(abs(first_number), abs(second_number), 1.0)) if finite else -1.0
            scalar_passed = finite and absolute <= threshold
            passed = passed and scalar_passed
            max_absolute = max(max_absolute, absolute)
            max_relative = max(max_relative, relative)
            if include_equal or not scalar_passed or absolute != 0.0:
                differences[location or "$"] = {
                    "run_a": _plain_number(first),
                    "run_b": _plain_number(second),
                    "absolute_difference": absolute,
                    "relative_difference": relative,
                    "atol": atol,
                    "rtol": rtol,
                    "finite": finite,
                }
            return
        if first != second:
            passed = False
            max_absolute = float("inf")
            max_relative = float("inf")
            differences[location or "$"] = {"run_a": first, "run_b": second}

    visit(left, right, path)
    return {
        "passed": passed,
        "max_absolute_difference": max_absolute,
        "max_relative_difference": max_relative,
        "differences": differences,
    }


def _array_numeric_compare(
    left: np.ndarray,
    right: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if left.shape != right.shape:
        return {
            "passed": False,
            "max_absolute_difference": float("inf"),
            "max_relative_difference": float("inf"),
            "difference": {"shape_a": list(left.shape), "shape_b": list(right.shape)},
        }
    if left.size == 0:
        return {
            "passed": True,
            "max_absolute_difference": 0.0,
            "max_relative_difference": 0.0,
            "difference": None,
        }
    try:
        first = left.astype(np.float64, copy=False)
        second = right.astype(np.float64, copy=False)
    except (TypeError, ValueError):
        equal = bool(np.array_equal(left, right))
        return {
            "passed": equal,
            "max_absolute_difference": 0.0 if equal else float("inf"),
            "max_relative_difference": 0.0 if equal else float("inf"),
            "difference": None if equal else {"arrays_equal": False},
        }
    finite = np.isfinite(first) & np.isfinite(second)
    if not bool(finite.all()):
        return {
            "passed": False,
            "max_absolute_difference": float("inf"),
            "max_relative_difference": float("inf"),
            "difference": {"finite": False},
        }
    absolute = np.abs(first - second)
    relative = absolute / np.maximum(np.maximum(np.abs(first), np.abs(second)), 1.0)
    threshold = np.maximum(atol, rtol * np.maximum(np.maximum(np.abs(first), np.abs(second)), 1.0))
    maximum_absolute = float(np.max(absolute))
    maximum_relative = float(np.max(relative))
    passed = bool(np.all(absolute <= threshold))
    return {
        "passed": passed,
        "max_absolute_difference": maximum_absolute,
        "max_relative_difference": maximum_relative,
        "difference": {
            "absolute_difference": maximum_absolute,
            "relative_difference": maximum_relative,
            "atol": atol,
            "rtol": rtol,
        }
        if maximum_absolute != 0.0 or not passed
        else None,
    }


def _values_close(left: Any, right: Any, *, atol: float, rtol: float = 0.0) -> bool:
    return bool(_numeric_compare(left, right, atol=atol, rtol=rtol)["passed"])


def _compare_one(
    *,
    model: str,
    first_dir: Path,
    second_dir: Path,
    prediction_atol: float,
    prediction_rtol: float,
    metric_atol: float,
    metric_rtol: float,
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
    first_model_config = yaml.safe_load((first_dir / "model_config.yaml").read_text(encoding="utf-8")) if (first_dir / "model_config.yaml").is_file() else None
    second_model_config = yaml.safe_load((second_dir / "model_config.yaml").read_text(encoding="utf-8")) if (second_dir / "model_config.yaml").is_file() else None
    checks.append(("model_config", first_model_config == second_model_config))
    checks.append(("initial_weights", first_info.get("initial_weight_hash") == second_info.get("initial_weight_hash")))
    checks.append(("parameter_count", first_info.get("parameter_count") == second_info.get("parameter_count")))
    checks.append(("data_split", first_info.get("data_split") == second_info.get("data_split")))
    checks.append(("train_batch_order", first_info.get("train_batch_order") == second_info.get("train_batch_order")))
    checks.append(("train_batch_count", first_info.get("train_batch_count") == second_info.get("train_batch_count")))
    for field in (
        "runtime_environment",
        "conda_env",
        "python_executable",
        "environment_resolution_source",
        "python_version",
        "pytorch_version",
        "cuda_version",
        "reproducibility_mode",
    ):
        checks.append((f"environment:{field}", first_info.get(field) == second_info.get(field)))
    first_history = _history(first_dir / "train_history.csv")
    second_history = _history(second_dir / "train_history.csv")
    checks.append(("epoch_sequence", [row["epoch"] for row in first_history] == [row["epoch"] for row in second_history]))
    checks.append(("train_updates", [row["train_updates"] for row in first_history] == [row["train_updates"] for row in second_history]))
    checks.append(("optimizer_update_count", first_info.get("train_update_count") == second_info.get("train_update_count")))
    history_numeric = _numeric_compare(
        [
            {field: row[field] for field in ("train_loss", "monitor", "learning_rate")}
            for row in first_history
        ],
        [
            {field: row[field] for field in ("train_loss", "monitor", "learning_rate")}
            for row in second_history
        ],
        atol=metric_atol,
        rtol=metric_rtol,
        include_equal=True,
    )
    checks.append(("short_training_loss_curve", bool(history_numeric["passed"])))
    first_step = _numeric_compare(
        first_info.get("first_step_loss"),
        second_info.get("first_step_loss"),
        atol=metric_atol,
        rtol=metric_rtol,
        include_equal=True,
    )
    checks.append(("first_step_loss", bool(first_step["passed"])))
    first_validation = json.loads((first_dir / "metrics_validation.json").read_text(encoding="utf-8"))
    second_validation = json.loads((second_dir / "metrics_validation.json").read_text(encoding="utf-8"))
    validation_numeric = _numeric_compare(
        first_validation,
        second_validation,
        atol=metric_atol,
        rtol=metric_rtol,
        include_equal=True,
    )
    checks.append(("validation_metrics", bool(validation_numeric["passed"])))
    max_prediction_diff = 0.0
    max_prediction_relative_diff = 0.0
    prediction_failed = False
    checks.append(("prediction_keys", set(first_npz.files) == set(second_npz.files)))
    for key in first_npz.files:
        if key not in second_npz.files:
            prediction_failed = True
            continue
        left = first_npz[key]
        right = second_npz[key]
        if any(token in key.casefold() for token in ("target", "mask", "starts")):
            equal = bool(np.array_equal(left, right))
            checks.append((f"data:{key}", equal))
            if not equal:
                prediction_failed = True
            continue
        result = _array_numeric_compare(left, right, atol=prediction_atol, rtol=prediction_rtol)
        prediction_failed = prediction_failed or not bool(result["passed"])
        max_prediction_diff = max(max_prediction_diff, float(result["max_absolute_difference"]))
        max_prediction_relative_diff = max(
            max_prediction_relative_diff,
            float(result["max_relative_difference"]),
        )
    checks.append(("predictions", not prediction_failed))
    first_test_files = _test_metric_files(first_dir)
    second_test_files = _test_metric_files(second_dir)
    first_horizons = set(first_test_files)
    second_horizons = set(second_test_files)
    horizon_set_matches = first_horizons == second_horizons
    checks.append(("test_horizon_set", horizon_set_matches))
    horizon_reports: dict[str, dict[str, Any]] = {}
    test_metrics_passed = True
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
            test_metrics_passed = False
            horizon_reports[str(horizon)] = horizon_report
            continue
        first_test = json.loads(first_path.read_text(encoding="utf-8"))
        second_test = json.loads(second_path.read_text(encoding="utf-8"))
        metric_differences: dict[str, Any] = {}
        horizon_passed = True
        for metric_name in _REPEATABILITY_METRICS:
            first_value = extract_metric_value(first_test, metric_name)
            second_value = extract_metric_value(second_test, metric_name)
            if first_value is None and second_value is None:
                continue
            metric_result = _numeric_compare(
                first_value,
                second_value,
                atol=metric_atol,
                rtol=metric_rtol,
                include_equal=True,
                path=metric_name,
            )
            metric_differences[metric_name] = {
                "passed": bool(metric_result["passed"]),
                "max_absolute_difference": float(metric_result["max_absolute_difference"]),
                "max_relative_difference": float(metric_result["max_relative_difference"]),
                "details": metric_result["differences"],
            }
            horizon_passed = horizon_passed and bool(metric_result["passed"])
        horizon_match = horizon_passed
        test_metrics_passed = test_metrics_passed and horizon_match
        checks.append((f"test_metrics:metrics_test_h{horizon}.json", horizon_match))
        horizon_reports[str(horizon)] = {
            "match": horizon_match,
            "metric_differences": metric_differences,
        }
    checks.append(("different_worker_pid", first_pid is not None and second_pid is not None and first_pid != second_pid))
    best_epoch_exact = first_info.get("best_epoch") == second_info.get("best_epoch")
    best_monitor = _numeric_compare(
        first_info.get("best_metric"),
        second_info.get("best_metric"),
        atol=metric_atol,
        rtol=metric_rtol,
        include_equal=True,
    )
    selection_differences = [
        {
            "epoch": first_row["epoch"],
            "run_a": first_row["checkpoint_selected"],
            "run_b": second_row["checkpoint_selected"],
        }
        for first_row, second_row in zip(first_history, second_history)
        if first_row["checkpoint_selected"] != second_row["checkpoint_selected"]
    ]
    selection_tie = (
        not best_epoch_exact
        and first_info.get("best_metric") is not None
        and second_info.get("best_metric") is not None
        and bool(best_monitor["passed"])
        and bool(validation_numeric["passed"])
        and test_metrics_passed
    )
    checks.append(("best_epoch", best_epoch_exact or selection_tie))
    numeric_changed = any(
        float(result["max_absolute_difference"]) != 0.0
        for result in (first_step, history_numeric, validation_numeric, best_monitor)
    ) or max_prediction_diff != 0.0 or any(
        report["metric_differences"]
        and any(
            isinstance(metric, Mapping)
            and float(metric.get("max_absolute_difference", float("inf"))) != 0.0
            for metric in report["metric_differences"].values()
        )
        for report in horizon_reports.values()
    )
    repeatability_level = (
        "NUMERICAL_WITH_SELECTION_TIE"
        if selection_tie
        else "NUMERICAL"
        if numeric_changed
        else "EXACT"
    )
    first_failure = next((name for name, passed in checks if not passed), None)
    return {
        "passed": first_failure is None,
        "model": model,
        "repeatability_level": repeatability_level,
        "checks": {name: passed for name, passed in checks},
        "first_failure": first_failure,
        "max_prediction_difference": max_prediction_diff,
        "max_prediction_absolute_difference": max_prediction_diff,
        "max_prediction_relative_difference": max_prediction_relative_diff,
        "first_step_loss_difference": float(first_step["max_absolute_difference"]),
        "train_history_max_differences": {
            field: {
                "absolute": float(
                    _numeric_compare(
                        [row[field] for row in first_history],
                        [row[field] for row in second_history],
                        atol=metric_atol,
                        rtol=metric_rtol,
                    )["max_absolute_difference"]
                ),
                "relative": float(
                    _numeric_compare(
                        [row[field] for row in first_history],
                        [row[field] for row in second_history],
                        atol=metric_atol,
                        rtol=metric_rtol,
                    )["max_relative_difference"]
                ),
            }
            for field in ("train_loss", "monitor", "learning_rate")
        },
        "validation_metric_differences": validation_numeric["differences"],
        "checkpoint_selection_differences": selection_differences,
        "best_epoch_exact": best_epoch_exact,
        "selection_tie": selection_tie,
        "prediction_atol": prediction_atol,
        "prediction_rtol": prediction_rtol,
        "metric_atol": metric_atol,
        "metric_rtol": metric_rtol,
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
    prediction_atol: float = 5e-3,
    prediction_rtol: float = 5e-3,
    metric_atol: float = 2e-4,
    metric_rtol: float = 2e-4,
    overwrite: bool = False,
    id_suffix: str | None = None,
    command_argv: list[str] | None = None,
    environment_preflight_only: bool = False,
) -> dict[str, Any]:
    if any(
        not math.isfinite(float(value)) or float(value) < 0
        for value in (prediction_atol, prediction_rtol, metric_atol, metric_rtol)
    ):
        raise ValueError("repeatability tolerances must be non-negative")
    if not models:
        raise ValueError("at least one model is required")
    validate_unique_models(models)
    if model_config_path is not None and len(models) > 1:
        raise ValueError("--model-config is only valid with exactly one --model; multi-model repeatability loads configs/models/<model>.yaml separately")
    config_file = Path(config_path).resolve()
    root = project_root_from_config(config_file)
    results_root = resolve_output_root(root, output_root)
    base_id = effective_run_id(run_id, id_suffix)
    if environment_preflight_only:
        model_configs = _validate_model_configs(models, config_file, model_config_path)
        model_environments, preflight_results = _prepare_batch_environments_with_seed(
            models=list(models),
            model_configs=model_configs,
            project_root=root,
            device=device,
            config_path=config_file,
            resolved_seed=_resolved_training_seed(
                config_file,
                cli_overrides,
                project_root=root,
            ),
        )
        results = [
            {
                "schema_version": 2,
                "model": model,
                "operation": "preflight",
                "profile": None,
                "status": PASS,
                "classification": None,
                "phase": "preflight",
                "error": None,
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
            "schema_version": 2,
            "passed": True,
            "run_id": base_id,
            "operation": "preflight",
            "status": PASS,
            "classification": None,
            "phase": "preflight",
            "error": None,
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
        models=list(reversed(models)),
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
    write_status(repeat_root / "run_a.json", first)
    write_status(repeat_root / "run_b.json", second)

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
        return status == PASS

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
                prediction_rtol=prediction_rtol,
                metric_atol=metric_atol,
                metric_rtol=metric_rtol,
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
                "max_prediction_absolute_difference": None,
                "max_prediction_relative_difference": None,
                "first_step_loss_difference": None,
                "train_history_max_differences": {},
                "validation_metric_differences": {},
                "repeatability_level": "NUMERICAL",
                "prediction_atol": prediction_atol,
                "prediction_rtol": prediction_rtol,
                "metric_atol": metric_atol,
                "metric_rtol": metric_rtol,
                "horizons": {},
            }
        comparison.update(
            {
                "schema_version": 2,
                "operation": "repeatability",
                "status": PASS if comparison["passed"] else FAILED,
                "classification": None if comparison["passed"] else REPEATABILITY,
                "profile": REPEATABILITY,
                "phase": "evaluation" if comparison["passed"] else "overall",
                "error": None
                if comparison["passed"]
                else {
                    "code": "REPEATABILITY_MISMATCH",
                    "type": None,
                    "message": comparison.get("first_failure"),
                    "traceback_tail": None,
                },
                "run_a": str(first_dir),
                "run_b": str(second_dir),
                "subprocess_pids": [
                    first_record.get("pid") if first_record else None,
                    second_record.get("pid") if second_record else None,
                ],
                "batch_run_a": str(first.get("run_root")),
                "batch_run_b": str(second.get("run_root")),
                "model_order_run_a": list(models),
                "model_order_run_b": list(reversed(models)),
            }
        )
        write_status(report_dir / "repeatability_report.json", comparison)
        reports.append(comparison)
    passed = bool(reports) and all(item["passed"] for item in reports)
    status = {
        "schema_version": 2,
        "run_id": base_id,
        "operation": "repeatability",
        "status": PASS if passed else FAILED,
        "classification": None if passed else REPEATABILITY,
        "profile": REPEATABILITY,
        "phase": "evaluation" if passed else "overall",
        "error": None
        if passed
        else {
            "code": "REPEATABILITY_MISMATCH",
            "type": None,
            "message": next(
                (item.get("first_failure") for item in reports if item.get("first_failure")),
                "repeatability mismatch",
            ),
            "traceback_tail": None,
        },
        "models": reports,
        "model_order_run_a": list(models),
        "model_order_run_b": list(reversed(models)),
    }
    write_status(repeat_root / "status.json", status)
    return {
        "passed": passed,
        "run_id": base_id,
        "models": reports,
        "repeatability_root": str(repeat_root),
        "run_a": first,
        "run_b": second,
    }

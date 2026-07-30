"""Parent-process orchestration for isolated model workers."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Mapping
from io import StringIO

import yaml

from runtime.environments import (
    ResolvedEnvironment,
    build_worker_environment,
    preflight_environment,
    preflight_model,
    resolve_model_environment,
)
from runtime.paths import (
    archive_directory,
    effective_run_id,
    formal_result_exists,
    is_completed_run,
    project_root_from_config,
    resolve_output_root,
    run_directory,
)
from runtime.run_info import utc_now, write_json, write_text_atomic
from runtime.status import (
    ENVIRONMENT_PREFLIGHT,
    FAIL_CONFIG,
    FAIL_SIGNAL,
    FAIL_WORKER_CRASH,
    FAILED,
    FORMAL_DEFAULT_SHAPE,
    INTERFACE_SMALL,
    MODEL_PREFLIGHT,
    PASS,
    PENDING,
    RESOLVED_SHAPE,
    RUNNING,
    SKIPPED,
    TOP_LEVEL_STATUSES,
    classify_validation_failure,
    failure_summary,
    finished_phase,
    pass_classification,
    phase_record,
    running_phase,
    normalize_status_payload,
    write_validation_status,
    write_status,
)


def _default_model_config(config_file: Path, model: str) -> Path:
    return config_file.parent / "models" / f"{model}.yaml"


def validate_unique_models(models: list[str]) -> None:
    """Reject duplicate work before creating a run directory or starting a worker."""

    duplicates = sorted({model for model in models if models.count(model) > 1})
    if duplicates:
        names = ", ".join(duplicates)
        raise ValueError(
            f"--model contains duplicate model names: {names}; "
            "use repeatability or separate --run-id/--id-suffix runs instead"
        )


def _validate_model_configs(models: list[str], config_file: Path, explicit: str | Path | None) -> dict[str, Path]:
    if explicit is not None and len(models) > 1:
        raise ValueError("--model-config is only valid with exactly one --model; multi-model runs load configs/models/<model>.yaml separately")
    path = Path(explicit).resolve() if explicit is not None else None
    return {model: path if path is not None else _default_model_config(config_file, model).resolve() for model in models}


def _worker_script(project_root: Path) -> Path:
    return project_root / "scripts" / "run.py"


def preflight_batch_environments(
    *,
    models: list[str],
    model_configs: Mapping[str, Path],
    model_environments: Mapping[str, ResolvedEnvironment],
    project_root: Path,
    device: str,
) -> dict[str, dict[str, Any]]:
    """Preflight each distinct environment in a model batch exactly once."""

    preflight_results: dict[str, dict[str, Any]] = {}
    for model in models:
        resolved = model_environments[model]
        if resolved.environment_id in preflight_results:
            continue
        preflight_results[resolved.environment_id] = preflight_environment(
            resolved,
            project_root=project_root,
            device=device,
            model_name=model,
            model_config_path=model_configs[model],
        )
    return preflight_results


def prepare_batch_environments(
    *,
    models: list[str],
    model_configs: Mapping[str, Path],
    project_root: Path,
    device: str,
    config_path: Path | None = None,
) -> tuple[dict[str, ResolvedEnvironment], dict[str, dict[str, Any]]]:
    """Resolve every model and preflight each distinct environment once."""

    model_environments: dict[str, ResolvedEnvironment] = {}
    for model in models:
        model_environments[model] = resolve_model_environment(
            model_configs[model],
            project_root=project_root,
        )
    preflight_results = preflight_batch_environments(
        models=models,
        model_configs=model_configs,
        model_environments=model_environments,
        project_root=project_root,
        device=device,
    )
    for model in models:
        model_result = preflight_model(
            model_environments[model],
            project_root=project_root,
            model_name=model,
            config_path=config_path or project_root / "configs" / "experiment.yaml",
            model_config_path=model_configs[model],
        )
        preflight_results[model_environments[model].environment_id].setdefault(
            "model_preflights", {}
        )[model] = model_result
    return model_environments, preflight_results


def _prepare_batch_environments(
    *,
    models: list[str],
    model_configs: Mapping[str, Path],
    project_root: Path,
    device: str,
    config_path: Path | None = None,
) -> tuple[dict[str, ResolvedEnvironment], dict[str, dict[str, Any]]]:
    """Backward-compatible seam for tests and callers of the old private name."""

    return prepare_batch_environments(
        models=models,
        model_configs=model_configs,
        project_root=project_root,
        device=device,
        config_path=config_path,
    )


def _runtime_record_fields(
    resolved: ResolvedEnvironment,
    preflight_result: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "runtime_environment": resolved.environment_id,
        "conda_env": resolved.conda_env,
        "python_executable": str(resolved.python_executable),
        "python_version": preflight_result.get("python_version"),
        "environment_resolution_source": resolved.resolution_source,
    }


def build_model_worker_command(
    *,
    project_root: Path,
    request_path: Path,
    model_name: str,
    resolved_environment: ResolvedEnvironment,
) -> list[str]:
    """Build the isolated worker command from the model's resolved Python."""

    return [
        str(resolved_environment.python_executable),
        str(_worker_script(project_root)),
        "_worker",
        str(request_path),
        model_name,
    ]


PAPER_HORIZONS = (3, 6, 10)
MODEL_COMPARISON_FIELDS = [
    "model",
    "status",
    "parameter_count",
    "best_epoch",
    "H3_MAE",
    "H3_RMSE",
    "H3_R2",
    "H3_Official_Score",
    "H3_MAPE",
    "H3_SMAPE",
    "H6_MAE",
    "H6_RMSE",
    "H6_R2",
    "H6_Official_Score",
    "H6_MAPE",
    "H6_SMAPE",
    "H10_MAE",
    "H10_RMSE",
    "H10_R2",
    "H10_Official_Score",
    "H10_MAPE",
    "H10_SMAPE",
    "training_wall_seconds",
    "test_model_forward_seconds",
    "mean_sample_latency_ms",
    "samples_per_second",
    "peak_gpu_allocated_mb",
    "runtime_environment",
    "python_executable",
    "result_dir",
    "metric_note",
]

_PAPER_COMPARISON_FIELDS = MODEL_COMPARISON_FIELDS[:-1]
_PAPER_GROUP_HEADER = [
    "Model Information", "", "", "",
    "3-step", "", "", "", "", "",
    "6-step", "", "", "", "", "",
    "10-step", "", "", "", "", "",
    "Efficiency", "", "", "", "",
    "Runtime", "", "",
]
_PAPER_FIELD_HEADER = [
    "Model", "Status", "Parameter Count", "Best Epoch",
    "MAE", "RMSE", "R2", "Score", "MAPE", "SMAPE",
    "MAE", "RMSE", "R2", "Score", "MAPE", "SMAPE",
    "MAE", "RMSE", "R2", "Score", "MAPE", "SMAPE",
    "Training Wall Seconds", "Test Forward Seconds", "Mean Sample Latency ms",
    "Samples Per Second", "Peak GPU Allocated MB",
    "Runtime Environment", "Python Executable", "Result Directory",
]

_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "MAE": ("MAE", "mae"),
    "RMSE": ("RMSE", "rmse"),
    "R2": ("R2", "r2"),
    "SMAPE": ("SMAPE", "smape"),
    "MAPE": ("MAPE", "mape"),
    "Official_Score": (
        "SDWPF Official Score",
        "Official_Score",
        "official_score",
        "official_align_score",
        "score",
    ),
}
_HORIZON_FILE = re.compile(r"^metrics_test_h(?P<horizon>[1-9][0-9]*)\.json$")


@dataclass(frozen=True)
class ModelRunSummary:
    """One parsed model result reused by all scheduler CSV outputs."""

    comparison_row: dict[str, Any]
    summary_row: dict[str, Any]
    performance_row: dict[str, Any]
    expected_horizons: tuple[int, ...]
    missing_horizons: tuple[int, ...]


def _read_json_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _first_value(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _metric_value(metrics: Mapping[str, Any], name: str) -> Any:
    """Read one Evaluator metric through the project's supported key aliases."""

    payload: Mapping[str, Any] = metrics
    nested = metrics.get("metrics")
    if isinstance(nested, Mapping):
        payload = nested
    aliases = _METRIC_ALIASES[name]
    for alias in aliases:
        if alias in payload:
            return payload[alias]
    folded = {str(key).casefold(): value for key, value in payload.items()}
    for alias in aliases:
        if alias.casefold() in folded:
            return folded[alias.casefold()]
    return None


def format_csv_float(value: Any) -> str:
    """Format finite numeric CSV values consistently without changing JSON metrics."""

    if value is None:
        return ""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(parsed):
        return ""
    return f"{parsed:.3f}"


def _finite_metric(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def extract_metric_value(metrics: Mapping[str, Any], name: str) -> Any:
    """Public shared metric reader used by aggregation and repeatability."""

    return _metric_value(metrics, name)


def _test_metric_files(result_dir: Path) -> dict[int, Path]:
    files: dict[int, Path] = {}
    for path in result_dir.glob("metrics_test_h*.json"):
        match = _HORIZON_FILE.fullmatch(path.name)
        if match:
            files[int(match.group("horizon"))] = path
    return files


def _configured_horizons(result_dir: Path, discovered: Mapping[int, Path]) -> tuple[int, ...]:
    path = result_dir / "resolved_config.yaml"
    if path.is_file():
        try:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            value = None
        if isinstance(value, Mapping):
            data = value.get("data")
            resolved = value.get("resolved")
            if not isinstance(data, Mapping) and isinstance(resolved, Mapping):
                data = resolved.get("data")
            horizons = data.get("eval_horizons") if isinstance(data, Mapping) else None
            if horizons is None and isinstance(resolved, Mapping):
                horizons = resolved.get("eval_horizons")
            if isinstance(horizons, list):
                parsed = sorted({int(item) for item in horizons if isinstance(item, int) and item > 0})
                if parsed:
                    return tuple(parsed)
    return tuple(sorted(discovered))


def collect_model_run_summary(record: Mapping[str, Any]) -> ModelRunSummary:
    """Parse one model directory once for summary, performance and paper CSVs."""

    result_dir = Path(str(record.get("result_dir", "")))
    run_info = _read_json_mapping(result_dir / "run_info.json")
    performance = _read_json_mapping(result_dir / "performance.json")
    test_performance = performance.get("test")
    test = test_performance if isinstance(test_performance, Mapping) else {}
    metric_files = _test_metric_files(result_dir)
    expected_horizons = _configured_horizons(result_dir, metric_files)
    missing_horizons = tuple(
        horizon for horizon in expected_horizons if horizon in PAPER_HORIZONS and horizon not in metric_files
    )
    metrics_by_horizon: dict[int, Mapping[str, Any]] = {}
    for horizon, path in metric_files.items():
        payload = _read_json_mapping(path)
        metrics_by_horizon[horizon] = payload

    runtime_environment = _first_value(
        record.get("runtime_environment"),
        run_info.get("runtime_environment"),
        performance.get("runtime_environment"),
    )
    python_executable = _first_value(
        record.get("python_executable"),
        run_info.get("python_executable"),
        performance.get("python_executable"),
    )
    status = _first_value(record.get("status"), run_info.get("status"))
    parameter_count = _first_value(
        performance.get("parameter_count"), run_info.get("parameter_count")
    )
    best_epoch = _first_value(performance.get("best_epoch"), run_info.get("best_epoch"))
    comparison: dict[str, Any] = {
        "model": record.get("model"),
        "status": status,
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "training_wall_seconds": performance.get("training_wall_seconds"),
        "test_model_forward_seconds": test.get("model_forward_seconds"),
        "mean_sample_latency_ms": test.get("mean_sample_latency_ms"),
        "samples_per_second": test.get("samples_per_second"),
        "peak_gpu_allocated_mb": performance.get("peak_gpu_allocated_mb"),
        "runtime_environment": runtime_environment,
        "python_executable": python_executable,
        "result_dir": str(result_dir),
    }
    for horizon in PAPER_HORIZONS:
        metrics = metrics_by_horizon.get(horizon, {})
        for metric_name in ("MAE", "RMSE", "R2", "MAPE", "SMAPE"):
            comparison[f"H{horizon}_{metric_name}"] = _metric_value(metrics, metric_name)
        comparison[f"H{horizon}_Official_Score"] = _metric_value(metrics, "Official_Score")
    notes: list[str] = []
    for metric_name in ("R2", "MAPE"):
        if any(
            horizon in metric_files
            and not _finite_metric(comparison[f"H{horizon}_{metric_name}"])
            for horizon in PAPER_HORIZONS
        ):
            notes.append(f"{metric_name}_UNDEFINED")
    comparison["metric_note"] = ";".join(notes)

    summary_row = {
        "model": comparison["model"],
        "runtime_environment": runtime_environment,
        "python_executable": python_executable,
        "status": status,
        "best_epoch": best_epoch,
        "main_metric": run_info.get("test_monitor"),
        "result_dir": str(result_dir),
        "exit_code": record.get("exit_code"),
    }
    performance_row = {
        "model": comparison["model"],
        "runtime_environment": runtime_environment,
        "python_executable": python_executable,
        "status": status,
        "parameter_count": parameter_count,
        "trainable_parameter_count": performance.get("trainable_parameter_count"),
        "checkpoint_size_mb": performance.get("checkpoint_size_mb"),
        "epochs_completed": performance.get("epochs_completed"),
        "best_epoch": best_epoch,
        "training_wall_seconds": performance.get("training_wall_seconds"),
        "mean_epoch_seconds": performance.get("mean_epoch_seconds"),
        "total_wall_seconds": performance.get("total_wall_seconds"),
        "test_model_forward_seconds": test.get("model_forward_seconds"),
        "mean_batch_latency_ms": test.get("mean_batch_latency_ms"),
        "mean_sample_latency_ms": test.get("mean_sample_latency_ms"),
        "samples_per_second": test.get("samples_per_second"),
        "forecast_values_per_second": test.get("forecast_values_per_second"),
        "peak_gpu_allocated_mb": performance.get("peak_gpu_allocated_mb"),
        "peak_gpu_reserved_mb": performance.get("peak_gpu_reserved_mb"),
        "result_dir": str(result_dir),
    }
    return ModelRunSummary(
        comparison_row=comparison,
        summary_row=summary_row,
        performance_row=performance_row,
        expected_horizons=tuple(expected_horizons),
        missing_horizons=missing_horizons,
    )


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    write_text_atomic(path, buffer.getvalue())


def _comparison_value(field: str, value: Any) -> str:
    if field in {"parameter_count", "best_epoch"}:
        if value is None:
            return ""
        try:
            return str(int(value))
        except (TypeError, ValueError):
            return ""
    if field in {
        "H3_MAE", "H3_RMSE", "H3_R2", "H3_Official_Score", "H3_MAPE", "H3_SMAPE",
        "H6_MAE", "H6_RMSE", "H6_R2", "H6_Official_Score", "H6_MAPE", "H6_SMAPE",
        "H10_MAE", "H10_RMSE", "H10_R2", "H10_Official_Score", "H10_MAPE", "H10_SMAPE",
        "training_wall_seconds", "test_model_forward_seconds", "mean_sample_latency_ms",
        "samples_per_second", "peak_gpu_allocated_mb",
    }:
        return format_csv_float(value)
    return "" if value is None else str(value)


def _write_model_comparisons(run_root: Path, rows: list[dict[str, Any]]) -> None:
    """Write paper and programmatic CSVs from the same normalized summaries."""

    flattened = [
        {field: _comparison_value(field, row.get(field)) for field in MODEL_COMPARISON_FIELDS}
        for row in rows
    ]
    _write_csv(run_root / "model_comparison_flat.csv", MODEL_COMPARISON_FIELDS, flattened)

    buffer = StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(_PAPER_GROUP_HEADER)
    writer.writerow(_PAPER_FIELD_HEADER)
    for row in flattened:
        writer.writerow([row[field] for field in _PAPER_COMPARISON_FIELDS])
    write_text_atomic(run_root / "model_comparison.csv", buffer.getvalue())


def _status_payload(run_id: str, operation: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "run_id": run_id,
        "operation": operation,
        "updated_at": utc_now(),
        "models": records,
    }


def _save_status(run_root: Path, run_id: str, operation: str, records: list[dict[str, Any]]) -> None:
    for record in records:
        parsed = collect_model_run_summary(record)
        phase = str(record.get("phase") or "")
        status = record.get("status")
        if operation in {"check", "preflight", "environment_preflight"} or phase in {
            "resolved_shape",
            "environment_preflight",
            "model_preflight",
        } or status in {PENDING, RUNNING}:
            record["metrics_complete"] = None
        elif operation in {"train", "evaluate"}:
            record["metrics_complete"] = not parsed.missing_horizons if status == PASS else False
        else:
            record["metrics_complete"] = None
        record["missing_horizons"] = list(parsed.missing_horizons)
    write_status(run_root / "status.json", _status_payload(run_id, operation, records))


def _write_summaries(run_root: Path, records: list[dict[str, Any]]) -> None:
    parsed_rows = [collect_model_run_summary(record) for record in records]
    performance_fields = [
        "model",
        "runtime_environment",
        "python_executable",
        "status",
        "parameter_count",
        "trainable_parameter_count",
        "checkpoint_size_mb",
        "epochs_completed",
        "best_epoch",
        "training_wall_seconds",
        "mean_epoch_seconds",
        "total_wall_seconds",
        "test_model_forward_seconds",
        "mean_batch_latency_ms",
        "mean_sample_latency_ms",
        "samples_per_second",
        "forecast_values_per_second",
        "peak_gpu_allocated_mb",
        "peak_gpu_reserved_mb",
        "result_dir",
    ]
    _write_csv(
        run_root / "summary.csv",
        [
            "model",
            "runtime_environment",
            "python_executable",
            "status",
            "best_epoch",
            "main_metric",
            "result_dir",
            "exit_code",
        ],
        [item.summary_row for item in parsed_rows],
    )
    _write_csv(
        run_root / "performance_summary.csv",
        performance_fields,
        [item.performance_row for item in parsed_rows],
    )
    _write_model_comparisons(run_root, [item.comparison_row for item in parsed_rows])


def summarize_existing_run(
    *, run_id: str, output_root: str | Path | None = None, project_root: Path | None = None
) -> dict[str, Any]:
    """Regenerate scheduler CSVs from completed artifacts without starting a worker."""

    root = (project_root or Path.cwd()).resolve()
    results_root = resolve_output_root(root, output_root)
    run_root = results_root / "_runs" / effective_run_id(run_id, None)
    status_path = run_root / "status.json"
    raw = _read_json_mapping(status_path)
    if not raw:
        raise FileNotFoundError(f"run status does not exist or is unreadable: {status_path}")
    payload = normalize_status_payload(raw)
    records_value = payload.get("models")
    if not isinstance(records_value, list):
        raise ValueError(f"run status has no model records: {status_path}")
    records = [dict(item) for item in records_value if isinstance(item, Mapping)]
    if not records:
        raise ValueError(f"run status has no usable model records: {status_path}")
    operation = str(payload.get("operation") or "train")
    _save_status(run_root, str(payload.get("run_id") or run_id), operation, records)
    _write_summaries(run_root, records)
    return {
        "run_id": str(payload.get("run_id") or run_id),
        "run_root": str(run_root),
        "model_count": len(records),
        "summary_csv": str(run_root / "summary.csv"),
        "performance_summary_csv": str(run_root / "performance_summary.csv"),
        "model_comparison_csv": str(run_root / "model_comparison.csv"),
        "model_comparison_flat_csv": str(run_root / "model_comparison_flat.csv"),
    }


def _new_record(
    model: str,
    result_dir: Path,
    log_path: Path,
    *,
    runtime_fields: Mapping[str, Any],
) -> dict[str, Any]:
    phases = {
        "environment_preflight": phase_record(phase="environment_preflight"),
        "model_preflight": phase_record(phase="model_preflight"),
        "resolved_shape": phase_record(phase="resolved_shape"),
        "training": phase_record(phase="training"),
        "checkpoint_write": phase_record(phase="checkpoint_write"),
        "checkpoint_reload": phase_record(phase="checkpoint_reload"),
        "validation": phase_record(phase="validation"),
        "test": phase_record(phase="test"),
        "overall": phase_record(phase="pending"),
    }
    return {
        "model": model,
        **dict(runtime_fields),
        "status": PENDING,
        "classification": None,
        "phase": "pending",
        "profile": None,
        "operation": "train",
        "pid": None,
        "exit_code": None,
        "started_at": None,
        "ended_at": None,
        "wall_seconds": None,
        "result_dir": str(result_dir),
        "log_path": str(log_path),
        "error_summary": None,
        "archive_path": None,
        "phases": phases,
    }


def _run_worker(
    *,
    project_root: Path,
    request_path: Path,
    model: str,
    log_path: Path,
    resolved_environment: ResolvedEnvironment,
) -> tuple[int, str, int]:
    command = build_model_worker_command(
        project_root=project_root,
        request_path=request_path,
        model_name=model,
        resolved_environment=resolved_environment,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output: list[str] = []
    with log_path.open("w", encoding="utf-8", newline="") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=build_worker_environment(resolved_environment, project_root=project_root),
        )
        assert process.stdout is not None
        for line in process.stdout:
            output.append(line)
            log_handle.write(line)
            log_handle.flush()
            print(f"[{model}] {line.rstrip()}")
        exit_code = process.wait()
    return exit_code, "".join(output)[-2000:], int(process.pid)


def _validation_profile_for_training(*, smoke: bool) -> str:
    """Training always validates the exact resolved shape before its worker."""

    del smoke
    return RESOLVED_SHAPE


def _shape_result_from_worker(
    *,
    status_path: Path,
    model: str,
    run_id: str,
    runtime_fields: Mapping[str, Any],
    exit_code: int,
    output_tail: str,
    wall_seconds: float,
    profile: str,
) -> dict[str, Any]:
    """Read worker-written validation JSON or synthesize an honest crash state."""

    payload = _read_json_mapping(status_path)
    if payload and payload.get("status") in TOP_LEVEL_STATUSES:
        payload.setdefault("wall_seconds", wall_seconds)
        payload.setdefault("exit_code", exit_code)
        return payload
    classification = classify_validation_failure(
        message=output_tail,
        phase="worker",
        exit_code=exit_code,
        worker_status_present=False,
    )
    payload = {
        "schema_version": 1,
        "model": model,
        "run_id": run_id,
        "operation": "train",
        "profile": profile,
        "status": FAILED,
        "classification": classification,
        "phase": "worker_completion_missing",
        "started_at": None,
        "ended_at": utc_now(),
        "wall_seconds": wall_seconds,
        **dict(runtime_fields),
        "exception_type": None,
        "error_message": output_tail or f"validation worker exited with code {exit_code}",
        "exit_code": exit_code,
    }
    write_validation_status(status_path, payload)
    return payload


def _copy_worker_run_phases(record: dict[str, Any]) -> None:
    info = _read_json_mapping(Path(record["result_dir"]) / "run_info.json")
    phases = info.get("phases")
    if isinstance(phases, Mapping):
        for source_name, target_name in (
            ("training", "training"),
            ("checkpoint", "checkpoint_write"),
            ("evaluation", "test"),
        ):
            value = phases.get(source_name)
            if isinstance(value, Mapping):
                record["phases"][target_name] = dict(value)


def run_training_models(
    *,
    models: list[str],
    config_path: str | Path,
    model_config_path: str | Path | None,
    run_id: str | None,
    device: str,
    output_root: str | Path | None,
    resume: bool,
    overwrite: bool,
    id_suffix: str | None,
    fail_fast: bool,
    smoke: bool,
    smoke_epochs: int | None,
    smoke_max_train_updates: int | None,
    smoke_max_eval_batches: int | None,
    cli_overrides: Mapping[str, Any] | None,
    command_argv: list[str] | None,
    environment_preflight_only: bool = False,
    environment_context: tuple[
        dict[str, ResolvedEnvironment], dict[str, dict[str, Any]]
    ] | None = None,
    environment_context_holder: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not models:
        raise ValueError("at least one model is required")
    validate_unique_models(models)
    if resume and overwrite or resume and id_suffix or overwrite and id_suffix:
        raise ValueError("--resume, --overwrite and --id-suffix are mutually exclusive")
    config_file = Path(config_path).resolve()
    project_root = project_root_from_config(config_file)
    model_configs = _validate_model_configs(models, config_file, model_config_path)
    effective_id = effective_run_id(run_id, id_suffix)
    root = resolve_output_root(project_root, output_root)
    run_root = root / "_runs" / effective_id
    if run_root.exists() and any(run_root.iterdir()):
        if overwrite:
            archive_directory(run_root, root / "_archive" / "_runs", label=effective_id)
        elif not resume:
            raise ValueError(f"scheduler run directory already contains files: {run_root}; choose --resume, --overwrite or --id-suffix")
    run_root.mkdir(parents=True, exist_ok=True)
    logs_root = project_root / "logs" / effective_id
    if environment_context is None and environment_context_holder is not None:
        cached = environment_context_holder.get("value")
        if cached is not None:
            environment_context = cached
    if environment_context is None:
        try:
            environment_context = _prepare_batch_environments(
                models=models,
                model_configs=model_configs,
                project_root=project_root,
                device=device,
                config_path=config_file,
            )
        except BaseException as exc:
            classification = classify_validation_failure(exc, phase="environment")
            records = [
                _new_record(
                    model,
                    run_directory(project_root, root, model, effective_id),
                    logs_root / f"{model}.log",
                    runtime_fields={},
                )
                for model in models
            ]
            for record in records:
                record.update(
                    {"status": FAILED, "classification": classification, "phase": "environment_preflight"}
                )
                record["error_summary"] = failure_summary(exc)
                record["phases"]["environment_preflight"] = finished_phase(
                    classification=classification,
                    phase="environment_preflight",
                    error_summary=record["error_summary"],
                )
                record["phases"]["overall"] = finished_phase(
                    classification=classification,
                    phase="environment_preflight",
                    error_summary=record["error_summary"],
                )
            _save_status(run_root, effective_id, "train", records)
            _write_summaries(run_root, records)
            return {
                "passed": False,
                "run_id": effective_id,
                "run_root": str(run_root),
                "models": records,
                "summary_csv": str(run_root / "summary.csv"),
                "performance_summary_csv": str(run_root / "performance_summary.csv"),
                "model_comparison_csv": str(run_root / "model_comparison.csv"),
            }
        if environment_context_holder is not None:
            environment_context_holder["value"] = environment_context
    model_environments, preflight_results = environment_context
    records = [
        _new_record(
            model,
            run_directory(project_root, root, model, effective_id),
            logs_root / f"{model}.log",
            runtime_fields=_runtime_record_fields(
                model_environments[model],
                preflight_results[model_environments[model].environment_id],
            ),
        )
        for model in models
    ]
    if environment_preflight_only:
        for record in records:
            record["status"] = PASS
            record["classification"] = pass_classification(ENVIRONMENT_PREFLIGHT)
            record["phase"] = "environment_preflight_complete"
            record["phases"]["environment_preflight"] = finished_phase(
                profile=ENVIRONMENT_PREFLIGHT,
                phase="environment_preflight_complete",
            )
            record["phases"]["model_preflight"] = finished_phase(
                profile=MODEL_PREFLIGHT,
                phase="model_preflight_complete",
            )
            record["phases"]["overall"] = finished_phase(
                profile=ENVIRONMENT_PREFLIGHT,
                phase="environment_preflight_complete",
            )
        _save_status(run_root, effective_id, "environment_preflight", records)
        _write_summaries(run_root, records)
        return {
            "passed": True,
            "run_id": effective_id,
            "run_root": str(run_root),
            "models": records,
            "summary_csv": str(run_root / "summary.csv"),
            "performance_summary_csv": str(run_root / "performance_summary.csv"),
            "model_comparison_csv": str(run_root / "model_comparison.csv"),
            "environment_preflight_only": True,
        }
    requests: dict[str, Any] = {
        "operation": "train",
        "run_id": effective_id,
        "config_path": str(config_file),
        "device": device,
        "output_root": str(root),
        "smoke": bool(smoke),
        "smoke_epochs": smoke_epochs,
        "smoke_max_train_updates": smoke_max_train_updates,
        "smoke_max_eval_batches": smoke_max_eval_batches,
        "cli_overrides": dict(cli_overrides or {}),
        "command_argv": list(command_argv or []),
        "models": {
            model: {
                "model_config_path": str(model_configs[model]),
                "resume_checkpoint": None,
                "environment": {
                    **_runtime_record_fields(
                        model_environments[model],
                        preflight_results[model_environments[model].environment_id],
                    ),
                    "source_roots": [
                        str(path) for path in model_environments[model].source_roots
                    ],
                    "required_imports": list(model_environments[model].required_imports),
                },
            }
            for model in models
        },
    }
    write_json(run_root / "request.json", requests)

    blocked = False
    for record in records:
        result_dir = Path(record["result_dir"])
        model = record["model"]
        record["phases"]["environment_preflight"] = finished_phase(
            profile=ENVIRONMENT_PREFLIGHT, phase="environment_preflight_complete"
        )
        record["phases"]["model_preflight"] = finished_phase(
            profile=MODEL_PREFLIGHT, phase="model_preflight_complete"
        )
        if resume:
            if not result_dir.exists() or not any(result_dir.iterdir()):
                continue
            if is_completed_run(result_dir):
                record["status"] = SKIPPED
                record["classification"] = None
                record["phase"] = "completed_run_reused"
                record["phases"]["overall"] = phase_record(
                    status=SKIPPED, phase="completed_run_reused", artifact=str(result_dir)
                )
                continue
            checkpoint = result_dir / "last.pt"
            if not checkpoint.is_file():
                record.update({"status": FAILED, "classification": FAIL_CONFIG, "phase": "resume"})
                record["error_summary"] = "--resume requires a valid last.pt; use --overwrite or --id-suffix"
                record["phases"]["overall"] = finished_phase(
                    classification=FAIL_CONFIG, phase="resume", error_summary=record["error_summary"]
                )
                blocked = True
                continue
            try:
                from engine.checkpoint import read_checkpoint_manifest

                read_checkpoint_manifest(checkpoint)
            except (OSError, RuntimeError, ValueError, TypeError) as exc:
                record.update({"status": FAILED, "classification": classify_validation_failure(exc, phase="checkpoint"), "phase": "resume"})
                record["error_summary"] = f"invalid last.pt: {exc}; use --overwrite or --id-suffix"
                record["phases"]["overall"] = finished_phase(
                    classification=record["classification"], phase="resume", error_summary=record["error_summary"]
                )
                blocked = True
                continue
            requests["models"][model]["resume_checkpoint"] = str(checkpoint)
        elif overwrite and result_dir.exists():
            if not result_dir.is_dir():
                record.update({"status": FAILED, "classification": FAIL_CONFIG, "phase": "output_directory"})
                record["error_summary"] = f"result path is not a directory: {result_dir}"
                record["phases"]["overall"] = finished_phase(
                    classification=FAIL_CONFIG, phase="output_directory", error_summary=record["error_summary"]
                )
                blocked = True
                continue
            if formal_result_exists(result_dir) or any(result_dir.iterdir()):
                archive = archive_directory(result_dir, root / "_archive" / model, label=effective_id)
                record["archive_path"] = str(archive)
        elif result_dir.exists() and (formal_result_exists(result_dir) or any(result_dir.iterdir())):
            record.update({"status": FAILED, "classification": FAIL_CONFIG, "phase": "output_directory"})
            record["error_summary"] = "result directory is not empty; choose --resume, --overwrite or --id-suffix"
            record["phases"]["overall"] = finished_phase(
                classification=FAIL_CONFIG, phase="output_directory", error_summary=record["error_summary"]
            )
            blocked = True
    write_json(run_root / "request.json", requests)
    _save_status(run_root, effective_id, "train", records)
    _write_summaries(run_root, records)

    failed = blocked
    for index, record in enumerate(records):
        if record["status"] not in {PENDING}:
            continue
        if fail_fast and failed:
            record.update({"status": SKIPPED, "phase": "fail_fast"})
            record["error_summary"] = "not started because --fail-fast stopped the scheduler"
            record["phases"]["overall"] = phase_record(
                status=SKIPPED, phase="fail_fast", error_summary=record["error_summary"]
            )
            continue
        model = record["model"]
        result_dir = Path(record["result_dir"])
        result_dir.mkdir(parents=True, exist_ok=True)
        profile = _validation_profile_for_training(smoke=smoke)
        status_path = run_root / f".{model}.resolved_shape.json"
        requests["models"][model]["status_path"] = str(status_path)
        requests["models"][model]["resume_checkpoint"] = requests["models"][model].get("resume_checkpoint")
        shape_request = dict(requests)
        shape_request["operation"] = "validate_shape"
        shape_request["profile"] = profile
        shape_request_path = run_root / f"{model}.resolved_shape.request.json"
        write_json(shape_request_path, shape_request)
        record["status"] = RUNNING
        record["profile"] = profile
        record["phase"] = "resolved_shape"
        record["phases"]["resolved_shape"] = running_phase(
            "resolved_shape", artifact=str(status_path)
        )
        _save_status(run_root, effective_id, "train", records)
        shape_started = time.perf_counter()
        shape_exit, shape_tail, shape_pid = _run_worker(
            project_root=project_root,
            request_path=shape_request_path,
            model=model,
            log_path=logs_root / f"{model}.resolved_shape.log",
            resolved_environment=model_environments[model],
        )
        record["resolved_shape_pid"] = shape_pid
        shape_seconds = time.perf_counter() - shape_started
        shape = _shape_result_from_worker(
            status_path=status_path,
            model=model,
            run_id=effective_id,
            runtime_fields=_runtime_record_fields(
                model_environments[model], preflight_results[model_environments[model].environment_id]
            ),
            exit_code=shape_exit,
            output_tail=shape_tail,
            wall_seconds=shape_seconds,
            profile=profile,
        )
        record["phases"]["resolved_shape"] = {
            "status": shape.get("status", FAILED),
            "classification": shape.get("classification"),
            "phase": shape.get("phase"),
            "started_at": shape.get("started_at"),
            "ended_at": shape.get("ended_at"),
            "wall_seconds": shape.get("wall_seconds"),
            "artifact": str(result_dir),
            "error_summary": (
                shape.get("error", {}).get("message")
                if isinstance(shape.get("error"), Mapping)
                else shape.get("error_message")
            ),
            "details": {
                key: shape.get(key)
                for key in (
                    "input_shape", "output_shape", "parameter_count",
                    "estimated_input_tensor_mb", "peak_gpu_allocated_mb",
                )
                if key in shape
            },
        }
        try:
            status_path.unlink(missing_ok=True)
        except OSError:
            pass
        if shape.get("status") != PASS:
            record.update(
                {
                    "status": FAILED,
                    "classification": shape.get("classification", FAIL_WORKER_CRASH),
                    "phase": "resolved_shape",
                    "exit_code": shape_exit,
                    "ended_at": utc_now(),
                    "wall_seconds": shape_seconds,
                    "error_summary": (
                        shape.get("error", {}).get("message")
                        if isinstance(shape.get("error"), Mapping)
                        else shape.get("error_message")
                    ) or shape_tail,
                }
            )
            record["phases"]["overall"] = finished_phase(
                classification=record["classification"],
                phase="resolved_shape",
                artifact=str(status_path),
                error_summary=record["error_summary"],
            )
            failed = True
            _save_status(run_root, effective_id, "train", records)
            _write_summaries(run_root, records)
            continue

        write_json(run_root / "request.json", requests)
        record["phase"] = "training"
        record["started_at"] = utc_now()
        record["pid"] = None
        record["phases"]["training"] = running_phase("training")
        _save_status(run_root, effective_id, "train", records)
        started = time.perf_counter()
        exit_code, tail, pid = _run_worker(
            project_root=project_root,
            request_path=run_root / "request.json",
            model=model,
            log_path=Path(record["log_path"]),
            resolved_environment=model_environments[model],
        )
        record["pid"] = pid
        record["exit_code"] = int(exit_code)
        record["ended_at"] = utc_now()
        record["wall_seconds"] = time.perf_counter() - started
        _copy_worker_run_phases(record)
        run_info = _read_json_mapping(result_dir / "run_info.json")
        if exit_code == 0 and run_info.get("status") == PASS:
            record.update(
                {
                    "status": PASS,
                    "classification": run_info.get("classification"),
                    "phase": run_info.get("phase", "complete"),
                    "error_summary": None,
                }
            )
            record["phases"]["overall"] = phase_record(
                status=PASS,
                phase=record["phase"],
                artifact=str(result_dir),
                wall_seconds=record["wall_seconds"],
            )
        else:
            worker_status_present = bool(run_info) and run_info.get("status") == FAILED
            classification = run_info.get("classification") if worker_status_present else classify_validation_failure(
                message=tail,
                phase="worker",
                exit_code=exit_code,
                worker_status_present=False,
            )
            record.update(
                {
                    "status": FAILED,
                    "classification": classification,
                    "phase": run_info.get("phase", "worker_completion_missing"),
                    "error_summary": run_info.get("error_message") or tail or f"worker exited with code {exit_code}",
                }
            )
            training_phase = record["phases"].get("training")
            if isinstance(training_phase, Mapping) and training_phase.get("status") == RUNNING:
                record["phases"]["training"] = finished_phase(
                    classification=classification,
                    phase=record["phase"],
                    artifact=str(result_dir),
                    error_summary=record["error_summary"],
                    started_at=training_phase.get("started_at"),
                    wall_seconds=record["wall_seconds"],
                )
            record["phases"]["overall"] = finished_phase(
                classification=classification,
                phase=record["phase"],
                artifact=str(result_dir),
                error_summary=record["error_summary"],
                wall_seconds=record["wall_seconds"],
            )
            failed = True
        _save_status(run_root, effective_id, "train", records)
        _write_summaries(run_root, records)
        if failed and fail_fast:
            for remaining in records[index + 1 :]:
                if remaining["status"] == PENDING:
                    remaining.update({"status": SKIPPED, "phase": "fail_fast"})
                    remaining["error_summary"] = "not started because --fail-fast stopped the scheduler"
                    remaining["phases"]["overall"] = phase_record(
                        status=SKIPPED, phase="fail_fast", error_summary=remaining["error_summary"]
                    )
            _save_status(run_root, effective_id, "train", records)
            _write_summaries(run_root, records)
            break
    _save_status(run_root, effective_id, "train", records)
    _write_summaries(run_root, records)
    return {
        "passed": not any(record["status"] == FAILED for record in records),
        "run_id": effective_id,
        "run_root": str(run_root),
        "models": records,
        "summary_csv": str(run_root / "summary.csv"),
        "performance_summary_csv": str(run_root / "performance_summary.csv"),
        "model_comparison_csv": str(run_root / "model_comparison.csv"),
    }


def run_isolated_checks(
    *,
    operation: str,
    models: list[str],
    config_path: str | Path,
    model_config_path: str | Path | None,
    device: str,
    cli_overrides: Mapping[str, Any] | None,
    run_id: str | None = None,
    output_root: str | Path | None = None,
    full_shape: bool = False,
    no_data: bool = False,
    environment_preflight_only: bool = False,
) -> dict[str, Any]:
    if operation not in {"check", "preflight"}:
        raise ValueError(f"unsupported isolated operation: {operation}")
    if not models:
        raise ValueError("at least one model is required")
    validate_unique_models(models)
    config_file = Path(config_path).resolve()
    project_root = project_root_from_config(config_file)
    root = resolve_output_root(project_root, output_root)
    run_name = effective_run_id(
        run_id or f"{operation}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
        None,
    )
    run_root = (root / ("_checks" if operation == "check" else "_runs") / run_name).resolve()
    if operation == "check" and run_root.exists() and any(run_root.iterdir()):
        raise ValueError(f"check run directory already contains files: {run_root}; choose a new --run-id")
    model_configs = _validate_model_configs(models, config_file, model_config_path)
    try:
        model_environments, preflight_results = _prepare_batch_environments(
            models=models,
            model_configs=model_configs,
            project_root=project_root,
            device=device,
            config_path=config_file,
        )
    except BaseException as exc:
        classification = classify_validation_failure(exc, phase="environment")
        run_root.mkdir(parents=True, exist_ok=True)
        results = [
            {
                "schema_version": 1,
                "model": model,
                "run_id": run_name,
                "operation": operation,
                "status": FAILED,
                "classification": classification,
                "phase": "environment_preflight",
                "exit_code": 1,
                "error_message": failure_summary(exc),
            }
            for model in models
        ]
        result = {
            "schema_version": 1,
            "passed": False,
            "operation": operation,
            "run_id": run_name,
            "results": results,
        }
        for item in results:
            write_validation_status(run_root / f"{item['model']}.json", item)
        _write_csv(
            run_root / "summary.csv",
            ["model", "status", "classification", "phase", "exit_code", "error_message"],
            results,
        )
        return result
    if environment_preflight_only:
        results = [
            {
                "model": model,
                **_runtime_record_fields(
                    model_environments[model],
                    preflight_results[model_environments[model].environment_id],
                ),
                "status": PASS,
                "classification": pass_classification(ENVIRONMENT_PREFLIGHT),
                "exit_code": 0,
            }
            for model in models
        ]
        return {
            "passed": True,
            "operation": operation,
            "results": results,
            "environment_preflight_only": True,
        }
    run_root.mkdir(parents=True, exist_ok=True)
    request: dict[str, Any] = {
        "operation": operation,
        "run_id": run_name,
        "config_path": str(config_file),
        "device": device,
        "cli_overrides": dict(cli_overrides or {}),
        "full_shape": bool(full_shape),
        "no_data": bool(no_data),
        "models": {
            model: {
                "model_config_path": str(model_configs[model]),
                "status_path": str(run_root / f"{model}.json"),
                "environment": {
                    **_runtime_record_fields(
                        model_environments[model],
                        preflight_results[model_environments[model].environment_id],
                    ),
                    "source_roots": [
                        str(path) for path in model_environments[model].source_roots
                    ],
                    "required_imports": list(model_environments[model].required_imports),
                },
            }
            for model in models
        },
    }
    request_path = run_root / "request.json"
    write_json(request_path, request)
    results: list[dict[str, Any]] = []
    for model in models:
        log_path = project_root / "logs" / run_name / f"{model}.log"
        started = time.perf_counter()
        exit_code, tail, pid = _run_worker(
            project_root=project_root,
            request_path=request_path,
            model=model,
            log_path=log_path,
            resolved_environment=model_environments[model],
        )
        wall_seconds = time.perf_counter() - started
        runtime_fields = _runtime_record_fields(
            model_environments[model], preflight_results[model_environments[model].environment_id]
        )
        if operation == "check":
            item = _shape_result_from_worker(
                status_path=run_root / f"{model}.json",
                model=model,
                run_id=run_name,
                runtime_fields=runtime_fields,
                exit_code=exit_code,
                output_tail=tail,
                wall_seconds=wall_seconds,
                profile=FORMAL_DEFAULT_SHAPE if full_shape and not cli_overrides else RESOLVED_SHAPE if full_shape else INTERFACE_SMALL,
            )
            item.update({**runtime_fields, "pid": pid, "log_path": str(log_path), "output_tail": tail})
        else:
            classification = pass_classification(MODEL_PREFLIGHT) if exit_code == 0 else classify_validation_failure(
                message=tail,
                phase="worker",
                exit_code=exit_code,
                worker_status_present=False,
            )
            item = {
                "schema_version": 1,
                "model": model,
                "run_id": run_name,
                "operation": operation,
                **runtime_fields,
                "status": PASS if exit_code == 0 else FAILED,
                "classification": classification,
                "phase": "model_preflight_complete" if exit_code == 0 else "worker_completion_missing",
                "exit_code": exit_code,
                "pid": pid,
                "log_path": str(log_path),
                "wall_seconds": wall_seconds,
                "error_message": None if exit_code == 0 else tail,
            }
            write_validation_status(run_root / f"{model}.json", item)
        results.append(item)
    result = {
        "schema_version": 1,
        "passed": not any(item["status"] == FAILED for item in results),
        "operation": operation,
        "run_id": run_name,
        "results": results,
    }
    _write_csv(
        run_root / "summary.csv",
        ["model", "profile", "status", "classification", "phase", "batch_size", "batch_size_source", "input_shape", "output_shape", "parameter_count", "exit_code", "wall_seconds"],
        results,
    )
    return result

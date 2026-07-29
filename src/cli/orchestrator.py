"""Parent-process orchestration for isolated model workers."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from dataclasses import dataclass
import json
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


def _default_model_config(config_file: Path, model: str) -> Path:
    return config_file.parent / "models" / f"{model}.yaml"


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
) -> tuple[dict[str, ResolvedEnvironment], dict[str, dict[str, Any]]]:
    """Resolve every model and preflight each distinct environment once."""

    model_environments: dict[str, ResolvedEnvironment] = {}
    for model in models:
        model_environments[model] = resolve_model_environment(
            model_configs[model],
            project_root=project_root,
        )
    return model_environments, preflight_batch_environments(
        models=models,
        model_configs=model_configs,
        model_environments=model_environments,
        project_root=project_root,
        device=device,
    )


def _prepare_batch_environments(
    *,
    models: list[str],
    model_configs: Mapping[str, Path],
    project_root: Path,
    device: str,
) -> tuple[dict[str, ResolvedEnvironment], dict[str, dict[str, Any]]]:
    """Backward-compatible seam for tests and callers of the old private name."""

    return prepare_batch_environments(
        models=models,
        model_configs=model_configs,
        project_root=project_root,
        device=device,
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

_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "MAE": ("MAE", "mae"),
    "RMSE": ("RMSE", "rmse"),
    "R2": ("R2", "r2"),
    "SMAPE": ("SMAPE", "smape"),
    "MAPE": ("MAPE", "mape"),
    "Official_Score": (
        "Official_Score",
        "official_score",
        "score",
        "SDWPF Official Score",
        "official_align_score",
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
        "mean_official_score": None,
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
        for metric_name in ("MAE", "RMSE", "R2", "SMAPE", "MAPE"):
            comparison[f"H{horizon}_{metric_name}"] = _metric_value(metrics, metric_name)
        comparison[f"H{horizon}_Official_Score"] = _metric_value(metrics, "Official_Score")

    expected_paper_horizons = [
        horizon for horizon in PAPER_HORIZONS if horizon in expected_horizons
    ]
    if not expected_paper_horizons:
        expected_paper_horizons = [horizon for horizon in PAPER_HORIZONS if horizon in metric_files]
    official_scores = [
        comparison[f"H{horizon}_Official_Score"]
        for horizon in expected_paper_horizons
    ]
    if official_scores and all(value is not None for value in official_scores):
        comparison["mean_official_score"] = sum(float(value) for value in official_scores) / len(official_scores)

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
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    write_text_atomic(path, buffer.getvalue())


def _status_payload(run_id: str, operation: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "operation": operation,
        "updated_at": utc_now(),
        "models": records,
    }


def _save_status(run_root: Path, run_id: str, operation: str, records: list[dict[str, Any]]) -> None:
    for record in records:
        parsed = collect_model_run_summary(record)
        record["metrics_complete"] = not parsed.missing_horizons
        record["missing_horizons"] = list(parsed.missing_horizons)
    write_json(run_root / "status.json", _status_payload(run_id, operation, records))
    write_json(run_root / "logs.json", {"run_id": run_id, "models": records})


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
    _write_csv(
        run_root / "model_comparison.csv",
        MODEL_COMPARISON_FIELDS,
        [item.comparison_row for item in parsed_rows],
    )


def _new_record(
    model: str,
    result_dir: Path,
    log_path: Path,
    *,
    runtime_fields: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model": model,
        **dict(runtime_fields),
        "status": "PENDING",
        "operation": "train",
        "pid": None,
        "exit_code": None,
        "started_at": None,
        "ended_at": None,
        "wall_seconds": None,
        "result_dir": str(result_dir),
        "log_path": str(log_path),
        "error_summary": None,
        "status_history": ["PENDING"],
        "archive_path": None,
    }


def _run_worker(
    *,
    project_root: Path,
    request_path: Path,
    model: str,
    log_path: Path,
    resolved_environment: ResolvedEnvironment,
) -> tuple[int, str]:
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
    return exit_code, "".join(output)[-2000:]


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
        environment_context = _prepare_batch_environments(
            models=models,
            model_configs=model_configs,
            project_root=project_root,
            device=device,
        )
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
            record["status"] = "PREFLIGHTED"
            record["status_history"].append("PREFLIGHTED")
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
        if resume:
            if not result_dir.exists() or not any(result_dir.iterdir()):
                continue
            if is_completed_run(result_dir):
                record["status"] = "SKIPPED_COMPLETED"
                continue
            checkpoint = result_dir / "last.pt"
            if not checkpoint.is_file():
                record["status"] = "FAILED"
                record["error_summary"] = "--resume requires a valid last.pt; use --overwrite or --id-suffix"
                blocked = True
                continue
            try:
                from engine.checkpoint import read_checkpoint_manifest

                read_checkpoint_manifest(checkpoint)
            except (OSError, RuntimeError, ValueError, TypeError) as exc:
                record["status"] = "FAILED"
                record["error_summary"] = f"invalid last.pt: {exc}; use --overwrite or --id-suffix"
                blocked = True
                continue
            requests["models"][model]["resume_checkpoint"] = str(checkpoint)
            record["status"] = "RESUMED"
            record["status_history"].append("RESUMED")
        elif overwrite:
            if result_dir.exists():
                if not result_dir.is_dir():
                    record["status"] = "FAILED"
                    record["error_summary"] = f"result path is not a directory: {result_dir}"
                    blocked = True
                    continue
                if formal_result_exists(result_dir) or any(result_dir.iterdir()):
                    archive = archive_directory(result_dir, root / "_archive" / model, label=effective_id)
                    record["archive_path"] = str(archive)
                    record["status"] = "OVERWRITTEN"
                    record["status_history"].append("OVERWRITTEN")
        elif result_dir.exists() and (formal_result_exists(result_dir) or any(result_dir.iterdir())):
            record["status"] = "FAILED"
            record["error_summary"] = "result directory is not empty; choose --resume, --overwrite or --id-suffix"
            blocked = True
    write_json(run_root / "request.json", requests)
    _save_status(run_root, effective_id, "train", records)
    _write_summaries(run_root, records)

    failed = blocked
    for index, record in enumerate(records):
        if record["status"] not in {"PENDING", "OVERWRITTEN", "RESUMED"}:
            continue
        if fail_fast and failed:
            record["status"] = "FAILED"
            record["error_summary"] = "not started because --fail-fast stopped the scheduler"
            failed = True
            continue
        model = record["model"]
        requests["models"][model]["resume_checkpoint"] = requests["models"][model].get("resume_checkpoint")
        write_json(run_root / "request.json", requests)
        execution_origin = record["status"]
        record["status"] = "RUNNING"
        record["started_at"] = utc_now()
        record["pid"] = None
        _save_status(run_root, effective_id, "train", records)
        started = time.perf_counter()
        resolved_environment = model_environments[model]
        command = build_model_worker_command(
            project_root=project_root,
            request_path=run_root / "request.json",
            model_name=model,
            resolved_environment=resolved_environment,
        )
        log_path = Path(record["log_path"])
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
            record["pid"] = int(process.pid)
            _save_status(run_root, effective_id, "train", records)
            assert process.stdout is not None
            for line in process.stdout:
                output.append(line)
                log_handle.write(line)
                log_handle.flush()
                print(f"[{model}] {line.rstrip()}")
            exit_code = process.wait()
        record["exit_code"] = int(exit_code)
        record["ended_at"] = utc_now()
        record["wall_seconds"] = time.perf_counter() - started
        if exit_code == 0:
            if execution_origin not in {"RESUMED", "OVERWRITTEN"}:
                record["status"] = "COMPLETED"
            else:
                record["status"] = execution_origin
            record["error_summary"] = record["error_summary"] or None
        else:
            record["status"] = "FAILED"
            record["error_summary"] = "".join(output)[-2000:] or f"worker exited with code {exit_code}"
            failed = True
        _save_status(run_root, effective_id, "train", records)
        _write_summaries(run_root, records)
        if failed and fail_fast:
            for remaining in records[index + 1 :]:
                if remaining["status"] == "PENDING":
                    remaining["status"] = "FAILED"
                    remaining["error_summary"] = "not started because --fail-fast stopped the scheduler"
            _save_status(run_root, effective_id, "train", records)
            _write_summaries(run_root, records)
            break
    _save_status(run_root, effective_id, "train", records)
    _write_summaries(run_root, records)
    return {
        "passed": not any(record["status"] == "FAILED" for record in records),
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
    full_shape: bool = False,
    no_data: bool = False,
    environment_preflight_only: bool = False,
) -> dict[str, Any]:
    if operation not in {"check", "preflight"}:
        raise ValueError(f"unsupported isolated operation: {operation}")
    config_file = Path(config_path).resolve()
    project_root = project_root_from_config(config_file)
    model_configs = _validate_model_configs(models, config_file, model_config_path)
    model_environments, preflight_results = _prepare_batch_environments(
        models=models,
        model_configs=model_configs,
        project_root=project_root,
        device=device,
    )
    if environment_preflight_only:
        results = [
            {
                "model": model,
                **_runtime_record_fields(
                    model_environments[model],
                    preflight_results[model_environments[model].environment_id],
                ),
                "status": "PREFLIGHTED",
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
    run_name = f"{operation}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    run_root = (project_root / "results" / "_runs" / run_name).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    request: dict[str, Any] = {
        "operation": operation,
        "config_path": str(config_file),
        "device": device,
        "cli_overrides": dict(cli_overrides or {}),
        "full_shape": bool(full_shape),
        "no_data": bool(no_data),
        "models": {
            model: {
                "model_config_path": str(model_configs[model]),
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
        exit_code, tail = _run_worker(
            project_root=project_root,
            request_path=request_path,
            model=model,
            log_path=log_path,
            resolved_environment=model_environments[model],
        )
        results.append(
            {
                "model": model,
                **_runtime_record_fields(
                    model_environments[model],
                    preflight_results[model_environments[model].environment_id],
                ),
                "status": "COMPLETED" if exit_code == 0 else "FAILED",
                "exit_code": exit_code,
                "log_path": str(log_path),
                "wall_seconds": time.perf_counter() - started,
                "output_tail": tail,
            }
        )
    result = {"passed": not any(item["status"] == "FAILED" for item in results), "operation": operation, "results": results}
    write_json(run_root / "status.json", result)
    return result

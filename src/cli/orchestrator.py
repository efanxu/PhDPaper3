"""Parent-process orchestration for isolated model workers."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping
from io import StringIO

from runtime.config import ConfigError
from runtime.paths import (
    archive_directory,
    effective_run_id,
    formal_result_exists,
    is_completed_run,
    project_root_from_config,
    resolve_output_root,
    run_directory,
    validate_run_id,
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
    write_json(run_root / "status.json", _status_payload(run_id, operation, records))
    write_json(run_root / "logs.json", {"run_id": run_id, "models": records})


def _write_summaries(run_root: Path, records: list[dict[str, Any]]) -> None:
    summary_rows: list[dict[str, Any]] = []
    performance_rows: list[dict[str, Any]] = []
    performance_fields = [
        "model",
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
    for record in records:
        result_dir = Path(record["result_dir"])
        run_info: dict[str, Any] = {}
        performance: dict[str, Any] = {}
        if (result_dir / "run_info.json").is_file():
            try:
                run_info = json.loads((result_dir / "run_info.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                run_info = {}
        if (result_dir / "performance.json").is_file():
            try:
                performance = json.loads((result_dir / "performance.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                performance = {}
        test = performance.get("test", {})
        summary_rows.append(
            {
                "model": record["model"],
                "status": record["status"],
                "best_epoch": run_info.get("best_epoch"),
                "main_metric": run_info.get("test_monitor"),
                "result_dir": record["result_dir"],
                "exit_code": record.get("exit_code"),
            }
        )
        performance_rows.append(
            {
                "model": record["model"],
                "status": record["status"],
                "parameter_count": performance.get("parameter_count"),
                "trainable_parameter_count": performance.get("trainable_parameter_count"),
                "checkpoint_size_mb": performance.get("checkpoint_size_mb"),
                "epochs_completed": performance.get("epochs_completed"),
                "best_epoch": performance.get("best_epoch"),
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
                "result_dir": record["result_dir"],
            }
        )
    _write_csv(
        run_root / "summary.csv",
        ["model", "status", "best_epoch", "main_metric", "result_dir", "exit_code"],
        summary_rows,
    )
    _write_csv(run_root / "performance_summary.csv", performance_fields, performance_rows)


def _new_record(model: str, result_dir: Path, log_path: Path) -> dict[str, Any]:
    return {
        "model": model,
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
) -> tuple[int, str]:
    command = [sys.executable, str(_worker_script(project_root)), "_worker", str(request_path), model]
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
    records = [
        _new_record(model, run_directory(project_root, root, model, effective_id), logs_root / f"{model}.log")
        for model in models
    ]
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
        if record["status"] not in {"PENDING", "OVERWRITTEN"}:
            continue
        if fail_fast and failed:
            record["status"] = "FAILED"
            record["error_summary"] = "not started because --fail-fast stopped the scheduler"
            failed = True
            continue
        model = record["model"]
        requests["models"][model]["resume_checkpoint"] = requests["models"][model].get("resume_checkpoint")
        write_json(run_root / "request.json", requests)
        record["status"] = "RUNNING"
        record["started_at"] = utc_now()
        record["pid"] = None
        _save_status(run_root, effective_id, "train", records)
        started = time.perf_counter()
        command = [sys.executable, str(_worker_script(project_root)), "_worker", str(run_root / "request.json"), model]
        log_path = Path(record["log_path"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        output: list[str] = []
        with log_path.open("w", encoding="utf-8", newline="") as log_handle:
            process = subprocess.Popen(command, cwd=project_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
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
            record["status"] = "COMPLETED"
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
) -> dict[str, Any]:
    if operation not in {"check", "preflight"}:
        raise ValueError(f"unsupported isolated operation: {operation}")
    config_file = Path(config_path).resolve()
    project_root = project_root_from_config(config_file)
    model_configs = _validate_model_configs(models, config_file, model_config_path)
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
        "models": {model: {"model_config_path": str(model_configs[model])} for model in models},
    }
    request_path = run_root / "request.json"
    write_json(request_path, request)
    results: list[dict[str, Any]] = []
    for model in models:
        log_path = project_root / "logs" / run_name / f"{model}.log"
        started = time.perf_counter()
        exit_code, tail = _run_worker(project_root=project_root, request_path=request_path, model=model, log_path=log_path)
        results.append({"model": model, "status": "COMPLETED" if exit_code == 0 else "FAILED", "exit_code": exit_code, "log_path": str(log_path), "wall_seconds": time.perf_counter() - started, "output_tail": tail})
    result = {"passed": not any(item["status"] == "FAILED" for item in results), "operation": operation, "results": results}
    write_json(run_root / "status.json", result)
    return result

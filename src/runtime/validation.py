"""One forward/backward shape validation implementation for every command."""

from __future__ import annotations

import sys
import time
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from data.dataset import ForecastBatch
from data.loader import load_data
from engine.model_execution import build_execution_plan, execute_training_backward
from engine.reproducibility import set_seed
from engine.precision import resolve_precision_policy
from models.base import DataInfoView, ModelInput
from models.loader import build_model
from runtime.config import (
    apply_cli_overrides,
    cli_overrides_as_nested,
    load_experiment_config,
    load_model_config,
)
from runtime.paths import project_root_from_config
from runtime.status import (
    FORMAL_DEFAULT_SHAPE,
    FAILED,
    INTERFACE_SMALL,
    PASS,
    RESOLVED_SHAPE,
    RUNNING,
    failure_details,
    stable_phase,
    write_status,
)
from runtime.run_info import utc_now


def _choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(value)


def _batch_size(profile: str, configured: int, cli_overrides: Mapping[str, Any]) -> tuple[int, str]:
    if profile == INTERFACE_SMALL:
        return min(2, configured), "interface_small"
    if "training.train_batch_size" in cli_overrides:
        return configured, "cli_override"
    return configured, "yaml_default"


def _memory_snapshot(device: torch.device) -> dict[str, float | None]:
    if device.type != "cuda":
        return {
            "gpu_total_mb": None,
            "gpu_allocated_mb": None,
            "gpu_reserved_mb": None,
            "peak_gpu_allocated_mb": None,
            "peak_gpu_reserved_mb": None,
        }
    free, total = torch.cuda.mem_get_info(device)
    del free
    scale = 1024.0 * 1024.0
    return {
        "gpu_total_mb": float(total) / scale,
        "gpu_allocated_mb": float(torch.cuda.memory_allocated(device)) / scale,
        "gpu_reserved_mb": float(torch.cuda.memory_reserved(device)) / scale,
        "peak_gpu_allocated_mb": float(torch.cuda.max_memory_allocated(device)) / scale,
        "peak_gpu_reserved_mb": float(torch.cuda.max_memory_reserved(device)) / scale,
    }


def _estimated_input_tensor_mb(batch: int, info: DataInfoView) -> float:
    values = batch * (
        info.lookback * info.num_nodes * info.num_features + 2 * info.num_nodes * info.max_pred_len
    )
    return float(values * 4) / (1024.0 * 1024.0)


def run_shape_validation(
    *,
    model_name: str,
    config_path: str | Path,
    model_config_path: str | Path | None,
    device: str,
    profile: str,
    cli_overrides: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    operation: str = "check",
    runtime_environment: str | None = None,
    status_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate actual resolved tensor dimensions in the current process.

    Callers intentionally run this function in a short-lived worker.  It loads
    real metadata and graph resources, but only passes synthetic ``x`` through
    ``ModelInput``; labels and masks remain owned by this validator.
    """

    if profile not in {INTERFACE_SMALL, RESOLVED_SHAPE, FORMAL_DEFAULT_SHAPE}:
        raise ValueError(f"unsupported shape validation profile: {profile}")
    started_at = utc_now()
    started = time.perf_counter()
    phase = "config"
    selected_device: torch.device | None = None
    model = None
    precision = None
    x = target = target_mask = output = loss = None
    overrides = dict(cli_overrides or {})
    payload: dict[str, Any] = {
        "schema_version": 2,
        "model": model_name,
        "run_id": run_id,
        "operation": operation,
        "profile": profile,
        "status": RUNNING,
        "classification": None,
        "phase": "preflight",
        "started_at": started_at,
        "ended_at": None,
        "wall_seconds": None,
        "runtime_environment": runtime_environment,
        "python_executable": sys.executable,
        "device": device,
        "batch_size": None,
        "batch_size_source": None,
        "cli_overrides": cli_overrides_as_nested(overrides),
        "input_shape": None,
        "output_shape": None,
        "parameter_count": None,
        "estimated_input_tensor_mb": None,
        "oom_requested_allocation_mb": None,
        "error": None,
        "details": {},
        "exit_code": 0,
    }
    try:
        config_file = Path(config_path).resolve()
        root = project_root_from_config(config_file)
        base_config = load_experiment_config(config_file)
        config = apply_cli_overrides(base_config, overrides, project_root=root)
        model_file = Path(model_config_path).resolve() if model_config_path else root / "configs" / "models" / f"{model_name}.yaml"
        model_config = load_model_config(model_file)
        selected_device = _choose_device(device)
        payload["device"] = str(selected_device)
        precision = resolve_precision_policy(
            device=selected_device,
            amp_configured=bool(config.training["amp"]),
            amp_dtype=str(config.training["amp_dtype"]),
            amp_cache_enabled=bool(config.training["amp_cache_enabled"]),
        )
        payload["details"]["precision"] = precision.as_dict()
        seed_details = set_seed(
            int(config.training["seed"]),
            reproducibility_mode=str(config.runtime["reproducibility_mode"]),
        )
        payload["reproducibility_mode"] = seed_details["reproducibility_mode"]
        payload["details"]["reproducibility"] = seed_details
        phase = "data"
        _, loaded_data_info = load_data(config, project_root=root)
        data_info = DataInfoView.from_object(loaded_data_info)
        batch_size, source = _batch_size(profile, int(config.training["train_batch_size"]), overrides)
        payload["batch_size"] = int(batch_size)
        payload["batch_size_source"] = source
        payload["estimated_input_tensor_mb"] = _estimated_input_tensor_mb(batch_size, data_info)
        payload["input_shape"] = [
            int(batch_size),
            int(data_info.lookback),
            int(data_info.num_nodes),
            int(data_info.num_features),
        ]
        if selected_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(selected_device)
        phase = "model_build"
        model = build_model(model_name, model_config, data_info).to(selected_device)
        execution_plan = build_execution_plan(
            model,
            total_nodes=int(data_info.num_nodes),
            node_shared_chunk_size=int(config.runtime["node_shared_chunk_size"]),
        )
        payload["details"]["execution"] = execution_plan.as_dict()
        payload["parameter_count"] = int(sum(parameter.numel() for parameter in model.parameters()))
        phase = "forward"
        x = torch.randn(batch_size, data_info.lookback, data_info.num_nodes, data_info.num_features, device=selected_device)
        target = torch.randn(batch_size, data_info.num_nodes, data_info.max_pred_len, device=selected_device)
        target_mask = torch.ones_like(target, dtype=torch.bool)
        model.train()
        batch = ForecastBatch(
            x=x,
            target=target,
            target_mask=target_mask,
            starts=torch.arange(batch_size, dtype=torch.int64),
        )
        execution = execute_training_backward(
            model,
            [batch],
            device=selected_device,
            plan=execution_plan,
            loss_name=str(config.training["loss"]),
            autocast=precision.autocast,
            backward=lambda contribution: contribution.backward(),
            capture_prediction=True,
        )
        output = execution.prediction
        if output is None:
            raise RuntimeError("execution did not return a validation prediction")
        payload["output_shape"] = [int(value) for value in output.shape]
        expected = (batch_size, data_info.num_nodes, data_info.max_pred_len)
        if tuple(output.shape) != expected:
            raise ValueError(f"output must have shape {expected}, got {tuple(output.shape)}")
        if not bool(torch.isfinite(output).all()):
            raise FloatingPointError("output contains NaN or Inf")
        phase = "loss"
        loss = float(execution.loss)
        if not torch.isfinite(torch.tensor(loss)):
            raise FloatingPointError("loss contains NaN or Inf")
        phase = "backward"
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        if not all(gradient is not None for gradient in gradients):
            raise RuntimeError("missing gradient after backward")
        if not all(bool(torch.isfinite(gradient).all()) for gradient in gradients if gradient is not None):
            raise FloatingPointError("gradient contains NaN or Inf")
        payload["status"] = PASS
        payload["classification"] = None
        payload["phase"] = "resolved_shape"
        payload["error"] = None
    except BaseException as exc:
        traceback_tail = "".join(traceback.format_exception(exc))[-4000:]
        details = failure_details(
            exc,
            phase=phase,
            traceback_tail=traceback_tail,
        )
        payload.update(
            {
                "status": FAILED,
                "classification": details["classification"],
                "phase": stable_phase(phase),
                "error": details["error"],
                "exit_code": 1,
            }
        )
    finally:
        if selected_device is not None:
            try:
                payload.update(_memory_snapshot(selected_device))
            except Exception as diagnostic_error:
                payload["diagnostic_error"] = str(diagnostic_error)
        payload["ended_at"] = utc_now()
        payload["wall_seconds"] = time.perf_counter() - started
        del model, x, target, target_mask, output, loss
        if selected_device is not None and selected_device.type == "cuda":
            torch.cuda.empty_cache()
        if status_path is not None:
            write_status(Path(status_path), payload)
    return payload

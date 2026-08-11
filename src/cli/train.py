"""Run one model through the shared data, training and evaluation pipeline."""

from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
import statistics
import time
import traceback
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from data.dataloader import build_dataloaders
from data.loader import load_data
from data.normalization import fit_normalization
from data.split import chronological_split
from data.window import build_window_index
from engine.checkpoint import load_checkpoint, read_checkpoint_manifest
from engine.evaluator import EvaluationResult, evaluate
from engine.model_execution import ExecutionPlan, build_execution_plan
from engine.reproducibility import set_seed
from engine.trainer import TrainResult, Trainer
from models.loader import build_model
from runtime.config import (
    ExperimentConfig,
    apply_cli_overrides,
    cli_overrides_as_nested,
    cli_overrides_from_namespace,
    load_experiment_config,
    load_model_config_document,
    resolved_config_values,
)
from runtime.environment import collect_environment
from runtime.environments import resolve_model_environment
from runtime.paths import project_root_from_config, resolve_output_root, run_directory
from runtime.run_info import utc_now, write_json, write_yaml
from runtime.status import (
    FAILED,
    FULL,
    PASS,
    RUNNING,
    SMOKE,
    failure_details,
    failure_summary,
    finished_phase,
    phase_record,
    running_phase,
    stable_phase,
    write_status,
)


def _choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(value)


def _default_model_config_path(config_path: Path, model_name: str) -> Path:
    return config_path.resolve().parent / "models" / f"{model_name}.yaml"


def _write_history(path: Path, history: list[dict[str, Any]]) -> None:
    fields = ["epoch", "train_loss", "monitor", "learning_rate", "checkpoint_selected", "train_updates"]
    rows = []
    for row in history:
        rows.append(
            {
                "epoch": row["epoch"],
                "train_loss": row["train_loss"],
                "monitor": row["validation"]["monitor"],
                "learning_rate": row["learning_rate"],
                "checkpoint_selected": row["checkpoint_selected"],
                "train_updates": row["train_updates"],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_evaluation_outputs(
    output_dir: Path,
    validation: EvaluationResult,
    test: EvaluationResult,
    *,
    save_predictions: bool,
    split: str = "both",
) -> None:
    if split in {"validation", "both"}:
        write_json(output_dir / "metrics_validation.json", validation.metrics)
    if split in {"test", "both"}:
        for horizon, metrics in test.metrics["by_horizon"].items():
            write_json(output_dir / f"metrics_test_h{horizon}.json", metrics)
    if save_predictions:
        values: dict[str, Any] = {}
        if split in {"validation", "both"}:
            values.update(
                {
                    "validation_prediction_kw": validation.prediction_kw,
                    "validation_target_kw": validation.target_kw,
                    "validation_target_mask": validation.target_mask,
                    "validation_starts": validation.starts,
                }
            )
        if split in {"test", "both"}:
            values.update(
                {
                    "test_prediction_kw": test.prediction_kw,
                    "test_target_kw": test.target_kw,
                    "test_target_mask": test.target_mask,
                    "test_starts": test.starts,
                }
            )
        np.savez(output_dir / "predictions.npz", **values)


def _prepare(config: ExperimentConfig, project_root: Path):
    arrays, data_info = load_data(config, project_root=project_root)
    splits = chronological_split(len(arrays.timestamps), config)
    windows = build_window_index(
        splits,
        lookback=int(config.data["lookback"]),
        horizon=int(config.data["max_pred_len"]),
        strides={
            "train": int(config.sampling["train_stride"]),
            "validation": int(config.sampling["val_stride"]),
            "test": int(config.sampling["test_stride"]),
        },
        target_mask=arrays.target_mask,
    )
    normalization = fit_normalization(arrays, splits.train)
    loaders = build_dataloaders(arrays, normalization, windows, splits, config)
    return arrays, data_info, splits, windows, normalization, loaders


def _checkpoint_path(value: str | Path) -> Path:
    path = Path(value).resolve()
    if path.is_dir():
        for name in ("last.pt", "best.pt"):
            candidate = path / name
            if candidate.is_file():
                return candidate
    return path


_CHECKPOINT_CONFIG_PATHS: tuple[tuple[str, ...], ...] = (
    ("data", "feature_columns"),
    ("data", "lookback"),
    ("data", "max_pred_len"),
    ("data", "num_nodes"),
    ("data", "eval_horizons"),
    ("data", "target_column"),
    ("data", "mask_column"),
    ("resources", "graph", "type"),
    ("resources", "graph", "k"),
    ("resources", "graph", "symmetrize"),
    ("resources", "graph", "self_loops"),
    ("resources", "graph", "weighting"),
    ("split", "method"),
    ("split", "train_ratio"),
    ("split", "val_ratio"),
    ("split", "test_ratio"),
    ("training", "loss"),
    ("training", "optimizer"),
    ("training", "train_batch_size"),
    ("training", "val_batch_size"),
    ("training", "test_batch_size"),
    ("training", "gradient_accumulation_steps"),
    ("training", "learning_rate"),
    ("training", "weight_decay"),
    ("training", "betas"),
    ("training", "epsilon"),
    ("training", "scheduler"),
    ("training", "amp"),
    ("training", "seed"),
    ("runtime", "reproducibility_mode"),
)
_GRAPH_CHECKPOINT_CONFIG_PATHS = frozenset(
    {
        ("resources", "graph", "type"),
        ("resources", "graph", "k"),
        ("resources", "graph", "symmetrize"),
        ("resources", "graph", "self_loops"),
        ("resources", "graph", "weighting"),
    }
)


def _at_path(value: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = value
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _saved_execution_plan(
    manifest: Mapping[str, Any], saved_config: Mapping[str, Any]
) -> Mapping[str, Any] | None:
    execution = manifest.get("execution_plan")
    if isinstance(execution, Mapping):
        return execution
    resolved = saved_config.get("resolved")
    if isinstance(resolved, Mapping) and isinstance(resolved.get("execution"), Mapping):
        return resolved["execution"]
    return None


def _check_checkpoint_compatibility(
    manifest: Mapping[str, Any],
    config: ExperimentConfig,
    model_config: Mapping[str, Any],
    checkpoint_path: Path,
    *,
    model_name: str,
    execution_plan: ExecutionPlan,
    for_resume: bool = True,
) -> None:
    if manifest.get("model") not in {None, model_name}:
        raise ValueError(f"checkpoint {checkpoint_path} belongs to model {manifest.get('model')!r}, not {model_name!r}")
    saved_config = (
        manifest.get("resolved_config")
        or manifest.get("resolved_config_identity")
        or manifest.get("experiment_config")
    )
    if not isinstance(saved_config, Mapping):
        raise ValueError(f"checkpoint {checkpoint_path} has no saved resolved experiment config; refusing compatibility-unsafe resume")
    reproducibility_path = ("runtime", "reproducibility_mode")
    saved_reproducibility_mode = _at_path(saved_config, reproducibility_path)
    current_reproducibility_mode = _at_path(config.values, reproducibility_path)
    if for_resume:
        if saved_reproducibility_mode is None:
            raise ValueError(
                f"checkpoint {checkpoint_path} is missing runtime.reproducibility_mode; "
                "old checkpoints cannot be resumed; create a new checkpoint with "
                "runtime.reproducibility_mode=controlled_nonstrict"
            )
        if saved_reproducibility_mode != current_reproducibility_mode:
            raise ValueError(
                f"checkpoint {checkpoint_path} is incompatible at "
                "runtime.reproducibility_mode: "
                f"saved={saved_reproducibility_mode!r}, "
                f"current={current_reproducibility_mode!r}"
            )
    if for_resume and execution_plan.uses_node_microbatch:
        saved_execution = _saved_execution_plan(manifest, saved_config)
        if (
            saved_execution is None
            or saved_execution.get("execution_mode") != "node_shared_microbatch"
        ):
            raise ValueError(
                f"checkpoint {checkpoint_path} predates node_shared_microbatch execution; "
                "it cannot be resumed with the current node_shared_microbatch plan"
            )
        saved_chunk = _at_path(saved_config, ("runtime", "node_shared_chunk_size"))
        if saved_chunk is None:
            raise ValueError(
                f"checkpoint {checkpoint_path} predates node_shared_microbatch execution; "
                "saved runtime.node_shared_chunk_size is missing"
            )
        current_chunk = _at_path(config.values, ("runtime", "node_shared_chunk_size"))
        if saved_chunk != current_chunk:
            raise ValueError(
                f"checkpoint {checkpoint_path} is incompatible at "
                "runtime.node_shared_chunk_size: "
                f"saved={saved_chunk!r}, current={current_chunk!r}"
            )
    for path in _CHECKPOINT_CONFIG_PATHS:
        if path == reproducibility_path:
            # Evaluate-only accepts old or differently configured checkpoints;
            # it never restores training RNG state.
            continue
        if model_name != "stcn" and path in _GRAPH_CHECKPOINT_CONFIG_PATHS:
            continue
        if not for_resume and path in {
            ("training", "train_batch_size"),
            ("training", "val_batch_size"),
            ("training", "test_batch_size"),
        }:
            continue
        saved = _at_path(saved_config, path)
        current = _at_path(config.values, path)
        if saved != current:
            dotted = ".".join(path)
            raise ValueError(f"checkpoint {checkpoint_path} is incompatible at {dotted}: saved={saved!r}, current={current!r}")
    saved_epoch = int(manifest.get("epoch", 0))
    if int(config.training["epochs"]) < saved_epoch:
        raise ValueError(f"resume epochs {config.training['epochs']} cannot be less than checkpoint epoch {saved_epoch}")
    saved_model_config = manifest.get("model_config") or manifest.get("model_config_identity")
    if not isinstance(saved_model_config, Mapping):
        raise ValueError(f"checkpoint {checkpoint_path} has no saved model config; refusing compatibility-unsafe resume")
    if dict(saved_model_config) != dict(model_config):
        raise ValueError(f"checkpoint {checkpoint_path} is incompatible with the current model config")
    runtime_state = manifest.get("runtime_state")
    if for_resume and runtime_state is not None and ("rng" not in runtime_state or "dataloader_generators" not in runtime_state):
        raise ValueError(f"checkpoint {checkpoint_path} does not contain complete resume RNG/DataLoader state")


def _print_config_summary(
    *,
    config_file: Path,
    model_file: Path,
    base_config: ExperimentConfig,
    cli_overrides: Mapping[str, Any],
    output_dir: Path,
) -> None:
    print(f"Public config: {config_file}")
    print(f"Model config: {model_file}")
    print("CLI overrides:")
    if not cli_overrides:
        print("  (none; using YAML values)")
    else:
        for dotted_path, new_value in cli_overrides.items():
            old_value = _at_path(base_config.values, tuple(dotted_path.split(".")))
            print(f"  {dotted_path}: {old_value!r} -> {new_value!r}")
    print(f"Resolved config saved to: {output_dir / 'resolved_config.yaml'}")


def _memory_mb(device: torch.device) -> tuple[float | None, float | None]:
    if device.type != "cuda":
        return None, None
    return (
        float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0),
        float(torch.cuda.max_memory_reserved(device)) / (1024.0 * 1024.0),
    )


def _performance(
    *,
    output_dir: Path,
    environment: Mapping[str, Any],
    data_prepare_seconds: float,
    model_build_seconds: float,
    checkpoint_reload_seconds: float,
    started: float,
    device: torch.device,
    data_info: Any,
    config: ExperimentConfig,
    train_result: TrainResult | None,
    validation: EvaluationResult,
    test: EvaluationResult,
    best_epoch: int | None,
    parameter_count: int,
    trainable_parameter_count: int,
    execution_plan: ExecutionPlan,
) -> dict[str, Any]:
    allocated_mb, reserved_mb = _memory_mb(device)
    total_mb = (
        float(torch.cuda.get_device_properties(device).total_memory) / (1024.0 * 1024.0)
        if device.type == "cuda"
        else None
    )
    best_path = output_dir / "best.pt"
    checkpoint_size_mb = float(best_path.stat().st_size) / (1024.0 * 1024.0) if best_path.is_file() else None
    epoch_seconds = train_result.epoch_seconds if train_result else []
    update_seconds = train_result.update_seconds if train_result else []
    training_wall_seconds = float(sum(epoch_seconds)) if epoch_seconds else None
    return {
        "runtime_environment": environment.get("runtime_environment"),
        "conda_env": environment.get("conda_env"),
        "python_executable": environment.get("python_executable"),
        "environment_resolution_source": environment.get("environment_resolution_source"),
        "parameter_count": int(parameter_count),
        "trainable_parameter_count": int(trainable_parameter_count),
        "checkpoint_size_mb": checkpoint_size_mb,
        "data_prepare_seconds": float(data_prepare_seconds),
        "model_build_seconds": float(model_build_seconds),
        "checkpoint_reload_seconds": float(checkpoint_reload_seconds),
        "training_wall_seconds": training_wall_seconds,
        "mean_epoch_seconds": float(statistics.mean(epoch_seconds)) if epoch_seconds else None,
        "median_epoch_seconds": float(statistics.median(epoch_seconds)) if epoch_seconds else None,
        "optimizer_updates": int(len(update_seconds)) if train_result else 0,
        "mean_update_seconds": float(statistics.mean(update_seconds)) if update_seconds else None,
        "best_epoch": best_epoch,
        "epochs_completed": train_result.epochs_completed if train_result else 0,
        "validation": validation.performance,
        "test": test.performance,
        "peak_gpu_allocated_mb": allocated_mb,
        "peak_gpu_reserved_mb": reserved_mb,
        "gpu_total_mb": total_mb,
        "total_wall_seconds": float(time.perf_counter() - started),
        "device": str(device),
        "gpu": environment.get("gpu"),
        "pytorch": environment.get("packages", {}).get("torch"),
        "cuda": environment.get("cuda"),
        "amp": bool(config.training["amp"]),
        "batch_size": {
            "train": int(config.training["train_batch_size"]),
            "validation": int(config.training["val_batch_size"]),
            "test": int(config.training["test_batch_size"]),
        },
        "input_shape": [
            int(config.training["test_batch_size"]),
            int(data_info.lookback),
            int(data_info.num_nodes),
            int(data_info.num_features),
        ],
        "forecast_shape": [int(data_info.num_nodes), int(data_info.max_pred_len)],
        "execution": execution_plan.as_dict(),
    }


def _write_model_invocation_artifacts(
    *,
    output_dir: Path,
    command_argv: list[str] | None,
    command_name: str,
    model_name: str,
    run_name: str,
    config_file: Path,
    model_file: Path,
    model_document: Mapping[str, Any],
    effective_overrides: Mapping[str, Any],
) -> None:
    """Persist original invocation provenance and its stable replay config."""

    replay_model_config_path = output_dir / "model_config.yaml"
    write_yaml(replay_model_config_path, dict(model_document))
    write_json(
        output_dir / "command.json",
        {
            "argv": list(command_argv or []),
            "command": command_name,
            "model": model_name,
            "run_id": run_name,
            "config_path": str(config_file),
            "model_config_path": str(model_file),
            "replay_model_config_path": "model_config.yaml",
            "cli_overrides": cli_overrides_as_nested(effective_overrides),
        },
    )


def _run_model_impl(
    *,
    model_name: str,
    config_path: str | Path = "configs/experiment.yaml",
    model_config_path: str | Path | None = None,
    run_id: str | None = None,
    device: str = "auto",
    output_root: str | Path | None = None,
    resume: str | Path | None = None,
    evaluate_only: bool = False,
    smoke: bool = False,
    smoke_epochs: int | None = None,
    smoke_max_train_updates: int | None = None,
    smoke_max_eval_batches: int | None = None,
    seed_override: int | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
    command_argv: list[str] | None = None,
    command_name: str = "train",
    evaluation_split: str = "both",
    runtime_environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    project_root = project_root_from_config(config_file)
    base_config = load_experiment_config(config_file)
    effective_overrides = cli_overrides_from_namespace(cli_overrides or {})
    config = apply_cli_overrides(base_config, effective_overrides, project_root=project_root)
    if seed_override is not None:
        requested_seed = effective_overrides.get("training.seed")
        if requested_seed is not None and int(requested_seed) != int(seed_override):
            raise ValueError("seed_override conflicts with the explicit training.seed override")
        if requested_seed is None:
            config = apply_cli_overrides(config, {"training.seed": int(seed_override)}, project_root=project_root)
            effective_overrides["training.seed"] = int(seed_override)
    model_file = Path(model_config_path).resolve() if model_config_path else _default_model_config_path(config_file, model_name)
    model_document = load_model_config_document(model_file)
    model_config = dict(model_document["model"])
    if runtime_environment is None:
        resolved_environment = resolve_model_environment(
            model_file,
            project_root=project_root,
        )
        runtime_metadata: dict[str, Any] = {
            "runtime_environment": resolved_environment.environment_id,
            "conda_env": resolved_environment.conda_env,
            "python_executable": str(resolved_environment.python_executable),
            "environment_resolution_source": resolved_environment.resolution_source,
            "source_roots": [str(path) for path in resolved_environment.source_roots],
            "required_imports": list(resolved_environment.required_imports),
        }
    else:
        runtime_metadata = dict(runtime_environment)
        if not runtime_metadata.get("runtime_environment"):
            raise ValueError("runtime_environment metadata must include runtime_environment")
    checkpoint_file = _checkpoint_path(resume) if resume is not None else None
    legacy_reproducibility_mode_unknown = False
    checkpoint_for_compatibility: dict[str, Any] | None = None
    if checkpoint_file is not None:
        checkpoint_for_compatibility = read_checkpoint_manifest(checkpoint_file)
        saved_config = (
            checkpoint_for_compatibility.get("resolved_config")
            or checkpoint_for_compatibility.get("resolved_config_identity")
            or checkpoint_for_compatibility.get("experiment_config")
        )
        legacy_reproducibility_mode_unknown = (
            isinstance(saved_config, Mapping)
            and _at_path(saved_config, ("runtime", "reproducibility_mode")) is None
        )
    run_name = run_id or datetime.now().strftime("run-%Y%m%d-%H%M%S")
    output_dir = run_directory(project_root, output_root, model_name, run_name)
    if output_dir.exists() and any(output_dir.iterdir()) and (resume is None or evaluate_only):
        precheck_artifacts = {"validation_status.json"}
        existing_names = {path.name for path in output_dir.iterdir()}
        if not existing_names.issubset(precheck_artifacts):
            raise ValueError(f"result directory already contains files: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = utc_now()
    started = time.perf_counter()
    selected_device = _choose_device(device)
    if selected_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(selected_device)
    seed_details = set_seed(
        int(config.training["seed"]),
        reproducibility_mode=str(config.runtime["reproducibility_mode"]),
    )
    environment = collect_environment(
        project_root,
        reproducibility_mode=str(config.runtime["reproducibility_mode"]),
        seed_details=seed_details,
    )
    environment.update(runtime_metadata)
    environment["runtime_environment"] = runtime_metadata["runtime_environment"]
    environment["conda_env"] = runtime_metadata.get("conda_env")
    environment["python_executable"] = runtime_metadata.get("python_executable", environment["python"]["executable"])
    environment["python_version"] = runtime_metadata.get(
        "python_version", environment["python"]["version"]
    )
    environment["environment_resolution_source"] = runtime_metadata.get("environment_resolution_source")
    resolved = resolved_config_values(config, project_root=project_root)
    resolved["resolved"].update({
        "model_name": model_name,
        "run_id": run_name,
        "device": str(selected_device),
        "command": command_name,
        "runtime_environment": runtime_metadata["runtime_environment"],
        "python_executable": runtime_metadata.get("python_executable"),
        "reproducibility": seed_details,
    })
    if smoke:
        resolved["resolved"]["smoke"] = {
            "epochs": int(smoke_epochs or 1),
            "max_train_updates": int(smoke_max_train_updates or 2),
            "max_eval_batches": int(smoke_max_eval_batches or 2),
        }
    write_yaml(output_dir / "resolved_config.yaml", resolved)
    write_yaml(output_dir / "cli_overrides.yaml", cli_overrides_as_nested(effective_overrides))
    _write_model_invocation_artifacts(
        output_dir=output_dir,
        command_argv=command_argv,
        command_name=command_name,
        model_name=model_name,
        run_name=run_name,
        config_file=config_file,
        model_file=model_file,
        model_document=model_document,
        effective_overrides=effective_overrides,
    )
    _print_config_summary(config_file=config_file, model_file=model_file, base_config=base_config, cli_overrides=effective_overrides, output_dir=output_dir)
    write_json(output_dir / "environment.json", environment)
    run_phases = {
        "preflight": phase_record(phase="preflight"),
        "resolved_shape": phase_record(phase="resolved_shape"),
        "training": phase_record(phase="training"),
        "checkpoint": phase_record(phase="checkpoint"),
        "evaluation": phase_record(phase="evaluation"),
        "overall": phase_record(phase="overall"),
    }
    run_info_path = output_dir / "run_info.json"
    write_status(
        run_info_path,
        {
            "schema_version": 2,
            "model": model_name,
            "run_id": run_name,
            "operation": "train",
            "profile": SMOKE if smoke else FULL,
            "status": RUNNING,
            "classification": None,
            "phase": "preflight",
            "start_time": start_time,
            "end_time": None,
            "wall_seconds": None,
            "exit_code": None,
            "exception_type": None,
            "error_message": None,
            "traceback_tail": None,
            "error": None,
            "metrics_complete": None,
            "phases": run_phases,
            **{
                key: runtime_metadata.get(key)
                for key in (
                    "runtime_environment",
                    "conda_env",
                    "python_executable",
                    "python_version",
                    "environment_resolution_source",
                )
            },
        },
    )

    def write_progress(phase: str) -> None:
        current = json.loads(run_info_path.read_text(encoding="utf-8"))
        current.update(
            {
                "status": RUNNING,
                "classification": None,
                "phase": stable_phase(phase),
                "error": None,
                "phases": run_phases,
            }
        )
        write_status(run_info_path, current)

    data_started = time.perf_counter()
    arrays, data_info, splits, windows, normalization, loaders = _prepare(config, project_root)
    data_prepare_seconds = time.perf_counter() - data_started
    normalization.save(output_dir / "normalization.npz")
    resolved["resolved"].update(
        {
            "data_info": data_info.as_dict(),
            "split_boundaries": splits.as_dict(arrays.timestamps),
            "window_index": windows.as_dict(),
            "window_starts": {
                "train": [int(value) for value in windows.train],
                "train_valid": [int(value) for value in windows.train_valid],
                "validation": [int(value) for value in windows.validation],
                "test": [int(value) for value in windows.test],
            },
            "normalization": normalization.as_dict(),
        }
    )
    write_yaml(output_dir / "resolved_config.yaml", resolved)
    model_started = time.perf_counter()
    model = build_model(model_name, model_config, data_info)
    model_build_seconds = time.perf_counter() - model_started
    execution_plan = build_execution_plan(
        model,
        total_nodes=int(data_info.num_nodes),
        node_shared_chunk_size=int(config.runtime["node_shared_chunk_size"]),
    )
    if checkpoint_for_compatibility is not None:
        _check_checkpoint_compatibility(
            checkpoint_for_compatibility,
            config,
            model_config,
            checkpoint_file,
            model_name=model_name,
            execution_plan=execution_plan,
            for_resume=not evaluate_only,
        )
    resolved["resolved"]["execution"] = execution_plan.as_dict()
    write_yaml(output_dir / "resolved_config.yaml", resolved)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    trainable_parameter_count = int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    run_phases["preflight"] = finished_phase(phase="preflight")
    checkpoint_extra = {
        "config_file": str(config_file),
        "model_config_file": str(model_file),
        "experiment_config": config.copy_values(),
        "resolved_config": resolved,
        "model_config": model_config,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "seed": int(config.training["seed"]),
        "reproducibility_mode": str(config.runtime["reproducibility_mode"]),
        "seed_details": seed_details,
        "cli_overrides": cli_overrides_as_nested(effective_overrides),
        "execution_plan": execution_plan.as_dict(),
    }
    train_result: TrainResult | None = None
    checkpoint_manifest: dict[str, Any] | None = None
    resume_manifest: dict[str, Any] | None = None
    final_eval_limit: int | None = None
    checkpoint_reload_seconds = 0.0
    if evaluate_only:
        assert checkpoint_file is not None
        run_phases["checkpoint"] = running_phase("checkpoint_reload", artifact=str(checkpoint_file))
        write_progress("checkpoint_reload")
        reload_started = time.perf_counter()
        checkpoint_manifest = load_checkpoint(checkpoint_file, model, device=selected_device)
        checkpoint_reload_seconds = time.perf_counter() - reload_started
        run_phases["checkpoint"] = finished_phase(
            profile=FULL,
            phase="checkpoint_reload",
            artifact=str(checkpoint_file),
            started_at=run_phases["checkpoint"]["started_at"],
            wall_seconds=checkpoint_reload_seconds,
        )
    else:
        trainer = Trainer(model, config, device=selected_device, model_name=model_name, normalization=normalization, output_dir=output_dir, dataloader_generators={"train": loaders.train.generator, "validation": loaders.validation.generator, "test": loaders.test.generator}, execution_plan=execution_plan)
        start_epoch = 1
        resume_state: Mapping[str, Any] | None = None
        if checkpoint_file is not None:
            resume_manifest = load_checkpoint(checkpoint_file, trainer.model, device=selected_device, optimizer=trainer.optimizer, scheduler=trainer.scheduler, scaler=trainer.scaler, restore_runtime=True, dataloader_generators={"train": loaders.train.generator, "validation": loaders.validation.generator, "test": loaders.test.generator})
            start_epoch = int(resume_manifest.get("epoch", 0)) + 1
            resume_state = resume_manifest.get("runtime_state", {}).get("trainer")
        if smoke:
            epochs = int(smoke_epochs or 1)
            max_train_updates = int(smoke_max_train_updates or 2)
            max_eval_batches = int(smoke_max_eval_batches or 2)
        else:
            epochs = None
            max_train_updates = None
            max_eval_batches = None
        profile = SMOKE if smoke else FULL
        run_phases["training"] = running_phase("training")
        write_progress("training")
        train_result = trainer.fit(loaders.train, loaders.validation, horizons=tuple(int(value) for value in config.data["eval_horizons"]), total_nodes=int(data_info.num_nodes), epochs=epochs, max_train_updates=max_train_updates, max_validation_batches=max_eval_batches, start_epoch=start_epoch, resume_state=resume_state, checkpoint_extra=checkpoint_extra)
        run_phases["training"] = finished_phase(
            profile=profile,
            phase="training_complete",
            started_at=run_phases["training"]["started_at"],
            wall_seconds=float(sum(train_result.epoch_seconds)),
        )
        checkpoint_paths = (output_dir / "best.pt", output_dir / "last.pt")
        if not all(path.is_file() for path in checkpoint_paths):
            raise RuntimeError("checkpoint write did not produce both best.pt and last.pt")
        run_phases["checkpoint"] = finished_phase(
            profile=profile,
            phase="checkpoint_write",
            artifact=str(output_dir),
        )
        final_eval_limit = max_eval_batches
        run_phases["checkpoint"] = running_phase("checkpoint_reload", artifact=str(output_dir / "best.pt"))
        write_progress("checkpoint_reload")
        reload_started = time.perf_counter()
        checkpoint_manifest = load_checkpoint(output_dir / "best.pt", model, device=selected_device)
        checkpoint_reload_seconds = time.perf_counter() - reload_started
        run_phases["checkpoint"] = finished_phase(
            profile=profile,
            phase="checkpoint_reload",
            artifact=str(output_dir / "best.pt"),
            started_at=run_phases["checkpoint"]["started_at"],
            wall_seconds=checkpoint_reload_seconds,
        )
    run_phases["evaluation"] = running_phase("validation")
    write_progress("validation")
    validation = evaluate(model, loaders.validation, device=selected_device, normalization=normalization, horizons=tuple(int(value) for value in config.data["eval_horizons"]), total_nodes=int(data_info.num_nodes), execution_plan=execution_plan, physical_clip=bool(config.evaluation["physical_clip"]), physical_min_kw=config.evaluation["physical_min_kw"], physical_max_kw=config.evaluation["physical_max_kw"], max_batches=final_eval_limit)
    run_phases["evaluation"] = finished_phase(
        profile=SMOKE if smoke else FULL,
        phase="validation_complete",
        started_at=run_phases["evaluation"]["started_at"],
    )
    run_phases["evaluation"] = running_phase("test")
    write_progress("test")
    test = evaluate(model, loaders.test, device=selected_device, normalization=normalization, horizons=tuple(int(value) for value in config.data["eval_horizons"]), total_nodes=int(data_info.num_nodes), execution_plan=execution_plan, physical_clip=bool(config.evaluation["physical_clip"]), physical_min_kw=config.evaluation["physical_min_kw"], physical_max_kw=config.evaluation["physical_max_kw"], max_batches=final_eval_limit)
    run_phases["evaluation"] = finished_phase(
        profile=SMOKE if smoke else FULL,
        phase="test_complete",
        started_at=run_phases["evaluation"]["started_at"],
    )
    _write_evaluation_outputs(output_dir, validation, test, save_predictions=bool(config.runtime["save_predictions"]), split=evaluation_split)
    _write_history(output_dir / "train_history.csv", train_result.history if train_result else [])
    performance = _performance(output_dir=output_dir, environment=environment, data_prepare_seconds=data_prepare_seconds, model_build_seconds=model_build_seconds, checkpoint_reload_seconds=checkpoint_reload_seconds, started=started, device=selected_device, data_info=data_info, config=config, train_result=train_result, validation=validation, test=test, best_epoch=checkpoint_manifest.get("epoch") if checkpoint_manifest else None, parameter_count=parameter_count, trainable_parameter_count=trainable_parameter_count, execution_plan=execution_plan)
    write_json(output_dir / "performance.json", performance)
    elapsed = time.perf_counter() - started
    peak_allocated = int(torch.cuda.max_memory_allocated(selected_device)) if selected_device.type == "cuda" else None
    public_checkpoint_manifest = {
        key: value for key, value in (checkpoint_manifest or {}).items() if key != "runtime_state"
    }
    completion_profile = SMOKE if smoke else FULL
    run_phases["overall"] = finished_phase(
        profile=completion_profile,
        phase="complete",
        artifact=str(output_dir),
        started_at=start_time,
        wall_seconds=elapsed,
    )
    run_info = {
        "schema_version": 2,
        "model": model_name,
        "run_id": run_name,
        "operation": "evaluate" if evaluate_only else "train",
        "profile": completion_profile,
        "status": PASS,
        "classification": None,
        "phase": "overall",
        "runtime_environment": runtime_metadata.get("runtime_environment"),
        "conda_env": runtime_metadata.get("conda_env"),
        "python_executable": runtime_metadata.get("python_executable"),
        "environment_resolution_source": runtime_metadata.get("environment_resolution_source"),
        "seed": int(config.training["seed"]),
        "git_commit": environment.get("git_commit"),
        "python_version": environment["python"]["version"],
        "pytorch_version": environment["packages"]["torch"],
        "cuda_version": environment["cuda"]["version"],
        "gpu": environment["gpu"]["name"],
        "device": str(selected_device),
        "start_time": start_time,
        "end_time": utc_now(),
        "wall_seconds": elapsed,
        "duration_seconds": elapsed,
        "total_wall_seconds": elapsed,
        "exit_code": 0,
        "exception_type": None,
        "error_message": None,
        "traceback_tail": None,
        "error": None,
        "phases": run_phases,
        "parameter_count": parameter_count,
        "peak_gpu_memory_bytes": peak_allocated,
        "best_epoch": checkpoint_manifest.get("epoch") if checkpoint_manifest else None,
        "best_metric": checkpoint_manifest.get("monitor") if checkpoint_manifest else None,
        "evaluate_only": bool(evaluate_only),
        "evaluation_split": evaluation_split,
        "cli_overrides": cli_overrides_as_nested(effective_overrides),
        "reproducibility_mode": str(config.runtime["reproducibility_mode"]),
        "seed_details": seed_details,
        "details": {
            "legacy_reproducibility_mode_unknown": bool(
                legacy_reproducibility_mode_unknown
            ),
            "reproducibility": seed_details,
            "execution": execution_plan.as_dict(),
        },
        "runtime_overrides": {"smoke": bool(smoke), "smoke_epochs": int(smoke_epochs or 1) if smoke else None, "smoke_max_train_updates": int(smoke_max_train_updates or 2) if smoke else None, "smoke_max_eval_batches": int(smoke_max_eval_batches or 2) if smoke else None},
        "initial_weight_hash": train_result.initial_state_hash if train_result else None,
        "train_batch_order": train_result.train_batch_order if train_result else None,
        "train_batch_count": (
            len(train_result.train_batch_order) if train_result else 0
        ),
        "train_update_count": (
            sum(int(row["train_updates"]) for row in train_result.history)
            if train_result
            else 0
        ),
        "data_split": {
            "split_boundaries": splits.as_dict(arrays.timestamps),
            "window_starts": {
                "train": [int(value) for value in windows.train],
                "train_valid": [int(value) for value in windows.train_valid],
                "validation": [int(value) for value in windows.validation],
                "test": [int(value) for value in windows.test],
            },
        },
        "first_step_loss": train_result.first_step_loss if train_result else None,
        "checkpoint_manifest": public_checkpoint_manifest,
        "execution": execution_plan.as_dict(),
        "validation_monitor": validation.metrics["monitor"],
        "test_monitor": test.metrics["monitor"],
        "final_evaluation_max_batches": final_eval_limit,
        "metrics_complete": True,
    }
    write_status(output_dir / "run_info.json", run_info)
    return {"output_dir": str(output_dir), "run_info": run_info, "validation": validation.metrics, "test": test.metrics, "selected_split": evaluation_split, "window_counts": windows.as_dict()["counts"], "performance": performance}


def _write_failed_run_info(*, model_name: str, config_path: str | Path, output_root: str | Path | None, run_id: str, started: float, error: BaseException) -> None:
    """Finalize an already-started model run without replacing earlier artifacts."""

    try:
        root = project_root_from_config(Path(config_path).resolve())
        output_dir = run_directory(root, output_root, model_name, run_id)
        path = output_dir / "run_info.json"
        if not path.is_file():
            return
        current = json.loads(path.read_text(encoding="utf-8"))
        phase = str(current.get("phase") or "unknown")
        phases = current.get("phases")
        if not isinstance(phases, dict):
            phases = {}
        phase_key = stable_phase(phase)
        traceback_tail = "".join(traceback.format_exception(error))[-4000:]
        details = failure_details(
            error,
            phase=phase,
            traceback_tail=traceback_tail,
        )
        phases[phase_key] = finished_phase(
            classification=details["classification"],
            phase=phase,
            error_summary=failure_summary(error),
            error=details["error"],
            started_at=phases.get(phase_key, {}).get("started_at") if isinstance(phases.get(phase_key), Mapping) else current.get("start_time"),
            wall_seconds=time.perf_counter() - started,
        )
        phases["overall"] = finished_phase(
            classification=details["classification"],
            phase=phase,
            artifact=str(output_dir),
            error_summary=failure_summary(error),
            error=details["error"],
            started_at=current.get("start_time"),
            wall_seconds=time.perf_counter() - started,
        )
        current.update(
            {
                "status": FAILED,
                "classification": details["classification"],
                "phase": stable_phase(phase),
                "end_time": utc_now(),
                "wall_seconds": time.perf_counter() - started,
                "duration_seconds": time.perf_counter() - started,
                "exit_code": 1,
                "exception_type": type(error).__name__,
                "error_message": failure_summary(error),
                "traceback_tail": traceback_tail,
                "error": details["error"],
                "phases": phases,
            }
        )
        current["metrics_complete"] = (
            False if phase in {"training", "validation", "test", "evaluation"} else None
        )
        write_status(path, current)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        # The parent scheduler records a worker crash if this emergency
        # finalizer cannot safely persist a worker completion state.
        return


def run_model(*args, **kwargs) -> dict[str, Any]:
    """Run one model and guarantee an existing ``run_info.json`` is terminal."""

    if args:
        raise TypeError("run_model accepts keyword arguments only")
    values = dict(kwargs)
    model_name = str(values.get("model_name"))
    config_path = values.get("config_path", "configs/experiment.yaml")
    output_root = values.get("output_root")
    run_id = values.get("run_id")
    if run_id is None:
        run_id = datetime.now().strftime("run-%Y%m%d-%H%M%S")
        values["run_id"] = run_id
    started = time.perf_counter()
    try:
        return _run_model_impl(**values)
    except BaseException as exc:
        _write_failed_run_info(
            model_name=model_name,
            config_path=config_path,
            output_root=output_root,
            run_id=str(run_id),
            started=started,
            error=exc,
        )
        raise

"""Train/evaluate one model using the shared formal pipeline."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time
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
from engine.reproducibility import set_seed
from engine.trainer import Trainer, TrainResult
from models.loader import build_model
from runtime.config import (
    ExperimentConfig,
    apply_cli_overrides,
    cli_overrides_as_nested,
    cli_overrides_from_namespace,
    load_experiment_config,
    load_model_config,
    resolved_config_values,
)
from runtime.environment import collect_environment
from runtime.paths import project_root_from_config, resolve_output_root, run_directory
from runtime.run_info import utc_now, write_json, write_yaml


def _choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(value)


def _default_model_config_path(config_path: Path, model_name: str) -> Path:
    return config_path.resolve().parent / "models" / f"{model_name}.yaml"


def _write_history(path: Path, history: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["epoch", "train_loss", "monitor", "learning_rate", "checkpoint_selected", "train_updates"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    "epoch": row["epoch"],
                    "train_loss": row["train_loss"],
                    "monitor": row["validation"]["monitor"],
                    "learning_rate": row["learning_rate"],
                    "checkpoint_selected": row["checkpoint_selected"],
                    "train_updates": row["train_updates"],
                }
            )


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
    ("split", "method"),
    ("split", "train_ratio"),
    ("split", "val_ratio"),
    ("split", "test_ratio"),
    ("training", "loss"),
)


def _at_path(value: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = value
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _check_checkpoint_compatibility(
    manifest: Mapping[str, Any],
    config: ExperimentConfig,
    model_config: Mapping[str, Any],
    checkpoint_path: Path,
) -> None:
    """Reject semantic config changes before evaluating a checkpoint."""

    saved_config = manifest.get("resolved_config") or manifest.get("experiment_config")
    if not isinstance(saved_config, Mapping):
        raise ValueError(
            f"checkpoint {checkpoint_path} has no saved resolved experiment config; "
            "refusing compatibility-unsafe evaluation"
        )
    for path in _CHECKPOINT_CONFIG_PATHS:
        saved = _at_path(saved_config, path)
        current = _at_path(config.values, path)
        if saved != current:
            dotted = ".".join(path)
            raise ValueError(
                f"checkpoint {checkpoint_path} is incompatible at {dotted}: "
                f"saved={saved!r}, current={current!r}"
            )
    saved_model_config = manifest.get("model_config")
    if not isinstance(saved_model_config, Mapping):
        raise ValueError(
            f"checkpoint {checkpoint_path} has no saved model config; "
            "refusing compatibility-unsafe evaluation"
        )
    if dict(saved_model_config) != dict(model_config):
        raise ValueError(
            f"checkpoint {checkpoint_path} is incompatible with the current model config"
        )


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
            path = tuple(dotted_path.split("."))
            old_value = _at_path(base_config.values, path)
            print(f"  {dotted_path}: {old_value!r} -> {new_value!r}")
    print(f"Resolved config saved to: {output_dir / 'resolved_config.yaml'}")


def run_model(
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
            config = apply_cli_overrides(
                config,
                {"training.seed": int(seed_override)},
                project_root=project_root,
            )
            effective_overrides["training.seed"] = int(seed_override)
    model_file = Path(model_config_path).resolve() if model_config_path else _default_model_config_path(config_file, model_name)
    model_config = load_model_config(model_file)
    if evaluate_only and resume is None:
        raise ValueError("--evaluate-only requires --resume")
    checkpoint_file = _checkpoint_path(resume) if resume is not None else None
    if checkpoint_file is not None:
        _check_checkpoint_compatibility(
            read_checkpoint_manifest(checkpoint_file),
            config,
            model_config,
            checkpoint_file,
        )
    run_name = run_id or datetime.now().strftime("run-%Y%m%d-%H%M%S")
    output_dir = run_directory(project_root, output_root, model_name, run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = utc_now()
    started = time.perf_counter()
    selected_device = _choose_device(device)
    seed_details = set_seed(int(config.training["seed"]), deterministic=bool(config.runtime["deterministic"]))
    environment = collect_environment(project_root)

    resolved = resolved_config_values(config, project_root=project_root)
    resolved["resolved"]["model_name"] = model_name
    resolved["resolved"]["run_id"] = run_name
    resolved["resolved"]["device"] = str(selected_device)
    resolved["resolved"]["command"] = command_name
    if smoke:
        resolved["resolved"]["smoke"] = {
            "epochs": int(smoke_epochs or 1),
            "max_train_updates": int(smoke_max_train_updates or 2),
            "max_eval_batches": int(smoke_max_eval_batches or 2),
        }
    write_yaml(output_dir / "resolved_config.yaml", resolved)
    write_yaml(output_dir / "cli_overrides.yaml", cli_overrides_as_nested(effective_overrides))
    write_json(
        output_dir / "command.json",
        {
            "argv": list(command_argv or []),
            "command": command_name,
            "model": model_name,
            "run_id": run_name,
            "config_path": str(config_file),
            "model_config_path": str(model_file),
            "cli_overrides": cli_overrides_as_nested(effective_overrides),
        },
    )
    _print_config_summary(
        config_file=config_file,
        model_file=model_file,
        base_config=base_config,
        cli_overrides=effective_overrides,
        output_dir=output_dir,
    )
    write_yaml(output_dir / "model_config.yaml", model_config)
    write_json(output_dir / "environment.json", environment)

    arrays, data_info, splits, windows, normalization, loaders = _prepare(config, project_root)
    normalization.save(output_dir / "normalization.npz")
    resolved["resolved"].update(
        {
            "data_info": data_info.as_dict(),
            "split_boundaries": splits.as_dict(arrays.timestamps),
            "window_index": windows.as_dict(),
            "normalization": normalization.as_dict(),
        }
    )
    write_yaml(output_dir / "resolved_config.yaml", resolved)

    model = build_model(model_name, model_config, data_info)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    checkpoint_extra = {
        "config_file": str(config_file),
        "model_config_file": str(model_file),
        "experiment_config": config.copy_values(),
        "resolved_config": resolved,
        "model_config": model_config,
        "parameter_count": int(parameter_count),
        "seed": int(config.training["seed"]),
    }
    train_result: TrainResult | None = None
    checkpoint_manifest: dict[str, Any] | None = None
    resume_manifest: dict[str, Any] | None = None
    final_eval_limit: int | None = None
    if evaluate_only:
        assert checkpoint_file is not None
        checkpoint_manifest = load_checkpoint(
            checkpoint_file,
            model,
            device=selected_device,
        )
    else:
        trainer = Trainer(
            model,
            config,
            device=selected_device,
            model_name=model_name,
            normalization=normalization,
            output_dir=output_dir,
        )
        start_epoch = 1
        if resume is not None:
            assert checkpoint_file is not None
            resume_manifest = load_checkpoint(
                checkpoint_file,
                trainer.model,
                device=selected_device,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
            )
            start_epoch = int(resume_manifest.get("epoch", 0)) + 1
        if smoke:
            epochs = int(smoke_epochs or 1)
            max_train_updates = int(smoke_max_train_updates or 2)
            max_eval_batches = int(smoke_max_eval_batches or 2)
        else:
            epochs = None
            max_train_updates = None
            max_eval_batches = None
        train_result = trainer.fit(
            loaders.train,
            loaders.validation,
            horizons=tuple(int(value) for value in config.data["eval_horizons"]),
            total_nodes=int(data_info.num_nodes),
            epochs=epochs,
            max_train_updates=max_train_updates,
            max_validation_batches=max_eval_batches,
            start_epoch=start_epoch,
            checkpoint_extra=checkpoint_extra,
        )
        final_eval_limit = max_eval_batches
        checkpoint_manifest = load_checkpoint(
            output_dir / "best.pt",
            model,
            device=selected_device,
        )

    validation = evaluate(
        model,
        loaders.validation,
        device=selected_device,
        normalization=normalization,
        horizons=tuple(int(value) for value in config.data["eval_horizons"]),
        total_nodes=int(data_info.num_nodes),
        physical_clip=bool(config.evaluation["physical_clip"]),
        physical_min_kw=config.evaluation["physical_min_kw"],
        physical_max_kw=config.evaluation["physical_max_kw"],
        max_batches=final_eval_limit,
    )
    test = evaluate(
        model,
        loaders.test,
        device=selected_device,
        normalization=normalization,
        horizons=tuple(int(value) for value in config.data["eval_horizons"]),
        total_nodes=int(data_info.num_nodes),
        physical_clip=bool(config.evaluation["physical_clip"]),
        physical_min_kw=config.evaluation["physical_min_kw"],
        physical_max_kw=config.evaluation["physical_max_kw"],
        max_batches=final_eval_limit,
    )
    _write_evaluation_outputs(
        output_dir,
        validation,
        test,
        save_predictions=bool(config.runtime["save_predictions"]),
        split=evaluation_split,
    )
    _write_history(output_dir / "train_history.csv", train_result.history if train_result else [])
    elapsed = time.perf_counter() - started
    peak_memory = int(torch.cuda.max_memory_allocated(selected_device)) if selected_device.type == "cuda" else 0
    run_info = {
        "model": model_name,
        "run_id": run_name,
        "seed": int(config.training["seed"]),
        "git_commit": environment.get("git_commit"),
        "python_version": environment["python"]["version"],
        "pytorch_version": environment["packages"]["torch"],
        "cuda_version": environment["cuda"]["version"],
        "gpu": environment["gpu"]["name"],
        "device": str(selected_device),
        "start_time": start_time,
        "end_time": utc_now(),
        "duration_seconds": elapsed,
        "parameter_count": int(parameter_count),
        "peak_gpu_memory_bytes": peak_memory,
        "best_epoch": checkpoint_manifest.get("epoch") if checkpoint_manifest else None,
        "best_metric": checkpoint_manifest.get("monitor") if checkpoint_manifest else None,
        "evaluate_only": bool(evaluate_only),
        "evaluation_split": evaluation_split,
        "cli_overrides": cli_overrides_as_nested(effective_overrides),
        "runtime_overrides": {
            "smoke": bool(smoke),
            "smoke_epochs": int(smoke_epochs or 1) if smoke else None,
            "smoke_max_train_updates": int(smoke_max_train_updates or 2) if smoke else None,
            "smoke_max_eval_batches": int(smoke_max_eval_batches or 2) if smoke else None,
        },
        "seed_details": seed_details,
        "initial_weight_hash": train_result.initial_state_hash if train_result else None,
        "first_step_loss": train_result.first_step_loss if train_result else None,
        "checkpoint_manifest": checkpoint_manifest,
        "validation_monitor": validation.metrics["monitor"],
        "test_monitor": test.metrics["monitor"],
        "final_evaluation_max_batches": final_eval_limit,
    }
    write_json(output_dir / "run_info.json", run_info)
    return {
        "output_dir": str(output_dir),
        "run_info": run_info,
        "validation": validation.metrics,
        "test": test.metrics,
        "selected_split": evaluation_split,
        "window_counts": windows.as_dict()["counts"],
    }

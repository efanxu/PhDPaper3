"""Train/evaluate one model using the shared formal pipeline."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import yaml

from data.dataloader import build_dataloaders
from data.loader import load_data
from data.normalization import fit_normalization
from data.split import chronological_split
from data.window import build_window_index
from engine.checkpoint import load_checkpoint
from engine.evaluator import EvaluationResult, evaluate
from engine.reproducibility import set_seed
from engine.trainer import Trainer, TrainResult
from models.loader import build_model
from runtime.config import ExperimentConfig, load_experiment_config, load_model_config, resolved_config_values
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
) -> None:
    write_json(output_dir / "metrics_validation.json", validation.metrics)
    for horizon, metrics in test.metrics["by_horizon"].items():
        write_json(output_dir / f"metrics_test_h{horizon}.json", metrics)
    if save_predictions:
        np.savez(
            output_dir / "predictions.npz",
            validation_prediction_kw=validation.prediction_kw,
            validation_target_kw=validation.target_kw,
            validation_target_mask=validation.target_mask,
            validation_starts=validation.starts,
            test_prediction_kw=test.prediction_kw,
            test_target_kw=test.target_kw,
            test_target_mask=test.target_mask,
            test_starts=test.starts,
        )


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
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    project_root = project_root_from_config(config_file)
    config = load_experiment_config(config_file)
    if seed_override is not None and int(seed_override) != int(config.training["seed"]):
        raise ValueError(
            "public training seed is frozen in configs/experiment.yaml; "
            "the requested seed does not match it"
        )
    model_file = Path(model_config_path).resolve() if model_config_path else _default_model_config_path(config_file, model_name)
    model_config = load_model_config(model_file)
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
    if smoke:
        resolved["resolved"]["smoke"] = {
            "epochs": int(smoke_epochs or 1),
            "max_train_updates": int(smoke_max_train_updates or 2),
            "max_eval_batches": int(smoke_max_eval_batches or 2),
        }
    write_yaml(output_dir / "resolved_config.yaml", resolved)
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
        "parameter_count": int(parameter_count),
        "seed": int(config.training["seed"]),
    }
    train_result: TrainResult | None = None
    checkpoint_manifest: dict[str, Any] | None = None
    resume_manifest: dict[str, Any] | None = None
    final_eval_limit: int | None = None
    if evaluate_only:
        if resume is None:
            raise ValueError("--evaluate-only requires --resume")
        checkpoint_manifest = load_checkpoint(
            _checkpoint_path(resume),
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
            resume_manifest = load_checkpoint(
                _checkpoint_path(resume),
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
        "window_counts": windows.as_dict()["counts"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one PhDPaper3 forecasting model")
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--model-config")
    parser.add_argument("--run-id")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-root")
    parser.add_argument("--resume")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-epochs", type=int)
    parser.add_argument("--smoke-max-train-updates", type=int)
    parser.add_argument("--smoke-max-eval-batches", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.smoke and any(
        value is not None
        for value in (args.smoke_epochs, args.smoke_max_train_updates, args.smoke_max_eval_batches)
    ):
        raise SystemExit("smoke-specific limits require --smoke")
    result = run_model(
        model_name=args.model,
        config_path=args.config,
        model_config_path=args.model_config,
        run_id=args.run_id,
        device=args.device,
        output_root=args.output_root,
        resume=args.resume,
        evaluate_only=args.evaluate_only,
        smoke=args.smoke,
        smoke_epochs=args.smoke_epochs,
        smoke_max_train_updates=args.smoke_max_train_updates,
        smoke_max_eval_batches=args.smoke_max_eval_batches,
    )
    print(json.dumps({
        "output_dir": result["output_dir"],
        "validation_monitor": result["validation"]["monitor"],
        "test_monitor": result["test"]["monitor"],
        "window_counts": result["window_counts"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

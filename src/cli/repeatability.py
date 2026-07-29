"""Repeatability orchestration kept separate from the ordinary training CLI."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .train import run_model
from runtime.config import (
    apply_cli_overrides,
    cli_overrides_as_nested,
    cli_overrides_from_namespace,
    load_experiment_config,
)
from runtime.paths import project_root_from_config


def _remove_run_id(value: dict[str, Any]) -> dict[str, Any]:
    copied = json.loads(json.dumps(value))
    copied.get("resolved", {}).pop("run_id", None)
    return copied


def _history(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _values_close(left: Any, right: Any, *, atol: float) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _values_close(left[key], right[key], atol=atol) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _values_close(item_left, item_right, atol=atol)
            for item_left, item_right in zip(left, right)
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=atol)
    return left == right


def compare_repeated_runs(
    *,
    model_name: str,
    config_path: str,
    model_config_path: str | None,
    seed: int | None = None,
    device: str = "auto",
    output_root: str | Path | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
    prediction_atol: float = 1e-6,
    metric_atol: float = 0.0,
    command_argv: list[str] | None = None,
) -> dict[str, Any]:
    if prediction_atol < 0 or metric_atol < 0:
        raise ValueError("repeatability tolerances must be non-negative")
    config_file = Path(config_path).resolve()
    root = project_root_from_config(config_file)
    base_config = load_experiment_config(config_file)
    effective_overrides = cli_overrides_from_namespace(cli_overrides or {})
    resolved_config = apply_cli_overrides(base_config, effective_overrides, project_root=root)
    resolved_seed = int(resolved_config.training["seed"])
    if seed is not None:
        requested_seed = effective_overrides.get("training.seed")
        if requested_seed is not None and int(requested_seed) != int(seed):
            raise ValueError("--seed conflicts with an explicit training.seed override")
        if requested_seed is None:
            effective_overrides["training.seed"] = int(seed)
            resolved_config = apply_cli_overrides(
                resolved_config,
                {"training.seed": int(seed)},
                project_root=root,
            )
            resolved_seed = int(seed)
    repeat_root = Path(output_root) if output_root else root / "results" / "repeatability"
    first = run_model(
        model_name=model_name,
        config_path=config_file,
        model_config_path=model_config_path,
        run_id=f"repeat_{resolved_seed}_a",
        device=device,
        output_root=repeat_root,
        smoke=True,
        smoke_epochs=1,
        smoke_max_train_updates=2,
        smoke_max_eval_batches=2,
        cli_overrides=effective_overrides,
        command_argv=command_argv,
        command_name="repeatability",
    )
    second = run_model(
        model_name=model_name,
        config_path=config_file,
        model_config_path=model_config_path,
        run_id=f"repeat_{resolved_seed}_b",
        device=device,
        output_root=repeat_root,
        smoke=True,
        smoke_epochs=1,
        smoke_max_train_updates=2,
        smoke_max_eval_batches=2,
        cli_overrides=effective_overrides,
        command_argv=command_argv,
        command_name="repeatability",
    )
    first_dir = Path(first["output_dir"])
    second_dir = Path(second["output_dir"])
    first_config = yaml.safe_load((first_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    second_config = yaml.safe_load((second_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    first_info = json.loads((first_dir / "run_info.json").read_text(encoding="utf-8"))
    second_info = json.loads((second_dir / "run_info.json").read_text(encoding="utf-8"))
    first_npz = np.load(first_dir / "predictions.npz", allow_pickle=False)
    second_npz = np.load(second_dir / "predictions.npz", allow_pickle=False)
    checks: list[tuple[str, bool, str]] = []
    checks.append(("resolved_config", _remove_run_id(first_config) == _remove_run_id(second_config), "resolved config"))
    checks.append(("initial_weights", first_info["initial_weight_hash"] == second_info["initial_weight_hash"], "initial weights"))
    checks.append(("first_step_loss", first_info["first_step_loss"] == second_info["first_step_loss"], "first step loss"))
    checks.append(("short_training_loss_curve", _history(first_dir / "train_history.csv") == _history(second_dir / "train_history.csv"), "loss curve"))
    checks.append(("best_epoch", first_info["best_epoch"] == second_info["best_epoch"], "best epoch"))
    checks.append(
        (
            "validation_metrics",
            _values_close(first["validation"], second["validation"], atol=metric_atol),
            "validation metrics",
        )
    )
    max_prediction_diff = 0.0
    for key in first_npz.files:
        if key not in second_npz.files:
            checks.append((f"predictions:{key}", False, f"missing prediction array {key}"))
            continue
        left = first_npz[key]
        right = second_npz[key]
        diff = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))) if left.size else 0.0
        max_prediction_diff = max(max_prediction_diff, diff)
    checks.append(("predictions", max_prediction_diff <= prediction_atol, "predictions"))
    checks.append(
        (
            "final_metrics",
            _values_close(first["test"], second["test"], atol=metric_atol),
            "final metrics",
        )
    )
    first_failure = next((name for name, passed, _ in checks if not passed), None)
    result = {
        "passed": first_failure is None,
        "seed": resolved_seed,
        "model": model_name,
        "runs": [str(first_dir), str(second_dir)],
        "checks": {name: passed for name, passed, _ in checks},
        "first_failure": first_failure,
        "max_prediction_difference": max_prediction_diff,
        "prediction_atol": prediction_atol,
        "metric_atol": metric_atol,
        "cli_overrides": cli_overrides_as_nested(effective_overrides),
        "max_metric_difference": 0.0
        if _values_close(first["test"], second["test"], atol=metric_atol)
        else float("inf"),
    }
    report_path = repeat_root / "repeatability" / model_name / f"seed_{resolved_seed}" / "repeatability_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result

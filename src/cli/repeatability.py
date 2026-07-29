"""Repeatability orchestration kept separate from the ordinary training CLI."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .train import run_model


def _remove_run_id(value: dict[str, Any]) -> dict[str, Any]:
    copied = json.loads(json.dumps(value))
    copied.get("resolved", {}).pop("run_id", None)
    return copied


def _history(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def compare_repeated_runs(
    *,
    model_name: str,
    config_path: str,
    model_config_path: str | None,
    seed: int,
    device: str = "auto",
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(config_path).resolve().parent.parent
    repeat_root = Path(output_root) if output_root else root / "results" / "repeatability"
    first = run_model(
        model_name=model_name,
        config_path=config_path,
        model_config_path=model_config_path,
        run_id=f"repeat_{seed}_a",
        device=device,
        output_root=repeat_root,
        smoke=True,
        smoke_epochs=1,
        smoke_max_train_updates=2,
        smoke_max_eval_batches=2,
        seed_override=seed,
    )
    second = run_model(
        model_name=model_name,
        config_path=config_path,
        model_config_path=model_config_path,
        run_id=f"repeat_{seed}_b",
        device=device,
        output_root=repeat_root,
        smoke=True,
        smoke_epochs=1,
        smoke_max_train_updates=2,
        smoke_max_eval_batches=2,
        seed_override=seed,
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
    checks.append(("validation_metrics", first["validation"] == second["validation"], "validation metrics"))
    max_prediction_diff = 0.0
    for key in first_npz.files:
        if key not in second_npz.files:
            checks.append((f"predictions:{key}", False, f"missing prediction array {key}"))
            continue
        left = first_npz[key]
        right = second_npz[key]
        diff = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))) if left.size else 0.0
        max_prediction_diff = max(max_prediction_diff, diff)
    checks.append(("predictions", max_prediction_diff <= 1e-6, "predictions"))
    checks.append(("final_metrics", first["test"] == second["test"], "final metrics"))
    first_failure = next((name for name, passed, _ in checks if not passed), None)
    result = {
        "passed": first_failure is None,
        "seed": seed,
        "model": model_name,
        "runs": [str(first_dir), str(second_dir)],
        "checks": {name: passed for name, passed, _ in checks},
        "first_failure": first_failure,
        "max_prediction_difference": max_prediction_diff,
        "max_metric_difference": 0.0 if first["test"] == second["test"] else float("inf"),
    }
    report_path = repeat_root / "repeatability" / model_name / f"seed_{seed}" / "repeatability_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result

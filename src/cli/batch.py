"""Business implementation for the unified ``batch`` command."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .train import run_model
from runtime.config import ConfigError
from runtime.paths import project_root_from_config, resolve_output_root


def run_batch(
    *,
    models: list[str],
    config_path: str | Path,
    model_config_path: str | Path | None,
    device: str,
    output_root: str | Path | None,
    smoke: bool,
    continue_on_error: bool,
    skip_completed: bool,
    smoke_epochs: int | None,
    smoke_max_train_updates: int | None,
    smoke_max_eval_batches: int | None,
    cli_overrides: Mapping[str, Any] | None = None,
    command_argv: list[str] | None = None,
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    project_root = project_root_from_config(config_file)
    resolved_root = resolve_output_root(project_root, output_root)
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for model_name in models:
        run_id = f"batch_{model_name}"
        expected = resolved_root / model_name / run_id / "best.pt"
        if skip_completed and expected.is_file():
            results.append({"model": model_name, "status": "skipped", "output_dir": str(expected.parent)})
            continue
        try:
            result = run_model(
                model_name=model_name,
                config_path=config_file,
                model_config_path=model_config_path,
                run_id=run_id,
                device=device,
                output_root=output_root,
                smoke=smoke,
                smoke_epochs=smoke_epochs,
                smoke_max_train_updates=smoke_max_train_updates,
                smoke_max_eval_batches=smoke_max_eval_batches,
                cli_overrides=cli_overrides,
                command_argv=command_argv,
                command_name="batch",
            )
        except (ConfigError, FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
            error = {"model": model_name, "error": str(exc)}
            errors.append(error)
            if not continue_on_error:
                raise
            results.append({"model": model_name, "status": "failed", "error": str(exc)})
            continue
        results.append(
            {
                "model": model_name,
                "status": "passed",
                "output_dir": result["output_dir"],
                "validation_monitor": result["validation"]["monitor"],
                "test_monitor": result["test"]["monitor"],
            }
        )
    return {
        "passed": not errors,
        "models": models,
        "results": results,
        "errors": errors,
    }

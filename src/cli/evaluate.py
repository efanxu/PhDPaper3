"""Evaluate-only command wrapper."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .orchestrator import run_evaluate_model
from runtime.paths import project_root_from_config, resolve_output_root


def evaluate_checkpoint(
    *,
    model_name: str,
    config_path: str | Path,
    model_config_path: str | Path | None,
    checkpoint: str | Path | None,
    run_id: str | None,
    device: str,
    output_root: str | Path | None,
    cli_overrides: Mapping[str, Any] | None = None,
    command_argv: list[str] | None = None,
    split: str = "both",
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    root = project_root_from_config(config_file)
    evaluation_run_id = run_id
    if checkpoint is None:
        if run_id is None:
            raise ValueError("evaluate requires --checkpoint or --run-id")
        source = resolve_output_root(root, output_root) / model_name / run_id / "best.pt"
        if not source.is_file():
            raise FileNotFoundError(f"no checkpoint found for --run-id {run_id}: {source}")
        checkpoint = source
        evaluation_run_id = f"{run_id}__evaluate"
    if evaluation_run_id is None:
        evaluation_run_id = "evaluate"
    return run_evaluate_model(
        model_name=model_name,
        config_path=config_file,
        model_config_path=model_config_path,
        resume=checkpoint,
        evaluate_only=True,
        run_id=evaluation_run_id,
        device=device,
        output_root=output_root,
        cli_overrides=cli_overrides,
        command_argv=command_argv,
        command_name="evaluate",
        evaluation_split=split,
    )

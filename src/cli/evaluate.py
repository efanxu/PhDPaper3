"""Evaluate-only command wrapper."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .train import run_model


def evaluate_checkpoint(
    *,
    model_name: str,
    config_path: str | Path,
    model_config_path: str | Path | None,
    checkpoint: str | Path,
    run_id: str | None,
    device: str,
    output_root: str | Path | None,
    cli_overrides: Mapping[str, Any] | None = None,
    command_argv: list[str] | None = None,
    split: str = "both",
) -> dict[str, Any]:
    return run_model(
        model_name=model_name,
        config_path=config_path,
        model_config_path=model_config_path,
        resume=checkpoint,
        evaluate_only=True,
        run_id=run_id,
        device=device,
        output_root=output_root,
        cli_overrides=cli_overrides,
        command_argv=command_argv,
        command_name="evaluate",
        evaluation_split=split,
    )

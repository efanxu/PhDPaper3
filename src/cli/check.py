"""Business implementation for the public ``check`` command."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from runtime.status import FORMAL_DEFAULT_SHAPE, INTERFACE_SMALL, RESOLVED_SHAPE
from runtime.validation import run_shape_validation


def run_check(
    *,
    model_name: str,
    config_path: str | Path,
    model_config_path: str | Path | None,
    device: str,
    full_shape: bool,
    cli_overrides: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    operation: str = "check",
    runtime_environment: str | None = None,
    status_path: str | Path | None = None,
) -> dict[str, Any]:
    overrides = dict(cli_overrides or {})
    profile = (
        FORMAL_DEFAULT_SHAPE
        if full_shape and not overrides
        else RESOLVED_SHAPE
        if full_shape
        else INTERFACE_SMALL
    )
    return run_shape_validation(
        model_name=model_name,
        config_path=config_path,
        model_config_path=model_config_path,
        device=device,
        profile=profile,
        cli_overrides=overrides,
        run_id=run_id,
        operation=operation,
        runtime_environment=runtime_environment,
        status_path=status_path,
    )

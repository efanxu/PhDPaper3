"""Hidden worker entry point used only by the parent scheduler."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if len(values) != 2:
        print("internal worker expects <request.json> <model>", file=sys.stderr)
        return 2
    request_path = Path(values[0]).resolve()
    model_name = values[1]
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        if not isinstance(request, dict):
            raise ValueError("worker request must be a JSON object")
        operation = request.get("operation")
        model_request = request.get("models", {}).get(model_name)
        if not isinstance(model_request, dict):
            raise ValueError(f"worker request has no model entry for {model_name}")
        if operation == "train":
            from .train import run_model

            result = run_model(
                model_name=model_name,
                config_path=request["config_path"],
                model_config_path=model_request["model_config_path"],
                run_id=request["run_id"],
                device=request["device"],
                output_root=request["output_root"],
                resume=model_request.get("resume_checkpoint"),
                smoke=bool(request.get("smoke", False)),
                smoke_epochs=request.get("smoke_epochs"),
                smoke_max_train_updates=request.get("smoke_max_train_updates"),
                smoke_max_eval_batches=request.get("smoke_max_eval_batches"),
                cli_overrides=request.get("cli_overrides", {}),
                command_argv=request.get("command_argv", []),
                command_name="train",
            )
            _print(
                {
                    "output_dir": result["output_dir"],
                    "validation_monitor": result["validation"]["monitor"],
                    "test_monitor": result["test"]["monitor"],
                    "performance": result.get("performance"),
                }
            )
            return 0
        if operation == "check":
            from .check import run_check

            result = run_check(
                model_name=model_name,
                config_path=request["config_path"],
                model_config_path=model_request["model_config_path"],
                device=request["device"],
                full_shape=bool(request.get("full_shape", False)),
                cli_overrides=request.get("cli_overrides", {}),
            )
            _print(result)
            return 0
        if operation == "preflight":
            from .preflight import run_preflight

            result = run_preflight(
                model_name=model_name,
                config_path=request["config_path"],
                model_config_path=model_request["model_config_path"],
                check_data=not bool(request.get("no_data", False)),
                device=request["device"],
                cli_overrides=request.get("cli_overrides", {}),
            )
            _print(result)
            return 0
        raise ValueError(f"unsupported worker operation: {operation!r}")
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        print(f"worker error [{model_name}]: {exc}", file=sys.stderr)
        return 2

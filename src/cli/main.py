"""Dispatch the one public parser to parent-process orchestration."""

from __future__ import annotations

import json
import sys
from typing import Sequence

from runtime.config import cli_overrides_from_namespace

from .command_schema import build_parser


def _validate_smoke_limits(args) -> None:
    limits = (args.smoke_epochs, args.smoke_max_train_updates, args.smoke_max_eval_batches)
    if not args.smoke and any(value is not None for value in limits):
        raise ValueError("smoke-specific limits require --smoke")
    if any(value is not None and value < 1 for value in limits):
        raise ValueError("smoke-specific limits must be positive")


def _print_json(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _dispatch(args, raw_argv: list[str]) -> int:
    cli_overrides = cli_overrides_from_namespace(args)
    if args.command == "summarize":
        from .orchestrator import summarize_existing_run

        result = summarize_existing_run(run_id=args.run_id, output_root=args.output_root)
        _print_json(result)
        return 0

    if args.command == "train":
        from .orchestrator import run_training_models

        _validate_smoke_limits(args)
        result = run_training_models(
            models=list(args.model),
            config_path=args.config,
            model_config_path=args.model_config,
            run_id=args.run_id,
            device=args.device,
            output_root=args.output_root,
            resume=args.resume,
            overwrite=args.overwrite,
            id_suffix=args.id_suffix,
            fail_fast=args.fail_fast,
            smoke=args.smoke,
            smoke_epochs=args.smoke_epochs,
            smoke_max_train_updates=args.smoke_max_train_updates,
            smoke_max_eval_batches=args.smoke_max_eval_batches,
            environment_preflight_only=args.environment_preflight_only,
            cli_overrides=cli_overrides,
            command_argv=raw_argv,
        )
        _print_json(result)
        return 0 if result["passed"] else 1

    if args.command == "evaluate":
        from .evaluate import evaluate_checkpoint

        result = evaluate_checkpoint(
            model_name=args.model,
            config_path=args.config,
            model_config_path=args.model_config,
            checkpoint=args.checkpoint,
            run_id=args.run_id,
            device=args.device,
            output_root=args.output_root,
            cli_overrides=cli_overrides,
            command_argv=raw_argv,
            split=args.split,
        )
        selected = result["validation"] if args.split == "validation" else result["test"]
        _print_json({"output_dir": result["output_dir"], "split": args.split, "metrics": selected if args.split != "both" else {"validation": result["validation"], "test": result["test"]}})
        return 0 if result.get("passed", True) else 1

    if args.command == "check":
        from .orchestrator import run_isolated_checks

        result = run_isolated_checks(
            operation="check",
            models=list(args.model),
            config_path=args.config,
            model_config_path=args.model_config,
            device=args.device,
            output_root=args.output_root,
            run_id=args.run_id,
            cli_overrides=cli_overrides,
            full_shape=args.full_shape,
            environment_preflight_only=args.environment_preflight_only,
        )
        _print_json(result)
        return 0 if result["passed"] else 1

    if args.command == "preflight":
        from .orchestrator import run_isolated_checks

        result = run_isolated_checks(operation="preflight", models=list(args.model), config_path=args.config, model_config_path=args.model_config, device=args.device, cli_overrides=cli_overrides, no_data=args.no_data, environment_preflight_only=args.environment_preflight_only)
        _print_json(result)
        return 0 if result["passed"] else 1

    if args.command == "repeatability":
        from .repeatability import compare_repeated_runs

        result = compare_repeated_runs(
            models=list(args.model),
            config_path=args.config,
            model_config_path=args.model_config,
            run_id=args.run_id,
            device=args.device,
            output_root=args.output_root,
            cli_overrides=cli_overrides,
            prediction_atol=args.prediction_atol,
            prediction_rtol=args.prediction_rtol,
            metric_atol=args.metric_atol,
            metric_rtol=args.metric_rtol,
            overwrite=args.overwrite,
            id_suffix=args.id_suffix,
            command_argv=raw_argv,
            environment_preflight_only=args.environment_preflight_only,
        )
        _print_json(result)
        return 0 if result["passed"] else 1

    raise ValueError(f"unsupported command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    """Parse and dispatch, returning a process exit code."""

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    try:
        return _dispatch(args, raw_argv)
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

"""Dispatch the single project parser to business implementations."""

from __future__ import annotations

import json
import sys
from typing import Sequence

from runtime.config import cli_overrides_from_namespace

from .command_schema import build_parser


def _validate_smoke_limits(args) -> None:
    limits = (
        args.smoke_epochs,
        args.smoke_max_train_updates,
        args.smoke_max_eval_batches,
    )
    if not args.smoke and any(value is not None for value in limits):
        raise ValueError("smoke-specific limits require --smoke")
    if any(value is not None and value < 1 for value in limits):
        raise ValueError("smoke-specific limits must be positive")


def _print_json(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _dispatch(args, raw_argv: list[str]) -> int:
    cli_overrides = cli_overrides_from_namespace(args)
    if args.command == "train":
        from .train import run_model

        _validate_smoke_limits(args)
        result = run_model(
            model_name=args.model,
            config_path=args.config,
            model_config_path=args.model_config,
            run_id=args.run_id,
            device=args.device,
            output_root=args.output_root,
            resume=args.resume,
            smoke=args.smoke,
            smoke_epochs=args.smoke_epochs,
            smoke_max_train_updates=args.smoke_max_train_updates,
            smoke_max_eval_batches=args.smoke_max_eval_batches,
            cli_overrides=cli_overrides,
            command_argv=raw_argv,
            command_name="train",
        )
        _print_json(
            {
                "output_dir": result["output_dir"],
                "validation_monitor": result["validation"]["monitor"],
                "test_monitor": result["test"]["monitor"],
                "window_counts": result["window_counts"],
            }
        )
        return 0

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
        _print_json(
            {
                "output_dir": result["output_dir"],
                "split": args.split,
                "metrics": selected if args.split != "both" else {
                    "validation": result["validation"],
                    "test": result["test"],
                },
            }
        )
        return 0

    if args.command == "check":
        from .check import run_check

        result = run_check(
            model_name=args.model,
            config_path=args.config,
            model_config_path=args.model_config,
            device=args.device,
            full_shape=args.full_shape,
            cli_overrides=cli_overrides,
        )
        _print_json(result)
        return 0

    if args.command == "preflight":
        from .preflight import run_preflight

        result = run_preflight(
            model_name=args.model,
            config_path=args.config,
            model_config_path=args.model_config,
            check_data=not args.no_data,
            device=args.device,
            cli_overrides=cli_overrides,
        )
        _print_json(result)
        return 0

    if args.command == "repeatability":
        from .repeatability import compare_repeated_runs

        result = compare_repeated_runs(
            model_name=args.model,
            config_path=args.config,
            model_config_path=args.model_config,
            seed=args.seed,
            device=args.device,
            output_root=args.output_root,
            cli_overrides=cli_overrides,
            prediction_atol=args.prediction_atol,
            metric_atol=args.metric_atol,
            command_argv=raw_argv,
        )
        _print_json(result)
        return 0 if result["passed"] else 1

    if args.command == "batch":
        from .batch import run_batch

        _validate_smoke_limits(args)
        result = run_batch(
            models=args.models,
            config_path=args.config,
            model_config_path=args.model_config,
            device=args.device,
            output_root=args.output_root,
            smoke=args.smoke,
            continue_on_error=args.continue_on_error,
            skip_completed=args.skip_completed,
            smoke_epochs=args.smoke_epochs,
            smoke_max_train_updates=args.smoke_max_train_updates,
            smoke_max_eval_batches=args.smoke_max_eval_batches,
            cli_overrides=cli_overrides,
            command_argv=raw_argv,
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

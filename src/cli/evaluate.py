"""Evaluate-only command wrapper."""

from __future__ import annotations

import argparse
import json

from .train import run_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a saved PhDPaper3 checkpoint")
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--model-config")
    parser.add_argument("--resume", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-root")
    args = parser.parse_args(argv)
    result = run_model(
        model_name=args.model,
        config_path=args.config,
        model_config_path=args.model_config,
        resume=args.resume,
        evaluate_only=True,
        run_id=args.run_id,
        device=args.device,
        output_root=args.output_root,
    )
    print(json.dumps({"output_dir": result["output_dir"], "test": result["test"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

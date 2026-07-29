from _bootstrap import ROOT

import argparse
import json

from cli.repeatability import compare_repeated_runs


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare two deterministic short runs")
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--model-config")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-root")
    args = parser.parse_args(argv)
    result = compare_repeated_runs(model_name=args.model, config_path=args.config, model_config_path=args.model_config, seed=args.seed, device=args.device, output_root=args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

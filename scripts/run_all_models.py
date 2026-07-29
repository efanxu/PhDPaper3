from _bootstrap import ROOT

import argparse

from cli.train import run_model


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run each explicitly listed model")
    parser.add_argument("--models", nargs="+", default=["node_shared_lstm"])
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    for model_name in args.models:
        run_model(model_name=model_name, config_path=args.config, device=args.device, smoke=args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

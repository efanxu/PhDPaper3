from _bootstrap import ROOT

import argparse
import json

from cli.preflight import run_preflight


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate config, data and model import")
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--model-config")
    parser.add_argument("--no-data", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(run_preflight(model_name=args.model, config_path=args.config, model_config_path=args.model_config, check_data=not args.no_data), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

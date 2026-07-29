from __future__ import annotations

import argparse
import json

import torch

from _bootstrap import ROOT
from engine.losses import masked_score_aligned_hybrid
from engine.reproducibility import set_seed
from models.base import DataInfoView, ModelInput
from models.loader import build_model
from runtime.config import load_experiment_config, load_model_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check a model import and full tensor shape")
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--model-config")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--full-shape", action="store_true")
    args = parser.parse_args(argv)
    config_path = ROOT / args.config if not __import__("pathlib").Path(args.config).is_absolute() else args.config
    config = load_experiment_config(config_path)
    model_path = args.model_config or str(ROOT / "configs" / "models" / f"{args.model}.yaml")
    model_config = load_model_config(model_path)
    set_seed(int(config.training["seed"]), deterministic=bool(config.runtime["deterministic"]))
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    data_info = DataInfoView(
        num_nodes=int(config.data["num_nodes"]),
        num_features=len(config.data["feature_columns"]),
        lookback=int(config.data["lookback"]),
        max_pred_len=int(config.data["max_pred_len"]),
    )
    model = build_model(args.model, model_config, data_info).to(device)
    batch_size = int(config.training["train_batch_size"] if args.full_shape else 2)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    x = torch.randn(
        batch_size,
        data_info.lookback,
        data_info.num_nodes,
        data_info.num_features,
        device=device,
    )
    target = torch.randn(batch_size, data_info.num_nodes, data_info.max_pred_len, device=device)
    target_mask = torch.ones_like(target, dtype=torch.bool)
    model.train()
    output = model(ModelInput(x=x))
    loss = masked_score_aligned_hybrid(output, target, target_mask)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    result = {
        "model": args.model,
        "full_shape": bool(args.full_shape),
        "device": str(device),
        "input_shape": list(x.shape),
        "output_shape": list(output.shape),
        "loss": float(loss.detach().cpu()),
        "output_finite": bool(torch.isfinite(output).all()),
        "gradients_present": all(gradient is not None for gradient in gradients),
        "gradients_finite": all(gradient is not None and bool(torch.isfinite(gradient).all()) for gradient in gradients),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
    }
    if not result["output_finite"] or not result["gradients_present"] or not result["gradients_finite"]:
        raise RuntimeError(json.dumps(result))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

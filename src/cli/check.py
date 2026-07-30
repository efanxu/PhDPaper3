"""Business implementation for the unified ``check`` command."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import torch

from data.loader import load_data
from engine.losses import masked_score_aligned_hybrid
from engine.reproducibility import set_seed
from models.base import DataInfoView, ModelInput
from models.loader import build_model
from runtime.config import (
    apply_cli_overrides,
    cli_overrides_as_nested,
    load_experiment_config,
    load_model_config,
    resolved_config_values,
)
from runtime.paths import project_root_from_config


def _choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(value)


def run_check(
    *,
    model_name: str,
    config_path: str | Path,
    model_config_path: str | Path | None,
    device: str,
    full_shape: bool,
    cli_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    root = project_root_from_config(config_file)
    base_config = load_experiment_config(config_file)
    config = apply_cli_overrides(base_config, cli_overrides, project_root=root)
    model_file = (
        Path(model_config_path).resolve()
        if model_config_path
        else root / "configs" / "models" / f"{model_name}.yaml"
    )
    model_config = load_model_config(model_file)
    selected_device = _choose_device(device)
    set_seed(int(config.training["seed"]), deterministic=bool(config.runtime["deterministic"]))
    # Shape checks use the real public node order so graph adapters exercise
    # the same resource alignment as training without consuming labels.
    _, loaded_data_info = load_data(config, project_root=root)
    data_info = DataInfoView.from_object(loaded_data_info)
    model = build_model(model_name, model_config, data_info).to(selected_device)
    batch_size = int(config.training["train_batch_size"] if full_shape else 2)
    if selected_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(selected_device)
    x = torch.randn(
        batch_size,
        data_info.lookback,
        data_info.num_nodes,
        data_info.num_features,
        device=selected_device,
    )
    target = torch.randn(batch_size, data_info.num_nodes, data_info.max_pred_len, device=selected_device)
    target_mask = torch.ones_like(target, dtype=torch.bool)
    model.train()
    output = model(ModelInput(x=x))
    loss = masked_score_aligned_hybrid(output, target, target_mask)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    result = {
        "model": model_name,
        "full_shape": bool(full_shape),
        "device": str(selected_device),
        "input_shape": list(x.shape),
        "output_shape": list(output.shape),
        "loss": float(loss.detach().cpu()),
        "output_finite": bool(torch.isfinite(output).all()),
        "gradients_present": all(gradient is not None for gradient in gradients),
        "gradients_finite": all(
            gradient is not None and bool(torch.isfinite(gradient).all()) for gradient in gradients
        ),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(selected_device))
        if selected_device.type == "cuda"
        else 0,
        "config": str(config_file),
        "model_config": str(model_file),
        "cli_overrides": cli_overrides_as_nested(cli_overrides),
        "resolved_config": resolved_config_values(config, project_root=root),
    }
    if not result["output_finite"] or not result["gradients_present"] or not result["gradients_finite"]:
        raise RuntimeError(json.dumps(result, ensure_ascii=False))
    return result

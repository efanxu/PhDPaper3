"""Small preflight: config, data, model import and shape metadata only."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from data.loader import load_data
from data.split import chronological_split
from data.window import build_window_index
from engine.model_execution import build_execution_plan
from models.loader import build_model
from runtime.config import (
    apply_cli_overrides,
    cli_overrides_as_nested,
    load_experiment_config,
    load_model_config,
    resolved_config_values,
)
from engine.reproducibility import set_seed
from runtime.paths import project_root_from_config


def run_preflight(
    *,
    model_name: str,
    config_path: str,
    model_config_path: str | Path | None = None,
    check_data: bool = True,
    device: str = "auto",
    cli_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    root = project_root_from_config(config_file)
    base_config = load_experiment_config(config_file)
    config = apply_cli_overrides(
        base_config,
        cli_overrides,
        project_root=root,
    )
    model_path = Path(model_config_path).resolve() if model_config_path else root / "configs" / "models" / f"{model_name}.yaml"
    model_config = load_model_config(model_path)
    seed_details = set_seed(
        int(config.training["seed"]),
        reproducibility_mode=str(config.runtime["reproducibility_mode"]),
    )
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    selected_device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    result: dict[str, Any] = {
        "passed": True,
        "model": model_name,
        "config": str(config.source),
        "model_config": str(model_path),
        "device": selected_device,
        "cli_overrides": cli_overrides_as_nested(cli_overrides),
        "resolved_config": resolved_config_values(config, project_root=root),
        "reproducibility_mode": seed_details["reproducibility_mode"],
        "seed_details": seed_details,
    }
    if check_data:
        arrays, info = load_data(config, project_root=root)
        splits = chronological_split(len(arrays.timestamps), config)
        windows = build_window_index(
            splits,
            lookback=int(config.data["lookback"]),
            horizon=int(config.data["max_pred_len"]),
            strides={
                "train": int(config.sampling["train_stride"]),
                "validation": int(config.sampling["val_stride"]),
                "test": int(config.sampling["test_stride"]),
            },
            target_mask=arrays.target_mask,
        )
        model = build_model(model_name, model_config, info)
        result["execution"] = build_execution_plan(
            model,
            total_nodes=int(info.num_nodes),
            node_shared_chunk_size=int(config.runtime["node_shared_chunk_size"]),
        ).as_dict()
        result["data_info"] = info.as_dict()
        result["window_counts"] = windows.as_dict()["counts"]
        result["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
    else:
        from models.base import DataInfoView

        info = DataInfoView(
            num_nodes=int(config.data["num_nodes"]),
            num_features=len(config.data["feature_columns"]),
            lookback=int(config.data["lookback"]),
            max_pred_len=int(config.data["max_pred_len"]),
            feature_columns=tuple(config.data["feature_columns"]),
            input_power_column=str(config.data["input_power_column"]),
            input_power_index=list(config.data["feature_columns"]).index(
                config.data["input_power_column"]
            ),
            # ``--no-data`` has no parquet node order to inspect. SDWPF's
            # public location table is keyed by the configured 1..N IDs; full
            # data preflight above remains the strict alignment check.
            node_ids=tuple(range(1, int(config.data["num_nodes"]) + 1)),
            graph_config=dict(config.resources["graph"]),
            project_root=root,
        )
        model = build_model(model_name, model_config, info)
        result["execution"] = build_execution_plan(
            model,
            total_nodes=int(info.num_nodes),
            node_shared_chunk_size=int(config.runtime["node_shared_chunk_size"]),
        ).as_dict()
        result["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
    return result

"""Small preflight: config, data, model import and shape metadata only."""

from __future__ import annotations

from typing import Any

from data.loader import load_data
from data.split import chronological_split
from data.window import build_window_index
from models.loader import build_model
from runtime.config import load_experiment_config, load_model_config
from runtime.paths import project_root_from_config


def run_preflight(
    *,
    model_name: str,
    config_path: str,
    model_config_path: str | None = None,
    check_data: bool = True,
) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    root = project_root_from_config(config_path)
    model_path = model_config_path or str(root / "configs" / "models" / f"{model_name}.yaml")
    model_config = load_model_config(model_path)
    result: dict[str, Any] = {
        "passed": True,
        "model": model_name,
        "config": str(config.source),
        "model_config": str(model_path),
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
        result["data_info"] = info.as_dict()
        result["window_counts"] = windows.as_dict()["counts"]
        result["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
    else:
        from models.base import DataInfoView

        info = DataInfoView(
            int(config.data["num_nodes"]),
            len(config.data["feature_columns"]),
            int(config.data["lookback"]),
            int(config.data["max_pred_len"]),
        )
        model = build_model(model_name, model_config, info)
        result["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
    return result

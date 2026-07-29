"""Convention-based model loading without a registry or capability manifest."""

from __future__ import annotations

import importlib
import re
from typing import Any

from .base import DataInfoView, ForecastModel


_MODEL_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def load_model_module(model_name: str):
    if not _MODEL_NAME.fullmatch(model_name):
        raise ValueError(f"invalid model name: {model_name!r}")
    try:
        return importlib.import_module(f"models.{model_name}.model")
    except ModuleNotFoundError as exc:
        if exc.name == f"models.{model_name}" or exc.name == f"models.{model_name}.model":
            raise ValueError(f"model implementation not found: models/{model_name}/model.py") from exc
        raise


def build_model(model_name: str, model_config: dict[str, Any], data_info: Any) -> ForecastModel:
    module = load_model_module(model_name)
    builder = getattr(module, "build_model", None)
    if builder is None or not callable(builder):
        raise TypeError(f"models/{model_name}/model.py must expose build_model(model_config, data_info)")
    model = builder(model_config, DataInfoView.from_object(data_info))
    if not isinstance(model, ForecastModel):
        raise TypeError(f"model {model_name} must subclass ForecastModel")
    return model

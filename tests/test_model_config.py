from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from models.base import DataInfoView, ForecastModel
from models import loader as model_loader
from runtime.config import (
    ConfigError,
    load_experiment_config,
    load_model_config,
    load_model_config_document,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_model(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def test_model_yaml_uses_new_shape_and_build_boundary_is_model_only() -> None:
    document = load_model_config_document(ROOT / "configs" / "models" / "lstm.yaml")
    assert document["runtime"] == {"environment": "tslib"}
    assert load_model_config(ROOT / "configs" / "models" / "lstm.yaml") == document["model"]


def test_node_shared_chunk_is_public_runtime_only() -> None:
    config = load_experiment_config(ROOT / "configs" / "experiment.yaml")
    assert config.runtime["node_shared_chunk_size"] == 32


def test_build_model_receives_only_model_mapping(monkeypatch, tmp_path: Path) -> None:
    received = {}

    class TinyModel(ForecastModel):
        def forward(self, inputs):
            return torch.zeros(inputs.x.shape[0], 1, 1)

    def builder(model_config, data_info):
        received.update(model_config)
        assert data_info.num_nodes == 1
        return TinyModel()

    monkeypatch.setattr(
        model_loader,
        "load_model_module",
        lambda name: SimpleNamespace(build_model=builder),
    )
    path = _write_model(
        tmp_path,
        {"runtime": {"environment": "tsl"}, "model": {"hidden_size": 64}},
    )
    model_loader.build_model(
        "fixture_model",
        load_model_config(path),
        DataInfoView(num_nodes=1, num_features=1, lookback=1, max_pred_len=1),
    )
    assert received == {"hidden_size": 64}


def test_runtime_omitted_defaults_to_tslib(tmp_path: Path) -> None:
    path = _write_model(tmp_path, {"model": {"hidden_size": 64, "layers": 2}})
    document = load_model_config_document(path)
    assert document["runtime"] == {}


@pytest.mark.parametrize("environment", ["tslib", "tsl"])
def test_supported_runtime_environment_is_accepted(tmp_path: Path, environment: str) -> None:
    path = _write_model(tmp_path, {"runtime": {"environment": environment}, "model": {"any_parameter": 1}})
    assert load_model_config(path) == {"any_parameter": 1}


def test_unknown_environment_is_rejected(tmp_path: Path) -> None:
    path = _write_model(tmp_path, {"runtime": {"environment": "other"}, "model": {"x": 1}})
    with pytest.raises(ConfigError, match="runtime.environment"):
        load_model_config(path)


def test_arbitrary_nested_model_parameters_are_allowed(tmp_path: Path) -> None:
    path = _write_model(
        tmp_path,
        {
            "model": {
                "e_layers": 3,
                "top_k": 5,
                "nested": [{"diffusion_steps": 2, "supports": [None, True]}],
            }
        },
    )
    assert load_model_config(path)["nested"][0]["diffusion_steps"] == 2


@pytest.mark.parametrize(
    "leaked",
    [
        {"nested": {"training": {"epochs": 2}}},
        {"items": [{"data": {"lookback": 12}}]},
        {"learning_rate": 0.001},
        {"node_shared_chunk_size": 16},
    ],
)
def test_public_parameters_are_rejected_recursively(tmp_path: Path, leaked: dict) -> None:
    path = _write_model(tmp_path, {"model": leaked})
    with pytest.raises(ConfigError, match="public parameter"):
        load_model_config(path)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_model_values_are_rejected(tmp_path: Path, value: float) -> None:
    path = _write_model(tmp_path, {"model": {"value": value}})
    with pytest.raises(ConfigError, match="finite"):
        load_model_config(path)


def test_python_object_yaml_tag_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "object.yaml"
    path.write_text("model:\n  value: !!python/object/apply:os.getcwd []\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid model YAML"):
        load_model_config(path)

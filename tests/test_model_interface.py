from __future__ import annotations

from pathlib import Path

import pytest
import torch

from engine.checkpoint import load_checkpoint, save_checkpoint
from engine.losses import masked_score_aligned_hybrid
from models.base import DataInfoView, ModelInput
from models.loader import build_model
from runtime.config import ConfigError, load_experiment_config, load_model_config


ROOT = Path(__file__).resolve().parents[1]


def _info() -> DataInfoView:
    return DataInfoView(num_nodes=4, num_features=3, lookback=12, max_pred_len=3)


def test_lstm_forward_backward_and_parameter_count() -> None:
    model = build_model("lstm", {"hidden_dim": 8, "num_layers": 1, "dropout": 0.0}, _info())
    x = torch.randn(2, 12, 4, 3)
    output = model(ModelInput(x=x))
    assert tuple(output.shape) == (2, 4, 3)
    loss = masked_score_aligned_hybrid(output, torch.zeros_like(output), torch.ones_like(output, dtype=torch.bool))
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_model_input_does_not_expose_target_or_mask() -> None:
    assert "target" not in ModelInput.__dataclass_fields__
    assert "target_mask" not in ModelInput.__dataclass_fields__
    model = build_model("lstm", {"hidden_dim": 8, "num_layers": 1, "dropout": 0.0}, _info())
    with pytest.raises(TypeError, match="ModelInput"):
        model(torch.randn(1, 12, 4, 3))


def test_checkpoint_save_reload_preserves_predictions(tmp_path: Path) -> None:
    model = build_model("lstm", {"hidden_dim": 8, "num_layers": 1, "dropout": 0.0}, _info())
    model.eval()
    x = torch.randn(1, 12, 4, 3)
    with torch.no_grad():
        expected = model(ModelInput(x=x)).clone()
    save_checkpoint(tmp_path / "best.pt", model, manifest={"epoch": 1})
    reloaded = build_model("lstm", {"hidden_dim": 8, "num_layers": 1, "dropout": 0.0}, _info())
    load_checkpoint(tmp_path / "best.pt", reloaded)
    reloaded.eval()
    with torch.no_grad():
        actual = reloaded(ModelInput(x=x))
    torch.testing.assert_close(actual, expected)


def test_config_and_model_config_boundaries_are_strict(tmp_path: Path) -> None:
    config = load_experiment_config(ROOT / "configs" / "experiment.yaml")
    altered = config.copy_values()
    altered["training"]["unexpected"] = True
    import yaml

    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(altered), encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown field"):
        load_experiment_config(path)
    model_path = tmp_path / "bad-model.yaml"
    model_path.write_text("batch_size: 1\nhidden_dim: 8\nnum_layers: 1\ndropout: 0.0\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="public parameter"):
        load_model_config(model_path)

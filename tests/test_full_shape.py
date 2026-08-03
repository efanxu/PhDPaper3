from __future__ import annotations

from pathlib import Path

import pytest
import torch

from engine.losses import masked_score_aligned_hybrid
from engine.reproducibility import set_seed
from models.base import DataInfoView, ModelInput
from models.loader import build_model
from runtime.config import load_experiment_config, load_model_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="full-shape GPU smoke requires CUDA")
def test_formal_full_shape_forward_backward(monkeypatch) -> None:
    config = load_experiment_config(ROOT / "configs" / "experiment.yaml")
    model_config = load_model_config(ROOT / "configs" / "models" / "node_shared_lstm.yaml")
    monkeypatch.setenv("PYTHONHASHSEED", str(config.training["seed"]))
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    set_seed(
        config.training["seed"],
        reproducibility_mode="controlled_nonstrict",
    )
    device = torch.device("cuda")
    info = DataInfoView(
        config.data["num_nodes"],
        len(config.data["feature_columns"]),
        config.data["lookback"],
        config.data["max_pred_len"],
    )
    model = build_model("node_shared_lstm", model_config, info).to(device)
    batch_size = config.training["train_batch_size"]
    torch.cuda.reset_peak_memory_stats(device)
    x = torch.randn(batch_size, info.lookback, info.num_nodes, info.num_features, device=device)
    target = torch.randn(batch_size, info.num_nodes, info.max_pred_len, device=device)
    mask = torch.ones_like(target, dtype=torch.bool)
    output = model(ModelInput(x=x))
    loss = masked_score_aligned_hybrid(output, target, mask)
    loss.backward()
    assert tuple(output.shape) == (batch_size, info.num_nodes, info.max_pred_len)
    assert torch.isfinite(output).all()
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)
    assert torch.cuda.max_memory_allocated(device) > 0

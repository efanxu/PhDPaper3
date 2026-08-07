from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import torch
from torch import nn

from models.base import DataInfoView, ForecastModel, ModelInput
from runtime import validation
from runtime.status import FORMAL_DEFAULT_SHAPE, PASS
from engine.precision import resolve_precision_policy


class _ValidationToy(ForecastModel):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        values = inputs.x.mean(dim=(1, 3)).unsqueeze(-1)
        return values.expand(-1, inputs.x.shape[2], 2) * self.scale


def test_cuda_amp_policy_resolves_float16_without_allocating_cuda_tensors() -> None:
    policy = resolve_precision_policy(
        device="cuda",
        amp_configured=True,
        amp_dtype="float16",
        amp_cache_enabled=True,
    )
    assert policy.amp_configured is True
    assert policy.amp_effective is True
    assert policy.amp_dtype is torch.float16
    assert policy.amp_cache_enabled is True


def test_cpu_amp_policy_is_configured_but_not_effective() -> None:
    policy = resolve_precision_policy(
        device="cpu",
        amp_configured=True,
        amp_dtype="float16",
        amp_cache_enabled=True,
    )
    assert policy.amp_configured is True
    assert policy.amp_effective is False
    assert policy.amp_dtype_name == "float16"


def test_shape_validation_uses_the_resolved_precision_callback(monkeypatch) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "2026")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    info = DataInfoView(
        num_nodes=2,
        num_features=1,
        lookback=3,
        max_pred_len=2,
        feature_columns=("f",),
    )
    toy = _ValidationToy()
    captured: dict[str, object] = {}

    class SpyPolicy:
        def as_dict(self):
            return {
                "amp_configured": True,
                "amp_effective": False,
                "amp_dtype": "float16",
                "amp_cache_enabled": True,
            }

        def autocast(self):
            captured["autocast_entered"] = True
            return nullcontext()

    spy = SpyPolicy()

    def resolve(**kwargs):
        captured["resolve_kwargs"] = kwargs
        return spy

    def fake_executor(model, batches, **kwargs):
        captured["autocast"] = kwargs["autocast"]
        batch = batches[0]
        with kwargs["autocast"]():
            prediction = model(batch.model_input())
            loss = prediction.square().mean()
        kwargs["backward"](loss)
        return SimpleNamespace(prediction=prediction.detach(), loss=float(loss.detach()))

    monkeypatch.setattr(validation, "load_data", lambda config, project_root: (None, info))
    monkeypatch.setattr(validation, "load_model_config", lambda path: {})
    monkeypatch.setattr(validation, "build_model", lambda name, config, data_info: toy)
    monkeypatch.setattr(validation, "resolve_precision_policy", resolve)
    monkeypatch.setattr(validation, "execute_training_backward", fake_executor)

    payload = validation.run_shape_validation(
        model_name="toy",
        config_path="configs/experiment.yaml",
        model_config_path=None,
        device="cpu",
        profile=FORMAL_DEFAULT_SHAPE,
    )

    assert payload["status"] == PASS, payload
    assert captured["resolve_kwargs"] == {
        "device": torch.device("cpu"),
        "amp_configured": True,
        "amp_dtype": "float16",
        "amp_cache_enabled": True,
    }
    assert captured["autocast"] == spy.autocast
    assert captured["autocast_entered"] is True
    assert payload["details"]["precision"]["amp_effective"] is False

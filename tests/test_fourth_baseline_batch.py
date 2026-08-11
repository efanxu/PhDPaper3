from __future__ import annotations

import inspect
import random
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn

from engine.model_execution import (
    build_execution_plan,
    has_batch_dependent_normalization,
)
from models.base import DataInfoView, ForecastModel, ModelInput, NodeSharedForecastModel
from models.loader import build_model
from runtime.config import load_model_config, load_model_config_document


ROOT = Path(__file__).resolve().parents[1]
TSLIB_NAMES = ("patchtst", "nonstationary_transformer", "fedformer")
ALL_NAMES = TSLIB_NAMES
PUBLIC_MODEL_KEYS = {
    "batch_size",
    "train_batch_size",
    "val_batch_size",
    "test_batch_size",
    "epochs",
    "loss",
    "learning_rate",
    "seed",
    "lookback",
    "max_pred_len",
    "num_nodes",
    "feature_columns",
    "graph",
    "k",
    "amp",
    "node_shared_chunk_size",
}


def _is_environment(name: str) -> bool:
    return Path(sys.executable).parent.name.casefold() == name.casefold()


def _contains_key(value: object, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(child, key) for child in value.values())
    if isinstance(value, list):
        return any(_contains_key(child, key) for child in value)
    return False


def _info(
    *,
    nodes: int = 5,
    lookback: int = 144,
    features: tuple[str, ...] = ("f0", "f1", "power", "f3"),
    project_root: Path = ROOT,
    graph_config: dict[str, object] | None = None,
    node_ids: tuple[int, ...] | None = None,
) -> DataInfoView:
    if node_ids is None:
        node_ids = tuple(range(1, nodes + 1))
    return DataInfoView(
        num_nodes=nodes,
        num_features=len(features),
        lookback=lookback,
        max_pred_len=10,
        feature_columns=features,
        input_power_column="power" if "power" in features else features[1],
        input_power_index=features.index("power") if "power" in features else 1,
        node_ids=node_ids,
        graph_config=graph_config,
        project_root=project_root,
    )


def _config(name: str) -> dict:
    return load_model_config(ROOT / "configs" / "models" / f"{name}.yaml")


def _build(name: str, info: DataInfoView) -> ForecastModel:
    return build_model(name, _config(name), info)


def test_fourth_batch_yaml_is_structure_only() -> None:
    expected_environment = {
        "patchtst": "tslib",
        "nonstationary_transformer": "tslib",
        "fedformer": "tslib",
    }
    for name in ALL_NAMES:
        document = load_model_config_document(ROOT / "configs" / "models" / f"{name}.yaml")
        assert document["runtime"] == {"environment": expected_environment[name]}
        assert document["model"]
        for key in PUBLIC_MODEL_KEYS:
            assert not _contains_key(document["model"], key), (name, key)
    assert "label_len" not in load_model_config_document(
        ROOT / "configs" / "models" / "fedformer.yaml"
    )["model"]


@pytest.mark.skipif(not _is_environment("env_tslib"), reason="requires the formal env_tslib interpreter")
def test_tslib_fourth_batch_build_forward_backward_and_structure() -> None:
    info = _info()
    inputs = ModelInput(x=torch.randn(2, 144, 5, 4))
    for name in TSLIB_NAMES:
        torch.manual_seed(2026)
        model = _build(name, info)
        assert isinstance(model, NodeSharedForecastModel), name
        assert type(model.upstream).__module__.startswith(
            "_phdpaper3_time_series_library.models"
        ), name

        model.train()
        model.zero_grad(set_to_none=True)
        output = model(inputs)
        loss = output.square().mean()
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        assert tuple(output.shape) == (2, 5, 10), name
        assert torch.isfinite(output).all(), name
        assert any(gradient is not None for gradient in gradients), name
        assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None), name

        if name == "patchtst":
            assert hasattr(model.upstream, "patch_embedding")
            assert any(isinstance(module, nn.BatchNorm1d) for module in model.upstream.modules())
            plan = build_execution_plan(model, total_nodes=134, node_shared_chunk_size=32)
            assert plan.execution_mode == "full_nodes"
            assert plan.node_chunk_count == 1
            assert plan.reason == "batch_dependent_normalization"
            assert has_batch_dependent_normalization(model) is True
        elif name == "nonstationary_transformer":
            assert hasattr(model.upstream, "tau_learner")
            assert hasattr(model.upstream, "delta_learner")
            assert has_batch_dependent_normalization(model) is False
        else:
            assert hasattr(model.upstream, "decomp")
            assert hasattr(model.upstream, "encoder")
            assert hasattr(model.upstream, "decoder")
            assert model.upstream.version == "fourier"
            assert model.upstream.mode_select == "random"
            assert model.label_len == model.lookback // 2


@pytest.mark.skipif(not _is_environment("env_tslib"), reason="requires the formal env_tslib interpreter")
def test_tslib_node_shared_chunk_equivalence_and_decoder_placeholders() -> None:
    inputs = ModelInput(x=torch.randn(2, 144, 5, 4))
    for name in ("nonstationary_transformer", "fedformer"):
        model = _build(name, _info()).eval()
        with torch.inference_mode():
            full = model(inputs)
            chunked = torch.cat(
                [
                    model.forward_node_chunk(inputs, 0, 2),
                    model.forward_node_chunk(inputs, 2, 4),
                    model.forward_node_chunk(inputs, 4, 5),
                ],
                dim=1,
            )
        torch.testing.assert_close(full, chunked, atol=1e-5, rtol=1e-5, msg=name)

        class RecordingUpstream(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.decoder_input: torch.Tensor | None = None

            def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
                del x_mark_enc, x_mark_dec
                self.decoder_input = x_dec.detach().clone()
                return torch.zeros(
                    x_enc.shape[0],
                    model.horizon,
                    x_enc.shape[2],
                    dtype=x_enc.dtype,
                    device=x_enc.device,
                )

        recorder = RecordingUpstream()
        model.upstream = recorder
        with torch.inference_mode():
            output = model.forward_node_chunk(inputs, 0, 2)
        assert tuple(output.shape) == (2, 2, 10)
        assert recorder.decoder_input is not None
        decoder_input = recorder.decoder_input
        assert tuple(decoder_input.shape) == (4, model.label_len + model.horizon, 4)
        expected_history = inputs.x[:, -model.label_len :, :2, :].permute(0, 2, 1, 3).reshape(
            4, model.label_len, 4
        )
        torch.testing.assert_close(decoder_input[:, : model.label_len], expected_history)
        assert torch.count_nonzero(decoder_input[:, model.label_len :]) == 0


@pytest.mark.skipif(not _is_environment("env_tslib"), reason="requires the formal env_tslib interpreter")
def test_tslib_public_override_rebuilds_all_dimensions() -> None:
    info = _info(
        lookback=120,
        features=("f0", "Patv_clean_for_input", "f2"),
    )
    inputs = ModelInput(x=torch.randn(2, 120, 5, 3))
    for name in TSLIB_NAMES:
        model = _build(name, info).eval()
        assert model.lookback == 120
        assert model.input_dim == 3
        assert model.input_power_index == 1
        assert model.upstream.seq_len == 120
        if name == "patchtst":
            assert model.upstream.head.n_vars == 3
        elif name == "nonstationary_transformer":
            assert model.upstream.tau_learner.series_conv.in_channels == 120
            assert model.upstream.dec_embedding.value_embedding.tokenConv.in_channels == 3
        else:
            assert model.label_len == 60
            assert model.upstream.label_len == 60
        with torch.inference_mode():
            output = model(inputs)
        assert tuple(output.shape) == (2, 5, 10)
        assert torch.isfinite(output).all()


@pytest.mark.skipif(not _is_environment("env_tslib"), reason="requires the formal env_tslib interpreter")
def test_fedformer_random_modes_follow_shared_seed() -> None:
    def mode_indices(model: ForecastModel) -> tuple[tuple[int, ...], ...]:
        values: list[tuple[int, ...]] = []
        for module in model.upstream.modules():
            for attribute in ("index", "index_q", "index_kv"):
                value = getattr(module, attribute, None)
                if isinstance(value, list):
                    values.append(tuple(int(item) for item in value))
        return tuple(values)

    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    first = _build("fedformer", _info())
    first_indices = mode_indices(first)
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    second = _build("fedformer", _info())
    assert first_indices == mode_indices(second)


def test_fourth_batch_adapters_do_not_cross_the_input_boundary() -> None:
    assert "target" not in ModelInput.__dataclass_fields__
    assert "target_mask" not in ModelInput.__dataclass_fields__
    for name in ALL_NAMES:
        module = __import__(f"models.{name}.model", fromlist=["build_model"])
        source = inspect.getsource(module).lower()
        assert "target" not in source, name

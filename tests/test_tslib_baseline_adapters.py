from __future__ import annotations

import inspect
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

from engine.model_execution import (
    build_execution_plan,
    has_batch_dependent_normalization,
)
from models.base import DataInfoView, ModelInput, NodeSharedForecastModel
from models.loader import build_model
from runtime.config import load_model_config, load_model_config_document


ROOT = Path(__file__).resolve().parents[1]


def _is_environment(name: str) -> bool:
    return Path(sys.executable).parent.name.casefold() == name.casefold()


TSLIB_RUNTIME = pytest.mark.skipif(
    not _is_environment("env_tslib"),
    reason="requires the formal env_tslib interpreter",
)


MODEL_NAMES = (
    "dlinear",
    "tsmixer",
    "segrnn",
    "itransformer",
    "timesnet",
    "timemixer",
    "transformer",
)


def _info(nodes: int = 5) -> DataInfoView:
    return DataInfoView(
        num_nodes=nodes,
        num_features=4,
        lookback=144,
        max_pred_len=10,
        feature_columns=("f0", "f1", "power", "f3"),
        input_power_column="power",
        input_power_index=2,
        node_ids=tuple(range(1, nodes + 1)),
        project_root=ROOT,
    )


def _build(name: str, *, nodes: int = 5):
    config = load_model_config(ROOT / "configs" / "models" / f"{name}.yaml")
    return build_model(name, config, _info(nodes))


@pytest.fixture(scope="module")
def models() -> dict[str, NodeSharedForecastModel]:
    torch.manual_seed(2026)
    return {name: _build(name).eval() for name in MODEL_NAMES}


@pytest.fixture(scope="module")
def inputs() -> ModelInput:
    torch.manual_seed(31415)
    return ModelInput(x=torch.randn(2, 144, 5, 4))


@TSLIB_RUNTIME
def test_all_adapters_build_and_return_finite_bnh(models, inputs) -> None:
    for name, model in models.items():
        assert isinstance(model, NodeSharedForecastModel), name
        with torch.inference_mode():
            output = model(inputs)
        assert tuple(output.shape) == (2, 5, 10), name
        assert torch.isfinite(output).all(), name


@TSLIB_RUNTIME
def test_timemixer_channel_independence_zero_builds_and_runs() -> None:
    model_config = dict(
        load_model_config(ROOT / "configs" / "models" / "timemixer.yaml")
    )
    model_config["channel_independence"] = 0
    model = build_model("timemixer", model_config, _info()).eval()
    assert model.upstream.configs.c_out == 4

    torch.manual_seed(2718)
    inputs = ModelInput(x=torch.randn(2, 144, 5, 4))
    with torch.inference_mode():
        output = model(inputs)
    assert tuple(output.shape) == (2, 5, 10)
    assert torch.isfinite(output).all()


@TSLIB_RUNTIME
def test_node_chunk_contract_only_consumes_requested_nodes(models, inputs) -> None:
    changed = inputs.x.clone()
    changed[:, :, 0, :] = changed[:, :, 0, :] + 1000.0
    changed[:, :, 4, :] = changed[:, :, 4, :] - 1000.0
    changed_input = ModelInput(x=changed)

    for name, model in models.items():
        with torch.inference_mode():
            output = model.forward_node_chunk(inputs, 1, 4)
            changed_output = model.forward_node_chunk(changed_input, 1, 4)
        assert tuple(output.shape) == (2, 3, 10), name
        torch.testing.assert_close(output, changed_output, atol=0.0, rtol=0.0)


@TSLIB_RUNTIME
def test_full_and_uneven_chunked_execution_are_equivalent(models, inputs) -> None:
    for name, model in models.items():
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
        assert tuple(chunked.shape) == (2, 5, 10), name
        torch.testing.assert_close(full, chunked, atol=1e-6, rtol=1e-6)


@TSLIB_RUNTIME
def test_formal_execution_plan_is_node_shared_for_all_new_models(models) -> None:
    for name, model in models.items():
        plan = build_execution_plan(model, total_nodes=134, node_shared_chunk_size=32)
        assert plan.execution_mode == "node_shared_microbatch", name
        assert plan.configured_node_chunk_size == 32, name
        assert plan.effective_node_chunk_size == 32, name
        assert plan.node_chunk_count == 5, name
        assert [end - start for start, end in plan.node_ranges()] == [32, 32, 32, 32, 6], name
        assert has_batch_dependent_normalization(model) is False, name


@TSLIB_RUNTIME
def test_segrnn_uses_the_horizon_compatible_segment_layout(models) -> None:
    model = models["segrnn"]
    assert model.upstream.seg_num_x == 72
    assert model.upstream.seg_num_y == 5


@TSLIB_RUNTIME
def test_input_power_channel_is_selected_by_metadata_not_zero(models, inputs) -> None:
    class FixedUpstream(nn.Module):
        def forward(self, x, *args):
            values = torch.arange(4, dtype=x.dtype, device=x.device).view(1, 1, 4)
            return values.expand(x.shape[0], 10, 4)

    for name in MODEL_NAMES:
        model = _build(name).eval()
        model.upstream = FixedUpstream()
        with torch.inference_mode():
            output = model(inputs)
        torch.testing.assert_close(output, torch.full((2, 5, 10), 2.0))


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_model_yaml_declares_tslib_and_only_model_structure(name: str) -> None:
    document = load_model_config_document(ROOT / "configs" / "models" / f"{name}.yaml")
    assert document["runtime"] == {"environment": "tslib"}
    assert document["model"]
    assert "lookback" not in document["model"]
    assert "max_pred_len" not in document["model"]
    assert "batch_size" not in document["model"]


_INVALID_CONFIG = {
    "dlinear": ("moving_avg", 24),
    "tsmixer": ("d_model", 0),
    "segrnn": ("seg_len", 12),
    "itransformer": ("d_model", 63),
    "timesnet": ("top_k", 0),
    "timemixer": ("down_sampling_window", 1),
    "transformer": ("output_attention", True),
}


@pytest.mark.parametrize("name", MODEL_NAMES)
@TSLIB_RUNTIME
def test_each_adapter_rejects_missing_and_unknown_fields(name: str) -> None:
    config = load_model_config(ROOT / "configs" / "models" / f"{name}.yaml")
    missing_name = next(iter(config))
    missing = dict(config)
    missing.pop(missing_name)
    with pytest.raises(ValueError, match="missing"):
        build_model(name, missing, _info())

    with pytest.raises(ValueError, match="unknown"):
        build_model(name, {**config, "unknown": 1}, _info())


@pytest.mark.parametrize("name", MODEL_NAMES)
@TSLIB_RUNTIME
def test_each_adapter_rejects_a_model_specific_invalid_value(name: str) -> None:
    config = load_model_config(ROOT / "configs" / "models" / f"{name}.yaml")
    field, value = _INVALID_CONFIG[name]
    with pytest.raises(ValueError):
        build_model(name, {**config, field: value}, _info())


@pytest.mark.parametrize("name", MODEL_NAMES)
@TSLIB_RUNTIME
def test_each_adapter_rejects_misaligned_input_metadata(name: str) -> None:
    config = load_model_config(ROOT / "configs" / "models" / f"{name}.yaml")
    invalid_info = DataInfoView(
        num_nodes=5,
        num_features=4,
        lookback=144,
        max_pred_len=10,
        feature_columns=("f0", "f1", "power", "f3"),
        input_power_column="power",
        input_power_index=0,
        node_ids=tuple(range(1, 6)),
        project_root=ROOT,
    )
    with pytest.raises(ValueError, match="does not match"):
        build_model(name, config, invalid_info)


def test_adapters_do_not_accept_labels_or_masks() -> None:
    assert "target" not in ModelInput.__dataclass_fields__
    assert "target_mask" not in ModelInput.__dataclass_fields__
    for name in MODEL_NAMES:
        module = __import__(f"models.{name}.model", fromlist=["build_model"])
        source = inspect.getsource(module)
        assert "target" not in source, name
        assert "target_mask" not in source, name

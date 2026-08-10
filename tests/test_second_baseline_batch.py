from __future__ import annotations

import inspect
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn

from engine.model_execution import (
    build_execution_plan,
    has_batch_dependent_normalization,
)
from integrations.time_series_library import run_time_series_library_forecast
from models.base import DataInfoView, ForecastModel, ModelInput, NodeSharedForecastModel
from models.loader import build_model
from runtime.config import (
    apply_cli_overrides,
    load_experiment_config,
    load_model_config,
    load_model_config_document,
)


ROOT = Path(__file__).resolve().parents[1]


def _is_environment(name: str) -> bool:
    return Path(sys.executable).parent.name.casefold() == name.casefold()


TSLIB_RUNTIME = pytest.mark.skipif(
    not _is_environment("env_tslib"),
    reason="requires the formal env_tslib interpreter",
)
TSL_RUNTIME = pytest.mark.skipif(
    not _is_environment("env_tsl"),
    reason="requires the formal env_tsl interpreter",
)


TSLIB_NAMES = ("lightts", "tide", "frets", "film", "informer", "autoformer")
ALL_NAMES = (*TSLIB_NAMES, "stid")
EXPECTED_ENVIRONMENTS = {name: "tslib" for name in TSLIB_NAMES} | {"stid": "tsl"}
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
    "split",
    "amp",
    "node_shared_chunk_size",
}


def _info(
    *,
    nodes: int = 5,
    features: tuple[str, ...] = ("f0", "f1", "power", "f3"),
    lookback: int = 144,
    horizon: int = 10,
    node_ids: tuple[int, ...] | None = None,
    input_power_column: str = "power",
) -> DataInfoView:
    if node_ids is None:
        node_ids = tuple(range(1, nodes + 1))
    return DataInfoView(
        num_nodes=nodes,
        num_features=len(features),
        lookback=lookback,
        max_pred_len=horizon,
        feature_columns=features,
        input_power_column=input_power_column,
        input_power_index=features.index(input_power_column),
        node_ids=node_ids,
        project_root=ROOT,
    )


def _config(name: str) -> dict:
    return load_model_config(ROOT / "configs" / "models" / f"{name}.yaml")


def _build(name: str, info: DataInfoView | None = None):
    return build_model(name, _config(name), info or _info())


def _contains_key(value, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(child, key) for child in value.values())
    if isinstance(value, list):
        return any(_contains_key(child, key) for child in value)
    return False


@pytest.fixture(scope="module")
def tslib_models() -> dict[str, NodeSharedForecastModel]:
    torch.manual_seed(2026)
    return {name: _build(name).eval() for name in TSLIB_NAMES}


@pytest.fixture(scope="module")
def default_inputs() -> ModelInput:
    torch.manual_seed(31415)
    return ModelInput(x=torch.randn(2, 144, 5, 4))


def test_second_batch_model_yaml_environment_and_structure() -> None:
    for name in ALL_NAMES:
        document = load_model_config_document(
            ROOT / "configs" / "models" / f"{name}.yaml"
        )
        assert document["runtime"] == {"environment": EXPECTED_ENVIRONMENTS[name]}
        assert document["model"]
        for key in PUBLIC_MODEL_KEYS:
            assert not _contains_key(document["model"], key), (name, key)


@TSLIB_RUNTIME
def test_second_batch_adapters_build_and_return_finite_bnh(
    tslib_models: dict[str, NodeSharedForecastModel],
    default_inputs: ModelInput,
) -> None:
    for name, model in tslib_models.items():
        assert isinstance(model, NodeSharedForecastModel), name
        with torch.inference_mode():
            output = model(default_inputs)
        assert tuple(output.shape) == (2, 5, 10), name
        assert torch.isfinite(output).all(), name


@TSLIB_RUNTIME
def test_second_batch_node_range_contract_and_isolation(
    tslib_models: dict[str, NodeSharedForecastModel],
    default_inputs: ModelInput,
) -> None:
    changed = default_inputs.x.clone()
    changed[:, :, 0, :] += 1000.0
    changed[:, :, 4, :] -= 1000.0
    changed_inputs = ModelInput(x=changed)
    for name, model in tslib_models.items():
        with torch.inference_mode():
            rng_state = torch.get_rng_state()
            output = model.forward_node_chunk(default_inputs, 1, 4)
            torch.set_rng_state(rng_state)
            changed_output = model.forward_node_chunk(changed_inputs, 1, 4)
        assert tuple(output.shape) == (2, 3, 10), name
        torch.testing.assert_close(output, changed_output, atol=0.0, rtol=0.0)


@TSLIB_RUNTIME
def test_second_batch_full_and_uneven_chunks_match_for_microbatch_models(
    tslib_models: dict[str, NodeSharedForecastModel],
    default_inputs: ModelInput,
) -> None:
    for name in ("lightts", "tide", "frets", "film", "autoformer"):
        model = tslib_models[name]
        with torch.inference_mode():
            full = model(default_inputs)
            chunked = torch.cat(
                [
                    model.forward_node_chunk(default_inputs, 0, 2),
                    model.forward_node_chunk(default_inputs, 2, 4),
                    model.forward_node_chunk(default_inputs, 4, 5),
                ],
                dim=1,
            )
        torch.testing.assert_close(full, chunked, atol=1e-5, rtol=1e-5, msg=name)


@TSLIB_RUNTIME
def test_second_batch_execution_plans_preserve_special_architectures(
    tslib_models: dict[str, NodeSharedForecastModel],
) -> None:
    for name in ("lightts", "tide", "frets", "film", "autoformer"):
        model = tslib_models[name]
        plan = build_execution_plan(model, total_nodes=134, node_shared_chunk_size=32)
        assert plan.execution_mode == "node_shared_microbatch", name
        assert plan.node_chunk_count == 5
        assert [end - start for start, end in plan.node_ranges()] == [32, 32, 32, 32, 6]
        assert has_batch_dependent_normalization(model) is False

    informer = tslib_models["informer"]
    informer_plan = build_execution_plan(
        informer,
        total_nodes=134,
        node_shared_chunk_size=32,
    )
    assert has_batch_dependent_normalization(informer) is True
    assert informer_plan.execution_mode == "full_nodes"
    assert informer_plan.node_chunk_count == 1
    assert informer_plan.reason == "batch_dependent_normalization"


@TSLIB_RUNTIME
def test_second_batch_resolved_lookback_and_feature_columns_are_model_inputs(
    tmp_path: Path,
) -> None:
    base = load_experiment_config(ROOT / "configs" / "experiment.yaml")
    resolved = apply_cli_overrides(
        base,
        {
            "lookback": 120,
            "feature_columns": ["f0", "Patv_clean_for_input", "f2"],
        },
        project_root=tmp_path,
    )
    assert resolved.data["lookback"] == 120
    assert resolved.data["feature_columns"] == ["f0", "Patv_clean_for_input", "f2"]
    info = _info(
        features=("f0", "Patv_clean_for_input", "f2"),
        lookback=int(resolved.data["lookback"]),
        input_power_column=str(resolved.data["input_power_column"]),
    )
    inputs = ModelInput(x=torch.randn(2, 120, 5, 3))
    for name in TSLIB_NAMES:
        model = _build(name, info).eval()
        assert model.lookback == 120, name
        assert model.input_dim == 3, name
        assert model.input_power_index == 1, name
        with torch.inference_mode():
            output = model(inputs)
        assert tuple(output.shape) == (2, 5, 10), name
        assert torch.isfinite(output).all(), name


def test_second_batch_public_override_is_not_reimplemented_by_adapters() -> None:
    forbidden_tokens = (
        "batch_size",
        "epochs",
        "learning_rate",
        "loss",
        "seed",
        "target_mask",
    )
    for name in ALL_NAMES:
        module = __import__(f"models.{name}.model", fromlist=["build_model"])
        source = inspect.getsource(module)
        for token in forbidden_tokens:
            assert token not in source, (name, token)


def test_second_batch_power_channel_is_selected_by_resolved_index() -> None:
    class FixedUpstream(nn.Module):
        def forward(self, x, *args):
            values = torch.arange(3, dtype=x.dtype, device=x.device).view(1, 1, 3)
            return values.expand(x.shape[0], 10, 3)

    info = _info(features=("f0", "power", "f2"))
    inputs = ModelInput(x=torch.randn(2, 144, 5, 3))
    upstream = FixedUpstream()
    output, batch, nodes = run_time_series_library_forecast(
        inputs.x[:, :, 1:4, :],
        upstream,
        horizon=10,
        input_power_index=info.input_power_index,
        model_name="second-batch-fixture",
    )
    assert (batch, nodes) == (2, 3)
    torch.testing.assert_close(output, torch.full((2, 3, 10), 1.0))


@TSLIB_RUNTIME
def test_second_batch_decoder_placeholders_and_tide_fallback_do_not_use_future_values() -> None:
    class RecordingUpstream(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
            self.calls.append((x_enc.detach().clone(), x_mark_enc, x_dec, x_mark_dec))
            return torch.zeros(x_enc.shape[0], 10, x_enc.shape[2], dtype=x_enc.dtype)

    inputs = ModelInput(x=torch.randn(2, 144, 5, 4))
    for name in ("tide", "informer", "autoformer"):
        model = _build(name).eval()
        recorder = RecordingUpstream()
        model.upstream = recorder
        with torch.inference_mode():
            model.forward_node_chunk(inputs, 0, 2)
        assert len(recorder.calls) == 1, name
        _, enc_marks, decoder_input, dec_marks = recorder.calls[0]
        assert enc_marks is None, name
        assert dec_marks is None, name
        if name == "tide":
            assert decoder_input is None
        else:
            expected_history = inputs.x[:, -model.label_len :, :2, :].permute(0, 2, 1, 3).reshape(
                2 * 2, model.label_len, 4
            )
            assert tuple(decoder_input.shape) == (4, model.label_len + 10, 4)
            torch.testing.assert_close(decoder_input[:, : model.label_len], expected_history)
            assert torch.count_nonzero(decoder_input[:, model.label_len :]) == 0


@TSLIB_RUNTIME
def test_film_cpu_device_isolation_even_when_cuda_is_visible() -> None:
    model = _build("film").eval()
    assert all(value.device.type == "cpu" for value in model.upstream.buffers())
    inputs = ModelInput(x=torch.randn(2, 144, 5, 4))
    with torch.inference_mode():
        output = model(inputs)
    assert tuple(output.shape) == (2, 5, 10)
    assert torch.isfinite(output).all()
    assert all(value.device.type == "cpu" for value in model.upstream.buffers())
    module = sys.modules[type(model.upstream).__module__]
    assert module.device == torch.device("cpu")


@pytest.mark.parametrize(
    ("name", "field", "value", "lookback"),
    [
        ("lightts", "chunk_size", 0, 144),
        ("tide", "label_len", 145, 144),
        ("frets", "channel_independence", 0, 144),
        ("film", "label_len", 145, 144),
        ("informer", "label_len", 145, 144),
        ("autoformer", "moving_avg", 24, 144),
    ],
)
@TSLIB_RUNTIME
def test_second_batch_model_math_conflicts_fail_closed(
    name: str,
    field: str,
    value,
    lookback: int,
) -> None:
    config = _config(name)
    config[field] = value
    with pytest.raises(ValueError):
        build_model(name, config, _info(lookback=lookback))


def test_command_reference_and_cli_schema_regression() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_command_reference.py"), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@TSL_RUNTIME
def test_stid_real_tsl_cpu_build_forward_and_identity_contract() -> None:
    info = _info(node_ids=(101, 205, 309, 412, 518))
    model = _build("stid", info).to("cpu").eval()
    assert isinstance(model, ForecastModel)
    assert not isinstance(model, NodeSharedForecastModel)
    assert model.execution_mode == "full_spatiotemporal"
    assert model.upstream.n_nodes == 5
    assert model.upstream.exog_size == []
    plan = build_execution_plan(model, total_nodes=5, node_shared_chunk_size=32)
    assert plan.execution_mode == "full_spatiotemporal"
    assert plan.node_chunk_count == 1
    inputs = ModelInput(x=torch.randn(2, 144, 5, 4))
    with torch.inference_mode():
        output = model(inputs)
    assert tuple(output.shape) == (2, 5, 10)
    assert torch.isfinite(output).all()


@TSL_RUNTIME
def test_stid_rejects_missing_or_duplicate_node_ids() -> None:
    with pytest.raises(ValueError, match="node_ids"):
        _build("stid", _info(node_ids=()))
    with pytest.raises(ValueError, match="duplicates"):
        _build("stid", _info(node_ids=(1, 2, 3, 4, 4)))

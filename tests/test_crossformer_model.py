from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
from torch import nn

from engine.model_execution import build_execution_plan, forward_with_execution_plan
from models.base import DataInfoView, ModelInput
from models.loader import build_model
from runtime.config import load_model_config_document


ROOT = Path(__file__).resolve().parents[1]


def _is_environment(name: str) -> bool:
    return Path(sys.executable).parent.name.casefold() == name.casefold()


TSLIB_RUNTIME = pytest.mark.skipif(
    not _is_environment("env_tslib"),
    reason="requires the formal env_tslib interpreter",
)


CONFIG = {"d_model": 16, "n_heads": 4, "d_ff": 32, "e_layers": 2, "dropout": 0.0, "factor": 2}


def _info(nodes: int = 3, *, root: Path = ROOT) -> DataInfoView:
    return DataInfoView(
        num_nodes=nodes,
        num_features=4,
        lookback=24,
        max_pred_len=3,
        feature_columns=("Wspd", "Patv_clean_for_input", "Wdir", "Etmp"),
        input_power_column="Patv_clean_for_input",
        input_power_index=1,
        node_ids=tuple(range(1, nodes + 1)),
        project_root=root,
    )


def _build(nodes: int = 3):
    return build_model("crossformer", dict(CONFIG), _info(nodes))


def test_crossformer_yaml_uses_tslib_runtime() -> None:
    assert load_model_config_document(ROOT / "configs" / "models" / "crossformer.yaml")["runtime"] == {"environment": "tslib"}


@TSLIB_RUNTIME
def test_crossformer_rejects_missing_unknown_and_invalid_input_metadata() -> None:
    with pytest.raises(ValueError, match="unknown"):
        build_model("crossformer", {**CONFIG, "unknown": 1}, _info())
    missing = dict(CONFIG)
    missing.pop("factor")
    with pytest.raises(ValueError, match="missing"):
        build_model("crossformer", missing, _info())
    invalid = _info()
    invalid = DataInfoView(**{**invalid.__dict__, "input_power_index": 3})
    with pytest.raises(ValueError, match="does not match"):
        build_model("crossformer", dict(CONFIG), invalid)


@TSLIB_RUNTIME
def test_crossformer_forward_backward_and_selects_configured_power_channel() -> None:
    model = _build()
    model.eval()
    x = torch.randn(1, 24, 3, 4)
    output = model(ModelInput(x=x))
    raw = model.upstream(x.permute(0, 2, 1, 3).reshape(3, 24, 4), None, None, None)
    expected = raw[..., 1].reshape(1, 3, 3)
    torch.testing.assert_close(output, expected)
    output.sum().backward()
    assert tuple(output.shape) == (1, 3, 3)
    assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)
    with pytest.raises(ValueError, match="input shape"):
        model(ModelInput(x=torch.randn(1, 23, 3, 4)))


@TSLIB_RUNTIME
def test_crossformer_chunked_execution_matches_full_eval_with_uneven_tail() -> None:
    model = _build(5).eval()
    inputs = ModelInput(x=torch.randn(2, 24, 5, 4))
    plan = build_execution_plan(model, total_nodes=5, node_shared_chunk_size=2)
    assert [end - start for start, end in plan.node_ranges()] == [2, 2, 1]

    with torch.inference_mode():
        full = model(inputs)
        chunked = forward_with_execution_plan(model, inputs, plan)

    assert tuple(chunked.shape) == tuple(full.shape) == (2, 5, 3)
    torch.testing.assert_close(chunked, full, atol=1e-6, rtol=1e-6)


@TSLIB_RUNTIME
def test_crossformer_node_shared_parameter_count_permutation_and_equal_histories() -> None:
    three = _build(3).eval()
    five = _build(5).eval()
    assert sum(parameter.numel() for parameter in three.parameters()) == sum(parameter.numel() for parameter in five.parameters())
    x = torch.randn(1, 24, 3, 4)
    x[:, :, 1] = x[:, :, 0]
    permutation = torch.tensor([2, 0, 1])
    with torch.no_grad():
        output = three(ModelInput(x=x))
        permuted = three(ModelInput(x=x[:, :, permutation]))
    torch.testing.assert_close(permuted, output[:, permutation])
    torch.testing.assert_close(output[:, 0], output[:, 1])


@TSLIB_RUNTIME
def test_crossformer_rejects_nonfinite_output_and_missing_upstream_source(tmp_path: Path) -> None:
    model = _build()

    class NanUpstream(nn.Module):
        def forward(self, x, *args):
            return torch.full((x.shape[0], 3, 4), float("nan"), device=x.device)

    model.upstream = NanUpstream()
    with pytest.raises(FloatingPointError, match="NaN or Inf"):
        model(ModelInput(x=torch.randn(1, 24, 3, 4)))
    with pytest.raises(FileNotFoundError, match="requires Time-Series-Library"):
        build_model("crossformer", dict(CONFIG), _info(root=tmp_path))

from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

from models.base import DataInfoView, ModelInput
from models.loader import build_model
from runtime.config import load_model_config_document


ROOT = Path(__file__).resolve().parents[1]


def _is_environment(name: str) -> bool:
    return Path(sys.executable).parent.name.casefold() == name.casefold()


TSL_RUNTIME = pytest.mark.skipif(
    not _is_environment("env_tsl"),
    reason="requires the formal env_tsl interpreter",
)


FIXTURE = ROOT / "tests" / "fixtures" / "turbine_locations_small.csv"
CONFIG = {
    "hidden_size": 8,
    "ff_size": 16,
    "n_layers": 2,
    "temporal_kernel_size": 3,
    "spatial_kernel_size": 2,
    "temporal_convs_layer": 2,
    "spatial_convs_layer": 1,
    "dilation": 1,
    "norm": "none",
    "gated": False,
    "activation": "relu",
    "dropout": 0.0,
}


def _info(tmp_path: Path, *, node_ids: tuple[int, ...] = (1, 2, 3, 4)) -> DataInfoView:
    location = tmp_path / "dataset" / "sdwpf_turb_location_elevation.csv"
    location.parent.mkdir(parents=True, exist_ok=True)
    location.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    return DataInfoView(
        num_nodes=4,
        num_features=3,
        lookback=24,
        max_pred_len=3,
        feature_columns=("Wspd", "Wdir", "Patv_clean_for_input"),
        input_power_column="Patv_clean_for_input",
        input_power_index=2,
        node_ids=node_ids,
        graph_config={
            "type": "physical_knn",
            "location_file": "sdwpf_turb_location_elevation.csv",
            "k": 1,
            "symmetrize": True,
            "self_loops": False,
            "weighting": "binary",
        },
        project_root=tmp_path,
    )


def test_stcn_yaml_uses_tsl_runtime() -> None:
    assert load_model_config_document(ROOT / "configs" / "models" / "stcn.yaml")["runtime"] == {"environment": "tsl"}


@TSL_RUNTIME
def test_stcn_uses_formal_class(tmp_path: Path) -> None:
    model = build_model("stcn", dict(CONFIG), _info(tmp_path))
    assert model.upstream.__class__.__module__.startswith("tsl.nn.models.stgn")
    assert "Time-Series-Library" not in (ROOT / "src" / "models" / "stcn" / "model.py").read_text(encoding="utf-8")


@TSL_RUNTIME
def test_stcn_forward_backward_output_layout_and_sparse_graph_buffers(tmp_path: Path) -> None:
    model = build_model("stcn", dict(CONFIG), _info(tmp_path)).eval()
    x = torch.randn(2, 24, 4, 3)
    output = model(ModelInput(x=x))
    raw = model.upstream(x, model.edge_index, model.edge_weight)
    torch.testing.assert_close(output, raw[..., 0].permute(0, 2, 1))
    output.sum().backward()
    assert tuple(output.shape) == (2, 4, 3)
    assert model.edge_index.dtype == torch.long
    assert model.edge_index.requires_grad is False and model.edge_weight.requires_grad is False
    assert "edge_index" in model.state_dict() and "edge_weight" in model.state_dict()
    assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)


@TSL_RUNTIME
def test_stcn_requires_complete_aligned_graph_and_rejects_unknown_model_fields(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="node_ids"):
        build_model("stcn", dict(CONFIG), _info(tmp_path, node_ids=(1, 2, 3)))
    with pytest.raises(ValueError, match="unknown"):
        build_model("stcn", {**CONFIG, "unknown": 1}, _info(tmp_path / "unknown"))
    missing_graph = _info(tmp_path / "missing-graph")
    missing_graph = DataInfoView(**{**missing_graph.__dict__, "graph_config": None})
    with pytest.raises(ValueError, match="resources.graph"):
        build_model("stcn", dict(CONFIG), missing_graph)
    model = build_model("stcn", dict(CONFIG), _info(tmp_path / "shape"))
    with pytest.raises(ValueError, match="input shape"):
        model(ModelInput(x=torch.randn(1, 23, 4, 3)))

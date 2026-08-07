from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from models.base import DataInfoView, ModelInput
from models.loader import build_model
from resources.graph import (
    build_graph_resource,
    build_physical_knn_adjacency,
    dense_adjacency_to_edges,
    validate_edge_tensors,
)
from runtime.config import load_experiment_config


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "turbine_locations_small.csv"


def _graph_root(tmp_path: Path, contents: str | None = None) -> Path:
    location = tmp_path / "dataset" / "sdwpf_turb_location_elevation.csv"
    location.parent.mkdir(parents=True)
    location.write_text(contents if contents is not None else FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    return tmp_path


def _config() -> dict[str, object]:
    return {
        "type": "physical_knn",
        "location_file": "sdwpf_turb_location_elevation.csv",
        "k": 1,
        "symmetrize": True,
        "self_loops": False,
        "weighting": "binary",
    }


def test_physical_knn_is_deterministic_aligned_symmetric_and_loop_free(tmp_path: Path) -> None:
    root = _graph_root(tmp_path)
    first = build_graph_resource(_config(), node_ids=(3, 1, 4, 2), num_nodes=4, project_root=root)
    second = build_graph_resource(_config(), node_ids=(3, 1, 4, 2), num_nodes=4, project_root=root)
    torch.testing.assert_close(first.adjacency, second.adjacency)
    torch.testing.assert_close(first.adjacency, first.adjacency.T)
    assert not bool(torch.diagonal(first.adjacency).any())
    assert first.edge_index.shape[0] == 2
    assert first.edge_weight.shape[0] == first.edge_index.shape[1]
    assert torch.equal(first.edge_weight, torch.ones_like(first.edge_weight))


def test_graph_rejects_duplicate_missing_and_extra_turbine_ids(tmp_path: Path) -> None:
    duplicate = "TurbID,x,y\n1,0,0\n1,1,0\n2,0,1\n3,1,1\n4,2,2\n"
    root = _graph_root(tmp_path / "duplicate", duplicate)
    with pytest.raises(ValueError, match="duplicate TurbID"):
        build_graph_resource(_config(), node_ids=(1, 2, 3, 4), num_nodes=4, project_root=root)
    missing = "TurbID,x,y\n1,0,0\n2,1,0\n3,0,1\n5,1,1\n"
    root = _graph_root(tmp_path / "missing", missing)
    with pytest.raises(ValueError, match="missing public data TurbID"):
        build_graph_resource(_config(), node_ids=(1, 2, 3, 4), num_nodes=4, project_root=root)
    extra = "TurbID,x,y\n1,0,0\n2,1,0\n3,0,1\n4,1,1\n5,2,2\n"
    root = _graph_root(tmp_path / "extra", extra)
    with pytest.raises(ValueError, match="absent from public data"):
        build_graph_resource(_config(), node_ids=(1, 2, 3, 4), num_nodes=4, project_root=root)


def test_dense_edges_validate_ranges_weights_and_nonempty_graph() -> None:
    adjacency = torch.tensor([[0.0, 1.0], [2.0, 0.0]])
    edge_index, edge_weight = dense_adjacency_to_edges(adjacency, num_nodes=2)
    assert torch.equal(edge_index, torch.tensor([[0, 1], [1, 0]], dtype=torch.long))
    assert torch.equal(edge_weight, torch.tensor([1.0, 2.0]))
    with pytest.raises(ValueError, match="outside"):
        validate_edge_tensors(torch.tensor([[0], [2]], dtype=torch.long), torch.tensor([1.0]), num_nodes=2)
    with pytest.raises(ValueError, match="length"):
        validate_edge_tensors(edge_index, torch.tensor([1.0]), num_nodes=2)
    with pytest.raises(ValueError, match="at least one"):
        dense_adjacency_to_edges(torch.zeros(2, 2), num_nodes=2)


def test_graph_builder_uses_no_target_and_non_graph_model_needs_no_location_file() -> None:
    assert "target" not in inspect.signature(build_physical_knn_adjacency).parameters
    info = DataInfoView(num_nodes=4, num_features=3, lookback=12, max_pred_len=3)
    model = build_model("lstm", {"hidden_dim": 8, "num_layers": 1, "dropout": 0.0}, info)
    output = model(ModelInput(x=torch.randn(1, 12, 4, 3)))
    assert tuple(output.shape) == (1, 4, 3)


def test_default_public_physical_graph_configuration_uses_k5() -> None:
    config = load_experiment_config(ROOT / "configs" / "experiment.yaml")
    assert config.resources["graph"]["k"] == 5

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from engine.model_execution import build_execution_plan
from models.base import DataInfoView, ForecastModel, ModelInput, NodeSharedForecastModel
from models.loader import build_model
from resources.graph import build_graph_resource, validate_edge_tensors
from runtime.config import (
    apply_cli_overrides,
    load_experiment_config,
    load_model_config,
    load_model_config_document,
)

ROOT = Path(__file__).resolve().parents[1]


def _is_environment(name: str) -> bool:
    return Path(sys.executable).parent.name.casefold() == name.casefold()


TSL_RUNTIME = pytest.mark.skipif(
    not _is_environment("env_tsl"),
    reason="requires the formal env_tsl interpreter",
)


FIXTURE = ROOT / "tests" / "fixtures" / "turbine_locations_small.csv"
MODEL_NAMES = ("dcrnn", "agcrn", "graphwavenet", "grugcn", "rnnencgcndec")
GRAPH_NAMES = ("dcrnn", "graphwavenet", "grugcn", "rnnencgcndec")
PUBLIC_MODEL_KEYS = {
    "lookback",
    "pred_len",
    "max_pred_len",
    "horizon",
    "num_nodes",
    "input_dim",
    "batch_size",
    "train_batch_size",
    "val_batch_size",
    "test_batch_size",
    "epochs",
    "loss",
    "learning_rate",
    "seed",
    "amp",
    "graph",
    "k",
    "node_shared_chunk_size",
}


def _contains_key(value: object, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(child, key) for child in value.values())
    if isinstance(value, list):
        return any(_contains_key(child, key) for child in value)
    return False


def _info(
    root: Path,
    *,
    nodes: int = 4,
    lookback: int = 144,
    features: tuple[str, ...] = ("f0", "f1", "f2", "f3"),
    node_ids: tuple[int, ...] | None = None,
    graph: bool = True,
) -> DataInfoView:
    location = root / "dataset" / "sdwpf_turb_location_elevation.csv"
    location.parent.mkdir(parents=True, exist_ok=True)
    location.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    if node_ids is None:
        node_ids = tuple(range(1, nodes + 1))
    return DataInfoView(
        num_nodes=nodes,
        num_features=len(features),
        lookback=lookback,
        max_pred_len=10,
        feature_columns=features,
        input_power_column=features[1],
        input_power_index=1,
        node_ids=node_ids,
        graph_config=(
            {
                "type": "physical_knn",
                "location_file": "sdwpf_turb_location_elevation.csv",
                "k": 1,
                "symmetrize": True,
                "self_loops": False,
                "weighting": "binary",
            }
            if graph
            else None
        ),
        project_root=root if graph else None,
    )


def _build(name: str, info: DataInfoView):
    return build_model(
        name,
        load_model_config(ROOT / "configs" / "models" / f"{name}.yaml"),
        info,
    )


@pytest.fixture(scope="module")
def model_bundle(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("graph-baseline")
    info = _info(root)
    models = {name: _build(name, info) for name in MODEL_NAMES}
    return info, models


def test_graph_batch_yaml_runtime_and_structure_only() -> None:
    for name in MODEL_NAMES:
        document = load_model_config_document(ROOT / "configs" / "models" / f"{name}.yaml")
        assert document["runtime"] == {"environment": "tsl"}
        assert document["model"]
        for key in PUBLIC_MODEL_KEYS:
            assert not _contains_key(document["model"], key), (name, key)


@TSL_RUNTIME
def test_graph_batch_models_are_full_spatiotemporal_and_real_upstream(model_bundle) -> None:
    _, models = model_bundle
    for name, model in models.items():
        assert isinstance(model, ForecastModel), name
        assert not isinstance(model, NodeSharedForecastModel), name
        assert model.execution_mode == "full_spatiotemporal", name
        assert not hasattr(model, "forward_node_chunk"), name
        assert model.upstream.__class__.__module__.startswith("tsl.nn.models.stgn"), name


@TSL_RUNTIME
def test_graph_batch_cpu_forward_backward_and_output_contract(model_bundle) -> None:
    _, models = model_bundle
    torch.manual_seed(2026)
    inputs = ModelInput(x=torch.randn(2, 144, 4, 4))
    for name, model in models.items():
        model.train()
        model.zero_grad(set_to_none=True)
        output = model(inputs)
        assert tuple(output.shape) == (2, 4, 10), name
        assert torch.isfinite(output).all(), name
        output.square().mean().backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        assert any(gradient is not None for gradient in gradients), name
        assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None), name


@TSL_RUNTIME
def test_graph_batch_execution_plan_never_chunks_nodes(model_bundle) -> None:
    _, models = model_bundle
    for name, model in models.items():
        plan = build_execution_plan(model, total_nodes=134, node_shared_chunk_size=32)
        assert plan.execution_mode == "full_spatiotemporal", name
        assert plan.node_chunk_count == 1, name
        assert plan.node_ranges() == ((0, 134),), name


@TSL_RUNTIME
def test_graph_models_use_public_graph_resource_and_persistent_buffers(monkeypatch, tmp_path: Path) -> None:
    for name in GRAPH_NAMES:
        root = tmp_path / name
        info = _info(root)
        module = importlib.import_module(f"models.{name}.model")
        original = module.build_graph_resource
        calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def wrapped(*args, **kwargs):
            calls.append((args, kwargs))
            return original(*args, **kwargs)

        monkeypatch.setattr(module, "build_graph_resource", wrapped)
        model = _build(name, info)
        assert calls, name
        resource = build_graph_resource(
            info.graph_config,
            node_ids=info.node_ids,
            num_nodes=info.num_nodes,
            project_root=info.project_root,
        )
        torch.testing.assert_close(model.edge_index, resource.edge_index)
        torch.testing.assert_close(model.edge_weight, resource.edge_weight)
        assert model.edge_index.dtype == torch.long
        assert not model.edge_index.requires_grad
        assert not model.edge_weight.requires_grad
        validate_edge_tensors(model.edge_index, model.edge_weight, num_nodes=info.num_nodes)
        assert "edge_index" in model.state_dict()
        assert "edge_weight" in model.state_dict()


@TSL_RUNTIME
def test_graph_resource_preserves_public_node_order(tmp_path: Path) -> None:
    info = _info(tmp_path, node_ids=(4, 1, 3, 2))
    model = _build("dcrnn", info)
    expected = build_graph_resource(
        info.graph_config,
        node_ids=info.node_ids,
        num_nodes=info.num_nodes,
        project_root=info.project_root,
    )
    torch.testing.assert_close(model.edge_index, expected.edge_index)
    torch.testing.assert_close(model.edge_weight, expected.edge_weight)


@TSL_RUNTIME
def test_agcrn_keeps_only_official_adaptive_graph(monkeypatch, tmp_path: Path) -> None:
    def fail(*args, **kwargs):
        del args, kwargs
        raise AssertionError("AGCRN must not build the public physical graph")

    graph_module = importlib.import_module("resources.graph")
    monkeypatch.setattr(graph_module, "build_graph_resource", fail)
    info = _info(tmp_path)
    model = _build("agcrn", info)
    assert "edge_index" not in model.state_dict()
    assert "edge_weight" not in model.state_dict()
    assert not hasattr(model, "edge_index")
    assert not hasattr(model, "edge_weight")
    assert model.upstream.agrn.node_emb.n_nodes == info.num_nodes
    assert tuple(model.upstream.agrn.node_emb.emb.shape) == (info.num_nodes, 10)


@TSL_RUNTIME
def test_graphwavenet_keeps_physical_plus_learned_graph(model_bundle) -> None:
    info, models = model_bundle
    model = models["graphwavenet"]
    assert model.model_config["learned_adjacency"] is True
    assert hasattr(model.upstream, "source_embeddings")
    assert hasattr(model.upstream, "target_embeddings")
    assert model.upstream.source_embeddings.n_nodes == info.num_nodes
    assert model.upstream.target_embeddings.n_nodes == info.num_nodes
    assert "edge_index" in model.state_dict()
    assert "edge_weight" in model.state_dict()


@TSL_RUNTIME
def test_graph_batch_preserves_explicit_upstream_structure(model_bundle) -> None:
    _, models = model_bundle
    assert models["dcrnn"].upstream.cache_support is False
    assert models["graphwavenet"].upstream.norms[0].norm_type == "batch"
    assert models["grugcn"].upstream.edge_encoder is None
    assert isinstance(models["rnnencgcndec"].upstream.encoder.rnn, torch.nn.GRU)


@pytest.mark.parametrize("name", MODEL_NAMES)
@TSL_RUNTIME
def test_graph_models_fail_closed_on_missing_or_duplicate_node_ids(name: str, tmp_path: Path) -> None:
    missing = _info(tmp_path / "missing", node_ids=())
    with pytest.raises(ValueError, match="node_ids"):
        _build(name, missing)
    duplicate = _info(tmp_path / "duplicate", node_ids=(1, 2, 3, 3))
    with pytest.raises(ValueError, match="duplicates"):
        _build(name, duplicate)


@TSL_RUNTIME
def test_graph_models_rebuild_from_public_lookback_and_feature_overrides(tmp_path: Path) -> None:
    base = load_experiment_config(ROOT / "configs" / "experiment.yaml")
    resolved = apply_cli_overrides(
        base,
        {"lookback": 120, "feature_columns": ["f0", "Patv_clean_for_input", "f2"]},
        project_root=tmp_path,
    )
    assert resolved.data["lookback"] == 120
    assert resolved.data["feature_columns"] == ["f0", "Patv_clean_for_input", "f2"]
    info = _info(
        tmp_path,
        lookback=int(resolved.data["lookback"]),
        features=tuple(resolved.data["feature_columns"]),
    )
    inputs = ModelInput(x=torch.randn(2, 120, 4, 3))
    for name in MODEL_NAMES:
        model = _build(name, info).eval()
        assert model.lookback == 120, name
        assert model.input_dim == 3, name
        with torch.inference_mode():
            output = model(inputs)
        assert tuple(output.shape) == (2, 4, 10), name
        assert torch.isfinite(output).all(), name


def test_graph_models_have_no_label_or_future_input_boundary() -> None:
    assert "target" not in ModelInput.__dataclass_fields__
    assert "target_mask" not in ModelInput.__dataclass_fields__
    for name in MODEL_NAMES:
        module = importlib.import_module(f"models.{name}.model")
        source = inspect.getsource(module)
        assert "target" not in source.lower(), name
        assert "NodeSharedForecastModel" not in source, name
        assert "node_shared_microbatch" not in source, name


def test_graph_batch_command_reference_is_fresh() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_command_reference.py"), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_public_graph_protocol_remains_k5() -> None:
    config = load_experiment_config(ROOT / "configs" / "experiment.yaml")
    assert config.resources["graph"]["type"] == "physical_knn"
    assert config.resources["graph"]["k"] == 5
    assert config.resources["graph"]["symmetrize"] is True
    assert config.resources["graph"]["self_loops"] is False
    assert config.resources["graph"]["weighting"] == "binary"

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch import nn

from cli.train import _check_checkpoint_compatibility
from engine.model_execution import ExecutionPlan, build_execution_plan
from models.base import ForecastModel, ModelInput, NodeSharedForecastModel
from runtime.config import load_experiment_config, load_model_config


ROOT = Path(__file__).resolve().parents[1]


class _NodeSharedToy(NodeSharedForecastModel):
    def forward_node_chunk(
        self, inputs: ModelInput, node_start: int, node_end: int
    ) -> torch.Tensor:
        values = inputs.x[:, :, node_start:node_end, :].mean(dim=(1, 3))
        return values.unsqueeze(-1).expand(-1, node_end - node_start, 2)


class _BatchNormNodeSharedToy(_NodeSharedToy):
    def __init__(self) -> None:
        super().__init__()
        self.normalization = nn.BatchNorm1d(1)


class _SpatialToy(ForecastModel):
    execution_mode = "full_spatiotemporal"

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        values = inputs.x.mean(dim=(1, 3))
        return values.unsqueeze(-1).expand(-1, inputs.x.shape[2], 2)


def _plan(model: ForecastModel, chunk_size: int = 32) -> ExecutionPlan:
    return build_execution_plan(model, total_nodes=134, node_shared_chunk_size=chunk_size)


def _manifest(
    experiment: dict,
    model: dict,
    model_name: str,
    execution_plan: ExecutionPlan | None = None,
) -> dict:
    manifest = {
        "model": model_name,
        "resolved_config": experiment,
        "model_config": model,
        "epoch": 0,
    }
    if execution_plan is not None:
        manifest["execution_plan"] = execution_plan.as_dict()
    return manifest


def test_stcn_k4_checkpoint_is_rejected_when_public_graph_k_is_5() -> None:
    config = load_experiment_config(ROOT / "configs" / "experiment.yaml")
    model = load_model_config(ROOT / "configs" / "models" / "stcn.yaml")
    old = deepcopy(config.values)
    old["resources"]["graph"]["k"] = 4
    with pytest.raises(ValueError, match=r"resources\.graph\.k"):
        _check_checkpoint_compatibility(
            _manifest(old, model, "stcn"),
            config,
            model,
            ROOT / "old-k4.pt",
            model_name="stcn",
            execution_plan=_plan(_SpatialToy()),
        )


def test_non_graph_checkpoint_is_not_rejected_only_for_graph_k_change() -> None:
    config = load_experiment_config(ROOT / "configs" / "experiment.yaml")
    model = load_model_config(ROOT / "configs" / "models" / "lstm.yaml")
    old = deepcopy(config.values)
    old["resources"]["graph"]["k"] = 4
    _check_checkpoint_compatibility(
        _manifest(old, model, "lstm", _plan(_NodeSharedToy())),
        config,
        model,
        ROOT / "old-k4.pt",
        model_name="lstm",
        execution_plan=_plan(_NodeSharedToy()),
    )


def test_spatial_resume_ignores_changed_chunk_and_missing_legacy_chunk() -> None:
    config = load_experiment_config(ROOT / "configs" / "experiment.yaml")
    model = load_model_config(ROOT / "configs" / "models" / "stcn.yaml")
    current_values = deepcopy(config.values)
    current_values["runtime"]["node_shared_chunk_size"] = 16
    current = type(config)(source=config.source, values=current_values)
    saved = deepcopy(config.values)
    saved["runtime"]["node_shared_chunk_size"] = 32
    _check_checkpoint_compatibility(
        _manifest(saved, model, "stcn"),
        current,
        model,
        ROOT / "spatial-chunk-change.pt",
        model_name="stcn",
        execution_plan=_plan(_SpatialToy(), 16),
    )

    saved_without_chunk = deepcopy(saved)
    saved_without_chunk["runtime"].pop("node_shared_chunk_size")
    _check_checkpoint_compatibility(
        _manifest(saved_without_chunk, model, "stcn"),
        config,
        model,
        ROOT / "spatial-legacy.pt",
        model_name="stcn",
        execution_plan=_plan(_SpatialToy()),
    )


@pytest.mark.parametrize(
    ("model_name", "model_file"),
    [("lstm", "lstm.yaml"), ("crossformer", "crossformer.yaml")],
)
def test_node_shared_resume_requires_matching_chunk_and_new_execution_metadata(
    model_name: str, model_file: str
) -> None:
    config = load_experiment_config(ROOT / "configs" / "experiment.yaml")
    model = load_model_config(ROOT / "configs" / "models" / model_file)
    saved = deepcopy(config.values)
    saved["runtime"]["node_shared_chunk_size"] = 32
    manifest = _manifest(saved, model, model_name, _plan(_NodeSharedToy(), 32))
    current_values = deepcopy(config.values)
    current_values["runtime"]["node_shared_chunk_size"] = 16
    current = type(config)(source=config.source, values=current_values)
    with pytest.raises(ValueError, match=r"runtime\.node_shared_chunk_size"):
        _check_checkpoint_compatibility(
            manifest,
            current,
            model,
            ROOT / "node-shared-chunk-change.pt",
            model_name=model_name,
            execution_plan=_plan(_NodeSharedToy(), 16),
        )

    with pytest.raises(ValueError, match="predates node_shared_microbatch execution"):
        _check_checkpoint_compatibility(
            _manifest(saved, model, model_name),
            config,
            model,
            ROOT / "node-shared-legacy.pt",
            model_name=model_name,
            execution_plan=_plan(_NodeSharedToy()),
        )


def test_batch_norm_node_shared_resume_ignores_chunk_change() -> None:
    config = load_experiment_config(ROOT / "configs" / "experiment.yaml")
    model = load_model_config(ROOT / "configs" / "models" / "lstm.yaml")
    saved = deepcopy(config.values)
    saved["runtime"]["node_shared_chunk_size"] = 32
    current_values = deepcopy(config.values)
    current_values["runtime"]["node_shared_chunk_size"] = 16
    current = type(config)(source=config.source, values=current_values)
    _check_checkpoint_compatibility(
        _manifest(saved, model, "lstm"),
        current,
        model,
        ROOT / "batch-norm-chunk-change.pt",
        model_name="lstm",
        execution_plan=_plan(_BatchNormNodeSharedToy(), 16),
    )


@pytest.mark.parametrize(
    ("model_name", "model_file"),
    [("lstm", "lstm.yaml"), ("crossformer", "crossformer.yaml")],
)
def test_evaluate_only_ignores_node_shared_chunk_change(
    model_name: str, model_file: str
) -> None:
    config = load_experiment_config(ROOT / "configs" / "experiment.yaml")
    model = load_model_config(ROOT / "configs" / "models" / model_file)
    saved = deepcopy(config.values)
    saved["runtime"]["node_shared_chunk_size"] = 32
    manifest = _manifest(saved, model, model_name, _plan(_NodeSharedToy(), 32))
    current_values = deepcopy(config.values)
    current_values["runtime"]["node_shared_chunk_size"] = 16
    current = type(config)(source=config.source, values=current_values)
    _check_checkpoint_compatibility(
        manifest,
        current,
        model,
        ROOT / "evaluate-only-chunk-change.pt",
        model_name=model_name,
        execution_plan=_plan(_NodeSharedToy(), 16),
        for_resume=False,
    )

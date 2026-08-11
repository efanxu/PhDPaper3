from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch
import yaml

from cli.repeatability import _values_close
from cli.train import _check_checkpoint_compatibility
from engine.model_execution import build_execution_plan
from runtime.config import ConfigError, load_experiment_config, load_model_config
from runtime.environments import ResolvedEnvironment, build_worker_environment
from engine.reproducibility import set_seed
from models.base import ModelInput, NodeSharedForecastModel


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "experiment.yaml"


class _CheckpointNodeSharedToy(NodeSharedForecastModel):
    def forward_node_chunk(
        self, inputs: ModelInput, node_start: int, node_end: int
    ) -> torch.Tensor:
        values = inputs.x[:, :, node_start:node_end, :].mean(dim=(1, 3))
        return values.unsqueeze(-1).expand(-1, node_end - node_start, 2)


def _checkpoint_plan():
    return build_execution_plan(
        _CheckpointNodeSharedToy(), total_nodes=134, node_shared_chunk_size=32
    )


def test_controlled_nonstrict_is_the_only_public_mode_and_legacy_field_fails(tmp_path: Path) -> None:
    base = load_experiment_config(CONFIG_PATH)
    assert base.runtime["reproducibility_mode"] == "controlled_nonstrict"

    legacy = base.copy_values()
    legacy["runtime"].pop("reproducibility_mode")
    legacy["runtime"]["deterministic"] = True
    legacy_path = tmp_path / "legacy.yaml"
    legacy_path.write_text(yaml.safe_dump(legacy, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"runtime\.deterministic.*reproducibility_mode=controlled_nonstrict"):
        load_experiment_config(legacy_path)

    unknown = base.copy_values()
    unknown["runtime"]["reproducibility_mode"] = "strict"
    unknown_path = tmp_path / "unknown.yaml"
    unknown_path.write_text(yaml.safe_dump(unknown, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigError, match="must be controlled_nonstrict"):
        load_experiment_config(unknown_path)


def test_seed_records_controlled_nonstrict_and_disables_global_strict(monkeypatch) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "2026")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    details = set_seed(2026, reproducibility_mode="controlled_nonstrict")
    assert details["reproducibility_mode"] == "controlled_nonstrict"
    assert details["global_deterministic_algorithms"] is False
    assert details["cudnn_deterministic"] is True
    assert details["cudnn_benchmark"] is False
    assert details["cuda_matmul_allow_tf32"] is False
    assert details["cudnn_allow_tf32"] is False
    assert torch.are_deterministic_algorithms_enabled() is False


def test_worker_seed_mismatch_fails_closed(monkeypatch) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "7")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with pytest.raises(RuntimeError, match="PYTHONHASHSEED"):
        set_seed(2026, reproducibility_mode="controlled_nonstrict")


def test_parent_worker_environment_contains_resolved_seed() -> None:
    resolved = ResolvedEnvironment(
        environment_id="tslib",
        conda_env="env_tslib",
        python_executable=Path(sys.executable),
        source_roots=(),
        required_imports=(),
        resolution_source="current_interpreter",
    )
    environment = build_worker_environment(
        resolved,
        project_root=ROOT,
        base_environment={},
        resolved_seed=2026,
    )
    assert environment["PYTHONHASHSEED"] == "2026"
    assert environment["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def _checkpoint_manifest(config_values: dict, model_config: dict, *, mode: str | None) -> dict:
    saved = deepcopy(config_values)
    if mode is None:
        saved["runtime"].pop("reproducibility_mode")
    else:
        saved["runtime"]["reproducibility_mode"] = mode
    return {
        "model": "lstm",
        "epoch": 1,
        "resolved_config": saved,
        "model_config": model_config,
        "runtime_state": {"rng": {}, "dataloader_generators": {}},
    }


def test_resume_requires_saved_reproducibility_mode_but_evaluate_only_accepts_legacy() -> None:
    config = load_experiment_config(CONFIG_PATH)
    model_config = load_model_config(ROOT / "configs" / "models" / "lstm.yaml")
    old = _checkpoint_manifest(config.values, model_config, mode=None)
    with pytest.raises(ValueError, match=r"missing runtime\.reproducibility_mode"):
        _check_checkpoint_compatibility(
            old,
            config,
            model_config,
            ROOT / "legacy.pt",
            model=_CheckpointNodeSharedToy(),
            model_name="lstm",
            execution_plan=_checkpoint_plan(),
            for_resume=True,
        )
    _check_checkpoint_compatibility(
        old,
        config,
        model_config,
        ROOT / "legacy.pt",
        model=_CheckpointNodeSharedToy(),
        model_name="lstm",
        execution_plan=_checkpoint_plan(),
        for_resume=False,
    )


def test_repeatability_uses_absolute_and_relative_tolerance_and_rejects_nonfinite() -> None:
    assert _values_close(1.0, 1.00005, atol=1e-5, rtol=1e-4)
    assert not _values_close(1.0, 1.01, atol=1e-5, rtol=1e-4)
    assert not _values_close(float("nan"), float("nan"), atol=1.0, rtol=1.0)
    assert not _values_close(float("inf"), float("inf"), atol=1.0, rtol=1.0)

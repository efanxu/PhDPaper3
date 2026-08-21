from __future__ import annotations

from copy import deepcopy
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn

from data.dataset import ForecastBatch
from cli.repeatability import _values_close
from cli.train import _check_checkpoint_compatibility
from engine.model_execution import build_execution_plan
from runtime.config import ConfigError, load_experiment_config, load_model_config
from runtime.environments import ResolvedEnvironment, build_worker_environment
from engine.reproducibility import set_seed, training_algorithm_context
from engine.trainer import Trainer
from models.base import ForecastModel, ModelInput, NodeSharedForecastModel


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


def test_forecast_model_default_deterministic_cuda_training_capability_is_false() -> None:
    assert ForecastModel().requires_deterministic_cuda_training is False


def test_disabled_capability_does_not_change_deterministic_algorithm_state() -> None:
    model = ForecastModel()
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(False)
        with training_algorithm_context(model, torch.device("cuda")):
            assert torch.are_deterministic_algorithms_enabled() is False
        assert torch.are_deterministic_algorithms_enabled() is False
    finally:
        torch.use_deterministic_algorithms(previous)


def test_relation_capability_temporarily_enables_deterministic_algorithms() -> None:
    model = ForecastModel()
    model.requires_deterministic_cuda_training = True
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(False)
        with training_algorithm_context(model, torch.device("cuda")):
            assert torch.are_deterministic_algorithms_enabled() is True
        assert torch.are_deterministic_algorithms_enabled() is False
    finally:
        torch.use_deterministic_algorithms(previous)


def test_deterministic_algorithm_context_restores_preexisting_true_state() -> None:
    model = ForecastModel()
    model.requires_deterministic_cuda_training = True
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        with training_algorithm_context(model, torch.device("cuda")):
            assert torch.are_deterministic_algorithms_enabled() is True
        assert torch.are_deterministic_algorithms_enabled() is True
    finally:
        torch.use_deterministic_algorithms(previous)


def test_deterministic_algorithm_context_restores_state_after_exception() -> None:
    model = ForecastModel()
    model.requires_deterministic_cuda_training = True
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(False)
        with pytest.raises(RuntimeError, match="scope failure"):
            with training_algorithm_context(model, torch.device("cuda")):
                assert torch.are_deterministic_algorithms_enabled() is True
                raise RuntimeError("scope failure")
        assert torch.are_deterministic_algorithms_enabled() is False
    finally:
        torch.use_deterministic_algorithms(previous)


def test_deterministic_algorithm_context_never_enables_strict_mode_on_cpu() -> None:
    model = ForecastModel()
    model.requires_deterministic_cuda_training = True
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(False)
        with training_algorithm_context(model, torch.device("cpu")):
            assert torch.are_deterministic_algorithms_enabled() is False
        assert torch.are_deterministic_algorithms_enabled() is False
    finally:
        torch.use_deterministic_algorithms(previous)


class _TrainerCapabilityToy(ForecastModel):
    def __init__(self, requires_deterministic_cuda_training: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.requires_deterministic_cuda_training = requires_deterministic_cuda_training

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        return inputs.x.mean(dim=(1, 3)).unsqueeze(-1).expand(-1, inputs.x.shape[2], 1) * self.weight


def _scope_probe_trainer(model: ForecastModel, model_name: str) -> Trainer:
    trainer = object.__new__(Trainer)
    trainer.model = model
    trainer.device = torch.device("cuda")
    trainer.model_name = model_name
    trainer.config = SimpleNamespace(
        training={
            "loss": "masked_score_aligned_hybrid",
            "gradient_clip": 5.0,
            "gradient_clip_norm_type": 2.0,
            "gradient_clip_error_if_nonfinite": False,
            "gradient_clip_foreach": False,
        }
    )
    trainer.execution_plan = None
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    trainer.scaler = None
    trainer.precision = SimpleNamespace(autocast=lambda: nullcontext())
    trainer.train_batch_order = []
    trainer.first_step_loss = None
    trainer.last_step_loss = None
    trainer.update_seconds = []
    return trainer


@pytest.mark.parametrize(
    ("model_name", "capability", "expected"),
    [
        ("ra_ds_pfd_crossformer", False, False),
        ("dummy", True, True),
    ],
)
def test_trainer_scope_uses_model_capability_not_model_name(
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
    capability: bool,
    expected: bool,
) -> None:
    model = _TrainerCapabilityToy(capability)
    trainer = _scope_probe_trainer(model, model_name)
    batch = ForecastBatch(
        x=torch.zeros(1, 1, 1, 1),
        target=torch.zeros(1, 1, 1),
        target_mask=torch.ones(1, 1, 1, dtype=torch.bool),
        starts=torch.tensor([0]),
    )
    observed: list[bool] = []

    def fake_executor(*args, **kwargs):  # type: ignore[no-untyped-def]
        observed.append(torch.are_deterministic_algorithms_enabled())
        return SimpleNamespace(loss=0.0)

    import engine.trainer as trainer_module

    monkeypatch.setattr(trainer_module, "execute_training_backward", fake_executor)
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(False)
        trainer._update([batch])
        assert observed == [expected]
        assert torch.are_deterministic_algorithms_enabled() is False
    finally:
        torch.use_deterministic_algorithms(previous)


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
            model_name="lstm",
            execution_plan=_checkpoint_plan(),
            for_resume=True,
        )
    _check_checkpoint_compatibility(
        old,
        config,
        model_config,
        ROOT / "legacy.pt",
        model_name="lstm",
        execution_plan=_checkpoint_plan(),
        for_resume=False,
    )


def test_repeatability_uses_absolute_and_relative_tolerance_and_rejects_nonfinite() -> None:
    assert _values_close(1.0, 1.00005, atol=1e-5, rtol=1e-4)
    assert not _values_close(1.0, 1.01, atol=1e-5, rtol=1e-4)
    assert not _values_close(float("nan"), float("nan"), atol=1.0, rtol=1.0)
    assert not _values_close(float("inf"), float("inf"), atol=1.0, rtol=1.0)

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from data.dataset import ForecastBatch
from data.normalization import NormalizationStats
from engine.checkpoint import load_checkpoint
from engine.reproducibility import set_seed
from engine.trainer import Trainer
from models.base import DataInfoView, ForecastModel, ModelInput
from runtime.config import ExperimentConfig, load_experiment_config


ROOT = Path(__file__).resolve().parents[1]


class _TinyDataset(Dataset):
    def __init__(self) -> None:
        self.x = torch.arange(24, dtype=torch.float32).reshape(4, 3, 2, 1) / 10.0
        self.target = torch.arange(16, dtype=torch.float32).reshape(4, 2, 2) / 10.0
        self.mask = torch.ones(4, 2, 2, dtype=torch.bool)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int):
        return self.x[index], self.target[index], self.mask[index], index


def _collate(items) -> ForecastBatch:
    x, target, mask, starts = zip(*items)
    return ForecastBatch(torch.stack(x), torch.stack(target), torch.stack(mask), torch.tensor(starts))


class _TinyModel(ForecastModel):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, value: ModelInput) -> torch.Tensor:
        batch, _, nodes, _ = value.x.shape
        prediction = value.x[:, -1, :, 0].unsqueeze(-1).expand(batch, nodes, 2) * self.scale
        return self.validate_output(prediction, batch=batch, nodes=nodes, horizon=2)


def _config() -> ExperimentConfig:
    base = load_experiment_config(ROOT / "configs" / "experiment.yaml")
    values = deepcopy(base.values)
    values["data"].update({"feature_columns": ["f"], "num_nodes": 2, "lookback": 3, "max_pred_len": 2, "eval_horizons": [1, 2]})
    values["training"].update({"epochs": 4, "effective_batch_size": 2, "train_batch_size": 2, "val_batch_size": 2, "test_batch_size": 2, "amp": False, "early_stopping_patience": 20})
    values["runtime"].update({"save_predictions": False})
    return ExperimentConfig(source=base.source, values=values)


def _loaders(seed: int):
    dataset = _TinyDataset()
    return (
        DataLoader(dataset, batch_size=2, shuffle=True, generator=torch.Generator().manual_seed(seed + 1), collate_fn=_collate),
        DataLoader(dataset, batch_size=2, shuffle=False, generator=torch.Generator().manual_seed(seed + 2), collate_fn=_collate),
        DataLoader(dataset, batch_size=2, shuffle=False, generator=torch.Generator().manual_seed(seed + 3), collate_fn=_collate),
    )


def _run(config: ExperimentConfig, output: Path, loaders, *, epochs: int, model: _TinyModel, resume_state=None, start_epoch: int = 1):
    normalization = NormalizationStats(torch.zeros(1), torch.ones(1), 0.0, 1.0)
    trainer = Trainer(
        model,
        config,
        device="cpu",
        model_name="tiny",
        normalization=normalization,
        output_dir=output,
        dataloader_generators={"train": loaders[0].generator, "validation": loaders[1].generator, "test": loaders[2].generator},
    )
    result = trainer.fit(
        loaders[0],
        loaders[1],
        horizons=(1, 2),
        total_nodes=2,
        epochs=epochs,
        start_epoch=start_epoch,
        resume_state=resume_state,
        checkpoint_extra={"resolved_config": config.copy_values(), "model_config": {}, "cli_overrides": {}},
    )
    return trainer, result


def _assert_nested_equal(left, right) -> None:
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(left_item, right_item)
    elif isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
    else:
        assert left == right


def test_epoch_resume_matches_continuous_cpu_training(tmp_path: Path) -> None:
    config = _config()
    seed = 991
    set_seed(seed, deterministic=True)
    continuous_loaders = _loaders(seed)
    continuous_model = _TinyModel()
    continuous_trainer, continuous_result = _run(config, tmp_path / "continuous", continuous_loaders, epochs=4, model=continuous_model)

    set_seed(seed, deterministic=True)
    interrupted_loaders = _loaders(seed)
    interrupted_model = _TinyModel()
    _run(config, tmp_path / "resumed", interrupted_loaders, epochs=2, model=interrupted_model)

    set_seed(seed, deterministic=True)
    resumed_loaders = _loaders(seed)
    resumed_model = _TinyModel()
    resumed_trainer = Trainer(
        resumed_model,
        config,
        device="cpu",
        model_name="tiny",
        normalization=NormalizationStats(torch.zeros(1), torch.ones(1), 0.0, 1.0),
        output_dir=tmp_path / "resumed",
        dataloader_generators={"train": resumed_loaders[0].generator, "validation": resumed_loaders[1].generator, "test": resumed_loaders[2].generator},
    )
    manifest = load_checkpoint(
        tmp_path / "resumed" / "last.pt",
        resumed_model,
        device="cpu",
        optimizer=resumed_trainer.optimizer,
        scheduler=resumed_trainer.scheduler,
        scaler=resumed_trainer.scaler,
        restore_runtime=True,
        dataloader_generators={"train": resumed_loaders[0].generator, "validation": resumed_loaders[1].generator, "test": resumed_loaders[2].generator},
    )
    resumed_result = resumed_trainer.fit(
        resumed_loaders[0],
        resumed_loaders[1],
        horizons=(1, 2),
        total_nodes=2,
        epochs=4,
        start_epoch=int(manifest["epoch"]) + 1,
        resume_state=manifest["runtime_state"]["trainer"],
        checkpoint_extra={"resolved_config": config.copy_values(), "model_config": {}, "cli_overrides": {}},
    )

    for name, value in continuous_model.state_dict().items():
        torch.testing.assert_close(value, resumed_model.state_dict()[name], rtol=0.0, atol=0.0)
    _assert_nested_equal(continuous_trainer.optimizer.state_dict(), resumed_trainer.optimizer.state_dict())
    assert continuous_trainer.scheduler.state_dict() == resumed_trainer.scheduler.state_dict()
    assert continuous_result.history == resumed_result.history
    assert continuous_result.best_epoch == resumed_result.best_epoch
    with torch.inference_mode():
        continuous_prediction = continuous_model(ModelInput(continuous_loaders[2].dataset.x))
        resumed_prediction = resumed_model(ModelInput(resumed_loaders[2].dataset.x))
    torch.testing.assert_close(continuous_prediction, resumed_prediction, rtol=0.0, atol=0.0)

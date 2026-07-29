"""Deterministic DataLoader construction for train/validation/test."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from runtime.config import ExperimentConfig
from .dataset import ForecastDataset, ForecastBatch, collate_forecast_batch
from .loader import DataArrays
from .normalization import NormalizationStats
from .split import SplitBoundaries
from .window import WindowIndex


@dataclass(frozen=True)
class DataLoaders:
    train: DataLoader[ForecastBatch]
    validation: DataLoader[ForecastBatch]
    test: DataLoader[ForecastBatch]

    def generator_states(self) -> dict[str, torch.Tensor]:
        return {
            "train": self.train.generator.get_state(),
            "validation": self.validation.generator.get_state(),
            "test": self.test.generator.get_state(),
        }

    def restore_generator_states(self, states: dict[str, torch.Tensor]) -> None:
        for name, loader in (("train", self.train), ("validation", self.validation), ("test", self.test)):
            if name in states:
                loader.generator.set_state(states[name])


def _worker_init(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    import random

    random.seed(seed)


def _loader(
    dataset: ForecastDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    drop_last: bool,
    pin_memory: bool,
) -> DataLoader[ForecastBatch]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=int(num_workers),
        worker_init_fn=_worker_init,
        drop_last=bool(drop_last),
        pin_memory=bool(pin_memory),
        collate_fn=collate_forecast_batch,
    )


def build_dataloaders(
    arrays: DataArrays,
    normalization: NormalizationStats,
    windows: WindowIndex,
    splits: SplitBoundaries,
    config: ExperimentConfig,
    *,
    seed: int | None = None,
) -> DataLoaders:
    del splits  # split ranges are already represented by the frozen starts.
    data = config.data
    training = config.training
    sampling = config.sampling
    runtime = config.runtime
    base_seed = int(training["seed"] if seed is None else seed)
    datasets = {
        "train": ForecastDataset(
            arrays,
            windows.train_valid,
            normalization,
            lookback=int(data["lookback"]),
            horizon=int(data["max_pred_len"]),
        ),
        "validation": ForecastDataset(
            arrays,
            windows.validation,
            normalization,
            lookback=int(data["lookback"]),
            horizon=int(data["max_pred_len"]),
        ),
        "test": ForecastDataset(
            arrays,
            windows.test,
            normalization,
            lookback=int(data["lookback"]),
            horizon=int(data["max_pred_len"]),
        ),
    }
    offsets = sampling["loader_seed_offsets"]
    return DataLoaders(
        train=_loader(
            datasets["train"],
            batch_size=int(training["train_batch_size"]),
            shuffle=bool(sampling["train_shuffle"]),
            seed=base_seed + int(offsets["train"]),
            num_workers=int(runtime["num_workers"]),
            drop_last=bool(sampling["drop_last"]),
            pin_memory=bool(runtime["pin_memory"]),
        ),
        validation=_loader(
            datasets["validation"],
            batch_size=int(training["val_batch_size"]),
            shuffle=bool(sampling["val_shuffle"]),
            seed=base_seed + int(offsets["validation"]),
            num_workers=int(runtime["num_workers"]),
            drop_last=bool(sampling["drop_last"]),
            pin_memory=bool(runtime["pin_memory"]),
        ),
        test=_loader(
            datasets["test"],
            batch_size=int(training["test_batch_size"]),
            shuffle=bool(sampling["test_shuffle"]),
            seed=base_seed + int(offsets["test"]),
            num_workers=int(runtime["num_workers"]),
            drop_last=bool(sampling["drop_last"]),
            pin_memory=bool(runtime["pin_memory"]),
        ),
    )

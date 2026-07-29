from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from data.dataset import ForecastDataset
from data.loader import load_data
from data.normalization import fit_normalization
from data.split import chronological_split
from data.window import build_window_index
from runtime.config import load_experiment_config


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / "tests" / "fixtures" / "legacy_manifest.json").read_text(encoding="utf-8"))
CONFIG_PATH = ROOT / "configs" / "experiment.yaml"
DATA_AVAILABLE = all(
    (ROOT / "dataset" / name).is_file()
    for name in ("sdwpf_model_input_base.parquet", "sdwpf_eval_target.parquet")
)


@pytest.mark.skipif(not DATA_AVAILABLE, reason="local SDWPF parquet files are not available")
def test_formal_data_pipeline_matches_audited_old_manifest() -> None:
    config = load_experiment_config(CONFIG_PATH)
    arrays, info = load_data(config, project_root=ROOT)
    assert list(arrays.x.shape) == FIXTURE["data"]["shape"]
    assert list(arrays.target.shape) == FIXTURE["data"]["target_shape"]
    assert info.start_timestamp == FIXTURE["data"]["start_timestamp"]
    assert info.end_timestamp == FIXTURE["data"]["end_timestamp"]
    assert list(arrays.node_ids) == FIXTURE["data"]["node_ids"]
    assert list(info.feature_columns) == FIXTURE["features"]
    assert arrays.target_mask.dtype == np.bool_
    assert np.isfinite(arrays.x).all()
    assert np.isfinite(arrays.target[arrays.target_mask]).all()

    splits = chronological_split(len(arrays.timestamps), config)
    assert [splits.train.count, splits.validation.count, splits.test.count] == FIXTURE["split"]["counts"]
    assert splits.train.end == FIXTURE["split"]["train_end"]
    assert splits.validation.end == FIXTURE["split"]["validation_end"]
    windows = build_window_index(
        splits,
        lookback=config.data["lookback"],
        horizon=config.data["max_pred_len"],
        strides={"train": 6, "validation": 3, "test": 1},
        target_mask=arrays.target_mask,
    )
    for name, value in (("train", windows.train), ("train_valid", windows.train_valid), ("validation", windows.validation), ("test", windows.test)):
        expected = FIXTURE["windows"][name]
        assert len(value) == expected["count"]
        assert int(value[0]) == expected["first"]
        assert int(value[-1]) == expected["last"]
    assert windows.invalid_train_count == FIXTURE["windows"]["invalid_train_count"]

    normalization = fit_normalization(arrays, splits.train)
    old = FIXTURE["normalization"]
    np.testing.assert_allclose(normalization.input_mean, old["input_mean"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(normalization.input_scale, old["input_scale"], rtol=1e-5, atol=1e-5)
    assert normalization.target_mean == pytest.approx(old["target_mean"], abs=1e-5)
    assert normalization.target_scale == pytest.approx(old["target_scale"], abs=1e-5)

    dataset = ForecastDataset(
        arrays,
        windows.train_valid[:1],
        normalization,
        lookback=config.data["lookback"],
        horizon=config.data["max_pred_len"],
    )
    x, target, mask, start = dataset[0]
    assert tuple(x.shape) == (144, 134, 16)
    assert tuple(target.shape) == (134, 10)
    assert tuple(mask.shape) == (134, 10)
    assert mask.dtype == torch.bool
    assert start == 144


def test_chronological_split_uses_floor_boundaries_on_tiny_arrays() -> None:
    config = load_experiment_config(CONFIG_PATH)
    boundaries = chronological_split(10, config)
    assert (boundaries.train.end, boundaries.validation.end) == (8, 9)

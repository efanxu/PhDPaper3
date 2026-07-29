from __future__ import annotations

from pathlib import Path

import pytest

from cli.command_schema import build_parser
from runtime.config import (
    ConfigError,
    apply_cli_overrides,
    cli_overrides_as_nested,
    cli_overrides_from_namespace,
    load_experiment_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "experiment.yaml"


def test_no_cli_values_inherit_yaml_defaults() -> None:
    base = load_experiment_config(CONFIG_PATH)
    resolved = apply_cli_overrides(base, {})
    assert resolved.values == base.values
    assert cli_overrides_as_nested({}) == {}


def test_explicit_public_values_map_to_yaml_paths() -> None:
    base = load_experiment_config(CONFIG_PATH)
    resolved = apply_cli_overrides(
        base,
        {
            "lookback": 96,
            "batch_size": 4,
            "epochs": 3,
            "loss": "masked_score_aligned_hybrid",
            "learning_rate": 0.0005,
            "train_ratio": 0.7,
            "val_ratio": 0.2,
            "test_ratio": 0.1,
            "eval_horizons": [3, 6],
            "feature_columns": ["Wspd", "Wdir"],
            "amp": False,
        },
    )
    assert resolved.data["lookback"] == 96
    assert resolved.data["eval_horizons"] == [3, 6]
    assert resolved.data["feature_columns"] == ["Wspd", "Wdir"]
    assert resolved.training["train_batch_size"] == 4
    assert resolved.training["effective_batch_size"] == 4
    assert resolved.training["epochs"] == 3
    assert resolved.training["loss"] == "masked_score_aligned_hybrid"
    assert resolved.training["learning_rate"] == 0.0005
    assert resolved.training["amp"] is False
    assert resolved.split == {"method": "chronological_ratio", "train_ratio": 0.7, "val_ratio": 0.2, "test_ratio": 0.1}


def test_namespace_records_only_explicit_values() -> None:
    args = build_parser().parse_args(
        [
            "train",
            "--model",
            "node_shared_lstm",
            "--batch-size",
            "4",
            "--eval-horizons",
            "3",
            "6",
            "--no-amp",
        ]
    )
    assert cli_overrides_from_namespace(args) == {
        "training.train_batch_size": 4,
        "data.eval_horizons": [3, 6],
        "training.amp": False,
    }
    assert cli_overrides_as_nested(args) == {
        "training": {"train_batch_size": 4, "amp": False},
        "data": {"eval_horizons": [3, 6]},
    }


def test_eval_batch_size_maps_to_validation_and_test_only() -> None:
    args = build_parser().parse_args(
        ["train", "--model", "node_shared_lstm", "--batch-size", "4", "--eval-batch-size", "7"]
    )
    resolved = apply_cli_overrides(load_experiment_config(CONFIG_PATH), args)
    assert resolved.training["train_batch_size"] == 4
    assert resolved.training["val_batch_size"] == 7
    assert resolved.training["test_batch_size"] == 7
    assert resolved.training["effective_batch_size"] == 4


def test_evaluate_exposes_eval_batch_size_but_not_training_batch_size() -> None:
    args = build_parser().parse_args(["evaluate", "--model", "node_shared_lstm", "--eval-batch-size", "5"])
    assert args.eval_batch_size == 5
    with pytest.raises(SystemExit):
        build_parser().parse_args(["evaluate", "--model", "node_shared_lstm", "--batch-size", "5"])


@pytest.mark.parametrize(
    "overrides",
    [
        {"batch_size": 0},
        {"epochs": -1},
        {"learning_rate": 0},
        {"train_ratio": 0.8, "val_ratio": 0.3, "test_ratio": 0.1},
        {"eval_horizons": [3, 20]},
        {"feature_columns": []},
        {"feature_columns": ["Wspd", "Wspd"]},
        {"loss": "unknown"},
    ],
)
def test_invalid_public_overrides_fail_before_execution(overrides: dict) -> None:
    base = load_experiment_config(CONFIG_PATH)
    with pytest.raises(ConfigError):
        apply_cli_overrides(base, overrides)

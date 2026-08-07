from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from cli.train import _check_checkpoint_compatibility
from runtime.config import load_experiment_config, load_model_config


ROOT = Path(__file__).resolve().parents[1]


def _manifest(experiment: dict, model: dict, model_name: str) -> dict:
    return {
        "model": model_name,
        "resolved_config": experiment,
        "model_config": model,
        "epoch": 0,
    }


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
        )


def test_non_graph_checkpoint_is_not_rejected_only_for_graph_k_change() -> None:
    config = load_experiment_config(ROOT / "configs" / "experiment.yaml")
    model = load_model_config(ROOT / "configs" / "models" / "lstm.yaml")
    old = deepcopy(config.values)
    old["resources"]["graph"]["k"] = 4
    _check_checkpoint_compatibility(
        _manifest(old, model, "lstm"),
        config,
        model,
        ROOT / "old-k4.pt",
        model_name="lstm",
    )

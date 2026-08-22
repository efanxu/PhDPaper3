from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from models.base import DataInfoView, ModelInput
from models.loader import build_model
from models.ra_ds_pfd_crossformer.model import RADSPFDCrossformerIA1
from models.ra_ds_pfd_crossformer.p3_ia_propagation import (
    canonical_candidate_names,
    validate_selected_candidates,
)
from models.ra_ds_pfd_crossformer.p3_ia_propagation import IAFixedPropagation
from models.ra_ds_pfd_crossformer.p3_ia_suite import (
    VARIANT_IDS,
    load_p3_ia1_suite,
    resolve_p3_ia1_variants,
)


ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = ROOT / "configs" / "experiments" / "ra_ds_pfd_p3_ia1.yaml"
FEATURE_COLUMNS = (
    "Wspd",
    "Wdir",
    "Etmp",
    "Itmp",
    "Ndir",
    "Pab1",
    "Pab2",
    "Pab3",
    "Prtv",
    "T2m",
    "Sp",
    "RelH",
    "Wspd_w",
    "Wdir_w",
    "Tp",
    "Patv_clean_for_input",
)


def _info() -> DataInfoView:
    return DataInfoView(
        num_nodes=3,
        num_features=len(FEATURE_COLUMNS),
        lookback=24,
        max_pred_len=3,
        feature_columns=FEATURE_COLUMNS,
        input_power_column="Patv_clean_for_input",
        input_power_index=15,
        node_ids=(1, 2, 3),
        project_root=ROOT,
    )


def _fixture(config: dict[str, object]) -> dict[str, object]:
    resolved = deepcopy(config)
    resolved["relation_resource"] = {
        "file": "tests/fixtures/ra_ds_pfd_relation_small_v1.npz"
    }
    return resolved


def test_fixed_candidate_validation_reuses_the_canonical_26_candidate_bank() -> None:
    names = canonical_candidate_names()
    assert len(names) == 26
    assert names[:2] == ("Wspd.level", "Wspd.diff1")
    assert names[-2:] == (
        "Patv_clean_for_input.level",
        "Patv_clean_for_input.diff1",
    )
    assert validate_selected_candidates(("Wspd.level", "Wspd.diff1")) == (
        "Wspd.level",
        "Wspd.diff1",
    )


@pytest.mark.parametrize(
    "selected",
    [
        (),
        ("Wspd.level", "Wspd.level"),
        ("not_a_feature.level",),
        ("Wspd.diff2",),
        ("Wspd",),
        ("Wspd.diff1.extra",),
        tuple(f"Wspd.level.{index}" for index in range(27)),
    ],
)
def test_fixed_candidate_validation_rejects_invalid_sets(selected: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        validate_selected_candidates(selected)


def test_ia1_suite_resolves_two_fixed_arms_from_frozen_r2() -> None:
    suite = load_p3_ia1_suite(SUITE_PATH)
    resolved = resolve_p3_ia1_variants(SUITE_PATH, project_root=ROOT)

    assert tuple(suite["variants"]) == VARIANT_IDS
    assert set(resolved) == set(VARIANT_IDS)
    assert all(config["pfd_mode"] == "pfd3_ia_fixed" for config in resolved.values())
    assert resolved["IA1_R2_PAIR"]["p3_ia"] == {
        "selection_mode": "fixed",
        "selected_candidates": ["Wspd.level", "Wspd.diff1"],
    }
    assert resolved["IA1_AUTO_K2_PAIR"]["p3_ia"] == {
        "selection_mode": "fixed",
        "selected_candidates": ["Wspd.level", "Patv_clean_for_input.diff1"],
    }

    left = dict(resolved["IA1_R2_PAIR"])
    right = dict(resolved["IA1_AUTO_K2_PAIR"])
    left.pop("pfd_mode")
    right.pop("pfd_mode")
    left_p3_ia = left.pop("p3_ia")
    right_p3_ia = right.pop("p3_ia")
    assert left == right
    left_p3_ia.pop("selected_candidates")
    right_p3_ia.pop("selected_candidates")
    assert left_p3_ia == right_p3_ia == {"selection_mode": "fixed"}


def test_ia1_suite_fails_closed_for_changed_arm_shape() -> None:
    suite = load_p3_ia1_suite(SUITE_PATH)
    invalid = deepcopy(suite)
    invalid["variants"]["IA1_R2_PAIR"]["selected_candidates"] = ["Wspd.level"]
    with pytest.raises(ValueError, match="exactly two fixed arms"):
        resolve_p3_ia1_variants(invalid, project_root=ROOT)


def test_ia1_build_keeps_full_self_view_and_frozen_r2_spatial_contract() -> None:
    resolved = resolve_p3_ia1_variants(SUITE_PATH, project_root=ROOT)["IA1_R2_PAIR"]
    model = build_model("ra_ds_pfd_crossformer", _fixture(resolved), _info())

    assert isinstance(model, RADSPFDCrossformerIA1)
    assert model.execution_mode == "full_spatiotemporal"
    assert model.pfd0 is None
    assert model.p3_propagation is None
    assert isinstance(model.ia_propagation, IAFixedPropagation)
    assert model.ia_propagation.effective_candidate_count == 2
    assert model.backbone.enc_in == len(FEATURE_COLUMNS)
    assert model.backbone.num_nodes == 3
    assert model.model_config["spatial_query_mode"] == "node_pooled"
    assert model.model_config["propagation_encoder_mode"] == "cross_time_then_fusion"
    assert model.model_config["turbine_embedding_mode"] == "temporal_and_relation"
    assert model.model_config["bias_scaling_mode"] == "learnable_per_scale"

    x = torch.randn(1, 24, 3, len(FEATURE_COLUMNS))
    with torch.no_grad():
        output = model(ModelInput(x=x))
        trace = model.forward_canonical_trace(ModelInput(x=x))
    assert tuple(output.shape) == (1, 3, 3)
    assert torch.isfinite(output).all()
    assert tuple(trace.pre_norm.shape) == (1, 3, len(FEATURE_COLUMNS), 2, 64)
    assert tuple(trace.scale0_spatial.shape) == (1, 3, len(FEATURE_COLUMNS), 2, 64)
    assert tuple(trace.scale1_spatial.shape) == (1, 3, len(FEATURE_COLUMNS), 1, 64)

    config_identity = model.canonical_model_config()
    assert config_identity["pfd_mode"] == "pfd3_ia_fixed"
    assert config_identity["p3_ia"] == resolved["p3_ia"]


def test_ia1_model_forward_backward_is_finite() -> None:
    resolved = resolve_p3_ia1_variants(SUITE_PATH, project_root=ROOT)["IA1_AUTO_K2_PAIR"]
    model = build_model("ra_ds_pfd_crossformer", _fixture(resolved), _info()).train()
    x = torch.randn(1, 24, 3, len(FEATURE_COLUMNS), requires_grad=True)
    output = model(ModelInput(x=x))
    assert tuple(output.shape) == (1, 3, 3)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.ia_propagation.parameters()
        if parameter.requires_grad
    )

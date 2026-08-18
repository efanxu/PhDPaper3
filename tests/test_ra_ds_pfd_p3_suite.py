from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from models.base import DataInfoView, ModelInput
from models.loader import build_model
from models.ra_ds_pfd_crossformer.model import (
    RADSPFDCrossformerP2,
    RADSPFDCrossformerP3,
)
from models.ra_ds_pfd_crossformer.pfd0 import CrossTimeThenFusionPFD0Propagation
from models.ra_ds_pfd_crossformer.p3_feature_bank import P3_BASE_FEATURES
from models.ra_ds_pfd_crossformer.p3_suite import (
    BASE_VARIANT,
    load_p3_suite,
    resolve_p3_model_config,
)
from models.ra_ds_pfd_crossformer.r0_r7_suite import resolve_r0_r7_variants


ROOT = Path(__file__).resolve().parents[1]
P3_SUITE_PATH = ROOT / "configs" / "experiments" / "ra_ds_pfd_p3.yaml"
R0_R7_SUITE_PATH = ROOT / "configs" / "experiments" / "ra_ds_pfd_r0_r7.yaml"
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


def _fixture(config: dict[str, object]) -> dict[str, object]:
    resolved = deepcopy(config)
    resolved["relation_resource"] = {
        "file": "tests/fixtures/ra_ds_pfd_relation_small_v1.npz"
    }
    return resolved


def _info() -> DataInfoView:
    return DataInfoView(
        num_nodes=3,
        num_features=16,
        lookback=24,
        max_pred_len=3,
        feature_columns=FEATURE_COLUMNS,
        input_power_column="Patv_clean_for_input",
        input_power_index=15,
        node_ids=(1, 2, 3),
        project_root=ROOT,
    )


def test_p3_resolves_only_from_frozen_r2_and_has_the_frozen_candidate_contract() -> None:
    suite = load_p3_suite(P3_SUITE_PATH)
    resolved = resolve_p3_model_config(P3_SUITE_PATH, project_root=ROOT)
    frozen_r2 = resolve_r0_r7_variants(
        R0_R7_SUITE_PATH,
        project_root=ROOT,
    )[BASE_VARIANT]
    assert suite["base"]["variant"] == "R2"
    assert resolved["pfd_mode"] == "pfd3_global_topk"
    assert resolved["p3"]["top_k"] == 2
    assert resolved["p3"]["selector_temperature"] == 0.1
    assert resolved["p3"]["selector_bisection_iterations"] == 64
    assert resolved["p3"]["candidate_features"] == list(P3_BASE_FEATURES)
    assert resolved["p3"]["candidate_transforms"] == ["level", "diff1"]
    assert len(P3_BASE_FEATURES) * 2 == 26
    for field in set(frozen_r2) | set(resolved):
        if field not in {"pfd_mode", "p3"}:
            assert resolved[field] == frozen_r2[field]
    assert resolved["relation_resource"] == frozen_r2["relation_resource"]
    assert resolved["spatial_edge_chunk_size"] == frozen_r2["spatial_edge_chunk_size"]
    assert tuple(
        resolved[field]
        for field in (
            "spatial_query_mode",
            "propagation_encoder_mode",
            "turbine_embedding_mode",
            "bias_scaling_mode",
        )
    ) == (
        "node_pooled",
        "cross_time_then_fusion",
        "temporal_and_relation",
        "learnable_per_scale",
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda p3: p3["candidate_features"].append("Wspd"), "duplicate"),
        (lambda p3: p3["candidate_features"].remove("Tp"), "missing"),
        (lambda p3: p3["candidate_features"].append("not_a_feature"), "illegal"),
        (lambda p3: p3["candidate_features"].append("Wdir"), "circular direction"),
        (lambda p3: p3.__setitem__("candidate_transforms", ["diff6"]), "unknown operator"),
        (lambda p3: p3.__setitem__("candidate_transforms", ["level", "level"]), "duplicate operator"),
        (lambda p3: p3.__setitem__("candidate_transforms", []), "non-empty"),
        (lambda p3: p3.__setitem__("batch_size", 1), "public experiment parameter"),
    ],
)
def test_p3_suite_rejects_invalid_candidate_or_public_configuration(change, message) -> None:
    suite = load_p3_suite(P3_SUITE_PATH)
    invalid = deepcopy(suite)
    change(invalid["p3"])
    with pytest.raises(ValueError, match=message):
        resolve_p3_model_config(invalid, project_root=ROOT)


@pytest.mark.parametrize("top_k", [0, -1, 27, True, 1.5])
def test_p3_suite_rejects_invalid_top_k(top_k: object) -> None:
    suite = load_p3_suite(P3_SUITE_PATH)
    invalid = deepcopy(suite)
    invalid["p3"]["top_k"] = top_k
    with pytest.raises(ValueError, match="top_k"):
        resolve_p3_model_config(invalid, project_root=ROOT)


def test_p3_suite_supports_level_only_with_dynamic_candidate_count() -> None:
    suite = load_p3_suite(P3_SUITE_PATH)
    level_only = deepcopy(suite)
    level_only["p3"]["candidate_transforms"] = ["level"]
    level_only["p3"]["top_k"] = 13
    resolved = resolve_p3_model_config(level_only, project_root=ROOT)
    assert resolved["p3"]["candidate_transforms"] == ["level"]
    assert resolved["p3"]["top_k"] == 13
    assert len(resolved["p3"]["candidate_features"]) == 13
    model = build_model(
        "ra_ds_pfd_crossformer",
        _fixture(resolved),
        _info(),
    )
    assert isinstance(model, RADSPFDCrossformerP3)
    assert model.p3_propagation is not None
    assert model.p3_propagation.candidate_count == 13
    assert tuple(model.p3_propagation.selector.logits.shape) == (13,)


def test_p3_propagation_inherits_spatial_dropout_not_backbone_dropout() -> None:
    p3_config = resolve_p3_model_config(P3_SUITE_PATH, project_root=ROOT)
    p3_config = _fixture(p3_config)
    p3_config["dropout"] = 0.07
    p3_config["spatial_dropout"] = 0.23
    model = build_model("ra_ds_pfd_crossformer", p3_config, _info())
    assert isinstance(model, RADSPFDCrossformerP3)
    assert model.backbone.dropout == 0.07
    assert model.p3_propagation is not None
    assert model.p3_propagation.spatial_dropout == 0.23
    assert model.p3_propagation.dropout.p == 0.23
    assert model.p3_propagation.scale0_cross_time.dropout.p == 0.23
    assert model.p3_propagation.scale1_cross_time.dropout.p == 0.23


def test_p3_build_keeps_self_full_and_legacy_r2_keeps_pfd0() -> None:
    frozen = resolve_r0_r7_variants(R0_R7_SUITE_PATH, project_root=ROOT)
    r2 = build_model("ra_ds_pfd_crossformer", _fixture(frozen["R2"]), _info())
    assert isinstance(r2, RADSPFDCrossformerP2)
    assert isinstance(r2.pfd0, CrossTimeThenFusionPFD0Propagation)
    assert r2.p3_propagation is None
    assert all(not key.startswith("p3_propagation.") for key in r2.state_dict())

    p3_config = resolve_p3_model_config(P3_SUITE_PATH, project_root=ROOT)
    p3_config["relation_resource"] = {
        "file": "tests/fixtures/ra_ds_pfd_relation_small_v1.npz"
    }
    p3 = build_model("ra_ds_pfd_crossformer", p3_config, _info())
    assert isinstance(p3, RADSPFDCrossformerP3)
    assert p3.pfd0 is None
    assert p3.p3_propagation is not None
    assert p3.input_dim == len(FEATURE_COLUMNS)
    assert p3.backbone.enc_in == len(FEATURE_COLUMNS)
    assert any(key.startswith("p3_propagation.") for key in p3.state_dict())
    assert p3.relation_resource is not None
    assert p3.selection_report()
    assert sum(item["selected"] for item in p3.selection_report()) == 2

    output = p3(ModelInput(torch.randn(1, 24, 3, 16)))
    assert tuple(output.shape) == (1, 3, 3)
    assert torch.isfinite(output).all()

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from engine.model_execution import build_execution_plan
from models.base import DataInfoView, ModelInput, NodeSharedForecastModel
from models.loader import build_model
from models.ra_ds_pfd_crossformer.model import (
    RADSPFDCrossformerP1,
    RADSPFDCrossformerP2,
)
from models.ra_ds_pfd_crossformer.r0_r7_suite import (
    VARIANT_IDS,
    load_r0_r7_suite,
    resolve_r0_r7_variants,
    validate_r0_r7_suite,
)
from runtime.config import load_model_config


ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = ROOT / "configs" / "experiments" / "ra_ds_pfd_r0_r7.yaml"
BASE_MODEL_PATH = ROOT / "configs" / "models" / "ra_ds_pfd_crossformer.yaml"
AXIS_FIELDS = (
    "spatial_query_mode",
    "propagation_encoder_mode",
    "turbine_embedding_mode",
    "bias_scaling_mode",
)
P2_COMMON_FIELDS = (
    "pfd_mode",
    "spatial_heads",
    "spatial_d_ff",
    "relation_dim",
    "base_turbine_dim",
    "spatial_dropout",
    "gamma_init",
    "relation_resource",
    "spatial_edge_chunk_size",
)
EXPECTED_MATRIX = {
    "R0": {"spatial_disabled": True},
    "R1": {
        "spatial_disabled": False,
        "spatial_query_mode": "per_variable",
        "propagation_encoder_mode": "segment_fusion",
        "turbine_embedding_mode": "relation_only",
        "bias_scaling_mode": "direct",
    },
    "R2": {
        "spatial_disabled": False,
        "spatial_query_mode": "node_pooled",
        "propagation_encoder_mode": "cross_time_then_fusion",
        "turbine_embedding_mode": "temporal_and_relation",
        "bias_scaling_mode": "learnable_per_scale",
    },
    "R3": {
        "spatial_disabled": False,
        "spatial_query_mode": "node_pooled",
        "propagation_encoder_mode": "segment_fusion",
        "turbine_embedding_mode": "relation_only",
        "bias_scaling_mode": "direct",
    },
    "R4": {
        "spatial_disabled": False,
        "spatial_query_mode": "per_variable",
        "propagation_encoder_mode": "cross_time_then_fusion",
        "turbine_embedding_mode": "relation_only",
        "bias_scaling_mode": "direct",
    },
    "R5": {
        "spatial_disabled": False,
        "spatial_query_mode": "per_variable",
        "propagation_encoder_mode": "segment_fusion",
        "turbine_embedding_mode": "temporal_and_relation",
        "bias_scaling_mode": "direct",
    },
    "R6": {
        "spatial_disabled": False,
        "spatial_query_mode": "per_variable",
        "propagation_encoder_mode": "segment_fusion",
        "turbine_embedding_mode": "relation_only",
        "bias_scaling_mode": "learnable_per_scale",
    },
    "R7": {
        "spatial_disabled": False,
        "spatial_query_mode": "node_pooled",
        "propagation_encoder_mode": "cross_time_then_fusion",
        "turbine_embedding_mode": "relation_only",
        "bias_scaling_mode": "direct",
    },
}


def _info(nodes: int = 3) -> DataInfoView:
    return DataInfoView(
        num_nodes=nodes,
        num_features=5,
        lookback=24,
        max_pred_len=3,
        feature_columns=("Patv_clean_for_input", "Prtv", "Wspd", "Pab1", "Wdir"),
        input_power_column="Patv_clean_for_input",
        input_power_index=0,
        node_ids=tuple(range(1, nodes + 1)),
        project_root=ROOT,
    )


def _resolved() -> dict[str, dict[str, object]]:
    return resolve_r0_r7_variants(SUITE_PATH, project_root=ROOT)


def _fixture_config(config: dict[str, object]) -> dict[str, object]:
    result = deepcopy(config)
    result["relation_resource"] = {
        "file": "tests/fixtures/ra_ds_pfd_relation_small_v1.npz"
    }
    return result


def test_exact_variant_set() -> None:
    suite = load_r0_r7_suite(SUITE_PATH)
    assert set(suite["variants"]) == set(VARIANT_IDS)
    assert len(suite["variants"]) == 8
    validate_r0_r7_suite(suite, project_root=ROOT)


def test_resolved_matrix_is_exact() -> None:
    resolved = _resolved()
    assert {variant: {field: resolved[variant][field] for field in EXPECTED_MATRIX[variant]}
            for variant in VARIANT_IDS} == EXPECTED_MATRIX


def test_r0_has_canonical_p1_identity_without_p2_fields() -> None:
    base = load_model_config(BASE_MODEL_PATH)
    resolved = _resolved()
    assert resolved["R0"] == base
    assert resolved["R0"]["spatial_disabled"] is True
    assert not set(P2_COMMON_FIELDS).intersection(resolved["R0"])
    assert not set(AXIS_FIELDS).intersection(resolved["R0"])


def test_bridge_isolation_is_checked_on_resolved_configs() -> None:
    resolved = _resolved()

    def differences(left: str, right: str) -> set[str]:
        return {
            field
            for field in set(resolved[left]) | set(resolved[right])
            if resolved[left].get(field) != resolved[right].get(field)
        }

    assert differences("R3", "R1") == {"spatial_query_mode"}
    assert differences("R4", "R1") == {"propagation_encoder_mode"}
    assert differences("R5", "R1") == {"turbine_embedding_mode"}
    assert differences("R6", "R1") == {"bias_scaling_mode"}
    assert differences("R7", "R1") == {
        "spatial_query_mode",
        "propagation_encoder_mode",
    }
    assert differences("R2", "R1") == set(AXIS_FIELDS)


def test_shared_p2_invariants_are_exact() -> None:
    suite = load_r0_r7_suite(SUITE_PATH)
    resolved = _resolved()
    base = load_model_config(BASE_MODEL_PATH)
    common = suite["p2_common"]

    for variant in VARIANT_IDS[1:]:
        for field in base:
            if field != "spatial_disabled":
                assert resolved[variant][field] == base[field]
        for field in P2_COMMON_FIELDS:
            assert resolved[variant][field] == common[field]
    assert {resolved[variant]["relation_resource"]["file"] for variant in VARIANT_IDS[1:]} == {
        common["relation_resource"]["file"]
    }
    assert {resolved[variant]["spatial_edge_chunk_size"] for variant in VARIANT_IDS[1:]} == {
        common["spatial_edge_chunk_size"]
    }


def test_public_experiment_fields_fail_closed() -> None:
    suite = load_r0_r7_suite(SUITE_PATH)
    invalid = deepcopy(suite)
    invalid["p2_common"]["batch_size"] = 1
    with pytest.raises(ValueError, match="unsupported field"):
        resolve_r0_r7_variants(invalid, project_root=ROOT)


def test_r0_execution_semantics() -> None:
    model = build_model("ra_ds_pfd_crossformer", _resolved()["R0"], _info(5))
    assert isinstance(model, RADSPFDCrossformerP1)
    assert isinstance(model, NodeSharedForecastModel)
    plan = build_execution_plan(model, total_nodes=5, node_shared_chunk_size=2)
    assert plan.execution_mode == "node_shared_microbatch"
    assert plan.node_ranges() == ((0, 2), (2, 4), (4, 5))


def test_r1_to_r7_execution_semantics() -> None:
    resolved = _resolved()
    for variant in VARIANT_IDS[1:]:
        model = build_model(
            "ra_ds_pfd_crossformer",
            _fixture_config(resolved[variant]),
            _info(),
        )
        assert isinstance(model, RADSPFDCrossformerP2)
        assert not isinstance(model, NodeSharedForecastModel)
        plan = build_execution_plan(model, total_nodes=3, node_shared_chunk_size=2)
        assert plan.execution_mode == "full_spatiotemporal"


def test_all_variants_small_cpu_build_forward_backward() -> None:
    resolved = _resolved()
    torch.manual_seed(2026)
    inputs = ModelInput(x=torch.randn(1, 24, 3, 5))
    for variant in VARIANT_IDS:
        config = resolved[variant] if variant == "R0" else _fixture_config(resolved[variant])
        model = build_model("ra_ds_pfd_crossformer", config, _info()).train()
        output = model(inputs)
        loss = output.square().mean()
        loss.backward()
        assert tuple(output.shape) == (1, 3, 3)
        assert torch.isfinite(output).all()
        assert torch.isfinite(loss)
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

from __future__ import annotations

import itertools
from pathlib import Path

import pytest
import torch

from models.base import DataInfoView, ModelInput
from models.loader import build_model
from models.ra_ds_pfd_crossformer.relation_spatial import LocalVariablePool


ROOT = Path(__file__).resolve().parents[1]


def _info() -> DataInfoView:
    return DataInfoView(
        num_nodes=3,
        num_features=5,
        lookback=24,
        max_pred_len=3,
        feature_columns=("Patv_clean_for_input", "Prtv", "Wspd", "Pab1", "Wdir"),
        input_power_column="Patv_clean_for_input",
        input_power_index=0,
        node_ids=(1, 2, 3),
        project_root=ROOT,
    )


def _config(**updates: object) -> dict[str, object]:
    config: dict[str, object] = {
        "d_model": 8,
        "n_heads": 2,
        "d_ff": 16,
        "e_layers": 2,
        "dropout": 0.0,
        "factor": 2,
        "seg_len": 12,
        "win_size": 2,
        "spatial_disabled": False,
        "pfd_mode": "pfd0",
        "spatial_heads": 2,
        "spatial_d_ff": 16,
        "relation_dim": 4,
        "base_turbine_dim": 4,
        "spatial_dropout": 0.0,
        "gamma_init": 0.1,
        "spatial_edge_chunk_size": 2,
        "spatial_query_mode": "per_variable",
        "propagation_encoder_mode": "segment_fusion",
        "turbine_embedding_mode": "relation_only",
        "bias_scaling_mode": "direct",
        "relation_resource": {"file": "tests/fixtures/ra_ds_pfd_relation_small_v1.npz"},
    }
    config.update(updates)
    return config


@pytest.mark.parametrize(
    "spatial_query_mode,propagation_encoder_mode,turbine_embedding_mode,bias_scaling_mode",
    itertools.product(
        ("per_variable", "node_pooled"),
        ("segment_fusion", "cross_time_then_fusion"),
        ("relation_only", "temporal_and_relation"),
        ("direct", "learnable_per_scale"),
    ),
)
def test_all_sixteen_mode_combinations_build_forward_backward(
    spatial_query_mode: str,
    propagation_encoder_mode: str,
    turbine_embedding_mode: str,
    bias_scaling_mode: str,
) -> None:
    torch.manual_seed(2026)
    model = build_model(
        "ra_ds_pfd_crossformer",
        _config(
            spatial_query_mode=spatial_query_mode,
            propagation_encoder_mode=propagation_encoder_mode,
            turbine_embedding_mode=turbine_embedding_mode,
            bias_scaling_mode=bias_scaling_mode,
        ),
        _info(),
    ).train()
    output = model(ModelInput(x=torch.randn(1, 24, 3, 5)))
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


def test_node_pool_formula_and_shapes() -> None:
    torch.manual_seed(7)
    pool = LocalVariablePool(4)
    tokens = torch.randn(2, 3, 5, 2, 4)
    weight, q_node = pool(tokens)
    assert tuple(weight.shape) == (2, 3, 5, 2, 1)
    assert tuple(q_node.shape) == (2, 3, 2, 4)
    torch.testing.assert_close(weight.sum(dim=2), torch.ones_like(weight.sum(dim=2)))


def test_temporal_identity_is_shared_and_bias_lambdas_are_per_scale() -> None:
    model = build_model(
        "ra_ds_pfd_crossformer",
        _config(turbine_embedding_mode="temporal_and_relation", bias_scaling_mode="learnable_per_scale"),
        _info(),
    )
    names = set(dict(model.named_parameters()))
    assert "turbine_identity.base_turbine_embedding" in names
    assert "turbine_identity.temporal_projection.weight" in names
    assert "turbine_identity.relation_projection.weight" in names
    assert not any(name.endswith("relation_embedding") for name in names)
    provider = model.relation_bias_provider
    assert provider is not None
    assert tuple(provider.lambda_edge.shape) == (2, 2)
    assert tuple(provider.lambda_relation.shape) == (2, 2)
    torch.testing.assert_close(provider.lambda_edge, torch.full((2, 2), 0.05))
    torch.testing.assert_close(provider.lambda_relation, torch.full((2, 2), 0.01))

    output = model(ModelInput(x=torch.randn(1, 24, 3, 5)))
    output.square().mean().backward()
    for parameter in (
        model.turbine_identity.temporal_projection.weight,
        model.turbine_identity.relation_projection.weight,
        provider.lambda_edge,
        provider.lambda_relation,
    ):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_cross_time_then_fusion_records_candidate_order() -> None:
    model = build_model(
        "ra_ds_pfd_crossformer",
        _config(propagation_encoder_mode="cross_time_then_fusion"),
        _info(),
    )
    model(ModelInput(x=torch.randn(1, 24, 3, 5)))
    assert model.pfd0.execution_trace == [
        "scale0_candidate_0_embedding",
        "scale0_candidate_0_cross_time",
        "scale0_candidate_1_embedding",
        "scale0_candidate_1_cross_time",
        "scale0_fusion",
        "scale1_candidate_0_merge",
        "scale1_candidate_0_cross_time",
        "scale1_candidate_1_merge",
        "scale1_candidate_1_cross_time",
        "scale1_fusion",
    ]


@pytest.mark.parametrize(
    "field,value",
    [
        ("spatial_query_mode", "unknown"),
        ("propagation_encoder_mode", "unknown"),
        ("turbine_embedding_mode", "unknown"),
        ("bias_scaling_mode", "unknown"),
    ],
)
def test_architecture_axes_reject_unknown_values(field: str, value: str) -> None:
    with pytest.raises(ValueError, match=field):
        build_model("ra_ds_pfd_crossformer", _config(**{field: value}), _info())


def test_relation_only_has_no_temporal_identity_parameters() -> None:
    model = build_model("ra_ds_pfd_crossformer", _config(), _info())
    names = set(dict(model.named_parameters()))
    assert not any("turbine_identity" in name for name in names)
    assert any(name.endswith("relation_embedding") for name in names)

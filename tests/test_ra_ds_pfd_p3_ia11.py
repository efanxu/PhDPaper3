from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from models.base import DataInfoView, ModelInput
from models.loader import build_model
from models.ra_ds_pfd_crossformer.model import (
    RADSPFDCrossformerIA11,
    RADSPFDCrossformerIA1,
)
from models.ra_ds_pfd_crossformer.p3_ia11_suite import (
    VARIANT_IDS,
    load_p3_ia11_suite,
    resolve_p3_ia11_variants,
)
from models.ra_ds_pfd_crossformer.p3_ia_propagation import (
    canonical_candidate_names,
)
from models.ra_ds_pfd_crossformer.p3_ia_suite import resolve_p3_ia1_variants
from models.ra_ds_pfd_crossformer.p3_ia_temporal import (
    IA11_OPERATOR_TYPES,
    IAIndependentTemporalPropagation,
    IAOperatorAdapterPropagation,
    OperatorResidualAdapter,
    SemanticCandidateIdentity,
)
from models.ra_ds_pfd_crossformer.pfd0 import (
    CanonicalCrossTime,
    CrossTimeThenFusionPFD0Propagation,
)


ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = ROOT / "configs" / "experiments" / "ra_ds_pfd_p3_ia11.yaml"
IA1_SUITE_PATH = ROOT / "configs" / "experiments" / "ra_ds_pfd_p3_ia1.yaml"
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


def _independent(
    selected: tuple[str, ...] = ("Wspd.level", "Wspd.diff1"),
) -> IAIndependentTemporalPropagation:
    return IAIndependentTemporalPropagation(
        feature_columns=FEATURE_COLUMNS,
        selected_candidates=selected,
        lookback=24,
        seg_len=12,
        win_size=2,
        d_model=16,
        n_heads=4,
        d_ff=32,
        factor=10,
        spatial_dropout=0.0,
        source_root=ROOT / "Time-Series-Library",
    )


def _adapter(
    selected: tuple[str, ...] = ("Wspd.level", "Wspd.diff1"),
) -> IAOperatorAdapterPropagation:
    return IAOperatorAdapterPropagation(
        feature_columns=FEATURE_COLUMNS,
        selected_candidates=selected,
        lookback=24,
        seg_len=12,
        win_size=2,
        d_model=16,
        n_heads=4,
        d_ff=32,
        factor=10,
        spatial_dropout=0.0,
        source_root=ROOT / "Time-Series-Library",
    )


def test_ia11_suite_has_exactly_two_frozen_r2_arms() -> None:
    suite = load_p3_ia11_suite(SUITE_PATH)
    resolved = resolve_p3_ia11_variants(SUITE_PATH, project_root=ROOT)

    assert tuple(suite["variants"]) == VARIANT_IDS
    assert set(resolved) == set(VARIANT_IDS)
    assert all(config["pfd_mode"] == "pfd3_ia_temporal" for config in resolved.values())
    assert resolved["IA11_INDEPENDENT_CT"]["p3_ia_temporal"] == {
        "selection_mode": "fixed",
        "selected_candidates": ["Wspd.level", "Wspd.diff1"],
        "temporal_encoder_mode": "independent_cross_time",
    }
    assert resolved["IA11_OPERATOR_ADAPTER"]["p3_ia_temporal"] == {
        "selection_mode": "fixed",
        "selected_candidates": ["Wspd.level", "Wspd.diff1"],
        "temporal_encoder_mode": "operator_adapter_shared_cross_time",
    }
    assert all("p3_ia" not in config for config in resolved.values())


def test_new_resolver_does_not_change_old_ia1_config_identity() -> None:
    old = resolve_p3_ia1_variants(IA1_SUITE_PATH, project_root=ROOT)
    new = resolve_p3_ia11_variants(SUITE_PATH, project_root=ROOT)

    assert set(old["IA1_R2_PAIR"]) == set(old["IA1_AUTO_K2_PAIR"])
    assert all("p3_ia_temporal" not in config for config in old.values())
    for config in new.values():
        frozen = dict(config)
        frozen.pop("pfd_mode")
        frozen.pop("p3_ia_temporal")
        old_r2 = dict(old["IA1_R2_PAIR"])
        old_r2.pop("pfd_mode")
        old_r2.pop("p3_ia")
        assert frozen == old_r2

    old_model = build_model(
        "ra_ds_pfd_crossformer",
        _fixture(old["IA1_R2_PAIR"]),
        _info(),
    )
    assert isinstance(old_model, RADSPFDCrossformerIA1)
    assert all(not key.startswith("ia11_propagation.") for key in old_model.state_dict())


def test_independent_ct_has_two_real_cross_time_instances_per_scale() -> None:
    module = _independent().eval()
    assert module.candidate_bank.candidate_count == 26
    assert module.effective_candidate_count == 2
    assert not hasattr(module, "candidate_identity")
    assert all(not key.startswith("candidate_identity.") for key in module.state_dict())
    assert len(module.candidate_projections) == 2
    assert len(module.scale0_cross_time) == 2
    assert len(module.scale1_cross_time) == 2
    assert module.scale0_cross_time[0] is not module.scale0_cross_time[1]
    assert module.scale1_cross_time[0] is not module.scale1_cross_time[1]
    assert all(isinstance(item, CanonicalCrossTime) for item in module.scale0_cross_time)
    assert all(isinstance(item, CanonicalCrossTime) for item in module.scale1_cross_time)

    calls: list[tuple[str, int, int]] = []
    handles = []
    for scale, encoders in (("scale0", module.scale0_cross_time), ("scale1", module.scale1_cross_time)):
        for index, encoder in enumerate(encoders):
            handles.append(
                encoder.register_forward_pre_hook(
                    lambda _module, args, scale=scale, index=index: calls.append(
                        (scale, index, int(args[0].shape[0]))
                    )
                )
            )
    try:
        with torch.no_grad():
            scale0, scale1 = module(torch.randn(2, 24, 3, len(FEATURE_COLUMNS)))
    finally:
        for handle in handles:
            handle.remove()

    assert calls == [
        ("scale0", 0, 2),
        ("scale0", 1, 2),
        ("scale1", 0, 2),
        ("scale1", 1, 2),
    ]
    assert module.cross_time_candidate_counts == (2, 2)
    assert module.execution_trace["scale0_cross_time_module_count"] == 2
    assert module.execution_trace["scale1_cross_time_module_count"] == 2
    assert tuple(scale0.shape) == (2, 3, 2, 16)
    assert tuple(scale1.shape) == (2, 3, 1, 16)


def test_operator_adapter_has_only_level_and_diff1_adapters_and_shared_cross_time() -> None:
    module = _adapter().eval()
    assert module.candidate_bank.candidate_count == 26
    assert module.effective_candidate_count == 2
    assert isinstance(module.candidate_identity, SemanticCandidateIdentity)
    assert len(module.candidate_projections) == 2
    assert tuple(module.scale0_operator_adapters) == IA11_OPERATOR_TYPES
    assert tuple(module.scale1_operator_adapters) == IA11_OPERATOR_TYPES
    assert len(module.scale0_operator_adapters) == 2
    assert len(module.scale1_operator_adapters) == 2
    assert isinstance(module.scale0_cross_time, CanonicalCrossTime)
    assert isinstance(module.scale1_cross_time, CanonicalCrossTime)

    seen_batch_sizes: list[int] = []
    handles = [
        encoder.register_forward_pre_hook(
            lambda _module, args: seen_batch_sizes.append(int(args[0].shape[0]))
        )
        for encoder in (module.scale0_cross_time, module.scale1_cross_time)
    ]
    try:
        with torch.no_grad():
            scale0, scale1 = module(torch.randn(2, 24, 3, len(FEATURE_COLUMNS)))
    finally:
        for handle in handles:
            handle.remove()

    assert seen_batch_sizes == [2 * 2, 2 * 2]
    assert module.cross_time_candidate_counts == (2, 2)
    assert module.execution_trace["operator_types"] == IA11_OPERATOR_TYPES
    assert tuple(scale0.shape) == (2, 3, 2, 16)
    assert tuple(scale1.shape) == (2, 3, 1, 16)


def test_independent_ct_matches_frozen_r2_temporal_propagation_after_weight_mapping() -> None:
    r2 = CrossTimeThenFusionPFD0Propagation(
        lookback=24,
        seg_len=12,
        win_size=2,
        d_model=16,
        n_heads=4,
        d_ff=32,
        factor=10,
        dropout=0.0,
        wspd_index=FEATURE_COLUMNS.index("Wspd"),
        source_root=ROOT / "Time-Series-Library",
    )
    independent = _independent()

    assert sum(parameter.numel() for parameter in r2.parameters()) == sum(
        parameter.numel() for parameter in independent.parameters()
    )
    with torch.no_grad():
        for index in range(2):
            r2_projection = r2.segment_embeddings[index].value_projection.weight
            ia_projection = independent.candidate_projections[index].weight
            assert ia_projection.shape == r2_projection.shape
            ia_projection.copy_(r2_projection)

            r2_position = r2.segment_embeddings[index].position_embedding
            ia_position = independent.candidate_position_embeddings[index]
            assert r2_position.shape == (
                1,
                1,
                1,
                independent.scale0_segments,
                independent.d_model,
            )
            assert ia_position.shape == r2_position[:, :, 0, :, :].shape
            ia_position.copy_(r2_position[:, :, 0, :, :])

    independent.scale0_cross_time[0].load_state_dict(
        r2.scale0_cross_time[0].state_dict(), strict=True
    )
    independent.scale0_cross_time[1].load_state_dict(
        r2.scale0_cross_time[1].state_dict(), strict=True
    )
    independent.scale1_cross_time[0].load_state_dict(
        r2.scale1_cross_time[0].state_dict(), strict=True
    )
    independent.scale1_cross_time[1].load_state_dict(
        r2.scale1_cross_time[1].state_dict(), strict=True
    )
    independent.scale1_merging.load_state_dict(r2.scale1_merging.state_dict(), strict=True)
    independent.scale0_fusion.load_state_dict(r2.wind_fusion.state_dict(), strict=True)
    independent.scale1_fusion.load_state_dict(r2.scale1_wind_fusion.state_dict(), strict=True)

    r2.eval()
    independent.eval()
    assert r2.segment_embeddings[0].dropout.p == independent.dropout.p
    x = torch.randn(2, 24, 3, len(FEATURE_COLUMNS))

    r2_history = r2.candidate_history(x)
    ia_history = independent.candidate_history(x)
    torch.testing.assert_close(r2_history, ia_history, rtol=1e-6, atol=1e-6)

    r2_scale0, r2_scale1 = r2(x)
    ia_scale0, ia_scale1 = independent(x)
    torch.testing.assert_close(r2_scale0, ia_scale0, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(r2_scale1, ia_scale1, rtol=1e-6, atol=1e-6)


def test_semantic_identity_is_base_variable_plus_operator_and_not_slot_order() -> None:
    identity = SemanticCandidateIdentity(16).eval()
    values = identity(("Wspd.level", "Wspd.diff1", "Prtv.level"))
    swapped = identity(("Prtv.level", "Wspd.level"))

    assert not torch.equal(values[0], values[1])
    assert torch.equal(swapped[0], values[2])
    assert torch.equal(swapped[1], values[0])
    assert torch.allclose(
        values[0] - identity.base_variable_embedding.weight[0],
        identity.operator_embedding.weight[0],
    )
    assert torch.allclose(
        values[2] - identity.base_variable_embedding.weight[
            identity.base_variable_names.index("Prtv")
        ],
        identity.operator_embedding.weight[0],
    )

    selected_swapped = _adapter(("Wspd.diff1", "Wspd.level"))
    swapped_ids = selected_swapped.candidate_identity(
        selected_swapped.selected_candidate_names
    )
    normal_ids = selected_swapped.candidate_identity(("Wspd.level", "Wspd.diff1"))
    assert torch.equal(swapped_ids[0], normal_ids[1])
    assert torch.equal(swapped_ids[1], normal_ids[0])


@pytest.mark.parametrize("factory", [_independent, _adapter])
def test_ia11_candidate_history_is_causal_and_selected_only(factory) -> None:
    module = factory().eval()
    x = torch.randn(1, 24, 2, len(FEATURE_COLUMNS))
    changed = x.clone()
    changed[:, -1, :, FEATURE_COLUMNS.index("Wspd")] += 1000.0

    original = module.candidate_history(x)
    updated = module.candidate_history(changed)
    wspd = x[..., FEATURE_COLUMNS.index("Wspd")]
    assert tuple(original.shape) == (1, 24, 2, 2)
    assert torch.equal(original[..., 0], wspd)
    assert torch.equal(original[:, 0, :, 1], torch.zeros_like(wspd[:, 0]))
    assert torch.equal(original[:, 1:, :, 1], wspd[:, 1:] - wspd[:, :-1])
    assert torch.equal(original[:, :-1], updated[:, :-1])


def test_operator_adapter_formula_is_identity_at_zero_gamma_and_can_make_distinct_corrections() -> None:
    level = OperatorResidualAdapter(8)
    diff1 = OperatorResidualAdapter(8)
    h = torch.randn(2, 3, 8)
    with torch.no_grad():
        level.down.weight.zero_()
        level.down.bias.fill_(1.0)
        level.up.weight.fill_(0.05)
        level.up.bias.zero_()
        diff1.down.weight.zero_()
        diff1.down.bias.fill_(-1.0)
        diff1.up.weight.fill_(0.05)
        diff1.up.bias.zero_()
    assert torch.equal(level(h), h)
    with torch.no_grad():
        level.gamma.fill_(1.0)
        diff1.gamma.fill_(1.0)
    level_out = level(h)
    diff1_out = diff1(h)
    assert torch.isfinite(level_out).all() and torch.isfinite(diff1_out).all()
    assert not torch.allclose(level_out, diff1_out)


def test_ia11_forward_backward_reaches_all_temporal_paths() -> None:
    for factory in (_independent, _adapter):
        module = factory().train()
        x = torch.randn(1, 24, 2, len(FEATURE_COLUMNS), requires_grad=True)
        scale0, scale1 = module(x)
        assert torch.isfinite(scale0).all() and torch.isfinite(scale1).all()
        (scale0.square().mean() + scale1.square().mean()).backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in module.parameters()
            if parameter.requires_grad
        )
        if isinstance(module, IAIndependentTemporalPropagation):
            assert all(
                encoder.MLP1[0].weight.grad is not None
                and torch.isfinite(encoder.MLP1[0].weight.grad).all()
                for encoder in (*module.scale0_cross_time, *module.scale1_cross_time)
            )
        else:
            assert all(
                adapter.gamma.grad is not None and torch.isfinite(adapter.gamma.grad).all()
                for adapters in (
                    module.scale0_operator_adapters,
                    module.scale1_operator_adapters,
                )
                for adapter in adapters.values()
            )
            assert module.scale0_cross_time.MLP1[0].weight.grad is not None
            assert module.scale1_cross_time.MLP1[0].weight.grad is not None


def test_ia11_model_dispatch_keeps_frozen_r2_spatial_fields() -> None:
    resolved = resolve_p3_ia11_variants(SUITE_PATH, project_root=ROOT)
    for variant in VARIANT_IDS:
        model = build_model(
            "ra_ds_pfd_crossformer",
            _fixture(resolved[variant]),
            _info(),
        )
        assert isinstance(model, RADSPFDCrossformerIA11)
        assert model.execution_mode == "full_spatiotemporal"
        assert model.pfd0 is None
        assert model.p3_propagation is None
        assert model.ia_propagation is None
        assert model.ia11_propagation is not None
        assert model.backbone.enc_in == len(FEATURE_COLUMNS)
        assert model.model_config["spatial_query_mode"] == "node_pooled"
        assert model.model_config["propagation_encoder_mode"] == "cross_time_then_fusion"
        assert model.model_config["turbine_embedding_mode"] == "temporal_and_relation"
        assert model.model_config["bias_scaling_mode"] == "learnable_per_scale"

        output = model(ModelInput(torch.randn(1, 24, 3, len(FEATURE_COLUMNS))))
        assert tuple(output.shape) == (1, 3, 3)
        assert torch.isfinite(output).all()
        assert model.canonical_model_config()["pfd_mode"] == "pfd3_ia_temporal"


def test_canonical_candidate_bank_remains_26_for_ia11() -> None:
    assert len(canonical_candidate_names()) == 26

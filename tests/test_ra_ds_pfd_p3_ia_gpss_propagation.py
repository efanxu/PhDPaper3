from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from models.ra_ds_pfd_crossformer.p3_feature_bank import (
    P3_BASE_FEATURES,
    P3_CANDIDATE_TRANSFORMS,
    P3CandidateBank,
)
from models.ra_ds_pfd_crossformer.p3_ia_gpss_propagation import (
    IAGPSSCandidateProjectionBank,
    IAGPSSPropagation,
    _hard_forward_soft_backward_gather,
    segmentize_candidate_history,
)
from models.ra_ds_pfd_crossformer.p3_ia_gpss_selector import IAGPSSSelectorOutput
from models.ra_ds_pfd_crossformer.p3_ia_temporal import (
    IA11_OPERATOR_TYPES,
    IAOperatorAdapterPropagation,
)


ROOT = Path(__file__).resolve().parents[1]

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


def _propagation(*, top_k: int = 2, refinement_rounds: int = 1) -> IAGPSSPropagation:
    return IAGPSSPropagation(
        feature_columns=FEATURE_COLUMNS,
        lookback=5,
        seg_len=3,
        win_size=2,
        d_model=4,
        n_heads=2,
        d_ff=8,
        factor=2,
        top_k=top_k,
        selector_temperature=0.5,
        refinement_rounds=refinement_rounds,
        spatial_dropout=0.0,
        source_root=ROOT / "Time-Series-Library",
    )


def _fixed_selection(
    propagation: IAGPSSPropagation,
    selected: tuple[int, ...],
    *,
    initial: tuple[int, ...] | None = None,
    soft: torch.Tensor | None = None,
) -> IAGPSSSelectorOutput:
    initial = selected if initial is None else initial
    hard = torch.zeros(propagation.top_k, propagation.candidate_count)
    for row, index in enumerate(selected):
        hard[row, index] = 1.0
    if soft is None:
        soft = hard.clone()
    st = hard - soft.detach() + soft
    initial_hard = torch.zeros_like(hard)
    for row, index in enumerate(initial):
        initial_hard[row, index] = 1.0
    initial_soft = initial_hard.clone()
    initial_st = initial_hard.clone()
    return IAGPSSSelectorOutput(
        hard_assignment=hard,
        soft_probabilities=soft,
        st_assignment=st,
        initial_hard_assignment=initial_hard,
        initial_soft_probabilities=initial_soft,
        initial_st_assignment=initial_st,
        initial_selected_indices=initial,
        initial_selected_names=tuple(propagation.candidate_names[index] for index in initial),
        selected_indices=selected,
        selected_names=tuple(propagation.candidate_names[index] for index in selected),
        selection_path=tuple(propagation.candidate_names[index] for index in initial),
        selection_path_indices=initial,
        refinement_trace=(),
    )


def _assert_metadata_only(value: object) -> None:
    if isinstance(value, torch.Tensor):
        raise AssertionError("execution trace retained a live tensor")
    if isinstance(value, dict):
        for item in value.values():
            _assert_metadata_only(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _assert_metadata_only(item)


def test_canonical_candidate_projection_bank_has_one_independent_row_per_candidate() -> None:
    bank = IAGPSSCandidateProjectionBank(candidate_count=26, seg_len=3, d_model=4)

    assert tuple(bank.weight.shape) == (26, 4, 3)
    assert bank.weight[0].data_ptr() != bank.weight[1].data_ptr()
    assert P3CandidateBank(feature_columns=P3_BASE_FEATURES).candidate_count == 26
    assert len(P3_BASE_FEATURES) * len(P3_CANDIDATE_TRANSFORMS) == 26


def test_candidate_projection_bank_matches_independent_bias_free_linear_layers() -> None:
    torch.manual_seed(2026)
    bank = IAGPSSCandidateProjectionBank(candidate_count=3, seg_len=4, d_model=5)
    segments = torch.randn(2, 3, 3, 2, 4)

    actual = bank(segments)
    expected = torch.stack(
        [F.linear(segments[:, :, index], bank.weight[index]) for index in range(3)],
        dim=2,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_candidate_segmentization_uses_ia11_left_zero_padding() -> None:
    history = torch.tensor(
        [
            [
                [[1.0, 10.0], [2.0, 20.0]],
                [[3.0, 30.0], [4.0, 40.0]],
                [[5.0, 50.0], [6.0, 60.0]],
                [[7.0, 70.0], [8.0, 80.0]],
                [[9.0, 90.0], [10.0, 100.0]],
            ]
        ]
    )

    segments = segmentize_candidate_history(history, seg_len=3)

    assert tuple(segments.shape) == (1, 2, 2, 2, 3)
    torch.testing.assert_close(
        segments[0, 0, 0],
        torch.tensor([[0.0, 1.0, 3.0], [5.0, 7.0, 9.0]]),
    )
    torch.testing.assert_close(
        segments[0, 1, 0],
        torch.tensor([[0.0, 2.0, 4.0], [6.0, 8.0, 10.0]]),
    )
    torch.testing.assert_close(
        segments[0, 0, 1],
        torch.tensor([[0.0, 10.0, 30.0], [50.0, 70.0, 90.0]]),
    )


def test_value_surrogate_is_exact_hard_forward_and_soft_backward() -> None:
    values = torch.arange(1.0, 5.0).reshape(1, 1, 4, 1, 1).requires_grad_()
    hard = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    soft = torch.tensor(
        [[0.7, 0.2, 0.1, 0.0], [0.0, 0.6, 0.2, 0.2]],
        requires_grad=True,
    )

    selected = _hard_forward_soft_backward_gather(values, hard, soft, candidate_dim=2)
    expected_hard = torch.tensor([[[[[1.0]], [[2.0]]]]])

    assert torch.equal(selected, expected_hard)
    selected.sum().backward()

    torch.testing.assert_close(
        values.grad.flatten(), torch.tensor([0.7, 0.8, 0.3, 0.2])
    )
    assert soft.grad is not None
    assert torch.isfinite(soft.grad).all()


def test_zero_soft_probability_has_no_hidden_projection_gradient() -> None:
    values = torch.randn(1, 1, 4, 1, 1, requires_grad=True)
    hard = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    soft = torch.tensor([[1.0, 0.0, 0.0, 0.0]], requires_grad=True)

    selected = _hard_forward_soft_backward_gather(values, hard, soft, candidate_dim=2)
    selected.square().mean().backward()

    assert values.grad is not None
    assert torch.equal(values.grad[:, :, 1:], torch.zeros_like(values.grad[:, :, 1:]))


def test_propagation_builds_all_26_projections_but_only_k_temporal_tokens() -> None:
    torch.manual_seed(2026)
    propagation = _propagation(top_k=2).eval()
    x = torch.randn(2, 5, 3, len(FEATURE_COLUMNS))

    projected = propagation._project_all_candidates(x)
    scale0, scale1 = propagation(x)

    assert propagation.candidate_bank.candidate_count == 26
    assert propagation.candidate_count == 26
    assert propagation.projected_candidate_count == 26
    assert tuple(projected.shape) == (2, 3, 26, 2, 4)
    assert tuple(scale0.shape) == (2, 3, 2, 4)
    assert tuple(scale1.shape) == (2, 3, 1, 4)
    assert torch.isfinite(scale0).all() and torch.isfinite(scale1).all()
    assert propagation.cross_time_candidate_counts == (2, 2)
    assert propagation.execution_trace["candidate_bank_count"] == 26
    assert propagation.execution_trace["projected_candidate_count"] == 26
    assert propagation.execution_trace["selected_candidate_count"] == 2
    assert propagation.execution_trace["scale0_cross_time_candidate_count"] == 2
    assert propagation.execution_trace["scale1_cross_time_candidate_count"] == 2
    assert propagation.execution_trace["scale0_cross_time_module_count"] == 1
    assert propagation.execution_trace["scale1_cross_time_module_count"] == 1
    assert propagation.execution_trace["operator_types"] == IA11_OPERATOR_TYPES
    _assert_metadata_only(propagation.execution_trace)


def test_selector_and_propagation_semantic_identities_do_not_share_parameters() -> None:
    propagation = _propagation()

    selector_base = propagation.selector.semantic_identity.base_variable_embedding.weight
    propagation_base = propagation.propagation_identity.base_variable_embedding.weight
    selector_operator = propagation.selector.semantic_identity.operator_embedding.weight
    propagation_operator = propagation.propagation_identity.operator_embedding.weight

    assert selector_base is not propagation_base
    assert selector_operator is not propagation_operator
    assert selector_base.data_ptr() != propagation_base.data_ptr()
    assert selector_operator.data_ptr() != propagation_operator.data_ptr()


def test_operator_incidence_is_derived_from_canonical_candidate_transforms() -> None:
    propagation = _propagation()

    assert tuple(propagation.operator_incidence.shape) == (26, 2)
    for index, name in enumerate(propagation.candidate_names):
        operator = name.rsplit(".", 1)[1]
        expected = torch.zeros(2)
        expected[IA11_OPERATOR_TYPES.index(operator)] = 1.0
        torch.testing.assert_close(propagation.operator_incidence[index], expected)


def test_scale_adapters_are_separate_and_production_gamma_starts_at_zero() -> None:
    propagation = _propagation()

    assert propagation.scale0_operator_adapters is not propagation.scale1_operator_adapters
    for operator in IA11_OPERATOR_TYPES:
        assert propagation.scale0_operator_adapters[operator] is not propagation.scale1_operator_adapters[operator]
        assert torch.equal(
            propagation.scale0_operator_adapters[operator].gamma,
            torch.zeros(()),
        )
        assert torch.equal(
            propagation.scale1_operator_adapters[operator].gamma,
            torch.zeros(()),
        )


@pytest.mark.parametrize("top_k", [1, 2, 3])
def test_k_values_have_finite_forward_and_exact_cross_time_candidate_count(top_k: int) -> None:
    propagation = _propagation(top_k=top_k).eval()
    scale0, scale1 = propagation(torch.randn(1, 5, 2, len(FEATURE_COLUMNS)))

    assert tuple(scale0.shape) == (1, 2, 2, 4)
    assert tuple(scale1.shape) == (1, 2, 1, 4)
    assert torch.isfinite(scale0).all() and torch.isfinite(scale1).all()
    assert propagation.execution_trace["selected_candidate_count"] == top_k
    assert propagation.execution_trace["scale0_cross_time_candidate_count"] == top_k
    assert propagation.execution_trace["scale1_cross_time_candidate_count"] == top_k


def test_propagation_level_loss_reaches_selector_all_projection_and_identity_groups() -> None:
    torch.manual_seed(2026)
    propagation = _propagation(top_k=2).train()
    x = torch.randn(1, 5, 2, len(FEATURE_COLUMNS), requires_grad=True)
    selection = propagation.selector()
    scale0, scale1 = propagation._propagate_with_selection(x, selection)

    (scale0.square().mean() + scale1.square().mean()).backward()

    selector_groups = (
        tuple(propagation.selector.semantic_identity.base_variable_embedding.parameters()),
        tuple(propagation.selector.semantic_identity.operator_embedding.parameters()),
        tuple(propagation.selector.unary_scorer.parameters()),
        tuple(propagation.selector.pairwise_interaction_mlp.parameters()),
    )
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for group in selector_groups
        for parameter in group
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for group in selector_groups
        for parameter in group
    )

    projection_gradient = propagation.candidate_projection_bank.weight.grad
    assert projection_gradient is not None
    assert torch.isfinite(projection_gradient).all()
    hard_indices = set(selection.selected_indices)
    soft_positive_unselected = [
        index
        for index in range(propagation.candidate_count)
        if index not in hard_indices and float(selection.soft_probabilities[:, index].sum()) > 0.0
    ]
    assert soft_positive_unselected
    assert any(
        projection_gradient[index].abs().sum() > 0
        for index in soft_positive_unselected
    )

    identity_groups = (
        tuple(propagation.propagation_identity.base_variable_embedding.parameters()),
        tuple(propagation.propagation_identity.operator_embedding.parameters()),
    )
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for group in identity_groups
        for parameter in group
    )
    assert propagation.scale0_cross_time.MLP1[0].weight.grad is not None
    assert propagation.scale1_cross_time.MLP1[0].weight.grad is not None
    assert propagation.scale0_fusion[0].weight.grad is not None
    assert propagation.scale1_fusion[0].weight.grad is not None


def test_propagation_identity_uses_exact_final_rows_and_soft_backward() -> None:
    propagation = _propagation(top_k=2)
    hard = torch.zeros(2, propagation.candidate_count)
    hard[0, 2] = 1.0
    hard[1, 4] = 1.0
    soft = torch.full((2, propagation.candidate_count), 0.0)
    soft[0] = 1.0 / propagation.candidate_count
    soft[1] = 1.0 / propagation.candidate_count
    soft.requires_grad_()
    identities = propagation.propagation_identity(propagation.candidate_names)

    selected = _hard_forward_soft_backward_gather(identities, hard, soft, candidate_dim=0)

    torch.testing.assert_close(selected, identities[[2, 4]])
    selected.square().mean().backward()
    assert soft.grad is not None and torch.isfinite(soft.grad).all()
    assert soft.grad.abs().sum() > 0


def test_operator_st_route_is_hard_forward_and_routes_gradient_to_selector_assignment() -> None:
    propagation = _propagation(top_k=2)
    hard = torch.zeros(2, propagation.candidate_count)
    hard[0, 0] = 1.0
    hard[1, 1] = 1.0
    soft = torch.full((2, propagation.candidate_count), 1.0 / propagation.candidate_count)
    soft.requires_grad_()
    st = hard - soft.detach() + soft
    tokens = torch.ones(1, 1, 2, 1, 4)
    with torch.no_grad():
        level = propagation.scale0_operator_adapters["level"]
        diff1 = propagation.scale0_operator_adapters["diff1"]
        level.down.weight.zero_()
        level.down.bias.fill_(1.0)
        level.up.weight.fill_(0.1)
        level.up.bias.zero_()
        level.gamma.fill_(1.0)
        diff1.down.weight.zero_()
        diff1.down.bias.fill_(-1.0)
        diff1.up.weight.fill_(0.1)
        diff1.up.bias.zero_()
        diff1.gamma.fill_(1.0)

    routed = propagation._apply_operator_adapters(
        tokens,
        propagation.scale0_operator_adapters,
        st,
    )
    expected_gate = hard @ propagation.operator_incidence
    actual_gate = st @ propagation.operator_incidence
    assert torch.equal(actual_gate.detach(), expected_gate)
    assert not torch.allclose(routed[:, :, 0], routed[:, :, 1])
    routed.sum().backward()
    assert soft.grad is not None and torch.isfinite(soft.grad).all()
    assert soft.grad.abs().sum() > 0


def test_final_selector_assignment_is_used_instead_of_initial_assignment() -> None:
    propagation = _propagation(top_k=2).eval()
    x = torch.randn(1, 5, 2, len(FEATURE_COLUMNS))
    selection = _fixed_selection(propagation, (2, 4), initial=(0, 2))

    propagation._propagate_with_selection(x, selection)

    assert propagation.execution_trace["selected_candidate_indices"] == (2, 4)
    assert propagation.execution_trace["selected_candidate_names"] == (
        "Etmp.level",
        "Itmp.level",
    )


def test_fixed_selector_forward_matches_ia11_operator_adapter_foundation() -> None:
    torch.manual_seed(2026)
    propagation = _propagation(top_k=2, refinement_rounds=0)
    ia11 = IAOperatorAdapterPropagation(
        feature_columns=FEATURE_COLUMNS,
        selected_candidates=("Wspd.level", "Wspd.diff1"),
        lookback=5,
        seg_len=3,
        win_size=2,
        d_model=4,
        n_heads=2,
        d_ff=8,
        factor=2,
        spatial_dropout=0.0,
        source_root=ROOT / "Time-Series-Library",
    )
    with torch.no_grad():
        propagation.candidate_projection_bank.weight[0].copy_(
            ia11.candidate_projections[0].weight
        )
        propagation.candidate_projection_bank.weight[1].copy_(
            ia11.candidate_projections[1].weight
        )
        propagation.position_embedding.copy_(ia11.position_embedding)
    propagation.propagation_identity.load_state_dict(ia11.candidate_identity.state_dict())
    for operator in IA11_OPERATOR_TYPES:
        propagation.scale0_operator_adapters[operator].load_state_dict(
            ia11.scale0_operator_adapters[operator].state_dict()
        )
        propagation.scale1_operator_adapters[operator].load_state_dict(
            ia11.scale1_operator_adapters[operator].state_dict()
        )
    propagation.scale0_cross_time.load_state_dict(ia11.scale0_cross_time.state_dict())
    propagation.scale1_merging.load_state_dict(ia11.scale1_merging.state_dict())
    propagation.scale1_cross_time.load_state_dict(ia11.scale1_cross_time.state_dict())
    propagation.scale0_fusion.load_state_dict(ia11.scale0_fusion.state_dict())
    propagation.scale1_fusion.load_state_dict(ia11.scale1_fusion.state_dict())
    propagation.eval()
    ia11.eval()
    x = torch.randn(2, 5, 3, len(FEATURE_COLUMNS))
    fixed = _fixed_selection(propagation, (0, 1))

    gpss_scale0, gpss_scale1 = propagation._propagate_with_selection(x, fixed)
    ia11_scale0, ia11_scale1 = ia11(x)

    torch.testing.assert_close(gpss_scale0, ia11_scale0, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(gpss_scale1, ia11_scale1, rtol=1e-6, atol=1e-6)


def test_eval_repeated_forward_is_deterministic_and_trace_has_no_tensors() -> None:
    propagation = _propagation().eval()
    x = torch.randn(1, 5, 2, len(FEATURE_COLUMNS))

    first = propagation(x)
    first_trace = dict(propagation.execution_trace)
    second = propagation(x)

    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert first_trace == propagation.execution_trace
    _assert_metadata_only(propagation.execution_trace)


@pytest.mark.parametrize("top_k", [0, 27, True, 1.5])
def test_invalid_k_fails_closed(top_k: object) -> None:
    with pytest.raises(ValueError, match="top_k"):
        _propagation(top_k=top_k)  # type: ignore[arg-type]


def test_invalid_shape_and_nan_fail_closed() -> None:
    propagation = _propagation()
    with pytest.raises(ValueError, match="input length"):
        propagation(torch.randn(1, 4, 2, len(FEATURE_COLUMNS)))
    invalid = torch.randn(1, 5, 2, len(FEATURE_COLUMNS))
    invalid[0, 0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="NaN"):
        propagation(invalid)

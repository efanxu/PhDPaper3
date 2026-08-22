from __future__ import annotations

import inspect

import pytest
import torch

from models.ra_ds_pfd_crossformer.p3_feature_bank import (
    P3_BASE_FEATURES,
    P3_CANDIDATE_TRANSFORMS,
    P3CandidateBank,
)
from models.ra_ds_pfd_crossformer.p3_ia_gpss_selector import (
    IAGPSSSelector,
    IAGPSSSemanticCandidateIdentity,
    canonical_candidate_names,
)


def _selector(
    *,
    top_k: int = 2,
    d_model: int = 8,
    temperature: float = 0.5,
    refinement_rounds: int = 0,
) -> IAGPSSSelector:
    return IAGPSSSelector(
        d_model=d_model,
        top_k=top_k,
        temperature=temperature,
        refinement_rounds=refinement_rounds,
    )


def _synergy_selector() -> IAGPSSSelector:
    """Build a deterministic A/B/C fixture with B-C complementarity."""

    selector = IAGPSSSelector(
        ("Wspd.level", "Etmp.level", "Itmp.level"),
        d_model=1,
        top_k=2,
        temperature=1.0,
        refinement_rounds=1,
    )
    with torch.no_grad():
        selector.semantic_identity.base_variable_embedding.weight.zero_()
        selector.semantic_identity.base_variable_embedding.weight[:3, 0] = torch.tensor(
            [1.0, 2.0, 3.0]
        )
        selector.semantic_identity.operator_embedding.weight.zero_()

        # Unary: A is strongest, followed by B and C.
        selector.unary_scorer.weight.fill_(-1.0)
        selector.unary_scorer.bias.zero_()

        # For scalar identities e, this first layer is
        # GELU(-sum(e_i,e_j) + 2*product(e_i,e_j) - 3).
        # It makes B-C strongly positive and A-B/A-C negative.
        selector.pairwise_interaction_mlp[0].weight.copy_(
            torch.tensor([[-1.0, 2.0, 0.0]])
        )
        selector.pairwise_interaction_mlp[0].bias.fill_(-3.0)
        selector.pairwise_interaction_mlp[2].weight.fill_(1.0)
        selector.pairwise_interaction_mlp[2].bias.zero_()
    return selector


def test_canonical_candidate_order_reuses_p3_candidate_bank() -> None:
    bank = P3CandidateBank(feature_columns=P3_BASE_FEATURES)
    expected = tuple(
        f"{base}.{operator}"
        for base in P3_BASE_FEATURES
        for operator in P3_CANDIDATE_TRANSFORMS
    )
    assert canonical_candidate_names() == bank.candidate_names
    assert canonical_candidate_names() == expected
    assert IAGPSSSelector(d_model=4).candidate_names == expected


def test_default_candidate_count_is_26_without_algorithm_constant() -> None:
    selector = IAGPSSSelector(d_model=4)
    assert selector.candidate_count == 26
    assert selector.candidate_count == len(P3_BASE_FEATURES) * len(
        P3_CANDIDATE_TRANSFORMS
    )


def test_semantic_identity_is_base_plus_operator_and_has_no_slot_embedding() -> None:
    selector = IAGPSSSelector(d_model=5, refinement_rounds=0)
    assert isinstance(selector.semantic_identity, IAGPSSSemanticCandidateIdentity)
    assert tuple(selector.semantic_identity.base_variable_embedding.weight.shape) == (13, 5)
    assert tuple(selector.semantic_identity.operator_embedding.weight.shape) == (2, 5)
    assert not hasattr(selector, "slot_embedding")
    assert not any(
        "slot" in name.lower() or "selection_order" in name.lower()
        for name, _ in selector.named_parameters()
    )

    embeddings = selector.semantic_embeddings()
    level = embeddings[selector.candidate_names.index("Wspd.level")]
    base = selector.semantic_identity.base_variable_embedding.weight[0]
    operator = selector.semantic_identity.operator_embedding.weight[0]
    torch.testing.assert_close(level, base + operator)


def test_pairwise_interaction_is_exactly_symmetric() -> None:
    selector = _selector()
    embeddings = selector.semantic_embeddings()
    left_right = selector.pairwise_interaction(embeddings[0], embeddings[3])
    right_left = selector.pairwise_interaction(embeddings[3], embeddings[0])
    assert torch.equal(left_right, right_left)

    interactions = selector.pairwise_interaction_matrix(embeddings)
    assert torch.equal(interactions, interactions.transpose(0, 1))


def test_set_utility_is_invariant_to_selected_set_permutation() -> None:
    selector = _selector()
    forward = selector.set_utility((0, 3, 6))
    reverse = selector.set_utility((6, 0, 3))
    by_name = selector.set_utility(
        (
            selector.candidate_names[6],
            selector.candidate_names[0],
            selector.candidate_names[3],
        )
    )
    assert torch.equal(forward, reverse)
    assert torch.equal(forward, by_name)


def test_empty_conditional_marginal_is_the_unary_score() -> None:
    selector = _selector()
    torch.testing.assert_close(
        selector.conditional_marginal_scores(), selector.unary_scores()
    )
    torch.testing.assert_close(
        selector.conditional_marginal_scores(torch.empty(0, selector.M)),
        selector.unary_scores(),
    )


def test_conditional_score_is_the_set_utility_difference_and_changes_with_context() -> None:
    selector = _synergy_selector()
    q_empty = selector.conditional_marginal_scores()
    q_after_a = selector.conditional_marginal_scores(selected_indices=(0,))
    q_after_c = selector.conditional_marginal_scores(selected_indices=(2,))

    assert not torch.equal(q_after_c[1], q_after_a[1])
    torch.testing.assert_close(
        q_after_c[1], selector.set_utility((1, 2)) - selector.set_utility((2,))
    )
    torch.testing.assert_close(q_empty, selector.unary_scores())


def test_previous_st_assignment_has_gradient_path_to_later_conditional_score() -> None:
    selector = _selector(top_k=2, refinement_rounds=0)
    output = selector()
    earlier = output.selection_st_assignment_rows[0]
    later = output.selection_st_assignment_rows[1]
    earlier.retain_grad()
    weights = torch.arange(selector.M, dtype=later.dtype)
    (later * weights).sum().backward()
    assert earlier.grad is not None
    assert torch.isfinite(earlier.grad).all()
    assert earlier.grad.abs().sum() > 0


@pytest.mark.parametrize("top_k", [1, 2, 3])
def test_k_values_produce_exact_unique_hard_selection(top_k: int) -> None:
    output = _selector(top_k=top_k)()
    assert tuple(output.hard_assignment.shape) == (top_k, 26)
    assert tuple(output.soft_probabilities.shape) == (top_k, 26)
    assert tuple(output.st_assignment.shape) == (top_k, 26)
    assert len(output.selected_indices) == top_k
    assert len(set(output.selected_indices)) == top_k
    assert len(set(output.initial_selected_indices)) == top_k
    assert torch.equal(
        output.hard_assignment.sum(dim=1),
        torch.ones(top_k, dtype=output.hard_assignment.dtype),
    )
    assert torch.allclose(
        output.soft_probabilities.sum(dim=1),
        torch.ones(top_k, dtype=output.soft_probabilities.dtype),
    )


def test_masked_candidates_cannot_be_selected_twice() -> None:
    output = _selector(top_k=3, refinement_rounds=1)()
    assert len(set(output.initial_selected_indices)) == 3
    assert len(set(output.selected_indices)) == 3
    assert torch.all(output.hard_assignment.sum(dim=0) <= 1)


def test_st_forward_is_exactly_hard_one_hot() -> None:
    output = _selector(top_k=3)()
    assert torch.equal(output.st_assignment, output.hard_assignment)
    assert torch.all((output.st_assignment == 0) | (output.st_assignment == 1))
    assert torch.isfinite(output.st_assignment).all()
    assert torch.isfinite(output.soft_probabilities).all()


def test_st_backward_reaches_all_selector_parameter_groups_with_finite_gradients() -> None:
    selector = _selector(top_k=2, refinement_rounds=1).train()
    output = selector()
    weights = torch.arange(selector.M, dtype=output.st_assignment.dtype)
    loss = (output.st_assignment * weights).sum()
    loss.backward()

    groups = (
        tuple(selector.semantic_identity.base_variable_embedding.parameters()),
        tuple(selector.semantic_identity.operator_embedding.parameters()),
        tuple(selector.unary_scorer.parameters()),
        tuple(selector.pairwise_interaction_mlp.parameters()),
    )
    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        for group in groups
        for parameter in group
    )


def test_exact_ties_use_the_first_canonical_bank_index() -> None:
    selector = _selector(top_k=3, refinement_rounds=0)
    with torch.no_grad():
        for parameter in selector.parameters():
            parameter.zero_()
    output = selector()
    assert output.initial_selected_indices == (0, 1, 2)
    assert output.selected_indices == (0, 1, 2)


def test_refinement_replaces_an_existing_slot() -> None:
    output = _synergy_selector()()
    assert output.initial_selected_indices == (0, 1)
    assert output.selected_indices == (1, 2)
    assert any(
        item["previous_candidate"] == "Wspd.level"
        and item["new_candidate"] == "Itmp.level"
        for item in output.refinement_trace
    )


def test_synthetic_synergy_prefers_bc_after_refinement() -> None:
    selector = _synergy_selector()
    unary = selector.unary_scores()
    interactions = selector.pairwise_interaction_matrix()
    assert unary[0] > unary[1] > unary[2]
    assert interactions[1, 2] > 0
    assert interactions[0, 1] < 0
    assert interactions[0, 2] < 0
    assert selector.set_utility((1, 2)) > selector.set_utility((0, 1))
    assert selector.set_utility((1, 2)) > selector.set_utility((0, 2))
    output = selector()
    assert output.initial_selected_indices == (0, 1)
    assert output.selected_indices == (1, 2)


def test_final_set_is_canonical_sorted_but_path_and_trace_keep_decision_order() -> None:
    output = _synergy_selector()()
    assert output.selection_path == ("Wspd.level", "Etmp.level")
    assert output.selection_path_indices == (0, 1)
    assert output.selected_indices == tuple(sorted(output.selected_indices))
    assert output.selected_names == ("Etmp.level", "Itmp.level")
    assert output.refinement_trace[0]["slot"] == 0
    assert output.refinement_trace[0]["condition_on"] == ("Etmp.level",)


def test_eval_repeated_calls_are_exactly_deterministic() -> None:
    selector = _selector(top_k=3, refinement_rounds=1).eval()
    first = selector()
    second = selector()
    for left, right in (
        (first.hard_assignment, second.hard_assignment),
        (first.soft_probabilities, second.soft_probabilities),
        (first.st_assignment, second.st_assignment),
    ):
        assert torch.equal(left, right)
    assert first.selected_indices == second.selected_indices
    assert first.selection_path == second.selection_path
    assert first.refinement_trace == second.refinement_trace


def test_same_seed_initialization_has_exact_same_output() -> None:
    torch.manual_seed(2026)
    first_selector = _selector(top_k=2, refinement_rounds=1)
    torch.manual_seed(2026)
    second_selector = _selector(top_k=2, refinement_rounds=1)
    first = first_selector()
    second = second_selector()
    assert all(
        torch.equal(first_parameter, second_parameter)
        for first_parameter, second_parameter in zip(
            first_selector.parameters(), second_selector.parameters(), strict=True
        )
    )
    assert torch.equal(first.hard_assignment, second.hard_assignment)
    assert torch.equal(first.soft_probabilities, second.soft_probabilities)
    assert torch.equal(first.st_assignment, second.st_assignment)
    assert first.selected_indices == second.selected_indices


@pytest.mark.parametrize("top_k", [0, -1, 27, True, 1.5])
def test_invalid_k_fails_closed(top_k: object) -> None:
    with pytest.raises(ValueError, match="top_k"):
        IAGPSSSelector(d_model=4, top_k=top_k)  # type: ignore[arg-type]


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf"), True, "0.5"])
def test_invalid_temperature_fails_closed(temperature: object) -> None:
    with pytest.raises(ValueError, match="temperature"):
        IAGPSSSelector(d_model=4, temperature=temperature)  # type: ignore[arg-type]


@pytest.mark.parametrize("rounds", [-1, True, 1.5, "1"])
def test_invalid_refinement_rounds_fail_closed(rounds: object) -> None:
    with pytest.raises(ValueError, match="refinement_rounds"):
        IAGPSSSelector(d_model=4, refinement_rounds=rounds)  # type: ignore[arg-type]


def test_nonfinite_parameter_fails_closed_before_selection() -> None:
    selector = _selector()
    with torch.no_grad():
        selector.semantic_identity.base_variable_embedding.weight[0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="parameter"):
        selector()


def test_selector_forward_has_no_dynamic_context_api() -> None:
    signature = inspect.signature(IAGPSSSelector.forward)
    assert tuple(signature.parameters) == ("self",)
    selector = _selector()
    assert selector.forward.__self__ is selector
    assert not any(
        name in {"x", "inputs", "batch", "node", "time", "edge", "target", "mask"}
        for name in signature.parameters
    )

"""IA-GPSS v1 global interaction-aware set selector.

This module deliberately stops at the global discrete-set assignment seam.  It
does not read a model input, build a propagation bank, or implement the later
``A_ST @ Z`` projection gather.  The three assignment tensors in
``IAGPSSSelectorOutput`` are the interface that the IA-2B propagation design
will consume.

The selector is global: its only inputs are its own parameters and the
canonical candidate identity.  ``Interaction-Aware`` therefore refers to
complementarity between candidate identities in the selected set, not to
sample-conditioned attention.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .p3_feature_bank import (
    P3_BASE_FEATURES,
    P3_CANDIDATE_TRANSFORMS,
    P3CandidateBank,
)


DEFAULT_IA_GPSS_D_MODEL = 64
DEFAULT_IA_GPSS_TOP_K = 2
DEFAULT_IA_GPSS_TEMPERATURE = 1.0
DEFAULT_IA_GPSS_REFINEMENT_ROUNDS = 1
IA_GPSS_SELECTOR_TYPE = "interaction_aware_global_pairwise_set_utility"


def canonical_candidate_names() -> tuple[str, ...]:
    """Return names in the existing P3 Candidate Bank order.

    ``P3CandidateBank`` owns the candidate construction semantics.  Keeping
    this small helper as a call through that module avoids a second candidate
    list in IA-GPSS.
    """

    return P3CandidateBank(feature_columns=P3_BASE_FEATURES).candidate_names


def _validate_d_model(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("IA-GPSS d_model must be a positive integer")
    return int(value)


def validate_top_k(top_k: Any, candidate_count: int) -> int:
    """Validate the exact cardinality against the resolved candidate count."""

    if not isinstance(candidate_count, int) or isinstance(candidate_count, bool):
        raise ValueError("IA-GPSS candidate_count must be an integer")
    if candidate_count < 1:
        raise ValueError("IA-GPSS candidate_count must be positive")
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise ValueError("IA-GPSS top_k must be an integer")
    if not 1 <= top_k <= candidate_count:
        raise ValueError("IA-GPSS top_k must satisfy 1 <= top_k <= candidate_count")
    return int(top_k)


def validate_temperature(value: Any) -> float:
    """Validate the positive softmax temperature."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("IA-GPSS temperature must be finite and positive")
    temperature = float(value)
    if not isfinite(temperature) or temperature <= 0.0:
        raise ValueError("IA-GPSS temperature must be finite and positive")
    return temperature


def validate_refinement_rounds(value: Any) -> int:
    """Validate the non-negative number of deterministic refinement passes."""

    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("IA-GPSS refinement_rounds must be an integer >= 0")
    return int(value)


def _validate_candidate_names(value: Any, *, field: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"IA-GPSS {field} must be a non-empty ordered sequence")
    names = tuple(value)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError(f"IA-GPSS {field} must contain non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError(f"IA-GPSS {field} must contain unique candidate names")
    return names


def _parse_candidate_name(value: str) -> tuple[str, str]:
    if value.count(".") != 1:
        raise ValueError(
            "IA-GPSS candidate names must use '<base_feature>.<operator>'"
        )
    feature, operator = value.split(".")
    if feature not in P3_BASE_FEATURES:
        raise ValueError(f"IA-GPSS candidate has unknown base feature: {feature}")
    if operator not in P3_CANDIDATE_TRANSFORMS:
        raise ValueError(f"IA-GPSS candidate has unknown operator: {operator}")
    return feature, operator


class IAGPSSSemanticCandidateIdentity(nn.Module):
    """Encode candidate meaning as base-variable plus operator embeddings.

    This identity has the same semantic definition as the existing IA11
    operator-adapter identity, but its parameters are intentionally local to
    IA-GPSS.  There is no slot, selection-order, or ``Embedding(K, D)``
    parameter here.
    """

    def __init__(
        self,
        d_model: int,
        candidate_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        self.d_model = _validate_d_model(d_model)
        self.base_variable_names = tuple(P3_BASE_FEATURES)
        self.operator_names = tuple(P3_CANDIDATE_TRANSFORMS)
        self._base_indices = {
            name: index for index, name in enumerate(self.base_variable_names)
        }
        self._operator_indices = {
            name: index for index, name in enumerate(self.operator_names)
        }
        names = (
            canonical_candidate_names()
            if candidate_names is None
            else _validate_candidate_names(candidate_names, field="candidate_names")
        )
        for name in names:
            _parse_candidate_name(name)
        self.candidate_names = names
        self.base_variable_embedding = nn.Embedding(
            len(self.base_variable_names), self.d_model
        )
        self.operator_embedding = nn.Embedding(len(self.operator_names), self.d_model)

    def forward(self, candidate_names: Sequence[str] | None = None) -> torch.Tensor:
        names = (
            self.candidate_names
            if candidate_names is None
            else _validate_candidate_names(candidate_names, field="candidate_names")
        )
        base_indices: list[int] = []
        operator_indices: list[int] = []
        for name in names:
            feature, operator = _parse_candidate_name(name)
            try:
                base_indices.append(self._base_indices[feature])
                operator_indices.append(self._operator_indices[operator])
            except KeyError as exc:  # pragma: no cover - guarded by parsing above
                raise ValueError(f"unsupported IA-GPSS candidate identity: {name}") from exc

        device = self.base_variable_embedding.weight.device
        base = self.base_variable_embedding(
            torch.tensor(base_indices, dtype=torch.long, device=device)
        )
        operator = self.operator_embedding(
            torch.tensor(operator_indices, dtype=torch.long, device=device)
        )
        result = base + operator
        if not torch.isfinite(result).all():
            raise FloatingPointError("IA-GPSS semantic embeddings contain NaN or Inf")
        return result

    def identity_for(self, candidate_name: str) -> torch.Tensor:
        """Return one semantic identity through the same lookup path."""

        return self((candidate_name,))[0]


@dataclass(frozen=True)
class IAGPSSSelectorOutput:
    """Assignments and readout traces returned by one selector invocation."""

    hard_assignment: torch.Tensor
    soft_probabilities: torch.Tensor
    st_assignment: torch.Tensor
    initial_selected_indices: tuple[int, ...]
    initial_selected_names: tuple[str, ...]
    selected_indices: tuple[int, ...]
    selected_names: tuple[str, ...]
    st_assignment_rows: tuple[torch.Tensor, ...]
    selection_st_assignment_rows: tuple[torch.Tensor, ...]
    selection_path: tuple[str, ...]
    refinement_trace: tuple[dict[str, Any], ...]
    selection_path_indices: tuple[int, ...]


@dataclass(frozen=True)
class _SelectionStep:
    raw_logits: torch.Tensor
    soft_probabilities: torch.Tensor
    hard_assignment: torch.Tensor
    st_assignment: torch.Tensor
    hard_index: int


class IAGPSSSelector(nn.Module):
    """Global, deterministic, interaction-aware exact-K selector.

    The default candidate bank is the canonical 13-base-feature by registered
    operator bank.  A canonical subset may be supplied for small deterministic
    fixtures; the default path remains sample-, node-, time-, edge- and
    target-independent.
    """

    def __init__(
        self,
        candidate_names: Sequence[str] | int | None = None,
        *,
        d_model: int = DEFAULT_IA_GPSS_D_MODEL,
        top_k: int = DEFAULT_IA_GPSS_TOP_K,
        temperature: float = DEFAULT_IA_GPSS_TEMPERATURE,
        refinement_rounds: int = DEFAULT_IA_GPSS_REFINEMENT_ROUNDS,
    ) -> None:
        super().__init__()

        # Accept ``IAGPSSSelector(16, top_k=...)`` as a compact d_model-only
        # construction while keeping the named candidate_names API explicit.
        if isinstance(candidate_names, int) and not isinstance(candidate_names, bool):
            if d_model != DEFAULT_IA_GPSS_D_MODEL:
                raise ValueError(
                    "IA-GPSS d_model was provided both positionally and by keyword"
                )
            d_model = candidate_names
            candidate_names = None

        self.d_model = _validate_d_model(d_model)
        self.temperature = validate_temperature(temperature)
        self.refinement_rounds = validate_refinement_rounds(refinement_rounds)

        resolved_names = (
            canonical_candidate_names()
            if candidate_names is None
            else _validate_candidate_names(candidate_names, field="candidate_names")
        )
        canonical_names = set(canonical_candidate_names())
        unknown = sorted(set(resolved_names) - canonical_names)
        if unknown:
            raise ValueError(
                "IA-GPSS candidate_names contains a candidate outside the canonical bank: "
                f"{unknown[0]}"
            )
        self.candidate_names = resolved_names
        self.candidate_count = len(self.candidate_names)
        self.top_k = validate_top_k(top_k, self.candidate_count)
        # Short structural aliases make the fixed-cardinality contract clear
        # without introducing another configurable selector dimension.
        self.M = self.candidate_count
        self.K = self.top_k

        self.semantic_identity = IAGPSSSemanticCandidateIdentity(
            self.d_model,
            candidate_names=self.candidate_names,
        )
        self.unary_scorer = nn.Linear(self.d_model, 1)
        self.pairwise_interaction_mlp = nn.Sequential(
            nn.Linear(3 * self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 1),
        )

        self._candidate_index = {
            name: index for index, name in enumerate(self.candidate_names)
        }

    @property
    def unary(self) -> nn.Linear:
        """Compatibility/readability alias for the unary scorer."""

        return self.unary_scorer

    @property
    def candidate_identity(self) -> IAGPSSSemanticCandidateIdentity:
        """Readability alias for the IA-GPSS-local semantic identity."""

        return self.semantic_identity

    @property
    def base_variable_embedding(self) -> nn.Embedding:
        """Expose the base-variable parameter group without duplicating it."""

        return self.semantic_identity.base_variable_embedding

    @property
    def operator_embedding(self) -> nn.Embedding:
        """Expose the operator parameter group without duplicating it."""

        return self.semantic_identity.operator_embedding

    @property
    def interaction_mlp(self) -> nn.Sequential:
        """Compatibility/readability alias for the pairwise MLP."""

        return self.pairwise_interaction_mlp

    def _validate_parameters_finite(self) -> None:
        for name, parameter in self.named_parameters():
            if not torch.isfinite(parameter).all():
                raise FloatingPointError(
                    f"IA-GPSS parameter {name} contains NaN or Inf"
                )

    def semantic_embeddings(self) -> torch.Tensor:
        """Return ``E`` with shape ``[M, D]``."""

        embeddings = self.semantic_identity()
        if not torch.isfinite(embeddings).all():
            raise FloatingPointError("IA-GPSS semantic embeddings contain NaN or Inf")
        return embeddings

    def pair_features(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> torch.Tensor:
        """Return the exact symmetric pair feature construction."""

        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise TypeError("IA-GPSS pair features require tensors")
        if left.shape[-1] != self.d_model or right.shape[-1] != self.d_model:
            raise ValueError(
                "IA-GPSS pair features require last dimension d_model="
                f"{self.d_model}"
            )
        if not torch.isfinite(left).all() or not torch.isfinite(right).all():
            raise FloatingPointError("IA-GPSS pair features contain NaN or Inf")
        result = torch.cat((left + right, left * right, (left - right).abs()), dim=-1)
        if not torch.isfinite(result).all():
            raise FloatingPointError("IA-GPSS pair features contain NaN or Inf")
        return result

    def pairwise_interaction(
        self,
        left: torch.Tensor | int,
        right: torch.Tensor | int,
        *,
        semantic_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate one or broadcastable pairs through the symmetric MLP."""

        embeddings = self.semantic_embeddings() if semantic_embeddings is None else semantic_embeddings
        left_tensor = self._resolve_embedding(left, embeddings)
        right_tensor = self._resolve_embedding(right, embeddings)
        result = self.pairwise_interaction_mlp(
            self.pair_features(left_tensor, right_tensor)
        ).squeeze(-1)
        if not torch.isfinite(result).all():
            raise FloatingPointError("IA-GPSS pairwise interaction contains NaN or Inf")
        return result

    def pairwise_interaction_matrix(
        self,
        semantic_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return exact-symmetric ``G`` with shape ``[M, M]``.

        Only the strict upper triangle represents valid distinct-candidate
        pairs.  Mirroring that triangle makes ``G[i, j]`` and ``G[j, i]`` the
        same tensor value and gives the set utility its explicit ``i < j``
        semantics.
        """

        embeddings = self.semantic_embeddings() if semantic_embeddings is None else semantic_embeddings
        self._validate_embedding_matrix(embeddings)
        pair = self.pairwise_interaction(
            embeddings.unsqueeze(1),
            embeddings.unsqueeze(0),
            semantic_embeddings=embeddings,
        )
        upper = torch.triu(pair, diagonal=1)
        result = upper + upper.transpose(0, 1)
        if not torch.isfinite(result).all():
            raise FloatingPointError("IA-GPSS interaction matrix contains NaN or Inf")
        return result

    def unary_scores(
        self,
        semantic_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the candidate unary scores ``u_i`` with shape ``[M]``."""

        embeddings = self.semantic_embeddings() if semantic_embeddings is None else semantic_embeddings
        self._validate_embedding_matrix(embeddings)
        result = self.unary_scorer(embeddings).squeeze(-1)
        if not torch.isfinite(result).all():
            raise FloatingPointError("IA-GPSS unary scores contain NaN or Inf")
        return result

    def set_utility(
        self,
        selected: Sequence[int | str] | torch.Tensor,
        *,
        semantic_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate ``U(S) = sum u_i + sum_{i<j} g(i,j)`` for a hard set."""

        indices = self._resolve_selected_indices(selected)
        embeddings = self.semantic_embeddings() if semantic_embeddings is None else semantic_embeddings
        self._validate_embedding_matrix(embeddings)
        unary = self.unary_scores(embeddings)
        if not indices:
            return unary.new_zeros(())
        index_tensor = torch.tensor(indices, dtype=torch.long, device=embeddings.device)
        interactions = self.pairwise_interaction_matrix(embeddings)
        selected_unary = unary.index_select(0, index_tensor).sum()
        selected_interactions = interactions.index_select(0, index_tensor).index_select(
            1, index_tensor
        )
        result = selected_unary + torch.triu(selected_interactions, diagonal=1).sum()
        if not torch.isfinite(result):
            raise FloatingPointError("IA-GPSS set utility contains NaN or Inf")
        return result

    # Keep the mathematical term discoverable under the shorter name used in
    # some downstream design notes.
    utility = set_utility

    def conditional_marginal_scores(
        self,
        condition_assignment: torch.Tensor | Sequence[int | str] | None = None,
        *,
        selected_indices: Sequence[int | str] | None = None,
        semantic_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``q(m | S) = U(S ∪ {m}) - U(S)`` for every candidate.

        A tensor assignment is intentionally supported so sequential and
        refinement decisions can pass their previous ST rows directly.  The
        hard selected indices are used only by the caller for duplicate
        masking; the interaction term below remains differentiable through
        ``condition_assignment``.
        """

        if selected_indices is not None and condition_assignment is not None:
            raise ValueError(
                "IA-GPSS conditional scores received both assignment and selected_indices"
            )
        embeddings = self.semantic_embeddings() if semantic_embeddings is None else semantic_embeddings
        self._validate_embedding_matrix(embeddings)
        unary = self.unary_scores(embeddings)
        interactions = self.pairwise_interaction_matrix(embeddings)

        if selected_indices is not None:
            condition = self._hard_assignment_from_indices(
                self._resolve_selected_indices(selected_indices), embeddings
            )
        elif condition_assignment is None:
            condition = None
        elif isinstance(condition_assignment, torch.Tensor):
            condition = condition_assignment
            if condition.ndim != 2 or condition.shape[1] != self.candidate_count:
                raise ValueError(
                    "IA-GPSS condition assignment must have shape [R, M]"
                )
            if condition.device != embeddings.device or condition.dtype != embeddings.dtype:
                condition = condition.to(device=embeddings.device, dtype=embeddings.dtype)
            if not torch.isfinite(condition).all():
                raise FloatingPointError(
                    "IA-GPSS condition assignment contains NaN or Inf"
                )
        else:
            condition = self._hard_assignment_from_indices(
                self._resolve_selected_indices(condition_assignment), embeddings
            )

        if condition is None:
            result = unary
        else:
            result = unary + interactions.matmul(condition.sum(dim=0))
        if not torch.isfinite(result).all():
            raise FloatingPointError("IA-GPSS conditional scores contain NaN or Inf")
        return result

    conditional_scores = conditional_marginal_scores

    def forward(self) -> IAGPSSSelectorOutput:
        """Select one deterministic exact-K set without dynamic model context."""

        self._validate_parameters_finite()
        embeddings = self.semantic_embeddings()
        unary = self.unary_scores(embeddings)
        interactions = self.pairwise_interaction_matrix(embeddings)

        hard_rows: list[torch.Tensor] = []
        soft_rows: list[torch.Tensor] = []
        st_rows: list[torch.Tensor] = []
        hard_indices: list[int] = []
        for _step in range(self.top_k):
            condition = self._stack_assignments(st_rows, embeddings)
            selected = self._selection_step(
                unary,
                interactions,
                condition,
                blocked_indices=hard_indices,
            )
            hard_rows.append(selected.hard_assignment)
            soft_rows.append(selected.soft_probabilities)
            st_rows.append(selected.st_assignment)
            hard_indices.append(selected.hard_index)

        initial_indices = tuple(hard_indices)
        initial_names = tuple(self.candidate_names[index] for index in initial_indices)

        refinement_trace: list[dict[str, Any]] = []
        for round_index in range(self.refinement_rounds):
            for slot in range(self.top_k):
                condition_indices = tuple(
                    hard_indices[index] for index in range(self.top_k) if index != slot
                )
                condition_names = tuple(
                    self.candidate_names[index] for index in condition_indices
                )
                condition_rows = [
                    st_rows[index] for index in range(self.top_k) if index != slot
                ]
                condition = self._stack_assignments(condition_rows, embeddings)
                selected = self._selection_step(
                    unary,
                    interactions,
                    condition,
                    blocked_indices=condition_indices,
                )
                previous_index = hard_indices[slot]
                hard_indices = [*hard_indices]
                hard_rows = [*hard_rows]
                soft_rows = [*soft_rows]
                st_rows = [*st_rows]
                hard_indices[slot] = selected.hard_index
                hard_rows[slot] = selected.hard_assignment
                soft_rows[slot] = selected.soft_probabilities
                st_rows[slot] = selected.st_assignment
                refinement_trace.append(
                    {
                        "round": round_index,
                        "slot": slot,
                        "condition_on": condition_names,
                        "previous_candidate": self.candidate_names[previous_index],
                        "new_candidate": self.candidate_names[selected.hard_index],
                    }
                )

        # Sorting is deliberately a detached hard readout.  It canonicalizes
        # the final propagation set while leaving the real sequential path and
        # refinement trace untouched.
        canonical_slot_order = tuple(
            sorted(range(self.top_k), key=lambda slot: hard_indices[slot])
        )
        final_indices = tuple(hard_indices[slot] for slot in canonical_slot_order)
        if len(set(final_indices)) != self.top_k:
            raise AssertionError("IA-GPSS final hard selection is not unique")
        hard_assignment = torch.stack(
            [hard_rows[slot] for slot in canonical_slot_order], dim=0
        )
        soft_probabilities = torch.stack(
            [soft_rows[slot] for slot in canonical_slot_order], dim=0
        )
        st_assignment = torch.stack(
            [st_rows[slot] for slot in canonical_slot_order], dim=0
        )
        st_assignment_rows = tuple(st_rows[slot] for slot in canonical_slot_order)
        selection_st_assignment_rows = tuple(st_rows)
        self._validate_output_assignments(
            hard_assignment,
            soft_probabilities,
            st_assignment,
        )
        final_names = tuple(self.candidate_names[index] for index in final_indices)
        return IAGPSSSelectorOutput(
            hard_assignment=hard_assignment,
            soft_probabilities=soft_probabilities,
            st_assignment=st_assignment,
            initial_selected_indices=initial_indices,
            initial_selected_names=initial_names,
            selected_indices=final_indices,
            selected_names=final_names,
            st_assignment_rows=st_assignment_rows,
            selection_st_assignment_rows=selection_st_assignment_rows,
            selection_path=initial_names,
            refinement_trace=tuple(refinement_trace),
            selection_path_indices=initial_indices,
        )

    def _selection_step(
        self,
        unary: torch.Tensor,
        interactions: torch.Tensor,
        condition_assignment: torch.Tensor | None,
        *,
        blocked_indices: Sequence[int],
    ) -> _SelectionStep:
        if condition_assignment is None:
            raw_logits = unary
        else:
            raw_logits = unary + interactions.matmul(condition_assignment.sum(dim=0))
        if not torch.isfinite(raw_logits).all():
            raise FloatingPointError("IA-GPSS raw logits contain NaN or Inf")

        blocked = torch.zeros(
            self.candidate_count,
            dtype=torch.bool,
            device=raw_logits.device,
        )
        if blocked_indices:
            blocked_tensor = torch.tensor(
                tuple(blocked_indices), dtype=torch.long, device=raw_logits.device
            )
            blocked[blocked_tensor] = True
        masked_logits = raw_logits.masked_fill(blocked, float("-inf"))
        probabilities = F.softmax(masked_logits / self.temperature, dim=-1)
        if not torch.isfinite(probabilities).all():
            raise FloatingPointError("IA-GPSS soft probabilities contain NaN or Inf")

        hard_index_tensor = torch.argmax(masked_logits, dim=-1)
        hard_index = int(hard_index_tensor.detach().cpu().item())
        hard = F.one_hot(hard_index_tensor, num_classes=self.candidate_count).to(
            dtype=probabilities.dtype
        )
        # This exact straight-through expression is the IA-GPSS gradient seam.
        st = hard - probabilities.detach() + probabilities
        if not torch.isfinite(st).all():
            raise FloatingPointError("IA-GPSS ST assignment contains NaN or Inf")
        return _SelectionStep(
            raw_logits=raw_logits,
            soft_probabilities=probabilities,
            hard_assignment=hard,
            st_assignment=st,
            hard_index=hard_index,
        )

    def _stack_assignments(
        self,
        rows: Sequence[torch.Tensor],
        embeddings: torch.Tensor,
    ) -> torch.Tensor | None:
        if not rows:
            return None
        result = torch.stack(tuple(rows), dim=0)
        if result.shape != (len(rows), self.candidate_count):
            raise ValueError("IA-GPSS assignment rows have an unexpected shape")
        if result.device != embeddings.device or result.dtype != embeddings.dtype:
            result = result.to(device=embeddings.device, dtype=embeddings.dtype)
        if not torch.isfinite(result).all():
            raise FloatingPointError("IA-GPSS assignment rows contain NaN or Inf")
        return result

    def _validate_embedding_matrix(self, embeddings: torch.Tensor) -> None:
        if not isinstance(embeddings, torch.Tensor) or embeddings.shape != (
            self.candidate_count,
            self.d_model,
        ):
            raise ValueError(
                "IA-GPSS semantic embeddings must have shape "
                f"({self.candidate_count}, {self.d_model})"
            )
        if not torch.isfinite(embeddings).all():
            raise FloatingPointError("IA-GPSS semantic embeddings contain NaN or Inf")

    def _resolve_embedding(
        self,
        value: torch.Tensor | int,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(value, int) and not isinstance(value, bool):
            if not 0 <= value < self.candidate_count:
                raise ValueError("IA-GPSS candidate index is outside the bank")
            return embeddings[value]
        if not isinstance(value, torch.Tensor):
            raise TypeError("IA-GPSS pair interaction expects an embedding or index")
        return value

    def _resolve_selected_indices(
        self,
        selected: Sequence[int | str] | torch.Tensor,
    ) -> tuple[int, ...]:
        if isinstance(selected, torch.Tensor):
            if selected.ndim != 1:
                raise ValueError("IA-GPSS selected indices must be one-dimensional")
            values = selected.detach().cpu().tolist()
        else:
            if isinstance(selected, (str, bytes)) or not isinstance(selected, Sequence):
                raise ValueError("IA-GPSS selected indices must be an ordered sequence")
            values = tuple(selected)
        result: list[int] = []
        for value in values:
            if isinstance(value, str):
                try:
                    index = self._candidate_index[value]
                except KeyError as exc:
                    raise ValueError(
                        f"IA-GPSS selected candidate is outside the bank: {value}"
                    ) from exc
            elif isinstance(value, int) and not isinstance(value, bool):
                index = int(value)
            else:
                raise ValueError("IA-GPSS selected indices must be integers or names")
            if not 0 <= index < self.candidate_count:
                raise ValueError("IA-GPSS selected index is outside the bank")
            result.append(index)
        if len(set(result)) != len(result):
            raise ValueError("IA-GPSS selected set contains duplicates")
        return tuple(sorted(result))

    def _hard_assignment_from_indices(
        self,
        indices: Sequence[int],
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if not indices:
            return embeddings.new_zeros((0, self.candidate_count))
        index_tensor = torch.tensor(indices, dtype=torch.long, device=embeddings.device)
        return F.one_hot(index_tensor, num_classes=self.candidate_count).to(
            dtype=embeddings.dtype
        )

    def _validate_output_assignments(
        self,
        hard_assignment: torch.Tensor,
        soft_probabilities: torch.Tensor,
        st_assignment: torch.Tensor,
    ) -> None:
        expected_shape = (self.top_k, self.candidate_count)
        if tuple(hard_assignment.shape) != expected_shape:
            raise ValueError("IA-GPSS hard assignment has an unexpected shape")
        if tuple(soft_probabilities.shape) != expected_shape:
            raise ValueError("IA-GPSS soft probabilities have an unexpected shape")
        if tuple(st_assignment.shape) != expected_shape:
            raise ValueError("IA-GPSS ST assignment has an unexpected shape")
        if not torch.isfinite(hard_assignment).all():
            raise FloatingPointError("IA-GPSS hard assignment contains NaN or Inf")
        if not torch.isfinite(soft_probabilities).all():
            raise FloatingPointError("IA-GPSS soft probabilities contain NaN or Inf")
        if not torch.isfinite(st_assignment).all():
            raise FloatingPointError("IA-GPSS ST assignment contains NaN or Inf")


# Explicit aliases keep the module easy to discover without creating separate
# implementations or parameter sets.
InteractionAwareGlobalSelector = IAGPSSSelector
GlobalInteractionAwareSelector = IAGPSSSelector
SemanticCandidateIdentity = IAGPSSSemanticCandidateIdentity


__all__ = [
    "DEFAULT_IA_GPSS_D_MODEL",
    "DEFAULT_IA_GPSS_REFINEMENT_ROUNDS",
    "DEFAULT_IA_GPSS_TEMPERATURE",
    "DEFAULT_IA_GPSS_TOP_K",
    "IA_GPSS_SELECTOR_TYPE",
    "IAGPSSSemanticCandidateIdentity",
    "IAGPSSSelector",
    "IAGPSSSelectorOutput",
    "InteractionAwareGlobalSelector",
    "GlobalInteractionAwareSelector",
    "SemanticCandidateIdentity",
    "canonical_candidate_names",
    "validate_refinement_rounds",
    "validate_temperature",
    "validate_top_k",
]

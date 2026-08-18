"""Global entropy-regularized differentiable Top-K selection for P3-A.

The selector solves the entropic Top-K relaxation on the capped simplex.  For
candidate logits ``alpha`` and a fixed cardinality ``K``, its relaxed gate is
the solution of the entropy-regularized selected/unselected transport problem
with row capacity one and total selected mass ``K``.  The scalar dual
threshold is solved by a fixed number of bisection iterations, so the forward
path stays differentiable and never executes a discrete Top-K operation.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from typing import Any

import torch
from torch import nn


SELECTOR_TYPE = "entropy_regularized_ot_topk"
DEFAULT_SELECTOR_TEMPERATURE = 0.1
DEFAULT_SELECTOR_BISECTION_ITERATIONS = 64
MAX_SELECTOR_BISECTION_ITERATIONS = 10_000


def validate_top_k(top_k: Any, candidate_count: int) -> int:
    """Validate one fixed-cardinality selector request."""

    if not isinstance(candidate_count, int) or isinstance(candidate_count, bool):
        raise ValueError("P3 selector candidate_count must be an integer")
    if candidate_count < 1:
        raise ValueError("P3 selector candidate_count must be positive")
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise ValueError("P3 selector top_k must be an integer")
    if not 1 <= top_k <= candidate_count:
        raise ValueError(
            "P3 selector top_k must be in [1, candidate_count]"
        )
    return int(top_k)


def validate_selector_temperature(value: Any) -> float:
    """Validate the positive entropy temperature used by the relaxation."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("P3 selector temperature must be a finite positive number")
    temperature = float(value)
    if not isfinite(temperature) or temperature <= 0.0:
        raise ValueError("P3 selector temperature must be a finite positive number")
    return temperature


def validate_selector_bisection_iterations(value: Any) -> int:
    """Validate the fixed iteration budget for the scalar OT dual solve."""

    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 1 <= value <= MAX_SELECTOR_BISECTION_ITERATIONS
    ):
        raise ValueError(
            "P3 selector bisection_iterations must be an integer in "
            f"[1, {MAX_SELECTOR_BISECTION_ITERATIONS}]"
        )
    return int(value)


class GlobalTopKSelector(nn.Module):
    """One global learnable differentiable fixed-cardinality selector."""

    def __init__(
        self,
        candidate_names: Sequence[str],
        *,
        top_k: int = 2,
        temperature: float = DEFAULT_SELECTOR_TEMPERATURE,
        bisection_iterations: int = DEFAULT_SELECTOR_BISECTION_ITERATIONS,
    ) -> None:
        super().__init__()
        if isinstance(candidate_names, (str, bytes)):
            raise ValueError("P3 selector candidate_names must be an ordered sequence")
        names = tuple(candidate_names)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("P3 selector candidate_names must contain non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("P3 selector candidate_names must be unique")
        self.candidate_names = names
        self.candidate_count = len(names)
        self.top_k = validate_top_k(top_k, self.candidate_count)
        self.temperature = validate_selector_temperature(temperature)
        self.bisection_iterations = validate_selector_bisection_iterations(
            bisection_iterations
        )
        self.logits = nn.Parameter(torch.zeros(self.candidate_count))

    def relaxed_gate(self) -> torch.Tensor:
        """Return the differentiable relaxed K-hot gate with shape ``[M]``.

        The gate is the selected mass in a two-column entropy-regularized
        transport problem: each candidate has unit mass, the selected column
        has capacity ``K`` and the unselected column has capacity ``M-K``.
        Solving its scalar dual gives a sigmoid at a shared threshold.  A
        fixed bisection budget makes the operation deterministic while
        retaining autograd connectivity to ``logits``.
        """

        if not torch.isfinite(self.logits).all():
            raise FloatingPointError("P3 selector logits contain NaN or Inf")

        # When every candidate is selected, the exact relaxation is the all-
        # ones gate.  Keep a zero-valued logits term so backward still returns
        # a finite zero gradient instead of ``None`` for this degenerate but
        # valid cardinality.
        if self.top_k == self.candidate_count:
            return torch.ones_like(self.logits) + self.logits * 0.0

        dtype = self.logits.dtype
        finfo = torch.finfo(dtype)
        temperature = self.logits.new_tensor(self.temperature)
        # The dual threshold is guaranteed to lie in this interval for a
        # sigmoid mass target between 1 and M-1.  Detaching the bounds keeps
        # the finite numerical bracket from adding a spurious max/min path to
        # the selector gradient.
        margin = temperature * 80.0
        low = torch.clamp(
            self.logits.detach().amin() - margin,
            min=finfo.min,
            max=finfo.max,
        )
        high = torch.clamp(
            self.logits.detach().amax() + margin,
            min=finfo.min,
            max=finfo.max,
        )
        target = self.logits.new_tensor(float(self.top_k))
        for _ in range(self.bisection_iterations):
            midpoint = (low + high) * 0.5
            scaled = torch.clamp(
                (self.logits - midpoint) / temperature,
                min=-80.0,
                max=80.0,
            )
            mass = torch.sigmoid(scaled).sum()
            too_much_selected = mass > target
            low = torch.where(too_much_selected, midpoint, low)
            high = torch.where(too_much_selected, high, midpoint)

        midpoint = (low + high) * 0.5
        gate = torch.sigmoid(
            torch.clamp(
                (self.logits - midpoint) / temperature,
                min=-80.0,
                max=80.0,
            )
        )
        if not torch.isfinite(gate).all():
            raise FloatingPointError("P3 selector relaxed gate contains NaN or Inf")
        # Sigmoid already enforces the per-candidate bounds.  This explicit
        # cardinality correction only removes the last bisection round-off;
        # it is differentiable and keeps the contract true even in float32.
        gate = gate * (target / gate.sum().clamp_min(torch.finfo(dtype).tiny))
        gate = gate.clamp(0.0, 1.0)
        if not torch.isfinite(gate).all():
            raise FloatingPointError("P3 selector relaxed gate contains NaN or Inf")
        return gate

    def mixture_weights(self, relaxed_gate: torch.Tensor | None = None) -> torch.Tensor:
        """Normalize a relaxed gate for propagation-only aggregation."""

        gate = self.relaxed_gate() if relaxed_gate is None else relaxed_gate
        if tuple(gate.shape) != (self.candidate_count,):
            raise ValueError("P3 selector gate has an unexpected shape")
        if not torch.isfinite(gate).all():
            raise FloatingPointError("P3 selector gate contains NaN or Inf")
        weights = gate / gate.sum().clamp_min(torch.finfo(gate.dtype).tiny)
        if not torch.isfinite(weights).all():
            raise FloatingPointError("P3 selector mixture weights contain NaN or Inf")
        return weights

    def scores(self) -> torch.Tensor:
        """Return normalized differentiable weights for report compatibility."""

        return self.mixture_weights()

    def forward(self) -> torch.Tensor:
        return self.relaxed_gate()

    def ranking(self, scores: torch.Tensor | None = None) -> tuple[int, ...]:
        values = self.scores() if scores is None else scores
        if tuple(values.shape) != (self.candidate_count,):
            raise ValueError("P3 selector ranking received an unexpected score shape")
        if not torch.isfinite(values).all():
            raise FloatingPointError("P3 selector ranking scores contain NaN or Inf")
        # Convert only the global readout to Python values. The forward path
        # stays fully differentiable and never uses this hard ranking.
        scores_cpu = values.detach().cpu().tolist()
        return tuple(
            sorted(
                range(self.candidate_count),
                key=lambda index: (-float(scores_cpu[index]), index),
            )
        )

    def hard_top_k(self) -> tuple[int, ...]:
        """Return deterministic hard indices for readout only."""

        return self.ranking(self.logits)[: self.top_k]

    def selection_report(self) -> list[dict[str, Any]]:
        """Return a JSON-serializable, deterministic ranking readout."""

        values = self.mixture_weights()
        # Rank by the learned global logits; report scores remain the
        # normalized differentiable weights below.
        order = self.ranking(self.logits)
        rank_by_index = {index: rank for rank, index in enumerate(order, start=1)}
        selected_indices = set(self.hard_top_k())
        report: list[dict[str, Any]] = []
        score_values = values.detach().cpu().tolist()
        for index, name in enumerate(self.candidate_names):
            score = float(score_values[index])
            if not isfinite(score):
                raise FloatingPointError("P3 selector report contains a non-finite score")
            report.append(
                {
                    "candidate_name": name,
                    "score": score,
                    "rank": int(rank_by_index[index]),
                    "selected": bool(index in selected_indices),
                }
            )
        return report

    # Keep the read-only report discoverable under a short name as well.
    def report(self) -> list[dict[str, Any]]:
        return self.selection_report()


P3GlobalSelector = GlobalTopKSelector
GlobalPropagationSelector = GlobalTopKSelector


__all__ = [
    "DEFAULT_SELECTOR_BISECTION_ITERATIONS",
    "DEFAULT_SELECTOR_TEMPERATURE",
    "GlobalPropagationSelector",
    "GlobalTopKSelector",
    "P3GlobalSelector",
    "SELECTOR_TYPE",
    "validate_selector_bisection_iterations",
    "validate_selector_temperature",
    "validate_top_k",
]

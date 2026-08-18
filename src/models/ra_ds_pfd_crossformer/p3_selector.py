"""Global entropy-regularized differentiable Top-K selection for P3-A.

The selector solves the entropic Top-K relaxation on the capped simplex.  For
candidate logits ``alpha`` and a fixed cardinality ``K``, its relaxed gate is
the solution of the entropy-regularized selected/unselected transport problem
with row capacity one and total selected mass ``K``.  The scalar dual
threshold is solved numerically by a fixed number of bisection iterations; the
custom backward supplies its implicit derivative and never executes a
discrete Top-K operation.
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


class _ImplicitTopKGateFunction(torch.autograd.Function):
    """Fixed-cardinality gate with an analytic implicit-differentiation backward."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        logits: torch.Tensor,
        top_k: int,
        temperature: float,
        bisection_iterations: int,
    ) -> torch.Tensor:
        if not torch.isfinite(logits).all():
            raise FloatingPointError("P3 selector logits contain NaN or Inf")

        ctx.top_k = int(top_k)
        ctx.temperature = float(temperature)
        ctx.all_selected = ctx.top_k == int(logits.numel())
        if ctx.all_selected:
            # When every candidate is selected, the constrained relaxation is
            # exactly one for every candidate and has no selection gradient.
            return torch.ones_like(logits)

        # The threshold is a numerical scalar solve.  Its bracket is allowed
        # to use detached logits because the backward below does not traverse
        # an unrolled bisection graph; it applies the analytic implicit
        # derivative of the fixed-cardinality constraint instead.
        with torch.no_grad():
            dtype = logits.dtype
            finfo = torch.finfo(dtype)
            temperature_tensor = logits.new_tensor(ctx.temperature)
            margin = temperature_tensor * 80.0
            low = torch.clamp(
                logits.detach().amin() - margin,
                min=finfo.min,
                max=finfo.max,
            )
            high = torch.clamp(
                logits.detach().amax() + margin,
                min=finfo.min,
                max=finfo.max,
            )
            target = logits.new_tensor(float(ctx.top_k))
            for _ in range(int(bisection_iterations)):
                midpoint = (low + high) * 0.5
                mass = torch.sigmoid((logits.detach() - midpoint) / temperature_tensor).sum()
                if bool(mass > target):
                    low = midpoint
                else:
                    high = midpoint
            threshold = (low + high) * 0.5
            gate = torch.sigmoid((logits.detach() - threshold) / temperature_tensor)

        if not torch.isfinite(gate).all():
            raise FloatingPointError("P3 selector relaxed gate contains NaN or Inf")
        ctx.save_for_backward(gate)
        return gate

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_gate: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None]:
        if ctx.all_selected:
            return torch.zeros_like(grad_gate), None, None, None
        if not torch.isfinite(grad_gate).all():
            raise FloatingPointError("P3 selector gate upstream gradient contains NaN or Inf")

        (gate,) = ctx.saved_tensors
        temperature = gate.new_tensor(ctx.temperature)
        s = gate * (1.0 - gate) / temperature
        denominator = s.sum()
        tiny = torch.finfo(gate.dtype).tiny
        if not bool(torch.isfinite(denominator)):
            raise FloatingPointError("P3 selector implicit-gradient denominator is non-finite")
        if bool(denominator <= tiny):
            raise FloatingPointError("P3 selector implicit-gradient denominator is too small")

        weighted_upstream = (grad_gate * s).sum() / denominator
        grad_logits = s * (grad_gate - weighted_upstream)
        if not torch.isfinite(grad_logits).all():
            raise FloatingPointError("P3 selector logits gradient contains NaN or Inf")
        return grad_logits, None, None, None


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
        Solving its scalar dual gives a sigmoid at a shared threshold.  The
        fixed bisection budget makes the numerical forward deterministic;
        ``_ImplicitTopKGateFunction.backward`` supplies the analytic implicit
        derivative instead of differentiating through the solver iterations.
        """

        return _ImplicitTopKGateFunction.apply(
            self.logits,
            self.top_k,
            self.temperature,
            self.bisection_iterations,
        )

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

"""The single execution policy shared by training, evaluation and checks.

The public model interface stays small: models return ``(B, N, H)`` and
NodeShared adapters implement one contiguous node range.  This module owns
the deeper policy: plan selection, node partitioning, evaluation concatenation
and the exact two-pass global masked-loss backward used by node micro-batches.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

import torch
from torch import nn

from data.dataset import ForecastBatch
from models.base import ForecastModel, ModelInput, NodeSharedForecastModel

from .losses import ScoreAlignedHybridTerms, resolve_loss, score_aligned_hybrid_terms
from .reproducibility import capture_rng_state, restore_rng_state


DEFAULT_NODE_SHARED_CHUNK_SIZE = 32


@dataclass(frozen=True)
class ExecutionPlan:
    """Resolved execution decision for one model and node count."""

    execution_mode: str
    total_nodes: int
    configured_node_chunk_size: int
    effective_node_chunk_size: int
    node_chunk_count: int
    batch_dependent_norm_detected: bool
    reason: str | None = None

    @property
    def uses_node_microbatch(self) -> bool:
        return self.execution_mode == "node_shared_microbatch"

    def node_ranges(self) -> tuple[tuple[int, int], ...]:
        """Return ordered, non-overlapping node ranges with no padding."""

        ranges = tuple(
            (start, min(start + self.effective_node_chunk_size, self.total_nodes))
            for start in range(0, self.total_nodes, self.effective_node_chunk_size)
        )
        if len(ranges) != self.node_chunk_count:
            raise RuntimeError("execution plan node range count is inconsistent")
        if not ranges or ranges[-1][1] != self.total_nodes:
            raise RuntimeError("execution plan does not cover all nodes")
        return ranges

    def as_dict(self) -> dict[str, object]:
        return {
            "execution_mode": self.execution_mode,
            "configured_node_chunk_size": self.configured_node_chunk_size,
            "effective_node_chunk_size": self.effective_node_chunk_size,
            "node_chunk_count": self.node_chunk_count,
            "node_chunk_sizes": [end - start for start, end in self.node_ranges()],
            "batch_dependent_norm_detected": self.batch_dependent_norm_detected,
            "reason": self.reason,
            "total_nodes": self.total_nodes,
        }


def has_batch_dependent_normalization(model: nn.Module) -> bool:
    """Detect PyTorch batch-dependent normalization by module type."""

    batch_norm_type = nn.modules.batchnorm._BatchNorm
    return any(isinstance(module, batch_norm_type) for module in model.modules())


def build_execution_plan(
    model: ForecastModel,
    *,
    total_nodes: int,
    node_shared_chunk_size: int = DEFAULT_NODE_SHARED_CHUNK_SIZE,
) -> ExecutionPlan:
    """Resolve one execution mode without model-name special cases."""

    total_nodes = int(total_nodes)
    configured = int(node_shared_chunk_size)
    if total_nodes < 1:
        raise ValueError("total_nodes must be positive")
    if configured < 1:
        raise ValueError("node_shared_chunk_size must be a positive integer")

    is_node_shared = isinstance(model, NodeSharedForecastModel)
    batch_norm_detected = has_batch_dependent_normalization(model) if is_node_shared else False
    if is_node_shared and not batch_norm_detected:
        effective = min(configured, total_nodes)
        return ExecutionPlan(
            execution_mode="node_shared_microbatch",
            total_nodes=total_nodes,
            configured_node_chunk_size=configured,
            effective_node_chunk_size=effective,
            node_chunk_count=(total_nodes + effective - 1) // effective,
            batch_dependent_norm_detected=False,
        )

    if is_node_shared:
        return ExecutionPlan(
            execution_mode="full_nodes",
            total_nodes=total_nodes,
            configured_node_chunk_size=configured,
            effective_node_chunk_size=total_nodes,
            node_chunk_count=1,
            batch_dependent_norm_detected=True,
            reason="batch_dependent_normalization",
        )

    return ExecutionPlan(
        execution_mode=getattr(model, "execution_mode", "full_spatiotemporal"),
        total_nodes=total_nodes,
        configured_node_chunk_size=configured,
        effective_node_chunk_size=total_nodes,
        node_chunk_count=1,
        batch_dependent_norm_detected=False,
    )


def _validate_prediction_shape(
    prediction: torch.Tensor,
    *,
    batch: int,
    nodes: int,
    name: str,
) -> torch.Tensor:
    if prediction.ndim != 3 or tuple(prediction.shape[:2]) != (batch, nodes):
        raise ValueError(
            f"{name} output must have shape (B, {nodes}, H), got {tuple(prediction.shape)}"
        )
    if not torch.isfinite(prediction).all():
        raise FloatingPointError(f"{name} output contains NaN or Inf")
    return prediction


def iter_execution_predictions(
    model: ForecastModel,
    inputs: ModelInput,
    plan: ExecutionPlan,
) -> Iterator[torch.Tensor]:
    """Yield one prediction per execution unit in deterministic node order."""

    if not isinstance(inputs, ModelInput):
        raise TypeError("execution requires ModelInput")
    if inputs.x.ndim != 4:
        raise ValueError("execution requires x with shape (B, L, N, C)")
    batch, _, nodes, _ = inputs.x.shape
    if int(nodes) != plan.total_nodes:
        raise ValueError(
            f"execution plan expects {plan.total_nodes} nodes, got {int(nodes)}"
        )
    if plan.uses_node_microbatch:
        for start, end in plan.node_ranges():
            prediction = model.forward_node_chunk(inputs, start, end)  # type: ignore[attr-defined]
            yield _validate_prediction_shape(
                prediction,
                batch=int(batch),
                nodes=end - start,
                name=type(model).__name__,
            )
        return
    prediction = model(inputs)
    yield _validate_prediction_shape(
        prediction,
        batch=int(batch),
        nodes=plan.total_nodes,
        name=type(model).__name__,
    )


def forward_with_execution_plan(
    model: ForecastModel,
    inputs: ModelInput,
    plan: ExecutionPlan,
) -> torch.Tensor:
    """Run the planned forward and restore the public ``(B, N, H)`` shape."""

    predictions = list(iter_execution_predictions(model, inputs, plan))
    if not predictions:
        raise RuntimeError("execution plan produced no predictions")
    output = predictions[0] if len(predictions) == 1 else torch.cat(predictions, dim=1)
    return _validate_prediction_shape(
        output,
        batch=int(inputs.x.shape[0]),
        nodes=plan.total_nodes,
        name=type(model).__name__,
    )


@dataclass(frozen=True)
class TrainingExecutionResult:
    """Detached reporting values from one optimizer update."""

    loss: float
    terms: ScoreAlignedHybridTerms
    prediction: torch.Tensor | None = None


def _detached_terms(terms: ScoreAlignedHybridTerms) -> ScoreAlignedHybridTerms:
    return ScoreAlignedHybridTerms(
        absolute_error_sum=terms.absolute_error_sum.detach().float(),
        squared_error_sum=terms.squared_error_sum.detach().float(),
        valid_count=terms.valid_count,
    )


def _accumulate_terms(
    total: ScoreAlignedHybridTerms | None,
    current: ScoreAlignedHybridTerms,
) -> ScoreAlignedHybridTerms:
    current = _detached_terms(current)
    return current if total is None else total + current


def execute_training_backward(
    model: ForecastModel,
    batches: Iterable[ForecastBatch],
    *,
    device: torch.device | str,
    plan: ExecutionPlan,
    loss_name: str,
    autocast: Callable[[], AbstractContextManager[None]],
    backward: Callable[[torch.Tensor], None],
    capture_prediction: bool = False,
) -> TrainingExecutionResult:
    """Compute one optimizer update's loss and gradients through the plan.

    For NodeShared micro-batching, Phase A computes global ``A/S/K`` under
    ``no_grad`` and Phase B replays the same dropout RNG while backwarding one
    exact surrogate contribution per node chunk.  The callback receives each
    contribution so Trainer can apply the current shared GradScaler exactly
    once around all backward calls.
    """

    batch_list = list(batches)
    if not batch_list:
        raise ValueError("optimizer update requires at least one micro-batch")
    if loss_name != "masked_score_aligned_hybrid":
        # Keep the existing registry as the source of truth for future loss
        # names, while making the exact node-microbatch contract explicit.
        resolve_loss(loss_name)
        if plan.uses_node_microbatch:
            raise ValueError(
                "node_shared_microbatch requires masked_score_aligned_hybrid"
            )

    if not plan.uses_node_microbatch:
        terms: ScoreAlignedHybridTerms | None = None
        scalar_losses: list[torch.Tensor] = []
        predictions: list[torch.Tensor] = []
        for batch in batch_list:
            device_batch = batch.to(device)
            with autocast():
                prediction = model(device_batch.model_input())
                _validate_prediction_shape(
                    prediction,
                    batch=int(device_batch.x.shape[0]),
                    nodes=plan.total_nodes,
                    name=type(model).__name__,
                )
                if loss_name == "masked_score_aligned_hybrid":
                    current = score_aligned_hybrid_terms(
                        prediction,
                        device_batch.target,
                        device_batch.target_mask,
                    )
                    terms = current if terms is None else terms + current
                else:
                    scalar_losses.append(
                        resolve_loss(loss_name)(
                            prediction,
                            device_batch.target,
                            device_batch.target_mask,
                        )
                    )
            if capture_prediction:
                predictions.append(prediction.detach())
        if terms is not None:
            loss_tensor = terms.loss()
            report_terms = _detached_terms(terms)
        elif scalar_losses:
            loss_tensor = torch.stack(scalar_losses).mean()
            report_terms = ScoreAlignedHybridTerms(
                absolute_error_sum=loss_tensor.detach().float(),
                squared_error_sum=loss_tensor.detach().float(),
                valid_count=1,
            )
        else:
            raise ValueError(f"loss registry entry produced no loss: {loss_name}")
        backward(loss_tensor)
        prediction = torch.cat(predictions, dim=0) if predictions else None
        return TrainingExecutionResult(
            loss=float(loss_tensor.detach().float().cpu()),
            terms=report_terms,
            prediction=prediction,
        )

    rng_before = capture_rng_state()
    global_terms: ScoreAlignedHybridTerms | None = None
    with torch.no_grad():
        for batch in batch_list:
            device_batch = batch.to(device)
            with autocast():
                model_input = device_batch.model_input()
                for node_start, node_end in plan.node_ranges():
                    prediction = model.forward_node_chunk(  # type: ignore[attr-defined]
                        model_input, node_start, node_end
                    )
                    _validate_prediction_shape(
                        prediction,
                        batch=int(device_batch.x.shape[0]),
                        nodes=node_end - node_start,
                        name=type(model).__name__,
                    )
                    current = score_aligned_hybrid_terms(
                        prediction,
                        device_batch.target[:, node_start:node_end, :],
                        device_batch.target_mask[:, node_start:node_end, :],
                        allow_empty=True,
                    )
                    global_terms = _accumulate_terms(global_terms, current)

    if global_terms is None or global_terms.valid_count <= 0:
        restore_rng_state(rng_before)
        raise ValueError("optimizer update contains no valid targets")
    global_loss = global_terms.loss().detach()
    global_rmse = torch.sqrt(global_terms.squared_error_sum / global_terms.valid_count).detach()
    global_count = float(global_terms.valid_count)
    restore_rng_state(rng_before)

    predictions: list[torch.Tensor] = []
    for batch in batch_list:
        device_batch = batch.to(device)
        with autocast():
            model_input = device_batch.model_input()
            for (node_start, node_end) in plan.node_ranges():
                prediction = model.forward_node_chunk(  # type: ignore[attr-defined]
                    model_input, node_start, node_end
                )
                _validate_prediction_shape(
                    prediction,
                    batch=int(device_batch.x.shape[0]),
                    nodes=node_end - node_start,
                    name=type(model).__name__,
                )
                target = device_batch.target[:, node_start:node_end, :]
                target_mask = device_batch.target_mask[:, node_start:node_end, :]
                current = score_aligned_hybrid_terms(
                    prediction,
                    target,
                    target_mask,
                    allow_empty=True,
                )
                if current.valid_count:
                    contribution = 0.5 * current.absolute_error_sum.float() / global_count
                    if float(global_rmse) != 0.0:
                        contribution = contribution + (
                            0.25
                            * current.squared_error_sum.float()
                            / (global_count * global_rmse)
                        )
                    backward(contribution)
                if capture_prediction:
                    predictions.append(prediction.detach())

    prediction = torch.cat(predictions, dim=1) if predictions else None
    return TrainingExecutionResult(
        loss=float(global_loss.float().cpu()),
        terms=global_terms,
        prediction=prediction,
    )

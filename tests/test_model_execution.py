from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn

from data.dataset import ForecastBatch
from engine.losses import masked_score_aligned_hybrid, score_aligned_hybrid_terms
from engine.model_execution import (
    build_execution_plan,
    execute_training_backward,
    forward_with_execution_plan,
)
from models.base import ForecastModel, ModelInput, NodeSharedForecastModel
from models.crossformer.model import Crossformer
from models.loader import build_model
from models.stcn.model import STCN
from runtime.config import load_experiment_config


class _ToyNodeShared(NodeSharedForecastModel):
    def __init__(self, nodes: int = 5, *, dropout: float = 0.0, batch_norm: bool = False) -> None:
        super().__init__()
        self.num_nodes = nodes
        self.input_dim = 1
        self.lookback = 3
        self.horizon = 2
        self.scale = nn.Parameter(torch.tensor(0.7))
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(1) if batch_norm else None

    def forward_node_chunk(self, inputs: ModelInput, node_start: int, node_end: int) -> torch.Tensor:
        x = self._node_chunk_x(inputs, node_start, node_end, model_name="Toy")
        value = x.mean(dim=(1, 3), keepdim=False).unsqueeze(-1) * self.scale
        value = self.dropout(value)
        if self.batch_norm is not None:
            value = self.batch_norm(value.reshape(-1, 1)).reshape_as(value)
        return value.expand(-1, -1, self.horizon)


class _SpatialToy(ForecastModel):
    execution_mode = "full_spatiotemporal"

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        return inputs.x.mean(dim=(1, 3)).unsqueeze(-1).expand(-1, inputs.x.shape[2], 2)


def _batch(nodes: int = 5, *, mask: torch.Tensor | None = None) -> ForecastBatch:
    x = torch.arange(2 * 3 * nodes, dtype=torch.float32).reshape(2, 3, nodes, 1) / 10.0
    target = torch.linspace(-0.4, 0.8, 2 * nodes * 2, dtype=torch.float32).reshape(2, nodes, 2)
    if mask is None:
        mask = torch.ones_like(target, dtype=torch.bool)
    return ForecastBatch(x=x, target=target, target_mask=mask, starts=torch.tensor([0, 1]))


def _backward(model: nn.Module, batch: ForecastBatch, chunk_size: int):
    plan = build_execution_plan(model, total_nodes=model.num_nodes, node_shared_chunk_size=chunk_size)
    result = execute_training_backward(
        model,
        [batch],
        device="cpu",
        plan=plan,
        loss_name="masked_score_aligned_hybrid",
        autocast=lambda: nullcontext(),
        backward=lambda contribution: contribution.backward(),
    )
    return plan, result


def test_node_chunk_partition_preserves_order_without_padding() -> None:
    model = _ToyNodeShared(nodes=134)
    plan = build_execution_plan(model, total_nodes=134, node_shared_chunk_size=32)
    assert [end - start for start, end in plan.node_ranges()] == [32, 32, 32, 32, 6]
    assert [node for start, end in plan.node_ranges() for node in range(start, end)] == list(range(134))
    assert plan.node_chunk_count == 5


def test_full_and_chunked_evaluation_are_equivalent_for_plain_node_shared_model() -> None:
    torch.manual_seed(2026)
    model = _ToyNodeShared(nodes=5).eval()
    batch = _batch()
    plan = build_execution_plan(model, total_nodes=5, node_shared_chunk_size=2)
    with torch.inference_mode():
        full = model(batch.model_input())
        chunked = forward_with_execution_plan(model, batch.model_input(), plan)
    torch.testing.assert_close(chunked, full, atol=1e-7, rtol=1e-7)


def test_chunked_terms_match_one_global_loss_with_uneven_mask() -> None:
    prediction = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]]
    )
    target = torch.zeros_like(prediction)
    mask = torch.tensor(
        [[[True, False], [True, True], [False, True], [True, False], [False, False]]]
    )
    full = score_aligned_hybrid_terms(prediction, target, mask)
    total = None
    for start, end in ((0, 2), (2, 4), (4, 5)):
        current = score_aligned_hybrid_terms(
            prediction[:, start:end], target[:, start:end], mask[:, start:end], allow_empty=True
        )
        total = current if total is None else total + current
    assert total is not None
    assert total.valid_count == full.valid_count
    torch.testing.assert_close(total.absolute_error_sum, full.absolute_error_sum)
    torch.testing.assert_close(total.squared_error_sum, full.squared_error_sum)
    torch.testing.assert_close(total.loss(), full.loss())


def test_exact_chunk_gradient_matches_full_global_loss() -> None:
    torch.manual_seed(7)
    full_model = _ToyNodeShared(nodes=5)
    chunk_model = _ToyNodeShared(nodes=5)
    chunk_model.load_state_dict(full_model.state_dict())
    batch = _batch(
        mask=torch.tensor(
            [
                [[True, True], [False, True], [True, False], [False, False], [True, True]],
                [[True, False], [True, True], [False, True], [True, False], [False, True]],
            ]
        )
    )

    full_output = full_model(batch.model_input())
    full_loss = masked_score_aligned_hybrid(full_output, batch.target, batch.target_mask)
    full_loss.backward()
    full_gradients = [parameter.grad.detach().clone() for parameter in full_model.parameters()]

    plan, result = _backward(chunk_model, batch, 2)
    assert plan.node_chunk_count == 3
    assert result.loss == float(full_loss.detach())
    for parameter, expected in zip(chunk_model.parameters(), full_gradients):
        assert parameter.grad is not None
        torch.testing.assert_close(parameter.grad, expected, atol=1e-6, rtol=1e-6)


def test_empty_node_chunk_is_skipped_but_empty_update_fails() -> None:
    model = _ToyNodeShared(nodes=5)
    mask = torch.ones(2, 5, 2, dtype=torch.bool)
    mask[:, 4] = False
    plan, result = _backward(model, _batch(mask=mask), 2)
    assert plan.node_chunk_count == 3
    assert result.terms.valid_count == 16

    empty = _batch(mask=torch.zeros(2, 5, 2, dtype=torch.bool))
    with torch.no_grad():
        model.zero_grad(set_to_none=True)
    try:
        _backward(model, empty, 2)
    except ValueError as exc:
        assert "no valid targets" in str(exc)
    else:
        raise AssertionError("an entirely invalid optimizer update must fail")


def test_zero_squared_error_has_finite_zero_rmse_contribution() -> None:
    model = _ToyNodeShared(nodes=5)
    batch = _batch()
    with torch.no_grad():
        target = model(batch.model_input()).detach()
    exact = ForecastBatch(batch.x, target, batch.target_mask, batch.starts)
    plan, result = _backward(model, exact, 2)
    assert plan.uses_node_microbatch
    assert result.loss == 0.0
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)


def test_dropout_two_pass_replays_rng_once_and_is_repeatable() -> None:
    torch.manual_seed(2026)
    template = _ToyNodeShared(nodes=5, dropout=0.25)
    state = template.state_dict()
    batch = _batch()

    def run() -> tuple[float, list[torch.Tensor], torch.Tensor]:
        model = _ToyNodeShared(nodes=5, dropout=0.25)
        model.load_state_dict(state)
        model.train()
        result = _backward(model, batch, 2)[1]
        gradients = [parameter.grad.detach().clone() for parameter in model.parameters()]
        return result.loss, gradients, torch.get_rng_state().clone()

    torch.manual_seed(99)
    first = run()
    torch.manual_seed(99)
    second = run()
    assert first[0] == second[0]
    for left, right in zip(first[1], second[1]):
        torch.testing.assert_close(left, right, atol=0.0, rtol=0.0)
    torch.testing.assert_close(first[2], second[2], atol=0, rtol=0)

    torch.manual_seed(99)
    model = _ToyNodeShared(nodes=5, dropout=0.25).train()
    model.load_state_dict(state)
    with torch.no_grad():
        for start, end in build_execution_plan(model, total_nodes=5, node_shared_chunk_size=2).node_ranges():
            model.forward_node_chunk(batch.model_input(), start, end)
    expected_rng = torch.get_rng_state().clone()
    assert torch.equal(first[2], expected_rng)


def test_batch_norm_node_shared_model_forces_full_nodes() -> None:
    model = _ToyNodeShared(nodes=5, batch_norm=True)
    plan = build_execution_plan(model, total_nodes=5, node_shared_chunk_size=2)
    assert plan.execution_mode == "full_nodes"
    assert plan.batch_dependent_norm_detected is True
    assert plan.node_chunk_count == 1
    assert plan.reason == "batch_dependent_normalization"


def test_spatial_model_and_current_model_boundaries() -> None:
    spatial_plan = build_execution_plan(_SpatialToy(), total_nodes=134, node_shared_chunk_size=32)
    assert spatial_plan.execution_mode == "full_spatiotemporal"
    assert spatial_plan.uses_node_microbatch is False
    assert issubclass(Crossformer, NodeSharedForecastModel)
    assert issubclass(STCN, ForecastModel)
    assert not issubclass(STCN, NodeSharedForecastModel)

    root_config = load_experiment_config()
    lstm = build_model(
        "lstm",
        {"hidden_dim": 4, "num_layers": 1, "dropout": 0.0},
        {
            "num_nodes": 134,
            "num_features": len(root_config.data["feature_columns"]),
            "lookback": root_config.data["lookback"],
            "max_pred_len": root_config.data["max_pred_len"],
        },
    )
    lstm_plan = build_execution_plan(
        lstm,
        total_nodes=134,
        node_shared_chunk_size=int(root_config.runtime["node_shared_chunk_size"]),
    )
    assert lstm_plan.execution_mode == "node_shared_microbatch"
    assert [end - start for start, end in lstm_plan.node_ranges()] == [32, 32, 32, 32, 6]

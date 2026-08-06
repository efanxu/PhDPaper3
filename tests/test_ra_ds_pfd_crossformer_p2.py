from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from engine.reproducibility import set_seed
from models.base import DataInfoView, ModelInput
from models.loader import build_model
from models.ra_ds_pfd_crossformer.pfd0 import build_wspd_level_diff1
from models.ra_ds_pfd_crossformer.relation_spatial import (
    RelationSpatialAttention,
    ordered_relation_pair_representation,
)


ROOT = Path(__file__).resolve().parents[1]


def _info() -> DataInfoView:
    # Wspd intentionally is not column zero: the model must resolve it by name.
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


def _config() -> dict[str, object]:
    return {
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
        "spatial_dropout": 0.0,
        "gamma_init": 0.1,
        "spatial_edge_chunk_size": 2,
        "relation_resource": {
            "file": "tests/fixtures/ra_ds_pfd_relation_small_v1.npz",
        },
    }


def _build() -> torch.nn.Module:
    return build_model("ra_ds_pfd_crossformer", _config(), _info())


def _dense_reference(
    layer: RelationSpatialAttention,
    self_tokens: torch.Tensor,
    propagation_tokens: torch.Tensor,
    edge_index: torch.Tensor,
    edge_bias: torch.Tensor,
    relation_bias: torch.Tensor,
) -> torch.Tensor:
    batch, nodes, channels, segments, _ = self_tokens.shape
    q = layer.q_projection(self_tokens).reshape(
        batch, nodes, channels, segments, layer.spatial_heads, layer.head_dim
    )
    k = layer.k_projection(propagation_tokens).reshape(
        batch, nodes, segments, layer.spatial_heads, layer.head_dim
    )
    v = layer.v_projection(propagation_tokens).reshape(
        batch, nodes, segments, layer.spatial_heads, layer.head_dim
    )
    source, target = edge_index
    message = torch.zeros(
        batch,
        nodes,
        channels,
        segments,
        layer.spatial_heads,
        layer.head_dim,
        dtype=self_tokens.dtype,
    )
    for batch_index in range(batch):
        for target_index in range(nodes):
            edge_ids = torch.nonzero(target == target_index, as_tuple=False).flatten()
            if edge_ids.numel() == 0:
                continue
            for channel in range(channels):
                for segment in range(segments):
                    for head in range(layer.spatial_heads):
                        logits = torch.stack(
                            [
                                (
                                    q[batch_index, target_index, channel, segment, head]
                                    * k[batch_index, source[edge_id], segment, head]
                                ).sum()
                                / math.sqrt(layer.head_dim)
                                + edge_bias[edge_id, head]
                                + relation_bias[edge_id, head]
                                for edge_id in edge_ids
                            ]
                        )
                        weights = torch.softmax(logits, dim=0)
                        message[batch_index, target_index, channel, segment, head] = sum(
                            weights[position]
                            * v[batch_index, source[edge_id], segment, head]
                            for position, edge_id in enumerate(edge_ids)
                        )
    return layer.out_projection(message.reshape(batch, nodes, channels, segments, layer.d_model))


def test_sparse_target_softmax_scatter_matches_dense_reference_and_gradients() -> None:
    torch.manual_seed(10)
    edge_index = torch.tensor([[1, 0, 2, 1], [0, 1, 1, 2]], dtype=torch.long)
    edge_bias = torch.randn(4, 2)
    relation_bias = torch.randn(4, 2)
    chunked = RelationSpatialAttention(4, 2, 0.0, edge_chunk_size=2)
    full = RelationSpatialAttention(4, 2, 0.0, edge_chunk_size=None)
    full.load_state_dict(chunked.state_dict())
    self_chunked = torch.randn(1, 3, 2, 2, 4, requires_grad=True)
    propagation_chunked = torch.randn(1, 3, 2, 4, requires_grad=True)
    self_full = self_chunked.detach().clone().requires_grad_(True)
    propagation_full = propagation_chunked.detach().clone().requires_grad_(True)

    sparse = chunked(
        self_chunked,
        propagation_chunked,
        edge_index,
        edge_bias=edge_bias,
        relation_bias=relation_bias,
    )
    unchunked = full(
        self_full,
        propagation_full,
        edge_index,
        edge_bias=edge_bias,
        relation_bias=relation_bias,
    )
    dense = _dense_reference(
        full,
        self_full,
        propagation_full,
        edge_index,
        edge_bias,
        relation_bias,
    )
    torch.testing.assert_close(sparse, unchunked, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(unchunked, dense, atol=1e-6, rtol=1e-6)

    weight = torch.randn_like(sparse)
    (sparse * weight).sum().backward()
    (unchunked * weight).sum().backward()
    assert self_chunked.grad is not None and self_full.grad is not None
    assert propagation_chunked.grad is not None and propagation_full.grad is not None
    torch.testing.assert_close(self_chunked.grad, self_full.grad, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(propagation_chunked.grad, propagation_full.grad, atol=1e-6, rtol=1e-6)
    for left, right in zip(chunked.parameters(), full.parameters()):
        assert left.grad is not None and right.grad is not None
        torch.testing.assert_close(left.grad, right.grad, atol=1e-6, rtol=1e-6)


def test_edge_permutation_preserves_node_output_and_biases_are_logits_only() -> None:
    torch.manual_seed(11)
    edge_index = torch.tensor([[1, 0, 2, 1], [0, 1, 1, 2]], dtype=torch.long)
    permutation = torch.tensor([2, 0, 3, 1])
    layer = RelationSpatialAttention(4, 2, 0.0, edge_chunk_size=2).eval()
    self_tokens = torch.randn(1, 3, 2, 2, 4)
    propagation = torch.randn(1, 3, 2, 4)
    edge_bias = torch.randn(4, 2)
    relation_bias = torch.randn(4, 2)
    output, base = layer(
        self_tokens,
        propagation,
        edge_index,
        edge_bias=edge_bias,
        relation_bias=relation_bias,
        return_diagnostics=True,
    )
    permuted_output, permuted = layer(
        self_tokens,
        propagation,
        edge_index[:, permutation],
        edge_bias=edge_bias[permutation],
        relation_bias=relation_bias[permutation],
        return_diagnostics=True,
    )
    torch.testing.assert_close(output, permuted_output, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(base["value"], permuted["value"][:, torch.argsort(permutation)], atol=1e-6, rtol=1e-6)

    zero_output, zero_bias = layer(
        self_tokens,
        propagation,
        edge_index,
        edge_bias=torch.zeros_like(edge_bias),
        relation_bias=torch.zeros_like(relation_bias),
        return_diagnostics=True,
    )
    assert not torch.allclose(base["attention"], zero_bias["attention"])
    torch.testing.assert_close(base["value"], zero_bias["value"], atol=0.0, rtol=0.0)
    assert not torch.allclose(output, zero_output)


def test_ordered_relation_pairs_distinguish_reversed_edges() -> None:
    relation_embedding = torch.tensor([[1.0, 2.0], [3.0, 5.0]])
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    pair = ordered_relation_pair_representation(relation_embedding, edge_index)
    assert pair.shape == (2, 10)
    assert not torch.equal(pair[0], pair[1])


def test_pfd0_is_name_resolved_causal_and_power_independent() -> None:
    model = _build().eval()
    assert model.pfd0 is not None
    x = torch.randn(1, 24, 3, 5)
    candidate = model.pfd0.candidate_history(x)
    expected = build_wspd_level_diff1(x, 2)
    torch.testing.assert_close(candidate, expected, atol=0.0, rtol=0.0)
    changed_non_wspd = x.clone()
    changed_non_wspd[..., [0, 1, 3, 4]] += 17.0
    torch.testing.assert_close(candidate, model.pfd0.candidate_history(changed_non_wspd), atol=0.0, rtol=0.0)

    before = model.pfd0(x)[0]
    changed_future = x.clone()
    changed_future[:, 12:] += 23.0
    after = model.pfd0(changed_future)[0]
    torch.testing.assert_close(before[:, :, :1], after[:, :, :1], atol=0.0, rtol=0.0)
    assert {"target", "target_mask"}.isdisjoint(ModelInput.__dataclass_fields__)


def test_p2_calls_each_scale_once_uses_local_tokens_and_has_gate_free_gradients() -> None:
    torch.manual_seed(12)
    model = _build().train()
    x = torch.randn(2, 24, 3, 5)
    calls: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {0: [], 1: []}

    def record(scale: int):
        def hook(_module, args, output):
            calls[scale].append((args[0], output))

        return hook

    handles = [
        model.backbone.scale0_spatial_insertion.register_forward_hook(record(0)),
        model.backbone.scale1_spatial_insertion.register_forward_hook(record(1)),
    ]
    try:
        output = model(ModelInput(x=x))
    finally:
        for handle in handles:
            handle.remove()
    assert tuple(output.shape) == (2, 3, 3)
    assert len(calls[0]) == 1 and len(calls[1]) == 1
    assert tuple(calls[0][0][0].shape) == (2, 3, 5, 2, 8)
    assert tuple(calls[1][0][0].shape) == (2, 3, 5, 1, 8)
    trace = model.forward_canonical_trace(ModelInput(x=x))
    assert tuple(trace.scale0_cross_time.shape) == (2, 3, 5, 2, 8)
    assert tuple(trace.scale1_cross_time.shape) == (2, 3, 5, 1, 8)
    assert [tuple(token.shape) for token in trace.decoder_tokens] == [
        (2, 3, 5, 2, 8),
        (2, 3, 5, 2, 8),
        (2, 3, 5, 1, 8),
    ]

    output.square().mean().backward()
    for insertion in (
        model.backbone.scale0_spatial_insertion,
        model.backbone.scale1_spatial_insertion,
    ):
        assert float(insertion.gamma.detach()) == pytest.approx(0.1)
        assert insertion.gamma.grad is not None and torch.isfinite(insertion.gamma.grad).all()
    provider = model.relation_bias_provider
    assert provider is not None
    for parameter in (
        provider.relation_embedding,
        provider.static_edge_mlp.net[0].weight,
        provider.relation_bias_mlp.net[0].weight,
        model.pfd0.segment_embedding.value_projection.weight,
    ):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    parameter_names = set(dict(model.named_parameters()))
    assert not any("gate_mlp" in name for name in parameter_names)
    assert "VariableConditionedResidualInjection" not in repr(model)


def test_p2_controlled_nonstrict_forward_backward_is_finite(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "2026")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    details = set_seed(2026, reproducibility_mode="controlled_nonstrict")
    assert details["global_deterministic_algorithms"] is False
    assert details["cudnn_deterministic"] is True
    assert details["cudnn_benchmark"] is False

    model = _build().train()
    output = model(ModelInput(x=torch.randn(2, 24, 3, 5)))
    loss = output.square().mean()
    loss.backward()

    assert torch.isfinite(output).all()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_legacy_spatial_disabled_config_has_no_p2_parameters_or_resource_read() -> None:
    legacy = {
        "d_model": 8,
        "n_heads": 2,
        "d_ff": 16,
        "e_layers": 2,
        "dropout": 0.0,
        "factor": 2,
        "seg_len": 12,
        "win_size": 2,
        "spatial_disabled": True,
        "pfd_mode": "pfd0",
        "relation_resource": {
            "file": "does/not/exist.npz",
        },
    }
    model = build_model("ra_ds_pfd_crossformer", legacy, _info())
    assert model.pfd0 is None
    assert model.relation_bias_provider is None
    assert not any("relation_bias_provider" in key or "pfd0" in key for key in model.state_dict())


def test_unimplemented_pfd_modes_fail_closed() -> None:
    config = _config()
    config["pfd_mode"] = "pfd1"
    with pytest.raises(ValueError, match="pfd_mode=pfd0"):
        build_model("ra_ds_pfd_crossformer", config, _info())

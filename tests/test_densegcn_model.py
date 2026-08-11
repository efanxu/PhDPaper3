from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from models.base import DataInfoView, ModelInput
from models.densegcn.model import (
    DenseGCN,
    WeightedGraphConv,
    _validate_config,
    dense_adjacency_from_edges,
)


CONFIG = {"hidden_dim": 8, "num_layers": 2, "dropout": 0.0}


def _info(nodes: int = 4) -> DataInfoView:
    return DataInfoView(
        num_nodes=nodes,
        num_features=3,
        lookback=5,
        max_pred_len=2,
        node_ids=tuple(range(1, nodes + 1)),
        graph_config={},
        project_root=Path.cwd(),
    )


def _adjacency(nodes: int = 4) -> torch.Tensor:
    adjacency = torch.zeros(nodes, nodes)
    for source in range(nodes):
        adjacency[source, (source + 1) % nodes] = 1.0
        adjacency[source, (source - 1) % nodes] = 1.0
    return adjacency


def test_dense_graph_layer_matches_legacy_propagation_order() -> None:
    torch.manual_seed(7)
    layer = WeightedGraphConv(5)
    x = torch.randn(2, 4, 5)
    adjacency = _adjacency()

    expected = layer.linear(torch.einsum("ij,bjd->bid", adjacency, x))
    actual = layer(x, adjacency)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_densegcn_matches_legacy_network_structure() -> None:
    torch.manual_seed(11)
    info = _info()
    adjacency = _adjacency()
    model = DenseGCN(data_info=info, model_config=CONFIG, adjacency=adjacency).eval()
    x = torch.randn(2, info.lookback, info.num_nodes, info.num_features)

    hidden = x.permute(0, 2, 1, 3).contiguous().reshape(2, info.num_nodes, -1)
    hidden = model.activation(model.input_proj(hidden))
    for conv, norm in zip(model.gcn_layers, model.norms):
        propagated = torch.einsum("ij,bjd->bid", adjacency, hidden)
        hidden = norm(hidden + model.dropout(model.activation(conv.linear(propagated))))
    expected = model.readout(hidden)

    actual = model(ModelInput(x=x))
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_densegcn_interface_adjacency_and_gradients() -> None:
    info = _info()
    adjacency = _adjacency()
    edge_index = torch.nonzero(adjacency, as_tuple=False).T.contiguous().long()
    edge_weight = adjacency[edge_index[0], edge_index[1]]
    rebuilt = dense_adjacency_from_edges(edge_index, edge_weight, num_nodes=info.num_nodes)
    model = DenseGCN(data_info=info, model_config=CONFIG, adjacency=rebuilt)
    x = torch.randn(2, info.lookback, info.num_nodes, info.num_features)

    output = model(ModelInput(x=x))
    output.square().mean().backward()

    assert model.uses_public_graph_resource is True
    assert model.execution_mode == "full_spatiotemporal"
    assert tuple(model.adjacency.shape) == (info.num_nodes, info.num_nodes)
    assert model.adjacency.dtype == torch.float32
    assert torch.equal(model.adjacency, adjacency)
    assert tuple(output.shape) == (2, info.num_nodes, info.max_pred_len)
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_densegcn_rejects_unknown_config() -> None:
    with pytest.raises(ValueError, match="unknown field"):
        _validate_config({**CONFIG, "unknown": 1})


def test_densegcn_has_no_sparse_graph_modules() -> None:
    model = DenseGCN(data_info=_info(), model_config=CONFIG, adjacency=_adjacency())
    names = [type(module).__module__ + "." + type(module).__name__ for module in model.modules()]
    assert not any(name.startswith(("torch_geometric.", "torch_scatter.", "tsl.")) for name in names)
    assert all(isinstance(layer.linear, nn.Linear) for layer in model.gcn_layers)

"""Sparse target-directed PFD0 relation spatial attention."""

from __future__ import annotations

import math

import torch
from torch import nn


class StaticEdgeMLP(nn.Module):
    """Map the 13 immutable edge features to per-head logit biases."""

    def __init__(self, in_features: int, hidden: int, heads: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_features), int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), int(heads)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(features))


def ordered_relation_pair_representation(
    relation_embedding: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """Build the ordered ``(target, source)`` relation representation."""

    source, target = edge_index[0].long(), edge_index[1].long()
    target_embedding = relation_embedding[target]
    source_embedding = relation_embedding[source]
    difference = target_embedding - source_embedding
    return torch.cat(
        (
            target_embedding,
            source_embedding,
            difference,
            difference.abs(),
            target_embedding * source_embedding,
        ),
        dim=-1,
    )


class OrderedRelationBias(nn.Module):
    """Map ordered relation pairs to per-head logit biases."""

    def __init__(self, relation_dim: int, hidden: int, heads: int) -> None:
        super().__init__()
        self.relation_dim = int(relation_dim)
        self.net = nn.Sequential(
            nn.Linear(5 * self.relation_dim, int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), int(heads)),
        )

    def forward(self, relation_embedding: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(ordered_relation_pair_representation(relation_embedding, edge_index)))


class RelationBiasProvider(nn.Module):
    """Shared static and ordered bias parameters for both encoder scales."""

    def __init__(
        self,
        *,
        edge_index: torch.Tensor,
        edge_static_features: torch.Tensor,
        num_nodes: int,
        spatial_heads: int,
        spatial_d_ff: int,
        relation_dim: int,
    ) -> None:
        super().__init__()
        if edge_index.ndim != 2 or tuple(edge_index.shape[:1]) != (2,):
            raise ValueError("relation edge_index must have shape (2, E)")
        if edge_static_features.ndim != 2 or tuple(edge_static_features.shape) != (
            edge_index.shape[1],
            13,
        ):
            raise ValueError("relation edge_static_features must have shape (E, 13)")
        if not torch.isfinite(edge_static_features).all():
            raise ValueError("relation edge_static_features contain NaN or Inf")
        self.num_nodes = int(num_nodes)
        self.spatial_heads = int(spatial_heads)
        self.register_buffer("edge_index", edge_index.detach().clone().long(), persistent=True)
        self.register_buffer(
            "edge_static_features",
            edge_static_features.detach().clone().float(),
            persistent=True,
        )
        self.relation_embedding = nn.Parameter(
            torch.randn(self.num_nodes, int(relation_dim)) * 0.02
        )
        self.static_edge_mlp = StaticEdgeMLP(13, int(spatial_d_ff), self.spatial_heads)
        self.relation_bias_mlp = OrderedRelationBias(
            int(relation_dim), int(spatial_d_ff), self.spatial_heads
        )

    @property
    def edge_count(self) -> int:
        return int(self.edge_index.shape[1])

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.static_edge_mlp(self.edge_static_features),
            self.relation_bias_mlp(self.relation_embedding, self.edge_index),
        )


def _group_reduce(
    values: torch.Tensor,
    target: torch.Tensor,
    *,
    num_nodes: int,
    reduce: str,
) -> torch.Tensor:
    """Reduce ``[B,E,C,S,H]`` values by target without a dense N-by-N tensor."""

    batch, edges, channels, segments, heads = values.shape
    rows = values.permute(0, 2, 3, 4, 1).reshape(-1, edges)
    index = target.view(1, edges).expand(rows.shape[0], edges)
    if reduce == "amax":
        result = torch.full(
            (rows.shape[0], int(num_nodes)),
            -torch.inf,
            dtype=rows.dtype,
            device=rows.device,
        )
        result.scatter_reduce_(1, index, rows, reduce="amax", include_self=True)
    elif reduce == "add":
        result = torch.zeros(
            (rows.shape[0], int(num_nodes)),
            dtype=rows.dtype,
            device=rows.device,
        )
        result.scatter_add_(1, index, rows)
    else:
        raise ValueError(f"unsupported relation group reduction: {reduce}")
    return result.reshape(batch, channels, segments, heads, int(num_nodes))


class RelationSpatialAttention(nn.Module):
    """Target-query/source-value sparse attention over ordered relation edges."""

    def __init__(
        self,
        d_model: int,
        spatial_heads: int,
        dropout: float,
        *,
        edge_chunk_size: int | None = 128,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.spatial_heads = int(spatial_heads)
        if self.d_model % self.spatial_heads:
            raise ValueError("relation spatial d_model must be divisible by spatial_heads")
        self.head_dim = self.d_model // self.spatial_heads
        if edge_chunk_size is not None and int(edge_chunk_size) < 1:
            raise ValueError("relation spatial edge_chunk_size must be positive or None")
        self.edge_chunk_size = None if edge_chunk_size is None else int(edge_chunk_size)
        self.q_projection = nn.Linear(self.d_model, self.d_model)
        self.k_projection = nn.Linear(self.d_model, self.d_model)
        self.v_projection = nn.Linear(self.d_model, self.d_model)
        self.out_projection = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(float(dropout))

    def _chunks(self, edge_count: int) -> list[tuple[int, int]]:
        width = self.edge_chunk_size or edge_count
        return [(start, min(start + width, edge_count)) for start in range(0, edge_count, width)]

    def _score_chunk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        edge_index: torch.Tensor,
        start: int,
        end: int,
        edge_bias: torch.Tensor | None,
        relation_bias: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source = edge_index[0, start:end].long()
        target = edge_index[1, start:end].long()
        q_target = q[:, target]
        k_source = k[:, source].unsqueeze(2)
        content = (q_target * k_source).sum(dim=-1) / math.sqrt(self.head_dim)
        score = content
        if edge_bias is not None:
            score = score + edge_bias[start:end].view(1, end - start, 1, 1, self.spatial_heads)
        if relation_bias is not None:
            score = score + relation_bias[start:end].view(1, end - start, 1, 1, self.spatial_heads)
        return score, content, source

    @staticmethod
    def _edge_values(values: torch.Tensor, target: torch.Tensor, maximum: torch.Tensor) -> torch.Tensor:
        # values: [B, E, C, S, H], maximum: [B, C, S, H, N]
        return maximum.index_select(-1, target).permute(0, 4, 1, 2, 3)

    def forward(
        self,
        self_tokens: torch.Tensor,
        propagation_tokens: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        edge_bias: torch.Tensor | None = None,
        relation_bias: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self_tokens.ndim != 5:
            raise ValueError("relation spatial self_tokens must have shape (B, N, C, S, D)")
        if propagation_tokens.ndim != 4:
            raise ValueError("relation spatial propagation_tokens must have shape (B, N, S, D)")
        batch, nodes, channels, segments, d_model = self_tokens.shape
        if tuple(propagation_tokens.shape) != (batch, nodes, segments, d_model):
            raise ValueError("relation spatial propagation tokens do not align with self tokens")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.shape[1] == 0:
            raise ValueError("relation spatial edge_index must have shape (2, E) with E > 0")
        edge_count = int(edge_index.shape[1])
        if edge_bias is not None and tuple(edge_bias.shape) != (edge_count, self.spatial_heads):
            raise ValueError("static relation bias must have shape (E, spatial_heads)")
        if relation_bias is not None and tuple(relation_bias.shape) != (edge_count, self.spatial_heads):
            raise ValueError("ordered relation bias must have shape (E, spatial_heads)")

        q = self.q_projection(self_tokens).reshape(
            batch, nodes, channels, segments, self.spatial_heads, self.head_dim
        )
        k = self.k_projection(propagation_tokens).reshape(
            batch, nodes, segments, self.spatial_heads, self.head_dim
        )
        v = self.v_projection(propagation_tokens).reshape(
            batch, nodes, segments, self.spatial_heads, self.head_dim
        )
        target_all = edge_index[1].long()
        chunks = self._chunks(edge_count)

        maxima = torch.full(
            (batch, channels, segments, self.spatial_heads, nodes),
            -torch.inf,
            dtype=self_tokens.dtype,
            device=self_tokens.device,
        )
        for start, end in chunks:
            score, _content, _source = self._score_chunk(
                q, k, edge_index, start, end, edge_bias, relation_bias
            )
            maxima = torch.maximum(
                maxima,
                _group_reduce(
                    score,
                    target_all[start:end],
                    num_nodes=nodes,
                    reduce="amax",
                ),
            )

        denominator = torch.zeros_like(maxima)
        for start, end in chunks:
            score, _content, _source = self._score_chunk(
                q, k, edge_index, start, end, edge_bias, relation_bias
            )
            target = target_all[start:end]
            shifted = score.permute(0, 2, 3, 4, 1) - self._edge_values(
                score, target, maxima
            ).permute(0, 2, 3, 4, 1)
            exp_score = shifted.exp().permute(0, 4, 1, 2, 3)
            denominator = denominator + _group_reduce(
                exp_score,
                target,
                num_nodes=nodes,
                reduce="add",
            )

        message = torch.zeros(
            (batch, channels, segments, self.spatial_heads, nodes, self.head_dim),
            dtype=self_tokens.dtype,
            device=self_tokens.device,
        )
        weight_chunks: list[torch.Tensor] = []
        content_chunks: list[torch.Tensor] = []
        score_chunks: list[torch.Tensor] = []
        value_chunks: list[torch.Tensor] = []
        tiny = torch.finfo(self_tokens.dtype).tiny
        for start, end in chunks:
            score, content, source = self._score_chunk(
                q, k, edge_index, start, end, edge_bias, relation_bias
            )
            target = target_all[start:end]
            shifted = score.permute(0, 2, 3, 4, 1) - self._edge_values(
                score, target, maxima
            ).permute(0, 2, 3, 4, 1)
            exp_score = shifted.exp().permute(0, 4, 1, 2, 3)
            denominator_edges = denominator.index_select(-1, target).permute(0, 4, 1, 2, 3)
            weights = exp_score / denominator_edges.clamp_min(tiny)
            source_values = v[:, source]
            dropped_weights = self.dropout(weights)
            weighted = dropped_weights.unsqueeze(-1) * source_values.unsqueeze(2)
            rows = weighted.permute(0, 2, 3, 4, 1, 5)
            scatter_index = target.view(1, 1, 1, 1, end - start, 1).expand_as(rows)
            message.scatter_add_(4, scatter_index, rows)
            if return_diagnostics:
                weight_chunks.append(weights)
                content_chunks.append(content)
                score_chunks.append(score)
                value_chunks.append(source_values)

        message = (
            message.permute(0, 4, 1, 2, 3, 5)
            .reshape(batch, nodes, channels, segments, self.d_model)
        )
        output = self.out_projection(message)
        if not torch.isfinite(output).all():
            raise FloatingPointError("relation spatial output contains NaN or Inf")
        if not return_diagnostics:
            return output
        attention = torch.cat(weight_chunks, dim=1)
        content = torch.cat(content_chunks, dim=1)
        score = torch.cat(score_chunks, dim=1)
        source_values = torch.cat(value_chunks, dim=1)
        entropy_values = attention.clamp_min(1e-12)
        diagnostics = {
            "attention": attention,
            "content": content,
            "score": score,
            "value": source_values,
            "message": message,
            "entropy": -(entropy_values * entropy_values.log()).mean(),
        }
        return output, diagnostics


class RelationSpatialInsertion(nn.Module):
    """Apply relation message output through the gate-free residual."""

    def __init__(
        self,
        *,
        d_model: int,
        spatial_heads: int,
        spatial_dropout: float,
        gamma_init: float,
        bias_provider: RelationBiasProvider,
        edge_chunk_size: int | None = 128,
    ) -> None:
        super().__init__()
        self.attention = RelationSpatialAttention(
            d_model,
            spatial_heads,
            spatial_dropout,
            edge_chunk_size=edge_chunk_size,
        )
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.output_dropout = nn.Dropout(float(spatial_dropout))
        # The provider is registered once on the model, so both scale
        # insertions share the one [N, relation_dim] parameter and its two
        # bias MLPs without duplicating state-dict entries.
        object.__setattr__(self, "_bias_provider", bias_provider)

    def forward(
        self,
        self_tokens: torch.Tensor,
        propagation_tokens: torch.Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        edge_bias, relation_bias = self._bias_provider()
        result = self.attention(
            self_tokens,
            propagation_tokens,
            self._bias_provider.edge_index,
            edge_bias=edge_bias,
            relation_bias=relation_bias,
            return_diagnostics=return_diagnostics,
        )
        if return_diagnostics:
            message, diagnostics = result
            residual = self_tokens + self.gamma * self.output_dropout(message)
            diagnostics = {**diagnostics, "residual": residual}
            return residual, diagnostics
        return self_tokens + self.gamma * self.output_dropout(result)


__all__ = [
    "OrderedRelationBias",
    "RelationBiasProvider",
    "RelationSpatialAttention",
    "RelationSpatialInsertion",
    "StaticEdgeMLP",
    "ordered_relation_pair_representation",
]

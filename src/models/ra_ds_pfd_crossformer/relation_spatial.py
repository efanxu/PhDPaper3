"""Sparse target-directed PFD0 relation spatial attention."""

from __future__ import annotations

import math

import torch
from torch import nn


class TurbineIdentityEmbedding(nn.Module):
    """One base identity used by temporal Self tokens and relation bias."""

    def __init__(self, num_nodes: int, base_turbine_dim: int, d_model: int, relation_dim: int) -> None:
        super().__init__()
        self.base_turbine_embedding = nn.Parameter(
            torch.randn(int(num_nodes), int(base_turbine_dim)) * 0.02
        )
        self.temporal_projection = nn.Linear(int(base_turbine_dim), int(d_model))
        self.relation_projection = nn.Linear(int(base_turbine_dim), int(relation_dim))

    def temporal_tokens(self) -> torch.Tensor:
        return self.temporal_projection(self.base_turbine_embedding).view(
            1,
            self.base_turbine_embedding.shape[0],
            1,
            1,
            -1,
        )

    def relation_embedding(self) -> torch.Tensor:
        return self.relation_projection(self.base_turbine_embedding)


class LocalVariablePool(nn.Module):
    """Old-prototype variable pooling used only by the node-pooled query mode."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.pool_projection = nn.Linear(int(d_model), int(d_model))
        self.pool_score = nn.Linear(int(d_model), 1)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 5:
            raise ValueError("LocalVariablePool expects [B,N,C,S,D]")
        hidden = torch.tanh(self.pool_projection(tokens))
        score = self.pool_score(hidden)
        weight = torch.softmax(score, dim=2)
        q_node = (weight * tokens).sum(dim=2)
        return weight, q_node


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
        turbine_embedding_mode: str = "relation_only",
        bias_scaling_mode: str = "direct",
        turbine_identity: TurbineIdentityEmbedding | None = None,
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
        self.turbine_embedding_mode = str(turbine_embedding_mode)
        self.bias_scaling_mode = str(bias_scaling_mode)
        if self.turbine_embedding_mode not in {"relation_only", "temporal_and_relation"}:
            raise ValueError(f"unsupported turbine_embedding_mode: {self.turbine_embedding_mode}")
        if self.bias_scaling_mode not in {"direct", "learnable_per_scale"}:
            raise ValueError(f"unsupported bias_scaling_mode: {self.bias_scaling_mode}")
        if self.turbine_embedding_mode == "temporal_and_relation" and turbine_identity is None:
            raise ValueError("temporal_and_relation requires one shared turbine identity embedding")
        if self.turbine_embedding_mode == "relation_only" and turbine_identity is not None:
            raise ValueError("relation_only must not receive a temporal turbine identity embedding")
        object.__setattr__(self, "_turbine_identity", turbine_identity)
        self.register_buffer("edge_index", edge_index.detach().clone().long(), persistent=True)
        self.register_buffer(
            "edge_static_features",
            edge_static_features.detach().clone().float(),
            persistent=True,
        )
        if self.turbine_embedding_mode == "relation_only":
            self.relation_embedding = nn.Parameter(
                torch.randn(self.num_nodes, int(relation_dim)) * 0.02
            )
        self.static_edge_mlp = StaticEdgeMLP(13, int(spatial_d_ff), self.spatial_heads)
        self.relation_bias_mlp = OrderedRelationBias(
            int(relation_dim), int(spatial_d_ff), self.spatial_heads
        )
        if self.bias_scaling_mode == "learnable_per_scale":
            self.lambda_edge = nn.Parameter(torch.full((2, self.spatial_heads), 0.05))
            self.lambda_relation = nn.Parameter(torch.full((2, self.spatial_heads), 0.01))

    @property
    def edge_count(self) -> int:
        return int(self.edge_index.shape[1])

    def forward(self, scale_id: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(scale_id, int) or isinstance(scale_id, bool) or scale_id not in (0, 1):
            raise ValueError("relation bias scale_id must be 0 or 1")
        if self.turbine_embedding_mode == "relation_only":
            relation_embedding = self.relation_embedding
        else:
            assert self._turbine_identity is not None
            relation_embedding = self._turbine_identity.relation_embedding()
        edge_bias = self.static_edge_mlp(self.edge_static_features)
        relation_bias = self.relation_bias_mlp(relation_embedding, self.edge_index)
        if self.bias_scaling_mode == "learnable_per_scale":
            edge_bias = edge_bias * self.lambda_edge[scale_id].view(1, self.spatial_heads)
            relation_bias = relation_bias * self.lambda_relation[scale_id].view(1, self.spatial_heads)
        return edge_bias, relation_bias


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
    if reduce == "amax":
        index = target.view(1, edges).expand(rows.shape[0], edges)
        result = torch.full(
            (rows.shape[0], int(num_nodes)),
            -torch.inf,
            dtype=rows.dtype,
            device=rows.device,
        )
        result.scatter_reduce_(1, index, rows, reduce="amax", include_self=True)
    elif reduce == "add":
        result = _deterministic_group_add_rows(rows, target, num_nodes=num_nodes)
    else:
        raise ValueError(f"unsupported relation group reduction: {reduce}")
    return result.reshape(batch, channels, segments, heads, int(num_nodes))


def _deterministic_group_add_rows(
    rows: torch.Tensor,
    target: torch.Tensor,
    *,
    num_nodes: int,
) -> torch.Tensor:
    """Sum duplicate target rows in a fixed edge order on every device.

    CUDA ``scatter_add_`` uses atomic accumulation for duplicate indices, so
    its floating-point reduction order can vary between otherwise identical
    forwards. Sorting the fixed relation edge list and taking a segmented sum
    gives each target a stable input-order reduction while preserving autograd.
    """

    edge_count = int(rows.shape[1])
    if target.ndim != 1 or int(target.shape[0]) != edge_count:
        raise ValueError("group-add target must be a vector aligned with rows")
    order = torch.argsort(target, stable=True)
    sorted_target = target.index_select(0, order)
    counts = torch.bincount(sorted_target, minlength=int(num_nodes))
    sorted_rows = rows.index_select(1, order)
    segment_reduce = getattr(torch, "segment_reduce", None)
    if segment_reduce is not None:
        return segment_reduce(
            sorted_rows.transpose(0, 1),
            "sum",
            lengths=counts,
            axis=0,
        ).transpose(0, 1)
    groups = []
    start = 0
    for count in counts:
        count_value = int(count)
        groups.append(sorted_rows[:, start : start + count_value].sum(dim=1))
        start += count_value
    return torch.stack(groups, dim=1)


def _group_add(
    values: torch.Tensor,
    target: torch.Tensor,
    *,
    edge_dim: int,
    num_nodes: int,
) -> torch.Tensor:
    """Aggregate one edge dimension into one target-node dimension."""

    moved = values.movedim(edge_dim, -1)
    rows = moved.reshape(-1, moved.shape[-1])
    reduced = _deterministic_group_add_rows(rows, target, num_nodes=num_nodes)
    return reduced.reshape(*moved.shape[:-1], int(num_nodes)).movedim(-1, edge_dim)


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
            message = message + _group_add(rows, target, edge_dim=4, num_nodes=nodes)
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


def _group_reduce_node(
    values: torch.Tensor,
    target: torch.Tensor,
    *,
    num_nodes: int,
    reduce: str,
) -> torch.Tensor:
    """Reduce ``[B,E,S,H]`` values by target without making an N-by-N tensor."""

    batch, edges = values.shape[:2]
    tail = values.shape[2:]
    rows = values.permute(0, *range(2, values.ndim), 1).reshape(-1, edges)
    if reduce == "amax":
        index = target.view(1, edges).expand(rows.shape[0], edges)
        result = torch.full(
            (rows.shape[0], int(num_nodes)),
            -torch.inf,
            dtype=values.dtype,
            device=values.device,
        )
        result.scatter_reduce_(1, index, rows, reduce="amax", include_self=True)
    elif reduce == "add":
        result = _deterministic_group_add_rows(rows, target, num_nodes=num_nodes)
    else:
        raise ValueError(f"unsupported relation group reduction: {reduce}")
    return result.reshape(batch, *tail, int(num_nodes))


class NodePooledRelationSpatialAttention(nn.Module):
    """Node-level target-query sparse attention with variable pooling."""

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
        self.pool = LocalVariablePool(self.d_model)
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
        k_source = k[:, source]
        content = (q_target * k_source).sum(dim=-1) / math.sqrt(self.head_dim)
        score = content
        if edge_bias is not None:
            score = score + edge_bias[start:end].view(1, end - start, 1, self.spatial_heads)
        if relation_bias is not None:
            score = score + relation_bias[start:end].view(1, end - start, 1, self.spatial_heads)
        return score, content, source

    @staticmethod
    def _edge_maximum(maximum: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return maximum.index_select(-1, target).permute(0, 3, 1, 2)

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
            raise ValueError("node-pooled spatial self_tokens must have shape (B,N,C,S,D)")
        if propagation_tokens.ndim != 4:
            raise ValueError("node-pooled propagation_tokens must have shape (B,N,S,D)")
        batch, nodes, _channels, segments, d_model = self_tokens.shape
        if tuple(propagation_tokens.shape) != (batch, nodes, segments, d_model):
            raise ValueError("node-pooled propagation tokens do not align with self tokens")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.shape[1] == 0:
            raise ValueError("node-pooled edge_index must have shape (2,E) with E > 0")
        edge_count = int(edge_index.shape[1])
        if edge_bias is not None and tuple(edge_bias.shape) != (edge_count, self.spatial_heads):
            raise ValueError("static relation bias must have shape (E, spatial_heads)")
        if relation_bias is not None and tuple(relation_bias.shape) != (edge_count, self.spatial_heads):
            raise ValueError("ordered relation bias must have shape (E, spatial_heads)")

        pool_weight, q_node = self.pool(self_tokens)
        q = self.q_projection(q_node).reshape(
            batch, nodes, segments, self.spatial_heads, self.head_dim
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
            (batch, segments, self.spatial_heads, nodes),
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
                _group_reduce_node(
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
            shifted = score - self._edge_maximum(maxima, target)
            exp_score = shifted.exp()
            denominator = denominator + _group_reduce_node(
                exp_score,
                target,
                num_nodes=nodes,
                reduce="add",
            )

        message = torch.zeros(
            (batch, segments, self.spatial_heads, nodes, self.head_dim),
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
            shifted = score - self._edge_maximum(maxima, target)
            weights = shifted.exp() / self._edge_maximum(denominator, target).clamp_min(tiny)
            source_values = v[:, source]
            dropped_weights = self.dropout(weights)
            weighted = dropped_weights.unsqueeze(-1) * source_values
            rows = weighted.permute(0, 2, 3, 1, 4)
            message = message + _group_add(rows, target, edge_dim=3, num_nodes=nodes)
            if return_diagnostics:
                weight_chunks.append(weights)
                content_chunks.append(content)
                score_chunks.append(score)
                value_chunks.append(source_values)

        message = message.permute(0, 3, 1, 2, 4).reshape(
            batch, nodes, segments, self.d_model
        )
        output = self.out_projection(message)
        if not torch.isfinite(output).all():
            raise FloatingPointError("node-pooled relation spatial output contains NaN or Inf")
        if not return_diagnostics:
            return output
        diagnostics = {
            "attention": torch.cat(weight_chunks, dim=1),
            "content": torch.cat(content_chunks, dim=1),
            "score": torch.cat(score_chunks, dim=1),
            "value": torch.cat(value_chunks, dim=1),
            "message": message,
            "pool_weight": pool_weight,
            "q_node": q_node,
        }
        attention = diagnostics["attention"].clamp_min(1e-12)
        diagnostics["entropy"] = -(attention * attention.log()).mean()
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
        scale_id: int = 0,
        spatial_query_mode: str = "per_variable",
        edge_chunk_size: int | None = 128,
    ) -> None:
        super().__init__()
        if spatial_query_mode not in {"per_variable", "node_pooled"}:
            raise ValueError(f"unsupported spatial_query_mode: {spatial_query_mode}")
        if not isinstance(scale_id, int) or isinstance(scale_id, bool) or scale_id not in (0, 1):
            raise ValueError("relation spatial scale_id must be 0 or 1")
        self.spatial_query_mode = spatial_query_mode
        self.scale_id = int(scale_id)
        attention_class = (
            RelationSpatialAttention
            if spatial_query_mode == "per_variable"
            else NodePooledRelationSpatialAttention
        )
        self.attention = attention_class(
            d_model, spatial_heads, spatial_dropout, edge_chunk_size=edge_chunk_size
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
        edge_bias, relation_bias = self._bias_provider(self.scale_id)
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
            if self.spatial_query_mode == "node_pooled":
                message_for_residual = message.unsqueeze(2)
            else:
                message_for_residual = message
            residual = self_tokens + self.gamma * self.output_dropout(message_for_residual)
            diagnostics = {**diagnostics, "residual": residual}
            return residual, diagnostics
        if self.spatial_query_mode == "node_pooled":
            result = result.unsqueeze(2)
        return self_tokens + self.gamma * self.output_dropout(result)


__all__ = [
    "LocalVariablePool",
    "OrderedRelationBias",
    "RelationBiasProvider",
    "RelationSpatialAttention",
    "RelationSpatialInsertion",
    "StaticEdgeMLP",
    "NodePooledRelationSpatialAttention",
    "TurbineIdentityEmbedding",
    "ordered_relation_pair_representation",
]

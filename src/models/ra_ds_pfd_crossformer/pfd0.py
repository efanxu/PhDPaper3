"""The PFD0 two-candidate causal propagation view.

PFD0 deliberately has a narrow input boundary: it resolves ``Wspd`` by name
and derives only the current/history level and first difference.  It never
sees labels, masks, power columns, or any other feature.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import ceil
from pathlib import Path
from typing import Any

import torch
from torch import nn

from integrations.time_series_library import load_time_series_library_model


def build_wspd_level_diff1(x: torch.Tensor, wspd_index: int) -> torch.Tensor:
    """Return ``[Wspd.level, Wspd.diff1]`` from a history-only input tensor.

    ``diff1[0]`` is explicitly zero.  Every later value uses exactly the
    current and immediately previous Wspd value, so no future value is read.
    """

    if not isinstance(x, torch.Tensor) or x.ndim != 4:
        raise ValueError("PFD0 expects model history x with shape (B, L, N, C)")
    if not isinstance(wspd_index, int) or isinstance(wspd_index, bool):
        raise TypeError("PFD0 wspd_index must be an integer resolved from feature_columns")
    if not 0 <= wspd_index < x.shape[-1]:
        raise ValueError("PFD0 Wspd index is outside the model input feature range")
    level = x[..., wspd_index]
    first = torch.zeros_like(level[:, :1])
    diff1 = torch.cat((first, level[:, 1:] - level[:, :-1]), dim=1)
    return torch.stack((level, diff1), dim=-1)


class PFD0SegmentEmbedding(nn.Module):
    """Old-prototype-compatible left-padded segment encoding for two candidates."""

    def __init__(self, seg_len: int, d_model: int, max_segments: int, dropout: float) -> None:
        super().__init__()
        self.seg_len = int(seg_len)
        self.max_segments = int(max_segments)
        self.value_projection = nn.Linear(self.seg_len, d_model, bias=False)
        self.position_embedding = nn.Parameter(
            torch.randn(1, 1, 1, self.max_segments, d_model) * 0.02
        )
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, candidates: torch.Tensor) -> torch.Tensor:
        if candidates.ndim != 4 or candidates.shape[-1] not in (1, 2):
            raise ValueError("PFD0 candidates must have shape (B, L, N, 1 or 2)")
        batch, length, nodes, candidate_count = candidates.shape
        segment_count = ceil(length / self.seg_len)
        if segment_count != self.max_segments:
            raise ValueError(
                f"PFD0 segment count changed from configured {self.max_segments} to {segment_count}"
            )
        pad = segment_count * self.seg_len - length
        if pad:
            # Left padding is deliberately history-only; no future values are
            # appended to the input window.
            candidates = torch.cat(
                (candidates.new_zeros(batch, pad, nodes, candidate_count), candidates),
                dim=1,
            )
        segments = candidates.permute(0, 2, 3, 1).reshape(
            batch, nodes, candidate_count, segment_count, self.seg_len
        )
        tokens = self.value_projection(segments) + self.position_embedding
        return self.dropout(tokens)


class PFD0SegmentMerging(nn.Module):
    """Merge only the segment axis, copying the final segment when it is odd."""

    def __init__(self, d_model: int, win_size: int) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.win_size = int(win_size)
        self.norm = nn.LayerNorm(self.win_size * self.d_model)
        self.linear = nn.Linear(self.win_size * self.d_model, self.d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 4:
            raise ValueError("PFD0 segment merging expects (B, N, S, D)")
        batch, nodes, segments, d_model = tokens.shape
        if d_model != self.d_model:
            raise ValueError("PFD0 segment merging received an unexpected d_model")
        pad = (-segments) % self.win_size
        if pad:
            tokens = torch.cat(
                (tokens, tokens[:, :, -1:, :].expand(batch, nodes, pad, d_model)),
                dim=2,
            )
            segments += pad
        tokens = tokens.reshape(batch, nodes, segments // self.win_size, self.win_size * d_model)
        return self.linear(self.norm(tokens))


class PFD0Propagation(nn.Module):
    """Encode the two safe Wspd candidates at Scale0 and Scale1."""

    def __init__(
        self,
        *,
        lookback: int,
        seg_len: int,
        win_size: int,
        d_model: int,
        dropout: float,
        wspd_index: int,
    ) -> None:
        super().__init__()
        self.lookback = int(lookback)
        self.seg_len = int(seg_len)
        self.win_size = int(win_size)
        self.wspd_index = int(wspd_index)
        self.scale0_segments = ceil(self.lookback / self.seg_len)
        self.scale1_segments = ceil(self.scale0_segments / self.win_size)
        self.segment_embedding = PFD0SegmentEmbedding(
            self.seg_len,
            int(d_model),
            self.scale0_segments,
            float(dropout),
        )
        self.wind_fusion = nn.Sequential(
            nn.Linear(2 * int(d_model), int(d_model)),
            nn.GELU(),
            nn.Linear(int(d_model), int(d_model)),
        )
        self.scale1_merging = PFD0SegmentMerging(int(d_model), self.win_size)

    def candidate_history(self, x: torch.Tensor) -> torch.Tensor:
        """Expose the exact two-candidate history used by PFD0 diagnostics."""

        return build_wspd_level_diff1(x, self.wspd_index)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        candidates = self.candidate_history(x)
        encoded = self.segment_embedding(candidates)
        # [B,N,2,S,D] -> one PFD0 token per node and segment.
        scale0 = self.wind_fusion(torch.cat((encoded[:, :, 0], encoded[:, :, 1]), dim=-1))
        scale1 = self.scale1_merging(scale0)
        if scale0.shape[2] != self.scale0_segments or scale1.shape[2] != self.scale1_segments:
            raise AssertionError("PFD0 segment schedule does not match the configured two scales")
        return scale0, scale1


class CanonicalCrossTime(nn.Module):
    """The upstream Crossformer Cross-Time stage without Cross-Dimension."""

    def __init__(
        self,
        *,
        source_root: Path,
        d_model: int,
        n_heads: int,
        d_ff: int,
        factor: int,
        dropout: float,
    ) -> None:
        super().__init__()
        upstream = load_time_series_library_model("Crossformer", source_root=source_root)
        self.time_attention = upstream.AttentionLayer(
            upstream.FullAttention(
                False,
                int(factor),
                attention_dropout=float(dropout),
                output_attention=False,
            ),
            int(d_model),
            int(n_heads),
        )
        self.dropout = nn.Dropout(float(dropout))
        self.norm1 = nn.LayerNorm(int(d_model))
        self.norm2 = nn.LayerNorm(int(d_model))
        self.MLP1 = nn.Sequential(
            nn.Linear(int(d_model), int(d_ff)),
            nn.GELU(),
            nn.Linear(int(d_ff), int(d_model)),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 4:
            raise ValueError("canonical propagation Cross-Time expects (B, N, S, D)")
        batch, nodes, segments, d_model = tokens.shape
        time_in = tokens.reshape(batch * nodes, segments, d_model)
        time_enc, _ = self.time_attention(
            time_in,
            time_in,
            time_in,
            attn_mask=None,
            tau=None,
            delta=None,
        )
        encoded = time_in + self.dropout(time_enc)
        encoded = self.norm1(encoded)
        encoded = encoded + self.dropout(self.MLP1(encoded))
        encoded = self.norm2(encoded)
        return encoded.reshape(batch, nodes, segments, d_model)


class CrossTimeThenFusionPFD0Propagation(nn.Module):
    """PFD0 variant that keeps level and diff1 independent through each scale."""

    def __init__(
        self,
        *,
        lookback: int,
        seg_len: int,
        win_size: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        factor: int,
        dropout: float,
        wspd_index: int,
        source_root: Path,
    ) -> None:
        super().__init__()
        self.lookback = int(lookback)
        self.seg_len = int(seg_len)
        self.win_size = int(win_size)
        self.wspd_index = int(wspd_index)
        self.scale0_segments = ceil(self.lookback / self.seg_len)
        self.scale1_segments = ceil(self.scale0_segments / self.win_size)
        # The candidate embeddings are intentionally separate.  This keeps
        # the old-concept candidate branches independent until Wind Fusion.
        self.segment_embeddings = nn.ModuleList(
            [
                PFD0SegmentEmbedding(
                    self.seg_len,
                    int(d_model),
                    self.scale0_segments,
                    float(dropout),
                )
                for _ in range(2)
            ]
        )
        self.scale0_cross_time = nn.ModuleList(
            [
                CanonicalCrossTime(
                    source_root=source_root,
                    d_model=int(d_model),
                    n_heads=int(n_heads),
                    d_ff=int(d_ff),
                    factor=int(factor),
                    dropout=float(dropout),
                )
                for _ in range(2)
            ]
        )
        self.scale1_merging = PFD0SegmentMerging(int(d_model), self.win_size)
        self.scale1_cross_time = nn.ModuleList(
            [
                CanonicalCrossTime(
                    source_root=source_root,
                    d_model=int(d_model),
                    n_heads=int(n_heads),
                    d_ff=int(d_ff),
                    factor=int(factor),
                    dropout=float(dropout),
                )
                for _ in range(2)
            ]
        )
        self.wind_fusion = nn.Sequential(
            nn.Linear(2 * int(d_model), int(d_model)),
            nn.GELU(),
            nn.Linear(int(d_model), int(d_model)),
        )
        self.scale1_wind_fusion = nn.Sequential(
            nn.Linear(2 * int(d_model), int(d_model)),
            nn.GELU(),
            nn.Linear(int(d_model), int(d_model)),
        )
        self.execution_trace: list[str] = []

    def candidate_history(self, x: torch.Tensor) -> torch.Tensor:
        return build_wspd_level_diff1(x, self.wspd_index)

    def _embed_candidate(self, candidate: torch.Tensor, index: int) -> torch.Tensor:
        # PFD0SegmentEmbedding keeps candidate axes explicit; squeeze only the
        # singleton candidate axis after its canonical patching/position path.
        embedded = self.segment_embeddings[index](candidate.unsqueeze(-1))
        return embedded.squeeze(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        candidates = self.candidate_history(x)
        self.execution_trace = []
        scale0_candidates: list[torch.Tensor] = []
        for index in range(2):
            embedded = self._embed_candidate(candidates[..., index], index)
            self.execution_trace.append(f"scale0_candidate_{index}_embedding")
            encoded = self.scale0_cross_time[index](embedded)
            self.execution_trace.append(f"scale0_candidate_{index}_cross_time")
            scale0_candidates.append(encoded)
        scale0 = self.wind_fusion(torch.cat(scale0_candidates, dim=-1))
        self.execution_trace.append("scale0_fusion")

        scale1_candidates: list[torch.Tensor] = []
        for index, candidate in enumerate(scale0_candidates):
            merged = self.scale1_merging(candidate)
            self.execution_trace.append(f"scale1_candidate_{index}_merge")
            encoded = self.scale1_cross_time[index](merged)
            self.execution_trace.append(f"scale1_candidate_{index}_cross_time")
            scale1_candidates.append(encoded)
        scale1 = self.scale1_wind_fusion(torch.cat(scale1_candidates, dim=-1))
        self.execution_trace.append("scale1_fusion")
        if scale0.shape[2] != self.scale0_segments or scale1.shape[2] != self.scale1_segments:
            raise AssertionError("PFD0 cross-time segment schedule does not match configured scales")
        return scale0, scale1


def build_pfd0_propagation(
    mode: str,
    *,
    lookback: int,
    seg_len: int,
    win_size: int,
    d_model: int,
    n_heads: int,
    d_ff: int,
    factor: int,
    dropout: float,
    wspd_index: int,
    source_root: Path,
) -> nn.Module:
    """Construct one of the two explicit PFD0 propagation encoder modes."""

    if mode == "segment_fusion":
        return PFD0Propagation(
            lookback=lookback,
            seg_len=seg_len,
            win_size=win_size,
            d_model=d_model,
            dropout=dropout,
            wspd_index=wspd_index,
        )
    if mode == "cross_time_then_fusion":
        return CrossTimeThenFusionPFD0Propagation(
            lookback=lookback,
            seg_len=seg_len,
            win_size=win_size,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            factor=factor,
            dropout=dropout,
            wspd_index=wspd_index,
            source_root=source_root,
        )
    raise ValueError(f"unsupported propagation_encoder_mode: {mode}")


__all__ = [
    "CanonicalCrossTime",
    "CrossTimeThenFusionPFD0Propagation",
    "PFD0Propagation",
    "PFD0SegmentEmbedding",
    "PFD0SegmentMerging",
    "build_pfd0_propagation",
    "build_wspd_level_diff1",
]

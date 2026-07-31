"""Local canonical Crossformer stages with explicit P1 insertion points.

The modules are constructed from the read-only upstream implementation in
``Time-Series-Library/models/Crossformer.py`` and its imported layer files.
The only local forward logic is the minimum copy of
``TwoStageAttentionLayer.forward`` needed to split Cross-Time from
Cross-Dimension and call the two strict identity insertion modules between
them.  The canonical module names and construction order intentionally match
the upstream ``Model`` so its state dict can be transferred strictly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from einops import rearrange, repeat
from torch import nn

from integrations.time_series_library import load_time_series_library_model


@dataclass(frozen=True)
class CanonicalTrace:
    """Small diagnostic view of the canonical stages used by P1 tests."""

    segment_embedding: torch.Tensor
    embedded_with_position: torch.Tensor
    pre_norm: torch.Tensor
    scale0_cross_time: torch.Tensor
    scale0_spatial: torch.Tensor
    scale0_cross_dimension: torch.Tensor
    scale1_merged: torch.Tensor
    scale1_cross_time: torch.Tensor
    scale1_spatial: torch.Tensor
    scale1_cross_dimension: torch.Tensor
    decoder_tokens: tuple[torch.Tensor, ...]
    decoder_output: torch.Tensor
    output: torch.Tensor


class IdentitySpatialInsertion(nn.Module):
    """A parameter-free, exact identity used by each P1 insertion point."""

    def __init__(self, scale_id: int) -> None:
        super().__init__()
        self.scale_id = int(scale_id)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value


class CanonicalBackbone(nn.Module):
    """The two-scale Crossformer encoder and canonical multi-scale decoder."""

    def __init__(
        self,
        *,
        source_root: Path,
        enc_in: int,
        seq_len: int,
        pred_len: int,
        model_config: Mapping[str, Any],
    ) -> None:
        super().__init__()
        if model_config.get("spatial_disabled") is not True:
            raise ValueError(
                "RA-DS-PFD Crossformer P1 requires spatial_disabled=true; "
                "spatial implementation is not available"
            )

        self.enc_in = int(enc_in)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.d_model = int(model_config["d_model"])
        self.n_heads = int(model_config["n_heads"])
        self.d_ff = int(model_config["d_ff"])
        self.e_layers = int(model_config["e_layers"])
        self.dropout = float(model_config["dropout"])
        self.factor = int(model_config["factor"])
        self.seg_len = int(model_config["seg_len"])
        self.win_size = int(model_config["win_size"])

        # The upstream model hard-codes these two values.  P1 validates that
        # the local configuration stays on that canonical two-scale path.
        self.pad_in_len = ceil(1.0 * self.seq_len / self.seg_len) * self.seg_len
        self.pad_out_len = ceil(1.0 * self.pred_len / self.seg_len) * self.seg_len
        self.in_seg_num = self.pad_in_len // self.seg_len
        self.out_seg_num = ceil(self.in_seg_num / (self.win_size ** (self.e_layers - 1)))

        upstream = load_time_series_library_model("Crossformer", source_root=source_root)
        upstream_config = SimpleNamespace(
            enc_in=self.enc_in,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            task_name="long_term_forecast",
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            e_layers=self.e_layers,
            dropout=self.dropout,
            factor=self.factor,
        )

        # This construction order mirrors Crossformer.Model.__init__ exactly.
        self.enc_value_embedding = upstream.PatchEmbedding(
            self.d_model,
            self.seg_len,
            self.seg_len,
            self.pad_in_len - self.seq_len,
            0,
        )
        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, self.enc_in, self.in_seg_num, self.d_model)
        )
        self.pre_norm = nn.LayerNorm(self.d_model)

        self.encoder = upstream.Encoder(
            [
                upstream.scale_block(
                    upstream_config,
                    1 if level == 0 else self.win_size,
                    self.d_model,
                    self.n_heads,
                    self.d_ff,
                    1,
                    self.dropout,
                    self.in_seg_num
                    if level == 0
                    else ceil(self.in_seg_num / self.win_size**level),
                    self.factor,
                )
                for level in range(self.e_layers)
            ]
        )

        # These modules have no state and intentionally do not alter the
        # upstream state-dict key set.
        self.scale0_spatial_insertion = IdentitySpatialInsertion(scale_id=0)
        self.scale1_spatial_insertion = IdentitySpatialInsertion(scale_id=1)

        self.dec_pos_embedding = nn.Parameter(
            torch.randn(1, self.enc_in, self.pad_out_len // self.seg_len, self.d_model)
        )
        self.decoder = upstream.Decoder(
            [
                upstream.DecoderLayer(
                    upstream.TwoStageAttentionLayer(
                        upstream_config,
                        self.pad_out_len // self.seg_len,
                        self.factor,
                        self.d_model,
                        self.n_heads,
                        self.d_ff,
                        self.dropout,
                    ),
                    upstream.AttentionLayer(
                        upstream.FullAttention(
                            False,
                            self.factor,
                            attention_dropout=self.dropout,
                            output_attention=False,
                        ),
                        self.d_model,
                        self.n_heads,
                    ),
                    self.seg_len,
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                )
                for _ in range(self.e_layers + 1)
            ]
        )

    @staticmethod
    def _forward_two_stage_with_spatial(
        layer: nn.Module,
        value: torch.Tensor,
        spatial_insertion: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the upstream TSA layer while exposing its two canonical stages."""

        # This is the upstream TwoStageAttentionLayer.forward sequence from
        # Time-Series-Library/layers/SelfAttention_Family.py.  No operation is
        # changed; the identity call is the only inserted operation.
        batch = value.shape[0]
        time_in = rearrange(value, "b ts_d seg_num d_model -> (b ts_d) seg_num d_model")
        time_enc, _ = layer.time_attention(
            time_in,
            time_in,
            time_in,
            attn_mask=None,
            tau=None,
            delta=None,
        )
        dim_in = time_in + layer.dropout(time_enc)
        dim_in = layer.norm1(dim_in)
        dim_in = dim_in + layer.dropout(layer.MLP1(dim_in))
        dim_in = layer.norm2(dim_in)
        cross_time = dim_in

        spatial_value = spatial_insertion(cross_time)

        dim_send = rearrange(
            spatial_value,
            "(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model",
            b=batch,
        )
        batch_router = repeat(
            layer.router,
            "seg_num factor d_model -> (repeat seg_num) factor d_model",
            repeat=batch,
        )
        dim_buffer, _ = layer.dim_sender(
            batch_router,
            dim_send,
            dim_send,
            attn_mask=None,
            tau=None,
            delta=None,
        )
        dim_receive, _ = layer.dim_receiver(
            dim_send,
            dim_buffer,
            dim_buffer,
            attn_mask=None,
            tau=None,
            delta=None,
        )
        dim_enc = dim_send + layer.dropout(dim_receive)
        dim_enc = layer.norm3(dim_enc)
        dim_enc = dim_enc + layer.dropout(layer.MLP2(dim_enc))
        dim_enc = layer.norm4(dim_enc)
        final_out = rearrange(
            dim_enc,
            "(b seg_num) ts_d d_model -> b ts_d seg_num d_model",
            b=batch,
        )
        return cross_time, spatial_value, final_out

    def forward_backbone(
        self,
        value: torch.Tensor,
        *,
        return_trace: bool = False,
    ) -> torch.Tensor | CanonicalTrace:
        if value.ndim != 3:
            raise ValueError("canonical Crossformer expects (batch_nodes, lookback, features)")
        expected = (self.seq_len, self.enc_in)
        if tuple(value.shape[1:]) != expected:
            raise ValueError(
                "unexpected canonical Crossformer input shape: "
                f"{tuple(value.shape)}; expected (*, {expected[0]}, {expected[1]})"
            )
        if not torch.isfinite(value).all():
            raise FloatingPointError("canonical Crossformer input contains NaN or Inf")

        embedded, n_vars = self.enc_value_embedding(value.permute(0, 2, 1))
        if n_vars != self.enc_in:
            raise ValueError(f"canonical Crossformer embedding returned {n_vars} variables")
        segment_embedding = embedded
        encoded = rearrange(
            embedded,
            "(b d) seg_num d_model -> b d seg_num d_model",
            d=n_vars,
        )
        embedded_with_position = encoded + self.enc_pos_embedding
        pre_norm = self.pre_norm(embedded_with_position)

        scale0_layer = self.encoder.encode_blocks[0].encode_layers[0]
        scale0_cross_time, scale0_spatial, scale0_cross_dimension = (
            self._forward_two_stage_with_spatial(
                scale0_layer,
                pre_norm,
                self.scale0_spatial_insertion,
            )
        )

        scale1_block = self.encoder.encode_blocks[1]
        scale1_merged = scale1_block.merge_layer(scale0_cross_dimension)
        scale1_layer = scale1_block.encode_layers[0]
        scale1_cross_time, scale1_spatial, scale1_cross_dimension = (
            self._forward_two_stage_with_spatial(
                scale1_layer,
                scale1_merged,
                self.scale1_spatial_insertion,
            )
        )

        decoder_tokens = (
            pre_norm,
            scale0_cross_dimension,
            scale1_cross_dimension,
        )
        decoder_input = repeat(
            self.dec_pos_embedding,
            "b ts_d l d -> (repeat b) ts_d l d",
            repeat=pre_norm.shape[0],
        )
        decoder_output = self.decoder(decoder_input, list(decoder_tokens))
        output = decoder_output[:, -self.pred_len :, :]
        if not torch.isfinite(output).all():
            raise FloatingPointError("canonical Crossformer output contains NaN or Inf")

        if not return_trace:
            return output
        return CanonicalTrace(
            segment_embedding=segment_embedding,
            embedded_with_position=embedded_with_position,
            pre_norm=pre_norm,
            scale0_cross_time=scale0_cross_time,
            scale0_spatial=scale0_spatial,
            scale0_cross_dimension=scale0_cross_dimension,
            scale1_merged=scale1_merged,
            scale1_cross_time=scale1_cross_time,
            scale1_spatial=scale1_spatial,
            scale1_cross_dimension=scale1_cross_dimension,
            decoder_tokens=decoder_tokens,
            decoder_output=decoder_output,
            output=output,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        result = self.forward_backbone(value)
        assert isinstance(result, torch.Tensor)
        return result

    def load_upstream_state_dict(
        self,
        upstream_state_dict: Mapping[str, torch.Tensor],
    ) -> None:
        """Load an upstream Crossformer state dict with a bijective strict check."""

        if not isinstance(upstream_state_dict, Mapping):
            raise TypeError("upstream canonical state must be a mapping")
        source = dict(upstream_state_dict)
        local = self.state_dict()
        local_keys = set(local)
        source_keys = set(source)
        missing = sorted(local_keys - source_keys)
        unexpected = sorted(source_keys - local_keys)
        if missing:
            raise ValueError(f"upstream canonical state is missing key: {missing[0]}")
        if unexpected:
            raise ValueError(f"upstream canonical state has unexpected key: {unexpected[0]}")
        for key in sorted(local_keys):
            source_value = source[key]
            if not isinstance(source_value, torch.Tensor):
                raise TypeError(f"upstream canonical state value is not a tensor: {key}")
            if tuple(source_value.shape) != tuple(local[key].shape):
                raise ValueError(
                    f"upstream canonical state shape mismatch for {key}: "
                    f"{tuple(source_value.shape)} != {tuple(local[key].shape)}"
                )
        self.load_state_dict(source, strict=True)

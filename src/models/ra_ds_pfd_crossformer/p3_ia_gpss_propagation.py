"""IA-GPSS v1 hard-sparse propagation core.

The selector and candidate bank remain global and cheap.  Only the final K
candidate streams enter the expensive temporal path; candidate-specific value
projections are deliberately kept separate so the selector can train them
through the fixed hard-forward/soft-backward value seam.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil, isfinite, sqrt
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .p3_feature_bank import (
    P3_BASE_FEATURES,
    P3_CANDIDATE_TRANSFORMS,
    P3CandidateBank,
)
from .p3_ia_gpss_selector import (
    DEFAULT_IA_GPSS_REFINEMENT_ROUNDS,
    DEFAULT_IA_GPSS_TEMPERATURE,
    IAGPSSSelector,
    IAGPSSSelectorOutput,
)
from .p3_ia_temporal import (
    IA11_OPERATOR_TYPES,
    OperatorResidualAdapter,
    SemanticCandidateIdentity,
)
from .pfd0 import CanonicalCrossTime, PFD0SegmentMerging


def _validate_positive_int(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"IA-GPSS {field} must be a positive integer")
    return int(value)


def segmentize_candidate_history(
    history: torch.Tensor,
    seg_len: int,
) -> torch.Tensor:
    """Left-pad ``[B,L,N,M]`` history into ``[B,N,M,S,seg_len]`` segments."""

    if not isinstance(history, torch.Tensor) or history.ndim != 4:
        raise ValueError("IA-GPSS history must have shape (B, L, N, M)")
    seg_len = _validate_positive_int(seg_len, field="seg_len")
    if history.shape[0] < 1 or history.shape[1] < 1 or history.shape[2] < 1:
        raise ValueError("IA-GPSS history dimensions must be positive")
    if not torch.isfinite(history).all():
        raise FloatingPointError("IA-GPSS history contains NaN or Inf")
    segment_count = ceil(history.shape[1] / seg_len)
    pad = segment_count * seg_len - history.shape[1]
    if pad:
        history = torch.cat(
            (history.new_zeros(history.shape[0], pad, history.shape[2], history.shape[3]), history),
            dim=1,
        )
    segments = history.permute(0, 2, 3, 1).reshape(
        history.shape[0],
        history.shape[2],
        history.shape[3],
        segment_count,
        seg_len,
    )
    if not torch.isfinite(segments).all():
        raise FloatingPointError("IA-GPSS segments contain NaN or Inf")
    return segments


def _hard_forward_soft_backward_gather(
    values: torch.Tensor,
    hard_assignment: torch.Tensor,
    soft_probabilities: torch.Tensor,
    candidate_dim: int | None = None,
) -> torch.Tensor:
    """Gather hard values in forward while using soft values in backward.

    The assignment dimension is moved to the front for the contraction so the
    helper works for both ``[B,N,M,S,D]`` value banks and ``[M,D]`` identities.
    """

    if not isinstance(values, torch.Tensor) or values.ndim < 1:
        raise ValueError("IA-GPSS value gather requires a non-empty value tensor")
    if not isinstance(hard_assignment, torch.Tensor) or hard_assignment.ndim != 2:
        raise ValueError("IA-GPSS hard assignment must have shape (K, M)")
    if not isinstance(soft_probabilities, torch.Tensor) or soft_probabilities.ndim != 2:
        raise ValueError("IA-GPSS soft probabilities must have shape (K, M)")
    if hard_assignment.shape != soft_probabilities.shape:
        raise ValueError("IA-GPSS hard and soft assignments must have the same shape")
    if candidate_dim is None:
        candidate_dim = 0 if values.ndim == 2 else 2
    candidate_dim = candidate_dim if candidate_dim >= 0 else values.ndim + candidate_dim
    if not 0 <= candidate_dim < values.ndim:
        raise ValueError("IA-GPSS candidate dimension is outside the value tensor")
    candidate_count = hard_assignment.shape[1]
    if values.shape[candidate_dim] != candidate_count:
        raise ValueError("IA-GPSS value tensor candidate count does not match assignment")
    if not torch.isfinite(values).all():
        raise FloatingPointError("IA-GPSS values contain NaN or Inf")
    if not torch.isfinite(hard_assignment).all() or not torch.isfinite(soft_probabilities).all():
        raise FloatingPointError("IA-GPSS assignments contain NaN or Inf")

    hard_assignment = hard_assignment.to(device=values.device, dtype=values.dtype)
    soft_probabilities = soft_probabilities.to(device=values.device, dtype=values.dtype)
    moved = values.movedim(candidate_dim, 0)
    hard_value = torch.tensordot(hard_assignment, moved, dims=([1], [0]))
    soft_value = torch.tensordot(soft_probabilities, moved, dims=([1], [0]))
    hard_value = hard_value.movedim(0, candidate_dim)
    soft_value = soft_value.movedim(0, candidate_dim)
    selected = hard_value.detach() + (soft_value - soft_value.detach())
    if not torch.isfinite(selected).all():
        raise FloatingPointError("IA-GPSS selected values contain NaN or Inf")
    return selected


class IAGPSSCandidateProjectionBank(nn.Module):
    """Independent cheap ``Linear(seg_len, d_model, bias=False)`` projections."""

    def __init__(self, candidate_count: int, seg_len: int, d_model: int) -> None:
        super().__init__()
        self.candidate_count = _validate_positive_int(candidate_count, field="candidate_count")
        self.seg_len = _validate_positive_int(seg_len, field="seg_len")
        self.d_model = _validate_positive_int(d_model, field="d_model")
        self.weight = nn.Parameter(
            torch.empty(self.candidate_count, self.d_model, self.seg_len)
        )
        nn.init.kaiming_uniform_(self.weight, a=sqrt(5))

    def forward(self, segments: torch.Tensor) -> torch.Tensor:
        """Project ``[B,N,M,S,seg_len]`` into ``[B,N,M,S,D]``."""

        if not isinstance(segments, torch.Tensor) or segments.ndim != 5:
            raise ValueError("IA-GPSS projection bank expects (B, N, M, S, seg_len)")
        if segments.shape[2] != self.candidate_count:
            raise ValueError("IA-GPSS projection bank received an unexpected candidate count")
        if segments.shape[-1] != self.seg_len:
            raise ValueError("IA-GPSS projection bank received an unexpected segment length")
        if not torch.isfinite(segments).all() or not torch.isfinite(self.weight).all():
            raise FloatingPointError("IA-GPSS projection bank received NaN or Inf")
        projected = torch.einsum("bnmsp,mdp->bnmsd", segments, self.weight)
        if not torch.isfinite(projected).all():
            raise FloatingPointError("IA-GPSS projected candidates contain NaN or Inf")
        return projected


class IAGPSSPropagation(nn.Module):
    """Hard-forward/soft-backward global propagation for the canonical bank."""

    def __init__(
        self,
        *,
        feature_columns: Sequence[str] | Any,
        lookback: int,
        seg_len: int,
        win_size: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        factor: int,
        top_k: int = 2,
        selector_temperature: float = DEFAULT_IA_GPSS_TEMPERATURE,
        refinement_rounds: int = DEFAULT_IA_GPSS_REFINEMENT_ROUNDS,
        spatial_dropout: float | None = None,
        source_root: Path | None = None,
        dropout: float | None = None,
    ) -> None:
        super().__init__()
        self.lookback = _validate_positive_int(lookback, field="lookback")
        self.seg_len = _validate_positive_int(seg_len, field="seg_len")
        self.win_size = _validate_positive_int(win_size, field="win_size")
        self.d_model = _validate_positive_int(d_model, field="d_model")
        self.n_heads = _validate_positive_int(n_heads, field="n_heads")
        self.d_ff = _validate_positive_int(d_ff, field="d_ff")
        self.factor = _validate_positive_int(factor, field="factor")
        self.scale0_segments = ceil(self.lookback / self.seg_len)
        self.scale1_segments = ceil(self.scale0_segments / self.win_size)

        if spatial_dropout is None:
            if dropout is None:
                raise ValueError("IA-GPSS propagation requires spatial_dropout")
            spatial_dropout = dropout
        elif dropout is not None and float(spatial_dropout) != float(dropout):
            raise ValueError("IA-GPSS propagation dropout aliases disagree")
        self.spatial_dropout = float(spatial_dropout)
        if not isfinite(self.spatial_dropout) or not 0.0 <= self.spatial_dropout < 1.0:
            raise ValueError("IA-GPSS propagation spatial_dropout must be finite and in [0, 1)")

        self.candidate_bank = P3CandidateBank(
            feature_columns,
            candidate_features=P3_BASE_FEATURES,
            candidate_transforms=P3_CANDIDATE_TRANSFORMS,
        )
        self.candidate_names = self.candidate_bank.candidate_names
        self.candidate_count = self.candidate_bank.candidate_count
        self.selector = IAGPSSSelector(
            d_model=self.d_model,
            top_k=top_k,
            temperature=selector_temperature,
            refinement_rounds=refinement_rounds,
        )
        if self.selector.candidate_names != self.candidate_names:
            raise AssertionError("IA-GPSS selector and candidate bank order diverged")
        self.top_k = self.selector.top_k
        self.selected_candidate_count = self.top_k
        self.projected_candidate_count = self.candidate_count
        self.candidate_projection_bank = IAGPSSCandidateProjectionBank(
            candidate_count=self.candidate_count,
            seg_len=self.seg_len,
            d_model=self.d_model,
        )
        # This is a new instance even though its semantic definition matches
        # IA11.  Selector and propagation identities must train independently.
        self.propagation_identity = SemanticCandidateIdentity(self.d_model)
        self.position_embedding = nn.Parameter(
            torch.randn(1, 1, 1, self.scale0_segments, self.d_model) * 0.02
        )
        self.dropout = nn.Dropout(self.spatial_dropout)

        self.operator_types = tuple(P3_CANDIDATE_TRANSFORMS)
        if self.operator_types != IA11_OPERATOR_TYPES:
            raise AssertionError("IA-GPSS operator basis diverged from IA11")
        incidence = torch.zeros(
            self.candidate_count,
            len(self.operator_types),
            dtype=torch.float32,
        )
        for candidate_index, candidate in enumerate(self.candidate_bank.candidates):
            incidence[candidate_index, self.operator_types.index(candidate.transform)] = 1.0
        self.register_buffer("operator_incidence", incidence, persistent=False)

        root = (
            Path(source_root).resolve()
            if source_root is not None
            else Path(__file__).resolve().parents[3] / "Time-Series-Library"
        )
        self.scale0_operator_adapters = nn.ModuleDict(
            {
                operator: OperatorResidualAdapter(self.d_model)
                for operator in self.operator_types
            }
        )
        self.scale1_operator_adapters = nn.ModuleDict(
            {
                operator: OperatorResidualAdapter(self.d_model)
                for operator in self.operator_types
            }
        )
        self.scale0_cross_time = CanonicalCrossTime(
            source_root=root,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            factor=self.factor,
            dropout=self.spatial_dropout,
        )
        self.scale1_merging = PFD0SegmentMerging(self.d_model, self.win_size)
        self.scale1_cross_time = CanonicalCrossTime(
            source_root=root,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            factor=self.factor,
            dropout=self.spatial_dropout,
        )
        self.scale0_fusion = self._build_fusion()
        self.scale1_fusion = self._build_fusion()

        self.cross_time_candidate_counts: tuple[int, ...] = ()
        self.execution_trace: dict[str, Any] = {}
        self._current_cross_time_candidate_counts: list[int] = []

    def _build_fusion(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(self.top_k * self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )

    def candidate_history(self, x: torch.Tensor) -> torch.Tensor:
        """Return the complete cheap ``[B,L,N,M]`` canonical history bank."""

        if not isinstance(x, torch.Tensor) or x.ndim != 4:
            raise ValueError("IA-GPSS propagation expects x with shape (B, L, N, C)")
        if x.shape[1] != self.lookback:
            raise ValueError(
                "IA-GPSS propagation input length does not match configured lookback"
            )
        return self.candidate_bank(x)

    def _project_all_candidates(self, x: torch.Tensor) -> torch.Tensor:
        history = self.candidate_history(x)
        segments = segmentize_candidate_history(history, seg_len=self.seg_len)
        if tuple(segments.shape[3:]) != (self.scale0_segments, self.seg_len):
            raise AssertionError("IA-GPSS segment schedule does not match configured lookback")
        projected = self.candidate_projection_bank(segments)
        if tuple(projected.shape) != (
            x.shape[0],
            x.shape[2],
            self.candidate_count,
            self.scale0_segments,
            self.d_model,
        ):
            raise AssertionError("IA-GPSS candidate projection shape drifted")
        return projected

    def _validate_selection(
        self,
        selection: IAGPSSSelectorOutput,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...], tuple[str, ...]]:
        hard = selection.hard_assignment
        soft = selection.soft_probabilities
        st = selection.st_assignment
        expected_shape = (self.top_k, self.candidate_count)
        if tuple(hard.shape) != expected_shape or tuple(soft.shape) != expected_shape:
            raise ValueError("IA-GPSS final selector assignments have an unexpected shape")
        if tuple(st.shape) != expected_shape:
            raise ValueError("IA-GPSS final selector ST assignment has an unexpected shape")
        if not torch.is_floating_point(hard) or not torch.is_floating_point(soft):
            raise ValueError("IA-GPSS selector assignments must be floating point")
        if not torch.isfinite(hard).all() or not torch.isfinite(soft).all() or not torch.isfinite(st).all():
            raise FloatingPointError("IA-GPSS final selector assignments contain NaN or Inf")
        if not torch.equal(st, hard):
            raise ValueError("IA-GPSS ST assignment is not exact hard-forward")
        if not torch.equal(hard, hard.round()):
            raise ValueError("IA-GPSS hard assignment is not one-hot")
        if not torch.equal(hard.sum(dim=1), torch.ones(self.top_k, dtype=hard.dtype, device=hard.device)):
            raise ValueError("IA-GPSS hard assignment does not have exact K rows")
        if torch.any(soft < 0):
            raise ValueError("IA-GPSS soft probabilities must be non-negative")
        if not torch.allclose(
            soft.sum(dim=1),
            torch.ones(self.top_k, dtype=soft.dtype, device=soft.device),
            rtol=1e-5,
            atol=1e-6,
        ):
            raise ValueError("IA-GPSS soft probabilities are not normalized")
        indices = tuple(int(index) for index in selection.selected_indices)
        names = tuple(str(name) for name in selection.selected_names)
        if len(indices) != self.top_k or len(set(indices)) != self.top_k:
            raise ValueError("IA-GPSS final selector indices are not exact-K")
        if indices != tuple(sorted(indices)):
            raise ValueError("IA-GPSS final selector indices are not canonical")
        if names != tuple(self.candidate_names[index] for index in indices):
            raise ValueError("IA-GPSS final selector names do not match canonical indices")
        row_indices = tuple(int(row.argmax().detach().cpu().item()) for row in hard)
        if row_indices != indices:
            raise ValueError("IA-GPSS final selector rows do not match selected indices")
        if len(set(row_indices)) != self.top_k:
            raise ValueError("IA-GPSS final selector rows select duplicate candidates")
        return hard, soft, st, indices, names

    def _apply_operator_adapters(
        self,
        tokens: torch.Tensor,
        adapters: nn.ModuleDict,
        st_assignment: torch.Tensor,
    ) -> torch.Tensor:
        if tokens.ndim != 5:
            raise ValueError("IA-GPSS operator adapters expect (B,N,K,S,D)")
        if tuple(st_assignment.shape) != (self.top_k, self.candidate_count):
            raise ValueError("IA-GPSS operator routing received an unexpected assignment shape")
        incidence = self.operator_incidence.to(
            device=st_assignment.device,
            dtype=st_assignment.dtype,
        )
        gates = st_assignment @ incidence
        if tuple(gates.shape) != (self.top_k, len(self.operator_types)):
            raise AssertionError("IA-GPSS operator incidence shape drifted")
        # G_ST is exact hard one-hot in forward. Adapter parameters therefore
        # follow the actual hard-routed operator branch; selector parameters
        # still receive routing gradients through G_ST.
        outputs = torch.stack(
            [adapters[operator](tokens) for operator in self.operator_types],
            dim=-1,
        )
        routed = (outputs * gates.view(1, 1, self.top_k, 1, 1, -1)).sum(dim=-1)
        if not torch.isfinite(routed).all():
            raise FloatingPointError("IA-GPSS routed operator output contains NaN or Inf")
        return routed

    def _shared_cross_time(self, encoder: CanonicalCrossTime, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 5:
            raise ValueError("IA-GPSS shared Cross-Time expects (B,N,K,S,D)")
        batch, nodes, candidates, segments, d_model = tokens.shape
        if candidates != self.top_k or d_model != self.d_model:
            raise ValueError("IA-GPSS shared Cross-Time received an unexpected shape")
        self._current_cross_time_candidate_counts.append(int(candidates))
        folded = tokens.permute(0, 2, 1, 3, 4).reshape(
            batch * candidates,
            nodes,
            segments,
            d_model,
        )
        encoded = encoder(folded)
        return encoded.reshape(batch, candidates, nodes, segments, d_model).permute(
            0, 2, 1, 3, 4
        )

    def _merge_scale1(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, nodes, candidates, segments, d_model = tokens.shape
        merged = self.scale1_merging(
            tokens.reshape(batch, nodes * candidates, segments, d_model)
        )
        return merged.reshape(batch, nodes, candidates, self.scale1_segments, d_model)

    def _fuse(self, tokens: torch.Tensor, fusion: nn.Module) -> torch.Tensor:
        batch, nodes, candidates, segments, d_model = tokens.shape
        if candidates != self.top_k or d_model != self.d_model:
            raise ValueError("IA-GPSS fusion received an unexpected selected shape")
        inputs = tokens.permute(0, 1, 3, 2, 4).reshape(
            batch,
            nodes,
            segments,
            candidates * d_model,
        )
        return fusion(inputs)

    def _propagate_with_selection(
        self,
        x: torch.Tensor,
        selector_output: IAGPSSSelectorOutput,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute one propagation path using the selector's final assignment."""

        hard, soft, st, selected_indices, selected_names = self._validate_selection(
            selector_output
        )
        projected_all = self._project_all_candidates(x)
        st = st.to(device=projected_all.device, dtype=projected_all.dtype)
        projected_selected = _hard_forward_soft_backward_gather(
            projected_all,
            hard,
            soft,
            candidate_dim=2,
        )
        all_identity = self.propagation_identity(self.candidate_names).to(
            device=projected_selected.device,
            dtype=projected_selected.dtype,
        )
        selected_identity = _hard_forward_soft_backward_gather(
            all_identity,
            hard,
            soft,
            candidate_dim=0,
        )
        identity = selected_identity.view(1, 1, self.top_k, 1, self.d_model)
        tokens = self.dropout(projected_selected + self.position_embedding + identity)

        self._current_cross_time_candidate_counts = []
        scale0_adapted = self._apply_operator_adapters(
            tokens,
            self.scale0_operator_adapters,
            st,
        )
        scale0_candidates = self._shared_cross_time(self.scale0_cross_time, scale0_adapted)
        scale1_merged = self._merge_scale1(scale0_candidates)
        scale1_adapted = self._apply_operator_adapters(
            scale1_merged,
            self.scale1_operator_adapters,
            st,
        )
        scale1_candidates = self._shared_cross_time(self.scale1_cross_time, scale1_adapted)
        scale0 = self._fuse(scale0_candidates, self.scale0_fusion)
        scale1 = self._fuse(scale1_candidates, self.scale1_fusion)

        self.cross_time_candidate_counts = tuple(self._current_cross_time_candidate_counts)
        if self.cross_time_candidate_counts != (self.top_k, self.top_k):
            raise AssertionError("IA-GPSS Cross-Time candidate axis is not K-sparse")
        if tuple(scale0.shape) != (
            x.shape[0],
            x.shape[2],
            self.scale0_segments,
            self.d_model,
        ):
            raise AssertionError("IA-GPSS Scale0 propagation contract drifted")
        if tuple(scale1.shape) != (
            x.shape[0],
            x.shape[2],
            self.scale1_segments,
            self.d_model,
        ):
            raise AssertionError("IA-GPSS Scale1 propagation contract drifted")
        if not torch.isfinite(scale0).all() or not torch.isfinite(scale1).all():
            raise FloatingPointError("IA-GPSS propagation output contains NaN or Inf")
        self.execution_trace = {
            "candidate_bank_count": int(self.candidate_count),
            "projected_candidate_count": int(self.projected_candidate_count),
            "selected_candidate_count": int(self.top_k),
            "scale0_cross_time_candidate_count": int(self.cross_time_candidate_counts[0]),
            "scale1_cross_time_candidate_count": int(self.cross_time_candidate_counts[1]),
            "scale0_cross_time_module_count": 1,
            "scale1_cross_time_module_count": 1,
            "operator_types": tuple(self.operator_types),
            "selected_candidate_indices": tuple(selected_indices),
            "selected_candidate_names": tuple(selected_names),
            "scale0_cross_time_candidate_names": tuple(selected_names),
            "scale1_cross_time_candidate_names": tuple(selected_names),
        }
        return scale0, scale1

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        selection = self.selector()
        return self._propagate_with_selection(x, selection)


__all__ = [
    "IAGPSSCandidateProjectionBank",
    "IAGPSSPropagation",
    "_hard_forward_soft_backward_gather",
    "segmentize_candidate_history",
]

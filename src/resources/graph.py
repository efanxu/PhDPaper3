"""Minimal graph helper for future models; NodeSharedLSTM does not use it."""

from __future__ import annotations

import torch


def validate_adjacency(adjacency: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if tuple(adjacency.shape[-2:]) != (num_nodes, num_nodes):
        raise ValueError(f"adjacency must end with ({num_nodes}, {num_nodes})")
    if not torch.isfinite(adjacency).all():
        raise ValueError("adjacency contains NaN or Inf")
    return adjacency

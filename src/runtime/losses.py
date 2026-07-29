"""Names of losses supported by the shared training engine.

This module intentionally has no PyTorch dependency so the command help can
load the real loss choices without importing the training runtime.
"""

from __future__ import annotations


LOSS_NAMES: tuple[str, ...] = ("masked_score_aligned_hybrid",)

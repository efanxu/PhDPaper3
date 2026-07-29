"""Optional static-feature loader seam, kept empty for the reference model."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_static_features(path: Path, *, num_nodes: int) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    if value.ndim != 2 or value.shape[0] != num_nodes:
        raise ValueError("static features must have shape (num_nodes, features)")
    return value.astype(np.float32, copy=False)

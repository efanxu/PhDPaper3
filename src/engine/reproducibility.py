"""Reproducibility controls used by every training and check entry point."""

from __future__ import annotations

import hashlib
import os
import random
from typing import Any

import numpy as np
import torch


CONTROLLED_NONSTRICT = "controlled_nonstrict"
CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def _validate_worker_environment(seed: int) -> None:
    expected_seed = str(seed)
    actual_seed = os.environ.get("PYTHONHASHSEED")
    if actual_seed != expected_seed:
        raise RuntimeError(
            "worker PYTHONHASHSEED does not match resolved training.seed: "
            f"expected {expected_seed!r}, got {actual_seed!r}"
        )
    actual_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if actual_cublas != CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "worker CUBLAS_WORKSPACE_CONFIG must be "
            f"{CUBLAS_WORKSPACE_CONFIG!r}, got {actual_cublas!r}"
        )


def set_seed(seed: int, *, reproducibility_mode: str) -> dict[str, Any]:
    """Apply the project's controlled non-strict reproducibility policy.

    The parent scheduler owns process-start environment variables.  A worker
    therefore validates and records them here instead of trying to change a
    Python hash seed after interpreter startup.
    """

    if seed < 0:
        raise ValueError("seed must be non-negative")
    if reproducibility_mode != CONTROLLED_NONSTRICT:
        raise ValueError(
            "reproducibility_mode must be controlled_nonstrict"
        )
    _validate_worker_environment(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(False)
    return {
        "seed": int(seed),
        "reproducibility_mode": reproducibility_mode,
        "global_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cuda_available": bool(torch.cuda.is_available()),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def loader_seed(seed: int, offset: int) -> int:
    return int(seed) + int(offset)


def capture_rng_state() -> dict[str, Any]:
    """Capture every process RNG used by the shared training path."""

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG state after model/optimizer construction is complete."""

    if not isinstance(state, dict):
        raise ValueError("checkpoint RNG state must be a mapping")
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    if isinstance(numpy_state, list):
        numpy_state = tuple(numpy_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def state_dict_hash(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        digest.update(name.encode("utf-8"))
        value = state_dict[name].detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(repr(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def max_tensor_difference(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    if left.size == 0:
        return 0.0
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))

"""Reproducibility controls used by every training and check entry point."""

from __future__ import annotations

import hashlib
import os
import random
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, *, deterministic: bool = True) -> dict[str, Any]:
    """Seed Python, NumPy, PyTorch, CUDA and deterministic cuDNN behavior."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    # This is required by deterministic CUDA matrix multiplication on supported
    # versions and is harmless on CPU-only runs.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = bool(deterministic)
    torch.use_deterministic_algorithms(bool(deterministic))
    return {
        "seed": int(seed),
        "python_hash_seed": os.environ["PYTHONHASHSEED"],
        "deterministic": bool(deterministic),
        "cuda_available": bool(torch.cuda.is_available()),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def loader_seed(seed: int, offset: int) -> int:
    return int(seed) + int(offset)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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

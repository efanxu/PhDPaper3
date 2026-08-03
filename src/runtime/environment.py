"""Ordinary environment metadata; no certificate or readiness machinery."""

from __future__ import annotations

import importlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch


def _version(module_name: str) -> str | None:
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None
    return str(getattr(module, "__version__", "unknown"))


def git_commit(project_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value or None


def collect_environment(
    project_root: Path,
    *,
    reproducibility_mode: str | None = None,
    seed_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda else None
    environment = {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": str(Path(sys.executable).resolve()),
        },
        "packages": {
            "numpy": _version("numpy"),
            "PyYAML": _version("yaml"),
            "pyarrow": _version("pyarrow"),
            "torch": _version("torch"),
        },
        "cuda": {
            "available": cuda,
            "version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version() if cuda else None,
        },
        "gpu": {
            "count": torch.cuda.device_count() if cuda else 0,
            "name": gpu_name,
        },
        "git_commit": git_commit(project_root),
        "determinism": {
            "reproducibility_mode": reproducibility_mode,
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "global_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
    }
    environment["reproducibility"] = dict(seed_details or environment["determinism"])
    encoded = json.dumps(environment, ensure_ascii=False, sort_keys=True, default=str).encode()
    import hashlib

    environment["environment_hash"] = hashlib.sha256(encoded).hexdigest()
    return environment

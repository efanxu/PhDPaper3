"""Portable best/last checkpoint save and reload helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .reproducibility import state_dict_hash


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state_dict = _cpu_state_dict(model)
    payload: dict[str, Any] = {
        "state_dict": state_dict,
        "manifest": dict(manifest or {}),
        "state_dict_hash": state_dict_hash(state_dict),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload["manifest"]


def read_checkpoint_manifest(path: Path) -> dict[str, Any]:
    """Read checkpoint metadata without loading weights into a model."""

    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"invalid checkpoint payload: {path}")
    manifest = dict(payload.get("manifest", {}))
    manifest.setdefault("state_dict_hash", payload.get("state_dict_hash"))
    return manifest


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    *,
    device: torch.device | str = "cpu",
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"invalid checkpoint payload: {path}")
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in payload:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if scaler is not None and "scaler_state_dict" in payload:
        scaler.load_state_dict(payload["scaler_state_dict"])
    manifest = dict(payload.get("manifest", {}))
    manifest.setdefault("state_dict_hash", payload.get("state_dict_hash"))
    return manifest

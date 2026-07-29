"""Portable, atomically written checkpoints with complete resume state."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import torch

from .reproducibility import capture_rng_state, restore_rng_state, state_dict_hash


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
    runtime_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state_dict = _cpu_state_dict(model)
    payload: dict[str, Any] = {
        "state_dict": state_dict,
        "manifest": dict(manifest or {}),
        "state_dict_hash": state_dict_hash(state_dict),
        "runtime_state": runtime_state if runtime_state is not None else {
            "rng": capture_rng_state(),
        },
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return payload["manifest"]


def _load_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"invalid checkpoint payload: {path}")
    return payload


def read_checkpoint_manifest(path: Path) -> dict[str, Any]:
    """Read checkpoint metadata without mutating a model or RNG."""

    payload = _load_payload(path)
    manifest = dict(payload.get("manifest", {}))
    manifest.setdefault("state_dict_hash", payload.get("state_dict_hash"))
    manifest.setdefault("runtime_state", payload.get("runtime_state"))
    return manifest


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    *,
    device: torch.device | str = "cpu",
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    restore_runtime: bool = False,
    dataloader_generators: Mapping[str, torch.Generator] | None = None,
) -> dict[str, Any]:
    payload = _load_payload(path)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    if optimizer is not None:
        if "optimizer_state_dict" not in payload:
            raise ValueError(f"checkpoint has no optimizer state: {path}")
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None:
        if "scheduler_state_dict" not in payload:
            raise ValueError(f"checkpoint has no scheduler state: {path}")
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if scaler is not None:
        if "scaler_state_dict" not in payload:
            raise ValueError(f"checkpoint has no AMP scaler state: {path}")
        scaler.load_state_dict(payload["scaler_state_dict"])
    runtime_state = payload.get("runtime_state", {})
    if restore_runtime:
        rng = runtime_state.get("rng")
        if rng is None:
            raise ValueError(f"checkpoint has no complete RNG state: {path}")
        restore_rng_state(rng)
        generator_states = runtime_state.get("dataloader_generators", {})
        if dataloader_generators is not None:
            for name, generator in dataloader_generators.items():
                if name in generator_states:
                    generator.set_state(generator_states[name])
    manifest = dict(payload.get("manifest", {}))
    manifest.setdefault("state_dict_hash", payload.get("state_dict_hash"))
    manifest["runtime_state"] = runtime_state
    return manifest

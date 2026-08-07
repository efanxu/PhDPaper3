"""Resolved training precision policy shared by training and validation."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PrecisionPolicy:
    """The effective autocast decision for one resolved device/config pair."""

    device_type: str
    amp_configured: bool
    amp_effective: bool
    amp_dtype: torch.dtype
    amp_dtype_name: str
    amp_cache_enabled: bool

    def autocast(self) -> AbstractContextManager[None]:
        """Return the context used by forward/backward execution."""

        if not self.amp_effective:
            return nullcontext()
        return torch.autocast(
            device_type=self.device_type,
            dtype=self.amp_dtype,
            enabled=True,
            cache_enabled=self.amp_cache_enabled,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "amp_configured": self.amp_configured,
            "amp_effective": self.amp_effective,
            "amp_dtype": self.amp_dtype_name,
            "amp_cache_enabled": self.amp_cache_enabled,
        }


def resolve_precision_policy(
    *,
    device: torch.device | str,
    amp_configured: bool,
    amp_dtype: str,
    amp_cache_enabled: bool,
) -> PrecisionPolicy:
    """Resolve public AMP settings without allocating any device tensors."""

    selected_device = torch.device(device)
    dtype_name = str(amp_dtype).lower()
    dtypes = {"float16": torch.float16}
    if dtype_name not in dtypes:
        raise ValueError(f"unsupported training AMP dtype: {amp_dtype!r}")
    return PrecisionPolicy(
        device_type=selected_device.type,
        amp_configured=bool(amp_configured),
        amp_effective=bool(amp_configured and selected_device.type == "cuda"),
        amp_dtype=dtypes[dtype_name],
        amp_dtype_name=dtype_name,
        amp_cache_enabled=bool(amp_cache_enabled),
    )

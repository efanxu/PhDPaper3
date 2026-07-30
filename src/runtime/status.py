"""Stable status values and atomic status helpers shared by all commands."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .run_info import utc_now, write_json


PENDING = "PENDING"
RUNNING = "RUNNING"
PASS = "PASS"
FAILED = "FAILED"
SKIPPED = "SKIPPED"

TOP_LEVEL_STATUSES = frozenset({PENDING, RUNNING, PASS, FAILED, SKIPPED})

INTERFACE_SMALL = "INTERFACE_SMALL"
RESOLVED_SHAPE = "RESOLVED_SHAPE"
FORMAL_DEFAULT_SHAPE = "FORMAL_DEFAULT_SHAPE"
SMOKE = "SMOKE"
FULL = "FULL"
REPEATABILITY = "REPEATABILITY"
ENVIRONMENT_PREFLIGHT = "ENVIRONMENT_PREFLIGHT"
MODEL_PREFLIGHT = "MODEL_PREFLIGHT"

PASS_INTERFACE_SMALL = "PASS_INTERFACE_SMALL"
PASS_RESOLVED_SHAPE = "PASS_RESOLVED_SHAPE"
PASS_FORMAL_DEFAULT_SHAPE = "PASS_FORMAL_DEFAULT_SHAPE"
PASS_SMOKE = "PASS_SMOKE"
PASS_FULL = "PASS_FULL"
PASS_REPEATABILITY = "PASS_REPEATABILITY"
PASS_ENVIRONMENT_PREFLIGHT = "PASS_ENVIRONMENT_PREFLIGHT"
PASS_MODEL_PREFLIGHT = "PASS_MODEL_PREFLIGHT"

FAIL_CONFIG = "FAIL_CONFIG"
FAIL_MISSING_RESOURCE = "FAIL_MISSING_RESOURCE"
FAIL_ENVIRONMENT = "FAIL_ENVIRONMENT"
FAIL_CUDA_UNAVAILABLE = "FAIL_CUDA_UNAVAILABLE"
FAIL_MODEL_IMPORT = "FAIL_MODEL_IMPORT"
FAIL_MODEL_BUILD = "FAIL_MODEL_BUILD"
FAIL_GRAPH = "FAIL_GRAPH"
FAIL_DATA = "FAIL_DATA"
FAIL_FORWARD = "FAIL_FORWARD"
FAIL_OUTPUT_SHAPE = "FAIL_OUTPUT_SHAPE"
FAIL_NONFINITE_OUTPUT = "FAIL_NONFINITE_OUTPUT"
FAIL_LOSS = "FAIL_LOSS"
FAIL_BACKWARD = "FAIL_BACKWARD"
FAIL_MISSING_GRADIENT = "FAIL_MISSING_GRADIENT"
FAIL_NONFINITE_GRADIENT = "FAIL_NONFINITE_GRADIENT"
FAIL_OOM = "FAIL_OOM"
FAIL_CHECKPOINT = "FAIL_CHECKPOINT"
FAIL_EVALUATION = "FAIL_EVALUATION"
FAIL_REPEATABILITY = "FAIL_REPEATABILITY"
FAIL_RESULT_WRITE = "FAIL_RESULT_WRITE"
FAIL_WORKER_CRASH = "FAIL_WORKER_CRASH"
FAIL_SIGNAL = "FAIL_SIGNAL"
FAIL_UNKNOWN = "FAIL_UNKNOWN"

PASS_CLASSIFICATIONS = frozenset(
    {
        PASS_INTERFACE_SMALL,
        PASS_RESOLVED_SHAPE,
        PASS_FORMAL_DEFAULT_SHAPE,
        PASS_SMOKE,
        PASS_FULL,
        PASS_REPEATABILITY,
        PASS_ENVIRONMENT_PREFLIGHT,
        PASS_MODEL_PREFLIGHT,
    }
)
FAIL_CLASSIFICATIONS = frozenset(
    {
        FAIL_CONFIG,
        FAIL_MISSING_RESOURCE,
        FAIL_ENVIRONMENT,
        FAIL_CUDA_UNAVAILABLE,
        FAIL_MODEL_IMPORT,
        FAIL_MODEL_BUILD,
        FAIL_GRAPH,
        FAIL_DATA,
        FAIL_FORWARD,
        FAIL_OUTPUT_SHAPE,
        FAIL_NONFINITE_OUTPUT,
        FAIL_LOSS,
        FAIL_BACKWARD,
        FAIL_MISSING_GRADIENT,
        FAIL_NONFINITE_GRADIENT,
        FAIL_OOM,
        FAIL_CHECKPOINT,
        FAIL_EVALUATION,
        FAIL_REPEATABILITY,
        FAIL_RESULT_WRITE,
        FAIL_WORKER_CRASH,
        FAIL_SIGNAL,
        FAIL_UNKNOWN,
    }
)

_PROFILE_PASS = {
    INTERFACE_SMALL: PASS_INTERFACE_SMALL,
    RESOLVED_SHAPE: PASS_RESOLVED_SHAPE,
    FORMAL_DEFAULT_SHAPE: PASS_FORMAL_DEFAULT_SHAPE,
    SMOKE: PASS_SMOKE,
    FULL: PASS_FULL,
    REPEATABILITY: PASS_REPEATABILITY,
    ENVIRONMENT_PREFLIGHT: PASS_ENVIRONMENT_PREFLIGHT,
    MODEL_PREFLIGHT: PASS_MODEL_PREFLIGHT,
}

_OOM_MARKERS = (
    "outofmemoryerror",
    "out of memory",
    "cuda oom",
    "cuda error: out of memory",
    "cublas_status_alloc_failed",
)


def pass_classification(profile: str) -> str:
    """Return the one canonical PASS classification for a profile."""

    try:
        return _PROFILE_PASS[profile]
    except KeyError as exc:
        raise ValueError(f"unsupported status profile: {profile}") from exc


def phase_record(*, status: str = PENDING, classification: str | None = None, phase: str | None = None, artifact: str | None = None, error_summary: str | None = None, started_at: str | None = None, ended_at: str | None = None, wall_seconds: float | None = None) -> dict[str, Any]:
    """Construct one JSON-safe stage record used in batch and model statuses."""

    if status not in TOP_LEVEL_STATUSES:
        raise ValueError(f"unsupported top-level status: {status}")
    return {
        "status": status,
        "classification": classification,
        "phase": phase,
        "started_at": started_at,
        "ended_at": ended_at,
        "wall_seconds": wall_seconds,
        "artifact": artifact,
        "error_summary": error_summary,
    }


def running_phase(phase: str, *, artifact: str | None = None) -> dict[str, Any]:
    return phase_record(status=RUNNING, phase=phase, artifact=artifact, started_at=utc_now())


def finished_phase(
    *,
    profile: str | None = None,
    classification: str | None = None,
    phase: str,
    artifact: str | None = None,
    error_summary: str | None = None,
    started_at: str | None = None,
    wall_seconds: float | None = None,
) -> dict[str, Any]:
    """Create a terminal phase without leaking ad-hoc status literals."""

    resolved_classification = classification or (pass_classification(profile) if profile else None)
    successful = resolved_classification in PASS_CLASSIFICATIONS
    return phase_record(
        status=PASS if successful else FAILED,
        classification=resolved_classification,
        phase=phase,
        artifact=artifact,
        error_summary=error_summary,
        started_at=started_at,
        ended_at=utc_now(),
        wall_seconds=wall_seconds,
    )


def is_oom_failure(value: BaseException | str | None) -> bool:
    """Recognize explicit PyTorch/CUDA OOM diagnostics, never exit codes alone."""

    if value is None:
        return False
    error_type = type(value).__name__.casefold() if isinstance(value, BaseException) else ""
    message = str(value).casefold()
    joined = f"{error_type}: {message}"
    return any(marker in joined for marker in _OOM_MARKERS)


def classify_validation_failure(
    error: BaseException | None = None,
    *,
    message: str | None = None,
    phase: str | None = None,
    exit_code: int | None = None,
    worker_status_present: bool = True,
) -> str:
    """Classify known validation and worker failures into the public schema."""

    detail = " ".join(part for part in (str(error) if error else "", message or "", phase or "") if part).casefold()
    error_name = type(error).__name__.casefold() if error is not None else ""
    if exit_code is not None and exit_code < 0:
        return FAIL_SIGNAL
    if is_oom_failure(error) or is_oom_failure(message):
        return FAIL_OOM
    if "cuda was requested" in detail or "cuda unavailable" in detail:
        return FAIL_CUDA_UNAVAILABLE
    if phase == "environment" or "environment" in error_name or "environment preflight" in detail or "conda" in detail:
        return FAIL_ENVIRONMENT
    if "modulenotfound" in error_name or "importerror" in error_name or "model implementation not found" in detail:
        return FAIL_MODEL_IMPORT
    if "filenotfound" in error_name or "does not exist" in detail or "missing resource" in detail:
        return FAIL_MISSING_RESOURCE
    if (
        "configerror" in error_name
        or phase == "config"
        or "configuration" in detail
        or "result directory" in detail
        or "--resume" in detail
    ):
        return FAIL_CONFIG
    if "graph" in detail:
        return FAIL_GRAPH
    if phase == "data" or "parquet" in detail or "data" in error_name:
        return FAIL_DATA
    if "output must have shape" in detail or "output shape" in detail:
        return FAIL_OUTPUT_SHAPE
    if "output contains nan" in detail or "output contains inf" in detail or "nonfinite output" in detail:
        return FAIL_NONFINITE_OUTPUT
    if phase == "loss" or "loss" in detail:
        return FAIL_LOSS
    if "missing gradient" in detail or "gradient is none" in detail:
        return FAIL_MISSING_GRADIENT
    if "gradient" in detail and ("nan" in detail or "inf" in detail or "nonfinite" in detail):
        return FAIL_NONFINITE_GRADIENT
    if phase == "backward" or "backward" in detail:
        return FAIL_BACKWARD
    if phase == "forward" or "forward" in detail:
        return FAIL_FORWARD
    if phase == "model_build" or "build_model" in detail:
        return FAIL_MODEL_BUILD
    if phase == "checkpoint" or "checkpoint" in detail:
        return FAIL_CHECKPOINT
    if phase == "evaluation" or "evaluation" in detail:
        return FAIL_EVALUATION
    if phase == "result_write" or "write" in detail and "result" in detail:
        return FAIL_RESULT_WRITE
    if not worker_status_present and exit_code not in (None, 0):
        return FAIL_WORKER_CRASH
    return FAIL_UNKNOWN


def failure_summary(error: BaseException | None = None, *, message: str | None = None, limit: int = 2000) -> str | None:
    value = message or (str(error) if error is not None else None)
    if not value:
        return None
    return value[-limit:]


def write_validation_status(path, value: Mapping[str, Any]) -> None:
    """Atomically write the portable per-model or standalone validation JSON."""

    write_json(path, dict(value))

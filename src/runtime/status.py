"""Stable status values and atomic status helpers shared by all commands."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import re
from typing import Any

from .run_info import utc_now, write_json


PENDING = "PENDING"
RUNNING = "RUNNING"
PASS = "PASS"
FAILED = "FAILED"
SKIPPED = "SKIPPED"

TOP_LEVEL_STATUSES = frozenset({PENDING, RUNNING, PASS, FAILED, SKIPPED})
STABLE_CLASSIFICATIONS = frozenset(
    {
        "CONFIG",
        "ENVIRONMENT",
        "DATA",
        "MODEL",
        "OOM",
        "TRAINING",
        "CHECKPOINT",
        "EVALUATION",
        "REPEATABILITY",
        "RUNTIME",
        "UNKNOWN",
    }
)
STABLE_PHASES = frozenset(
    {"preflight", "resolved_shape", "training", "checkpoint", "evaluation", "overall"}
)

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

_LEGACY_CLASSIFICATION_MAP: dict[str, tuple[str, str]] = {
    FAIL_CONFIG: ("CONFIG", "CONFIGURATION_ERROR"),
    FAIL_MISSING_RESOURCE: ("DATA", "MISSING_RESOURCE"),
    FAIL_ENVIRONMENT: ("ENVIRONMENT", "ENVIRONMENT_FAILURE"),
    FAIL_CUDA_UNAVAILABLE: ("ENVIRONMENT", "CUDA_UNAVAILABLE"),
    FAIL_MODEL_IMPORT: ("MODEL", "MODEL_IMPORT_ERROR"),
    FAIL_MODEL_BUILD: ("MODEL", "MODEL_BUILD_ERROR"),
    FAIL_GRAPH: ("MODEL", "GRAPH_CONFIGURATION_ERROR"),
    FAIL_DATA: ("DATA", "DATA_ERROR"),
    FAIL_FORWARD: ("MODEL", "FORWARD_FAILURE"),
    FAIL_OUTPUT_SHAPE: ("MODEL", "OUTPUT_SHAPE_MISMATCH"),
    FAIL_NONFINITE_OUTPUT: ("MODEL", "NONFINITE_OUTPUT"),
    FAIL_LOSS: ("TRAINING", "LOSS_FAILURE"),
    FAIL_BACKWARD: ("TRAINING", "BACKWARD_FAILURE"),
    FAIL_MISSING_GRADIENT: ("TRAINING", "MISSING_GRADIENT"),
    FAIL_NONFINITE_GRADIENT: ("TRAINING", "NONFINITE_GRADIENT"),
    FAIL_OOM: ("OOM", "CUDA_OUT_OF_MEMORY"),
    FAIL_CHECKPOINT: ("CHECKPOINT", "CHECKPOINT_FAILURE"),
    FAIL_EVALUATION: ("EVALUATION", "EVALUATION_FAILURE"),
    FAIL_REPEATABILITY: ("REPEATABILITY", "REPEATABILITY_FAILURE"),
    FAIL_RESULT_WRITE: ("RUNTIME", "RESULT_WRITE_FAILURE"),
    FAIL_WORKER_CRASH: ("RUNTIME", "WORKER_CRASH"),
    FAIL_SIGNAL: ("RUNTIME", "WORKER_SIGNAL"),
    FAIL_UNKNOWN: ("UNKNOWN", "UNKNOWN_FAILURE"),
}
_LEGACY_PHASE_MAP = {
    "environment_preflight": "preflight",
    "model_preflight": "preflight",
    "config": "preflight",
    "data": "preflight",
    "model_build": "preflight",
    "starting": "preflight",
    "forward": "resolved_shape",
    "loss": "resolved_shape",
    "backward": "resolved_shape",
    "checkpoint_write": "checkpoint",
    "checkpoint_reload": "checkpoint",
    "validation": "evaluation",
    "test": "evaluation",
    "complete": "overall",
    "pending": "overall",
    "fail_fast": "overall",
    "resume": "overall",
    "output_directory": "overall",
    "worker_completion_missing": "overall",
}
_OOM_REQUEST = re.compile(r"tried to allocate\s+([0-9]+(?:\.[0-9]+)?)\s*(ki?b|mi?b|gi?b)", re.IGNORECASE)


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


def normalize_legacy_classification(classification: Any) -> tuple[str | None, str | None]:
    """Map schema-v1 detail classifications to schema-v2 category/code pairs."""

    if classification is None or classification in PASS_CLASSIFICATIONS:
        return None, None
    if isinstance(classification, str) and classification in STABLE_CLASSIFICATIONS:
        return classification, None
    return _LEGACY_CLASSIFICATION_MAP.get(str(classification), ("UNKNOWN", "UNKNOWN_FAILURE"))


def _stable_phase(value: Any) -> str:
    phase = str(value or "overall")
    if phase in STABLE_PHASES:
        return phase
    return _LEGACY_PHASE_MAP.get(phase, "overall")


def _oom_requested_allocation_mb(message: Any) -> float | None:
    if not isinstance(message, str):
        return None
    match = _OOM_REQUEST.search(message)
    if match is None:
        return None
    value = float(match.group(1))
    unit = match.group(2).casefold()
    if unit.startswith("g"):
        return value * 1024.0
    if unit.startswith("k"):
        return value / 1024.0
    return value


def _normalize_status_record(value: Mapping[str, Any], *, phase_record_value: bool = False) -> dict[str, Any]:
    """Normalize one v1/v2 status record without requiring optional legacy fields."""

    result = deepcopy(dict(value))
    status = result.get("status")
    if status not in TOP_LEVEL_STATUSES:
        status = FAILED if result.get("classification") in FAIL_CLASSIFICATIONS else PENDING
    result["status"] = status
    legacy_classification = result.get("classification")
    classification, derived_code = normalize_legacy_classification(legacy_classification)
    result["classification"] = classification if status == FAILED else None
    result["phase"] = _stable_phase(result.get("phase"))

    legacy_error = result.get("error")
    error_source = legacy_error if isinstance(legacy_error, Mapping) else {}
    message = error_source.get("message") or result.get("error_message")
    error_type = error_source.get("type") or result.get("exception_type")
    traceback_tail = error_source.get("traceback_tail") or result.get("traceback_tail")
    code = error_source.get("code") or derived_code
    if status == FAILED:
        result["error"] = {
            "code": str(code or "UNKNOWN_FAILURE"),
            "type": error_type,
            "message": message,
            "traceback_tail": traceback_tail,
        }
        if result["classification"] is None:
            result["classification"] = "UNKNOWN"
        if result["classification"] == "OOM":
            result["oom_requested_allocation_mb"] = _oom_requested_allocation_mb(message)
    else:
        result.pop("error", None)
        result.pop("oom_requested_allocation_mb", None)

    if "requested_allocation_mb" in result and "estimated_input_tensor_mb" not in result:
        result["estimated_input_tensor_mb"] = result["requested_allocation_mb"]
    result.pop("requested_allocation_mb", None)
    for field in ("status_history", "validation_status", "exception_type", "error_message", "traceback_tail"):
        result.pop(field, None)

    phases = result.get("phases")
    if isinstance(phases, Mapping) and not phase_record_value:
        normalized_phases: dict[str, dict[str, Any]] = {}
        for legacy_name, phase_value in phases.items():
            if not isinstance(phase_value, Mapping):
                continue
            stable_name = _stable_phase(legacy_name)
            normalized = _normalize_status_record(phase_value, phase_record_value=True)
            normalized["phase"] = stable_name
            existing = normalized_phases.get(stable_name)
            priority = {PENDING: 0, RUNNING: 1, PASS: 2, SKIPPED: 2, FAILED: 3}
            if existing is None or priority[normalized["status"]] >= priority[existing["status"]]:
                normalized_phases[stable_name] = normalized
        result["phases"] = normalized_phases
    return result


def normalize_status_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    """Read schema-v1, schema-v2 or partial status JSON as a schema-v2 payload.

    This is deliberately tolerant: status readers need only the stable public
    fields, while unknown legacy details remain ignorable rather than blocking
    existing result directories.
    """

    result = deepcopy(dict(value))
    for collection_name in ("models", "results"):
        collection = result.get(collection_name)
        if isinstance(collection, list):
            result[collection_name] = [
                _normalize_status_record(item) if isinstance(item, Mapping) else item
                for item in collection
            ]
    if "status" in result:
        result = _normalize_status_record(result)
    result["schema_version"] = 2
    return result


def write_status(path, value: Mapping[str, Any]) -> None:
    """Persist schema-v2 status while accepting v1-shaped writer inputs."""

    write_json(path, normalize_status_payload(value))


def write_validation_status(path, value: Mapping[str, Any]) -> None:
    """Compatibility alias for shape/check status writers."""

    write_status(path, value)

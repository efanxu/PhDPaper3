"""Stable status values and the single atomic status writer.

The public status contract is intentionally small.  The legacy classification
names below are kept only for reading old result directories; new callers use
stable classifications plus a structured ``error`` object.
"""

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
STABLE_OPERATIONS = frozenset(
    {"train", "check", "preflight", "evaluate", "repeatability", "summarize"}
)
PROFILES = frozenset(
    {
        "INTERFACE_SMALL",
        "RESOLVED_SHAPE",
        "FORMAL_DEFAULT_SHAPE",
        "SMOKE",
        "FULL",
        "REPEATABILITY",
    }
)

INTERFACE_SMALL = "INTERFACE_SMALL"
RESOLVED_SHAPE = "RESOLVED_SHAPE"
FORMAL_DEFAULT_SHAPE = "FORMAL_DEFAULT_SHAPE"
SMOKE = "SMOKE"
FULL = "FULL"
REPEATABILITY = "REPEATABILITY"


# This is a read-only compatibility table.  These names must never be
# imported by another production module or emitted by a new run.
_LEGACY_CLASSIFICATION_MAP: dict[str, tuple[str, str]] = {
    "FAIL_CONFIG": ("CONFIG", "INVALID_CONFIG"),
    "FAIL_MISSING_RESOURCE": ("DATA", "MISSING_RESOURCE"),
    "FAIL_ENVIRONMENT": ("ENVIRONMENT", "ENVIRONMENT_PREFLIGHT_FAILED"),
    "FAIL_CUDA_UNAVAILABLE": ("ENVIRONMENT", "CUDA_UNAVAILABLE"),
    "FAIL_MODEL_IMPORT": ("MODEL", "MODEL_IMPORT_FAILED"),
    "FAIL_MODEL_BUILD": ("MODEL", "MODEL_BUILD_FAILED"),
    "FAIL_GRAPH": ("DATA", "GRAPH_BUILD_FAILED"),
    "FAIL_DATA": ("DATA", "DATA_PREPARATION_FAILED"),
    "FAIL_FORWARD": ("MODEL", "FORWARD_FAILED"),
    "FAIL_OUTPUT_SHAPE": ("MODEL", "OUTPUT_SHAPE_MISMATCH"),
    "FAIL_NONFINITE_OUTPUT": ("MODEL", "NONFINITE_OUTPUT"),
    "FAIL_LOSS": ("TRAINING", "LOSS_FAILED"),
    "FAIL_BACKWARD": ("TRAINING", "BACKWARD_FAILED"),
    "FAIL_MISSING_GRADIENT": ("TRAINING", "MISSING_GRADIENT"),
    "FAIL_NONFINITE_GRADIENT": ("TRAINING", "NONFINITE_GRADIENT"),
    "FAIL_OOM": ("OOM", "CUDA_OUT_OF_MEMORY"),
    "FAIL_CHECKPOINT": ("CHECKPOINT", "CHECKPOINT_FAILED"),
    "FAIL_EVALUATION": ("EVALUATION", "EVALUATION_FAILED"),
    "FAIL_REPEATABILITY": ("REPEATABILITY", "REPEATABILITY_MISMATCH"),
    "FAIL_RESULT_WRITE": ("RUNTIME", "RESULT_WRITE_FAILED"),
    "FAIL_WORKER_CRASH": ("RUNTIME", "WORKER_CRASH"),
    "FAIL_SIGNAL": ("RUNTIME", "WORKER_SIGNAL"),
    "FAIL_UNKNOWN": ("UNKNOWN", "UNKNOWN_ERROR"),
}
_LEGACY_PASS_PROFILE_MAP = {
    "PASS_INTERFACE_SMALL": INTERFACE_SMALL,
    "PASS_RESOLVED_SHAPE": RESOLVED_SHAPE,
    "PASS_FORMAL_DEFAULT_SHAPE": FORMAL_DEFAULT_SHAPE,
    "PASS_SMOKE": SMOKE,
    "PASS_FULL": FULL,
    "PASS_REPEATABILITY": REPEATABILITY,
    # These old preflight values had no useful stable profile.
    "PASS_ENVIRONMENT_PREFLIGHT": None,
    "PASS_MODEL_PREFLIGHT": None,
}

_LEGACY_PHASE_MAP = {
    "environment_preflight": "preflight",
    "environment_preflight_complete": "preflight",
    "model_preflight": "preflight",
    "model_preflight_complete": "preflight",
    "config": "preflight",
    "data": "preflight",
    "model_build": "preflight",
    "starting": "preflight",
    "forward": "resolved_shape",
    "backward_complete": "resolved_shape",
    "loss": "resolved_shape",
    "backward": "resolved_shape",
    "checkpoint_write": "checkpoint",
    "checkpoint_reload": "checkpoint",
    "validation": "evaluation",
    "validation_complete": "evaluation",
    "test": "evaluation",
    "test_complete": "evaluation",
    "training_complete": "training",
    "complete": "overall",
    "completed": "overall",
    "pending": "overall",
    "fail_fast": "overall",
    "resume": "overall",
    "output_directory": "overall",
    "worker_completion_missing": "overall",
    "completed_run_reused": "overall",
}
_OOM_MARKERS = (
    "outofmemoryerror",
    "out of memory",
    "cuda oom",
    "cuda error: out of memory",
    "cublas_status_alloc_failed",
)
_OOM_REQUEST = re.compile(
    r"tried to allocate\s+([0-9]+(?:\.[0-9]+)?)\s*(ki?b|mi?b|gi?b)",
    re.IGNORECASE,
)


def _stable_phase(value: Any) -> str:
    phase = str(value or "overall")
    if phase in STABLE_PHASES:
        return phase
    return _LEGACY_PHASE_MAP.get(phase, "overall")


def stable_phase(value: Any) -> str:
    """Return the coarse public phase for an implementation phase."""

    return _stable_phase(value)


def _failure_kind(
    error: BaseException | None = None,
    *,
    message: str | None = None,
    phase: str | None = None,
    exit_code: int | None = None,
    worker_status_present: bool = True,
) -> tuple[str, str]:
    detail = " ".join(
        part for part in (str(error) if error else "", message or "", phase or "") if part
    ).casefold()
    error_name = type(error).__name__.casefold() if error is not None else ""
    if exit_code is not None and exit_code < 0:
        return "RUNTIME", "WORKER_SIGNAL"
    if is_oom_failure(error) or is_oom_failure(message):
        return "OOM", "CUDA_OUT_OF_MEMORY"
    if "cuda was requested" in detail or "cuda unavailable" in detail:
        return "ENVIRONMENT", "CUDA_UNAVAILABLE"
    if (
        phase == "environment"
        or "environment" in error_name
        or "environment preflight" in detail
        or "conda" in detail
    ):
        return "ENVIRONMENT", "ENVIRONMENT_PREFLIGHT_FAILED"
    if (
        "modulenotfound" in error_name
        or "importerror" in error_name
        or "model implementation not found" in detail
    ):
        return "MODEL", "MODEL_IMPORT_FAILED"
    if (
        "filenotfound" in error_name
        or "does not exist" in detail
        or "missing resource" in detail
    ):
        return "DATA", "MISSING_RESOURCE"
    if "graph" in detail:
        return "DATA", "GRAPH_BUILD_FAILED"
    if (
        "configerror" in error_name
        or phase in {"config", "resume", "output_directory"}
        or "configuration" in detail
        or "result directory" in detail
        or "--resume" in detail
    ):
        return "CONFIG", "INVALID_CONFIG"
    if phase == "data" or "parquet" in detail or "data" in error_name:
        return "DATA", "DATA_PREPARATION_FAILED"
    if "output must have shape" in detail or "output shape" in detail:
        return "MODEL", "OUTPUT_SHAPE_MISMATCH"
    if "output contains nan" in detail or "output contains inf" in detail or "nonfinite output" in detail:
        return "MODEL", "NONFINITE_OUTPUT"
    if phase == "loss" or "loss" in detail:
        return "TRAINING", "LOSS_FAILED"
    if "missing gradient" in detail or "gradient is none" in detail:
        return "TRAINING", "MISSING_GRADIENT"
    if "gradient" in detail and (
        "nan" in detail or "inf" in detail or "nonfinite" in detail
    ):
        return "TRAINING", "NONFINITE_GRADIENT"
    if phase == "backward" or "backward" in detail:
        return "TRAINING", "BACKWARD_FAILED"
    if phase == "forward" or "forward" in detail:
        return "MODEL", "FORWARD_FAILED"
    if phase == "model_build" or "build_model" in detail:
        return "MODEL", "MODEL_BUILD_FAILED"
    if phase == "checkpoint" or "checkpoint" in detail:
        return "CHECKPOINT", "CHECKPOINT_FAILED"
    if phase == "evaluation" or "evaluation" in detail:
        return "EVALUATION", "EVALUATION_FAILED"
    if phase == "repeatability" or "repeatability" in detail:
        return "REPEATABILITY", "REPEATABILITY_MISMATCH"
    if phase == "result_write" or ("write" in detail and "result" in detail):
        return "RUNTIME", "RESULT_WRITE_FAILED"
    if not worker_status_present and exit_code not in (None, 0):
        return "RUNTIME", "WORKER_CRASH"
    return "UNKNOWN", "UNKNOWN_ERROR"


def _fallback_error_code(classification: str, *, phase: Any = None, message: Any = None, exit_code: Any = None) -> str:
    inferred_classification, inferred_code = _failure_kind(
        message=str(message) if message is not None else None,
        phase=str(phase) if phase is not None else None,
        exit_code=int(exit_code) if isinstance(exit_code, int) else None,
        worker_status_present=False if classification == "RUNTIME" else True,
    )
    if inferred_classification == classification:
        return inferred_code
    return {
        "CONFIG": "INVALID_CONFIG",
        "ENVIRONMENT": "ENVIRONMENT_PREFLIGHT_FAILED",
        "DATA": "DATA_PREPARATION_FAILED",
        "MODEL": "MODEL_BUILD_FAILED",
        "OOM": "CUDA_OUT_OF_MEMORY",
        "TRAINING": "BACKWARD_FAILED",
        "CHECKPOINT": "CHECKPOINT_FAILED",
        "EVALUATION": "EVALUATION_FAILED",
        "REPEATABILITY": "REPEATABILITY_MISMATCH",
        "RUNTIME": "WORKER_CRASH",
        "UNKNOWN": "UNKNOWN_ERROR",
    }.get(classification, "UNKNOWN_ERROR")


def failure_details(
    error: BaseException | None = None,
    *,
    message: str | None = None,
    phase: str | None = None,
    exit_code: int | None = None,
    worker_status_present: bool = True,
    traceback_tail: str | None = None,
    limit: int = 2000,
) -> dict[str, Any]:
    """Return stable classification and a canonical error object."""

    classification, code = _failure_kind(
        error,
        message=message,
        phase=phase,
        exit_code=exit_code,
        worker_status_present=worker_status_present,
    )
    return {
        "classification": classification,
        "error": {
            "code": code,
            "type": type(error).__name__ if error is not None else None,
            "message": failure_summary(error, message=message, limit=limit),
            "traceback_tail": traceback_tail,
        },
    }


def classify_validation_failure(
    error: BaseException | None = None,
    *,
    message: str | None = None,
    phase: str | None = None,
    exit_code: int | None = None,
    worker_status_present: bool = True,
) -> str:
    """Classify a failure using only the stable public categories."""

    return str(
        failure_details(
            error,
            message=message,
            phase=phase,
            exit_code=exit_code,
            worker_status_present=worker_status_present,
        )["classification"]
    )


def is_oom_failure(value: BaseException | str | None) -> bool:
    """Recognize explicit PyTorch/CUDA OOM diagnostics, never exit codes alone."""

    if value is None:
        return False
    error_type = type(value).__name__.casefold() if isinstance(value, BaseException) else ""
    message = str(value).casefold()
    joined = f"{error_type}: {message}"
    return any(marker in joined for marker in _OOM_MARKERS)


def failure_summary(
    error: BaseException | None = None,
    *,
    message: str | None = None,
    limit: int = 2000,
) -> str | None:
    value = message or (str(error) if error is not None else None)
    if not value:
        return None
    return value[-limit:]


def normalize_legacy_classification(classification: Any) -> tuple[str | None, str | None]:
    """Map a legacy classification to a stable category and error code."""

    if classification is None:
        return None, None
    if isinstance(classification, str) and classification in STABLE_CLASSIFICATIONS:
        return classification, None
    if isinstance(classification, str) and classification in _LEGACY_PASS_PROFILE_MAP:
        return None, None
    return _LEGACY_CLASSIFICATION_MAP.get(
        str(classification), ("UNKNOWN", "LEGACY_UNKNOWN_CLASSIFICATION")
    )


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


def _canonical_error(
    source: Mapping[str, Any] | None,
    *,
    derived_code: str | None,
    classification: str,
    phase: Any,
    message: Any,
    error_type: Any,
    traceback_tail: Any,
    exit_code: Any,
) -> dict[str, Any]:
    source = source or {}
    code = source.get("code") or derived_code or _fallback_error_code(
        classification,
        phase=phase,
        message=message,
        exit_code=exit_code,
    )
    return {
        "code": str(code),
        "type": source.get("type") or error_type,
        "message": source.get("message") or message,
        "traceback_tail": source.get("traceback_tail") or traceback_tail,
    }


def _normalize_status_record(
    value: Mapping[str, Any], *, phase_record_value: bool = False
) -> dict[str, Any]:
    """Normalize one v1/v2 record without requiring optional fields."""

    result = deepcopy(dict(value))
    if result.get("operation") in {"environment_preflight", "model_preflight"}:
        result["operation"] = "preflight"
    raw_status = result.get("status")
    raw_classification = result.get("classification")
    legacy_profile = _LEGACY_PASS_PROFILE_MAP.get(str(raw_classification))
    if result.get("profile") not in PROFILES:
        result["profile"] = legacy_profile
    if raw_status == "COMPLETED":
        status = PASS
        if result.get("profile") is None:
            result["profile"] = FULL
    elif raw_status == "ERROR":
        status = FAILED
    elif raw_status in TOP_LEVEL_STATUSES:
        status = raw_status
    elif raw_status is None and legacy_profile is not None:
        status = PASS
    else:
        status = FAILED if str(raw_classification) in _LEGACY_CLASSIFICATION_MAP else PENDING
    result["status"] = status

    classification, derived_code = normalize_legacy_classification(raw_classification)
    phase_value = result.get("phase")
    result["phase"] = _stable_phase(phase_value)
    if status != FAILED:
        result["classification"] = None
    else:
        result["classification"] = classification if classification in STABLE_CLASSIFICATIONS else "UNKNOWN"

    legacy_error = result.get("error")
    error_source = legacy_error if isinstance(legacy_error, Mapping) else None
    message = (
        error_source.get("message") if error_source is not None else None
    ) or result.get("error_message") or result.get("error_summary")
    error_type = (
        error_source.get("type") if error_source is not None else None
    ) or result.get("exception_type")
    traceback_tail = (
        error_source.get("traceback_tail") if error_source is not None else None
    ) or result.get("traceback_tail")
    if status == FAILED:
        if result["classification"] == "UNKNOWN" and derived_code is None:
            derived_code = "LEGACY_UNKNOWN_CLASSIFICATION" if raw_classification is not None else None
        result["error"] = _canonical_error(
            error_source,
            derived_code=derived_code,
            classification=result["classification"],
            phase=phase_value,
            message=message,
            error_type=error_type,
            traceback_tail=traceback_tail,
            exit_code=result.get("exit_code"),
        )
        if result["classification"] == "OOM":
            result["oom_requested_allocation_mb"] = _oom_requested_allocation_mb(
                result["error"].get("message")
            )
    else:
        result["error"] = None
        result.pop("oom_requested_allocation_mb", None)

    if "requested_allocation_mb" in result and "estimated_input_tensor_mb" not in result:
        result["estimated_input_tensor_mb"] = result["requested_allocation_mb"]
    result.pop("requested_allocation_mb", None)
    for field in (
        "status_history",
        "validation_status",
        "exception_type",
        "error_message",
        "traceback_tail",
    ):
        result.pop(field, None)

    phases = result.get("phases")
    if isinstance(phases, Mapping) and not phase_record_value:
        normalized_phases: dict[str, dict[str, Any]] = {}
        priority = {PENDING: 0, RUNNING: 1, PASS: 2, SKIPPED: 2, FAILED: 3}
        for legacy_name, phase_value_record in phases.items():
            if not isinstance(phase_value_record, Mapping):
                continue
            stable_name = _stable_phase(legacy_name)
            normalized = _normalize_status_record(
                phase_value_record, phase_record_value=True
            )
            normalized["phase"] = stable_name
            existing = normalized_phases.get(stable_name)
            if existing is None or priority[normalized["status"]] >= priority[existing["status"]]:
                normalized_phases[stable_name] = normalized
        result["phases"] = normalized_phases
    return result


def normalize_status_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    """Read schema-v1, schema-v2 or partial status JSON as schema v2.

    Unknown fields are retained so a newer writer can be read by an older
    command, but no optional legacy field is required and the input mapping is
    never modified.
    """

    result = deepcopy(dict(value))
    for collection_name in ("models", "results"):
        collection = result.get(collection_name)
        if isinstance(collection, list):
            result[collection_name] = [
                _normalize_status_record(item) if isinstance(item, Mapping) else item
                for item in collection
            ]
    if "status" in result or "classification" in result or "phase" in result:
        result = _normalize_status_record(result)
    elif result.get("operation") in {"environment_preflight", "model_preflight"}:
        result["operation"] = "preflight"
    result["schema_version"] = 2
    return result


def phase_record(
    *,
    status: str = PENDING,
    classification: str | None = None,
    phase: str | None = None,
    artifact: str | None = None,
    error_summary: str | None = None,
    error: Mapping[str, Any] | None = None,
    details: Mapping[str, Any] | None = None,
    started_at: str | None = None,
    ended_at: str | None = None,
    wall_seconds: float | None = None,
) -> dict[str, Any]:
    """Construct one JSON-safe coarse phase record."""

    if status not in TOP_LEVEL_STATUSES:
        raise ValueError(f"unsupported top-level status: {status}")
    stable_classification = (
        classification if classification in STABLE_CLASSIFICATIONS else None
    )
    result: dict[str, Any] = {
        "status": status,
        "classification": stable_classification if status == FAILED else None,
        "phase": _stable_phase(phase),
        "started_at": started_at,
        "ended_at": ended_at,
        "wall_seconds": wall_seconds,
        "artifact": artifact,
        "error_summary": error_summary,
        "error": deepcopy(dict(error)) if isinstance(error, Mapping) else None,
    }
    if isinstance(details, Mapping):
        result["details"] = deepcopy(dict(details))
    return result


def running_phase(phase: str, *, artifact: str | None = None) -> dict[str, Any]:
    return phase_record(status=RUNNING, phase=phase, artifact=artifact, started_at=utc_now())


def finished_phase(
    *,
    profile: str | None = None,
    classification: str | None = None,
    phase: str,
    artifact: str | None = None,
    error_summary: str | None = None,
    error: Mapping[str, Any] | None = None,
    details: Mapping[str, Any] | None = None,
    started_at: str | None = None,
    wall_seconds: float | None = None,
) -> dict[str, Any]:
    """Create a terminal coarse phase without profile-specific classes."""

    if profile is not None and profile not in PROFILES:
        raise ValueError(f"unsupported status profile: {profile}")
    stable_classification = (
        classification if classification in STABLE_CLASSIFICATIONS else None
    )
    if stable_classification is None and classification is not None:
        stable_classification, legacy_code = normalize_legacy_classification(classification)
        if error is None and stable_classification is not None:
            error = {
                "code": legacy_code or _fallback_error_code(
                    stable_classification, phase=phase, message=error_summary
                ),
                "type": None,
                "message": error_summary,
                "traceback_tail": None,
            }
    elif stable_classification is not None and error is None:
        error = {
            "code": _fallback_error_code(
                stable_classification, phase=phase, message=error_summary
            ),
            "type": None,
            "message": error_summary,
            "traceback_tail": None,
        }
    successful = stable_classification is None and error is None and not error_summary
    return phase_record(
        status=PASS if successful else FAILED,
        classification=stable_classification,
        phase=phase,
        artifact=artifact,
        error_summary=error_summary,
        error=error,
        details=details,
        started_at=started_at,
        ended_at=utc_now(),
        wall_seconds=wall_seconds,
    )


def write_status(path, value: Mapping[str, Any]) -> None:
    """Persist a schema-v2 status atomically after central normalization."""

    write_json(path, normalize_status_payload(value))


def write_validation_status(path, value: Mapping[str, Any]) -> None:
    """Keep the worker-file name as a compatibility alias to ``write_status``."""

    write_status(path, value)

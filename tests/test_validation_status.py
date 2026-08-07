from __future__ import annotations

import json
from pathlib import Path

import pytest

from cli import orchestrator
from cli import train
from runtime.config import ConfigError
from runtime.paths import is_completed_run
from runtime.status import (
    FAILED,
    FORMAL_DEFAULT_SHAPE,
    FULL,
    INTERFACE_SMALL,
    PASS,
    RESOLVED_SHAPE,
    SMOKE,
    STABLE_CLASSIFICATIONS,
    classify_validation_failure,
    failure_details,
    normalize_status_payload,
    write_status,
    write_validation_status,
)
from runtime.validation import _batch_size


@pytest.mark.parametrize(
    ("error", "message", "phase", "expected_classification", "expected_code"),
    [
        (ConfigError("bad"), None, "config", "CONFIG", "INVALID_CONFIG"),
        (FileNotFoundError("missing graph"), None, "data", "DATA", "MISSING_RESOURCE"),
        (ModuleNotFoundError("model"), None, "model_build", "MODEL", "MODEL_IMPORT_FAILED"),
        (RuntimeError("build_model failed"), None, "model_build", "MODEL", "MODEL_BUILD_FAILED"),
        (ValueError("graph edge mismatch"), None, "model_build", "DATA", "GRAPH_BUILD_FAILED"),
        (ValueError("bad parquet"), None, "data", "DATA", "DATA_PREPARATION_FAILED"),
        (ValueError("output must have shape (1, 2, 3)"), None, "forward", "MODEL", "OUTPUT_SHAPE_MISMATCH"),
        (FloatingPointError("output contains NaN or Inf"), None, "forward", "MODEL", "NONFINITE_OUTPUT"),
        (RuntimeError("loss invalid"), None, "loss", "TRAINING", "LOSS_FAILED"),
        (RuntimeError("backward failed"), None, "backward", "TRAINING", "BACKWARD_FAILED"),
        (RuntimeError("missing gradient after backward"), None, "backward", "TRAINING", "MISSING_GRADIENT"),
        (FloatingPointError("gradient contains NaN"), None, "backward", "TRAINING", "NONFINITE_GRADIENT"),
        (RuntimeError("CUDA out of memory. Tried to allocate 1 GiB"), None, "forward", "OOM", "CUDA_OUT_OF_MEMORY"),
        (RuntimeError("CUDA was requested but is unavailable"), None, "config", "ENVIRONMENT", "CUDA_UNAVAILABLE"),
        (RuntimeError("result write failed"), None, "result_write", "RUNTIME", "RESULT_WRITE_FAILED"),
        (None, "worker killed", "worker", "RUNTIME", "WORKER_SIGNAL"),
        (None, "worker crashed", "worker", "RUNTIME", "WORKER_CRASH"),
    ],
)
def test_failure_classification_and_error_code(
    error, message, phase, expected_classification, expected_code
) -> None:
    kwargs = {"message": message, "phase": phase}
    if expected_code == "WORKER_SIGNAL":
        kwargs.update(exit_code=-9, worker_status_present=False)
    if expected_code == "WORKER_CRASH":
        kwargs.update(exit_code=3, worker_status_present=False)
    assert classify_validation_failure(error, **kwargs) == expected_classification
    details = failure_details(error, **kwargs)
    assert details["classification"] == expected_classification
    assert details["error"]["code"] == expected_code


def test_shape_profiles_and_batch_sources_are_not_conflated() -> None:
    assert normalize_status_payload(
        {"status": PASS, "operation": "check", "profile": INTERFACE_SMALL}
    )["classification"] is None
    assert normalize_status_payload(
        {"status": PASS, "operation": "check", "profile": RESOLVED_SHAPE}
    )["profile"] == RESOLVED_SHAPE
    assert normalize_status_payload(
        {"status": PASS, "operation": "check", "profile": FORMAL_DEFAULT_SHAPE}
    )["profile"] == FORMAL_DEFAULT_SHAPE
    assert _batch_size(RESOLVED_SHAPE, 1, {"training.train_batch_size": 1}) == (1, "cli_override")
    assert _batch_size(FORMAL_DEFAULT_SHAPE, 32, {}) == (32, "yaml_default")
    assert _batch_size(INTERFACE_SMALL, 32, {}) == (2, "interface_small")


def test_validation_status_write_is_atomic_and_always_v2(tmp_path: Path) -> None:
    path = tmp_path / "validation_status.json"
    write_validation_status(
        path,
        {
            "schema_version": 1,
            "model": "lstm",
            "status": PASS,
            "classification": "PASS_RESOLVED_SHAPE",
            "phase": "backward_complete",
        },
    )
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["schema_version"] == 2
    assert saved["status"] == PASS
    assert saved["classification"] is None
    assert saved["profile"] == RESOLVED_SHAPE
    assert saved["error"] is None
    assert not list(tmp_path.glob(".*.tmp"))


def test_missing_shape_worker_status_becomes_worker_crash(tmp_path: Path) -> None:
    result = orchestrator._shape_result_from_worker(
        status_path=tmp_path / "missing.json",
        model="crossformer",
        run_id="run",
        runtime_fields={},
        exit_code=7,
        output_tail="worker disappeared",
        wall_seconds=1.0,
        profile=RESOLVED_SHAPE,
    )
    assert result["classification"] == "RUNTIME"
    assert result["error"]["code"] == "WORKER_CRASH"
    saved = json.loads((tmp_path / "missing.json").read_text(encoding="utf-8"))
    assert saved["schema_version"] == 2
    assert saved["status"] == FAILED


def test_train_worker_failure_finalizes_existing_run_info(monkeypatch, tmp_path: Path) -> None:
    config = Path(__file__).resolve().parents[1] / "configs" / "experiment.yaml"
    output = tmp_path / "results" / "lstm" / "failure"
    output.mkdir(parents=True)
    (output / "run_info.json").write_text(
        json.dumps({"status": "RUNNING", "phase": "forward", "start_time": "start", "phases": {}}),
        encoding="utf-8",
    )

    def fail(**kwargs):
        del kwargs
        raise RuntimeError("synthetic forward failure")

    monkeypatch.setattr(train, "_run_model_impl", fail)
    with pytest.raises(RuntimeError, match="synthetic forward failure"):
        train.run_model(
            model_name="lstm",
            config_path=config,
            output_root=tmp_path / "results",
            run_id="failure",
        )
    result = json.loads((output / "run_info.json").read_text(encoding="utf-8"))
    assert result["schema_version"] == 2
    assert result["status"] == FAILED
    assert result["classification"] == "MODEL"
    assert result["error"]["code"] == "FORWARD_FAILED"
    assert result["phases"]["overall"]["status"] == FAILED


def test_environment_preflight_failure_is_persisted_at_batch_boundary(monkeypatch, tmp_path: Path) -> None:
    config = Path(__file__).resolve().parents[1] / "configs" / "experiment.yaml"

    def fail(**kwargs):
        del kwargs
        raise RuntimeError("Conda environment is unavailable")

    monkeypatch.setattr(orchestrator, "_prepare_batch_environments", fail)
    result = orchestrator.run_training_models(
        models=["lstm"],
        config_path=config,
        model_config_path=None,
        run_id="environment-failure",
        device="cuda",
        output_root=tmp_path / "results",
        resume=False,
        overwrite=False,
        id_suffix=None,
        fail_fast=False,
        smoke=False,
        smoke_epochs=None,
        smoke_max_train_updates=None,
        smoke_max_eval_batches=None,
        cli_overrides={},
        command_argv=[],
    )
    assert result["passed"] is False
    assert result["models"][0]["classification"] == "ENVIRONMENT"
    assert result["models"][0]["error"]["code"] == "ENVIRONMENT_PREFLIGHT_FAILED"
    saved = json.loads((tmp_path / "results" / "_runs" / "environment-failure" / "status.json").read_text(encoding="utf-8"))
    assert saved["schema_version"] == 2
    assert saved["models"][0]["phases"]["preflight"]["status"] == FAILED


@pytest.mark.parametrize(
    ("legacy_classification", "expected_profile", "expected_classification", "expected_code"),
    [
        ("PASS_RESOLVED_SHAPE", RESOLVED_SHAPE, None, None),
        ("PASS_SMOKE", SMOKE, None, None),
        ("PASS_FULL", FULL, None, None),
        ("FAIL_NONFINITE_OUTPUT", None, "MODEL", "NONFINITE_OUTPUT"),
        ("FAIL_OOM", None, "OOM", "CUDA_OUT_OF_MEMORY"),
        ("FAIL_WORKER_CRASH", None, "RUNTIME", "WORKER_CRASH"),
        ("NEW_LEGACY_VALUE", None, "UNKNOWN", "LEGACY_UNKNOWN_CLASSIFICATION"),
    ],
)
def test_legacy_schema_normalization(
    legacy_classification, expected_profile, expected_classification, expected_code
) -> None:
    payload = {
        "schema_version": 1,
        "status": PASS if legacy_classification.startswith("PASS_") else FAILED,
        "classification": legacy_classification,
        "phase": "forward",
        "error_message": "output contains NaN or Inf",
        "exception_type": "ValueError",
        "future_field": {"anything": True},
    }
    normalized = normalize_status_payload(payload)
    assert normalized["schema_version"] == 2
    assert normalized["profile"] == expected_profile
    assert normalized["classification"] == expected_classification
    if expected_code is None:
        assert normalized["error"] is None
    else:
        assert normalized["error"]["code"] == expected_code


def test_legacy_completed_and_unknown_fields_are_tolerated() -> None:
    normalized = normalize_status_payload(
        {
            "status": "COMPLETED",
            "classification": "PASS_FULL",
            "future_field": {"anything": True},
        }
    )
    assert normalized["schema_version"] == 2
    assert normalized["status"] == PASS
    assert normalized["profile"] == FULL
    assert normalized["classification"] is None
    assert normalized["error"] is None
    assert normalized["future_field"] == {"anything": True}


def test_new_classifications_are_stable_and_never_profile_specific() -> None:
    payloads = [
        {"status": PASS, "profile": profile}
        for profile in (INTERFACE_SMALL, RESOLVED_SHAPE, FORMAL_DEFAULT_SHAPE, SMOKE, FULL)
    ]
    payloads.extend(
        [
            {"status": FAILED, "classification": classification, "error": {"code": "X"}}
            for classification in STABLE_CLASSIFICATIONS
        ]
    )
    for payload in payloads:
        normalized = normalize_status_payload(payload)
        assert normalized.get("classification") in (None, *STABLE_CLASSIFICATIONS)
        assert not str(normalized.get("classification")).startswith(("PASS_", "FAIL_"))


def _write_complete_artifacts(path: Path, info: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for name in ("best.pt", "last.pt", "performance.json", "metrics_validation.json", "metrics_test_h6.json"):
        (path / name).write_text("{}", encoding="utf-8")
    (path / "run_info.json").write_text(json.dumps(info), encoding="utf-8")


@pytest.mark.parametrize(
    "info",
    [
        {"schema_version": 2, "status": PASS, "operation": "train", "profile": FULL},
        {"schema_version": 1, "status": PASS, "classification": "PASS_FULL"},
        {"status": "COMPLETED"},
    ],
)
def test_resume_completion_does_not_depend_on_internal_validation_or_error_code(
    tmp_path: Path, info: dict
) -> None:
    path = tmp_path / "model" / "run"
    _write_complete_artifacts(path, info)
    assert is_completed_run(path)
    (path / "validation_status.json").write_text("{}", encoding="utf-8")
    assert is_completed_run(path)
    if info.get("schema_version") == 2:
        info["error"] = {"code": "A_DIFFERENT_DETAIL"}
        (path / "run_info.json").write_text(json.dumps(info), encoding="utf-8")
        assert is_completed_run(path)


def test_write_status_does_not_mutate_input(tmp_path: Path) -> None:
    payload = {"schema_version": 1, "status": FAILED, "classification": "FAIL_OOM"}
    before = json.loads(json.dumps(payload))
    write_status(tmp_path / "status.json", payload)
    assert payload == before

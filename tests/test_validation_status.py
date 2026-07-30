from __future__ import annotations

import json
from pathlib import Path

import pytest

from cli import orchestrator
from cli import train
from runtime.config import ConfigError
from runtime.status import (
    FAILED,
    FAIL_BACKWARD,
    FAIL_CONFIG,
    FAIL_CUDA_UNAVAILABLE,
    FAIL_DATA,
    FAIL_ENVIRONMENT,
    FAIL_GRAPH,
    FAIL_LOSS,
    FAIL_MISSING_GRADIENT,
    FAIL_MISSING_RESOURCE,
    FAIL_MODEL_BUILD,
    FAIL_MODEL_IMPORT,
    FAIL_NONFINITE_GRADIENT,
    FAIL_NONFINITE_OUTPUT,
    FAIL_OOM,
    FAIL_OUTPUT_SHAPE,
    FAIL_RESULT_WRITE,
    FAIL_SIGNAL,
    FAIL_WORKER_CRASH,
    FORMAL_DEFAULT_SHAPE,
    INTERFACE_SMALL,
    PASS,
    PASS_FORMAL_DEFAULT_SHAPE,
    PASS_INTERFACE_SMALL,
    PASS_RESOLVED_SHAPE,
    RESOLVED_SHAPE,
    classify_validation_failure,
    normalize_status_payload,
    pass_classification,
    write_validation_status,
)
from runtime.validation import _batch_size


@pytest.mark.parametrize(
    ("error", "message", "phase", "expected"),
    [
        (ConfigError("bad"), None, "config", FAIL_CONFIG),
        (FileNotFoundError("missing graph"), None, "data", FAIL_MISSING_RESOURCE),
        (ModuleNotFoundError("model"), None, "model_build", FAIL_MODEL_IMPORT),
        (RuntimeError("build_model failed"), None, "model_build", FAIL_MODEL_BUILD),
        (ValueError("graph edge mismatch"), None, "model_build", FAIL_GRAPH),
        (ValueError("bad parquet"), None, "data", FAIL_DATA),
        (ValueError("output must have shape (1, 2, 3)"), None, "forward", FAIL_OUTPUT_SHAPE),
        (FloatingPointError("output contains NaN or Inf"), None, "forward", FAIL_NONFINITE_OUTPUT),
        (RuntimeError("loss invalid"), None, "loss", FAIL_LOSS),
        (RuntimeError("backward failed"), None, "backward", FAIL_BACKWARD),
        (RuntimeError("missing gradient after backward"), None, "backward", FAIL_MISSING_GRADIENT),
        (FloatingPointError("gradient contains NaN"), None, "backward", FAIL_NONFINITE_GRADIENT),
        (RuntimeError("CUDA out of memory. Tried to allocate 1 GiB"), None, "forward", FAIL_OOM),
        (RuntimeError("CUDA was requested but is unavailable"), None, "config", FAIL_CUDA_UNAVAILABLE),
        (RuntimeError("result write failed"), None, "result_write", FAIL_RESULT_WRITE),
        (None, "worker killed", "worker", FAIL_SIGNAL),
        (None, "worker crashed", "worker", FAIL_WORKER_CRASH),
    ],
)
def test_failure_classification_is_specific(error, message, phase, expected) -> None:
    kwargs = {"message": message, "phase": phase}
    if expected == FAIL_SIGNAL:
        kwargs["exit_code"] = -9
        kwargs["worker_status_present"] = False
    if expected == FAIL_WORKER_CRASH:
        kwargs["exit_code"] = 3
        kwargs["worker_status_present"] = False
    assert classify_validation_failure(error, **kwargs) == expected


def test_shape_profiles_and_batch_sources_are_not_conflated() -> None:
    assert pass_classification(INTERFACE_SMALL) == PASS_INTERFACE_SMALL
    assert pass_classification(RESOLVED_SHAPE) == PASS_RESOLVED_SHAPE
    assert pass_classification(FORMAL_DEFAULT_SHAPE) == PASS_FORMAL_DEFAULT_SHAPE
    assert _batch_size(RESOLVED_SHAPE, 1, {"training.train_batch_size": 1}) == (1, "cli_override")
    assert _batch_size(FORMAL_DEFAULT_SHAPE, 32, {}) == (32, "yaml_default")
    assert _batch_size(INTERFACE_SMALL, 32, {}) == (2, "interface_small")


def test_validation_status_write_is_atomic_and_readable(tmp_path: Path) -> None:
    path = tmp_path / "validation_status.json"
    write_validation_status(
        path,
        {
            "schema_version": 1,
            "model": "node_shared_lstm",
            "status": PASS,
            "classification": PASS_RESOLVED_SHAPE,
            "phase": "backward_complete",
        },
    )
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == PASS
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
    assert result["classification"] == FAIL_WORKER_CRASH
    assert json.loads((tmp_path / "missing.json").read_text(encoding="utf-8"))["status"] == "FAILED"


def test_train_worker_failure_finalizes_existing_run_info(monkeypatch, tmp_path: Path) -> None:
    config = Path(__file__).resolve().parents[1] / "configs" / "experiment.yaml"
    output = tmp_path / "results" / "node_shared_lstm" / "failure"
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
            model_name="node_shared_lstm",
            config_path=config,
            output_root=tmp_path / "results",
            run_id="failure",
        )
    result = json.loads((output / "run_info.json").read_text(encoding="utf-8"))
    assert result["status"] == "FAILED"
    assert result["classification"] == "MODEL"
    assert result["error"]["code"] == "FORWARD_FAILURE"
    assert result["phases"]["overall"]["status"] == "FAILED"


def test_environment_preflight_failure_is_persisted_at_batch_boundary(monkeypatch, tmp_path: Path) -> None:
    config = Path(__file__).resolve().parents[1] / "configs" / "experiment.yaml"

    def fail(**kwargs):
        del kwargs
        raise RuntimeError("Conda environment is unavailable")

    monkeypatch.setattr(orchestrator, "_prepare_batch_environments", fail)
    result = orchestrator.run_training_models(
        models=["node_shared_lstm"],
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
    assert result["models"][0]["classification"] == FAIL_ENVIRONMENT
    saved = json.loads((tmp_path / "results" / "_runs" / "environment-failure" / "status.json").read_text(encoding="utf-8"))
    assert saved["models"][0]["phases"]["preflight"]["status"] == "FAILED"


def test_schema_v1_failure_normalizes_to_v2_error_code_without_duplicate_history() -> None:
    normalized = normalize_status_payload(
        {
            "schema_version": 1,
            "status": FAILED,
            "classification": FAIL_NONFINITE_OUTPUT,
            "phase": "forward",
            "error_message": "output contains NaN or Inf",
            "exception_type": "FloatingPointError",
            "traceback_tail": "trace",
            "status_history": ["PENDING", FAILED],
            "metrics_complete": False,
        }
    )
    assert normalized["schema_version"] == 2
    assert normalized["classification"] == "MODEL"
    assert normalized["phase"] == "resolved_shape"
    assert normalized["error"] == {
        "code": "NONFINITE_OUTPUT",
        "type": "FloatingPointError",
        "message": "output contains NaN or Inf",
        "traceback_tail": "trace",
    }
    assert "status_history" not in normalized


def test_schema_v1_oom_and_partial_v2_payload_are_backward_compatible() -> None:
    old_oom = normalize_status_payload(
        {
            "status": FAILED,
            "classification": FAIL_OOM,
            "phase": "forward",
            "error_message": "CUDA out of memory. Tried to allocate 1.5 GiB",
        }
    )
    assert old_oom["classification"] == "OOM"
    assert old_oom["error"]["code"] == "CUDA_OUT_OF_MEMORY"
    assert old_oom["oom_requested_allocation_mb"] == 1536.0

    partial_v2 = normalize_status_payload(
        {"schema_version": 2, "status": PASS, "unknown_future_field": {"ok": True}}
    )
    assert partial_v2["schema_version"] == 2
    assert partial_v2["classification"] is None
    assert partial_v2["unknown_future_field"] == {"ok": True}

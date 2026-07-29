from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from cli import orchestrator


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiment.yaml"


class _FakeProcess:
    next_pid = 41000
    started: list[tuple[str, str]] = []

    def __init__(self, command, **kwargs):
        del kwargs
        self.pid = _FakeProcess.next_pid
        _FakeProcess.next_pid += 1
        self.stdout = iter(["worker output\n"])
        self._returncode = 0
        _FakeProcess.started.append((command[0], command[-1]))

    def wait(self):
        return self._returncode


class _SelectiveFakeProcess(_FakeProcess):
    def __init__(self, command, **kwargs):
        super().__init__(command, **kwargs)
        self._returncode = 1 if command[-1] == "node_shared_lstm" else 0


def test_isolated_checks_use_sys_executable_and_distinct_worker_pids(monkeypatch, tmp_path: Path) -> None:
    _FakeProcess.started.clear()
    monkeypatch.setattr(orchestrator.subprocess, "Popen", _FakeProcess)
    result = orchestrator.run_isolated_checks(
        operation="check",
        models=["node_shared_lstm", "node_shared_lstm"],
        config_path=CONFIG,
        model_config_path=None,
        device="cpu",
        cli_overrides={},
    )
    assert result["passed"]
    assert [item["exit_code"] for item in result["results"]] == [0, 0]
    assert all(command == sys.executable for command, _ in _FakeProcess.started)
    assert len(_FakeProcess.started) == 2
    assert _FakeProcess.next_pid >= 41002


def test_default_conflict_fails_before_starting_worker(monkeypatch, tmp_path: Path) -> None:
    output_root = tmp_path / "results"
    result_dir = output_root / "node_shared_lstm" / "conflict"
    result_dir.mkdir(parents=True)
    (result_dir / "best.pt").write_bytes(b"old")
    calls: list[object] = []

    def fake_popen(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("worker must not start for a preflight conflict")

    monkeypatch.setattr(orchestrator.subprocess, "Popen", fake_popen)
    result = orchestrator.run_training_models(
        models=["node_shared_lstm"],
        config_path=CONFIG,
        model_config_path=None,
        run_id="conflict",
        device="cpu",
        output_root=output_root,
        resume=False,
        overwrite=False,
        id_suffix=None,
        fail_fast=False,
        smoke=True,
        smoke_epochs=1,
        smoke_max_train_updates=1,
        smoke_max_eval_batches=1,
        cli_overrides={},
        command_argv=[],
    )
    assert result["passed"] is False
    assert result["models"][0]["status"] == "FAILED"
    assert calls == []


def test_overwrite_archives_old_result_and_suffix_is_distinct(monkeypatch, tmp_path: Path) -> None:
    output_root = tmp_path / "results"
    old = output_root / "node_shared_lstm" / "rerun"
    old.mkdir(parents=True)
    (old / "best.pt").write_bytes(b"old")
    monkeypatch.setattr(orchestrator.subprocess, "Popen", _FakeProcess)
    overwritten = orchestrator.run_training_models(
        models=["node_shared_lstm"],
        config_path=CONFIG,
        model_config_path=None,
        run_id="rerun",
        device="cpu",
        output_root=output_root,
        resume=False,
        overwrite=True,
        id_suffix=None,
        fail_fast=False,
        smoke=True,
        smoke_epochs=1,
        smoke_max_train_updates=1,
        smoke_max_eval_batches=1,
        cli_overrides={},
        command_argv=[],
    )
    assert overwritten["passed"]
    assert Path(overwritten["models"][0]["archive_path"]).is_dir()
    suffixed = orchestrator.run_training_models(
        models=["node_shared_lstm"],
        config_path=CONFIG,
        model_config_path=None,
        run_id="rerun",
        device="cpu",
        output_root=output_root,
        resume=False,
        overwrite=False,
        id_suffix="rerun1",
        fail_fast=False,
        smoke=True,
        smoke_epochs=1,
        smoke_max_train_updates=1,
        smoke_max_eval_batches=1,
        cli_overrides={},
        command_argv=[],
    )
    assert suffixed["run_id"] == "rerun__rerun1"
    assert Path(suffixed["models"][0]["result_dir"]).name == "rerun__rerun1"


def test_failed_model_continues_by_default_and_fail_fast_stops(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(orchestrator.subprocess, "Popen", _SelectiveFakeProcess)
    common = {
        "models": ["node_shared_lstm", "dlinear"],
        "config_path": CONFIG,
        "model_config_path": None,
        "device": "cpu",
        "output_root": tmp_path / "results",
        "resume": False,
        "overwrite": False,
        "id_suffix": None,
        "smoke": True,
        "smoke_epochs": 1,
        "smoke_max_train_updates": 1,
        "smoke_max_eval_batches": 1,
        "cli_overrides": {},
        "command_argv": [],
    }
    continued = orchestrator.run_training_models(run_id="continue", fail_fast=False, **common)
    assert [item["status"] for item in continued["models"]] == ["FAILED", "COMPLETED"]
    stopped = orchestrator.run_training_models(run_id="stop", fail_fast=True, **common)
    assert stopped["models"][0]["status"] == "FAILED"
    assert stopped["models"][1]["status"] == "FAILED"
    assert "fail-fast" in stopped["models"][1]["error_summary"]

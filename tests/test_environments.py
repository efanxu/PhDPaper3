from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import yaml

from cli import orchestrator
from runtime.environments import (
    EnvironmentResolutionError,
    EnvironmentSpec,
    ResolvedEnvironment,
    load_environment_config,
    preflight_environment,
    preflight_model,
    resolve_model_environment,
    resolve_python_executable,
)


ROOT = Path(__file__).resolve().parents[1]


def _config(tmp_path: Path, *, default: str = "tslib") -> Path:
    path = tmp_path / "environments.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "default_environment": default,
                "environments": {
                    "tslib": {
                        "conda_env": "env_tslib",
                        "python_override_env_var": "PHDPAPER3_TSLIB_PYTHON",
                        "source_roots": ["Time-Series-Library"],
                        "required_imports": ["yaml"],
                    },
                    "tsl": {
                        "conda_env": "env_tsl",
                        "python_override_env_var": "PHDPAPER3_TSL_PYTHON",
                        "source_roots": [],
                        "required_imports": ["yaml", "tsl"],
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _spec(environment_id: str = "tslib") -> EnvironmentSpec:
    return EnvironmentSpec(
        environment_id=environment_id,
        conda_env="env_tslib" if environment_id == "tslib" else "env_tsl",
        python_override_env_var=f"PHDPAPER3_{environment_id.upper()}_PYTHON",
        source_roots=(),
        required_imports=(),
    )


def test_environment_variable_has_priority(tmp_path: Path) -> None:
    override = tmp_path / "override-python"
    resolved = resolve_python_executable(
        _spec(),
        project_root=tmp_path,
        current_executable=tmp_path / "wrong" / "python",
        current_prefix=tmp_path / "wrong",
        environment_variables={"PHDPAPER3_TSLIB_PYTHON": str(override)},
    )
    assert resolved.python_executable == override
    assert resolved.resolution_source == "environment_variable"


def test_current_interpreter_is_used_when_it_belongs_to_target() -> None:
    resolved = resolve_python_executable(
        _spec(),
        current_executable=Path("C:/conda/envs/env_tslib/python.exe"),
        current_prefix=Path("C:/conda/envs/env_tslib"),
        environment_variables={},
    )
    assert str(resolved.python_executable).replace("\\", "/").endswith("env_tslib/python.exe")
    assert resolved.resolution_source == "current_interpreter"


def test_conda_env_list_exact_name_is_used(monkeypatch, tmp_path: Path) -> None:
    env_dir = tmp_path / "envs" / "env_tsl"
    python = env_dir / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")

    def fake_run(command, environment):
        assert command[1:] == ["env", "list", "--json"]
        del environment
        return SimpleNamespace(returncode=0, stdout=json.dumps({"envs": [str(env_dir)]}), stderr="")

    monkeypatch.setattr("runtime.environments._run_conda", fake_run)
    resolved = resolve_python_executable(
        _spec("tsl"),
        project_root=tmp_path,
        current_executable=tmp_path / "wrong" / "python",
        current_prefix=tmp_path / "wrong",
        environment_variables={"CONDA_EXE": "conda"},
    )
    assert resolved.python_executable == python.resolve()
    assert resolved.resolution_source == "conda_env_list"


def test_conda_base_fallback_is_used(monkeypatch, tmp_path: Path) -> None:
    base = tmp_path / "miniconda"
    python = base / "envs" / "env_tslib" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")

    def fake_run(command, environment):
        del environment
        if command[1:4] == ["env", "list", "--json"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout=f"{base}\n", stderr="")

    monkeypatch.setattr("runtime.environments._run_conda", fake_run)
    resolved = resolve_python_executable(
        _spec(),
        project_root=tmp_path,
        current_executable=tmp_path / "wrong" / "python",
        current_prefix=tmp_path / "wrong",
        environment_variables={"CONDA_EXE": str(tmp_path / "conda" / "conda.exe")},
    )
    assert resolved.python_executable == python.resolve()
    assert resolved.resolution_source == "conda_base"


def test_missing_environment_fails_closed(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("runtime.environments._run_conda", lambda command, environment: None)
    with pytest.raises(EnvironmentResolutionError, match="could not resolve Python"):
        resolve_python_executable(
            _spec("tsl"),
            project_root=tmp_path,
            current_executable=tmp_path / "wrong" / "python",
            current_prefix=tmp_path / "wrong",
            environment_variables={},
        )


def test_model_environment_defaults_and_tsl_selection(tmp_path: Path) -> None:
    environment_config = _config(tmp_path)
    default_model = tmp_path / "default.yaml"
    default_model.write_text("model:\n  width: 2\n", encoding="utf-8")
    tsl_model = tmp_path / "tsl.yaml"
    tsl_model.write_text("runtime:\n  environment: tsl\nmodel:\n  width: 2\n", encoding="utf-8")
    default = resolve_model_environment(
        default_model,
        environment_config_path=environment_config,
        project_root=tmp_path,
        current_executable=sys.executable,
        current_prefix=sys.prefix,
        environment_variables={"PHDPAPER3_TSLIB_PYTHON": sys.executable},
    )
    assert default.environment_id == "tslib"
    assert resolve_model_environment(
        tsl_model,
        environment_config_path=environment_config,
        project_root=tmp_path,
        current_executable=sys.executable,
        current_prefix=sys.prefix,
        environment_variables={"PHDPAPER3_TSL_PYTHON": sys.executable},
    ).environment_id == "tsl"


def test_environment_yaml_does_not_contain_windows_absolute_paths() -> None:
    source = (ROOT / "configs" / "environments.yaml").read_text(encoding="utf-8")
    assert "D:\\Apps" not in source
    assert load_environment_config(ROOT / "configs" / "environments.yaml").default_environment == "tslib"


def test_preflight_checks_target_python_imports_and_model_config() -> None:
    resolved = ResolvedEnvironment(
        environment_id="tslib",
        conda_env="env_tslib",
        python_executable=Path(sys.executable),
        source_roots=(),
        required_imports=("yaml",),
        resolution_source="current_interpreter",
    )
    result = preflight_environment(
        resolved,
        project_root=ROOT,
        device="cpu",
        model_name="lstm",
        model_config_path=ROOT / "configs" / "models" / "lstm.yaml",
    )
    assert result["environment_id"] == "tslib"
    assert result["python_executable"] == sys.executable


def test_pure_tsl_environment_preflight_does_not_require_time_series_library(monkeypatch, tmp_path: Path) -> None:
    resolved = ResolvedEnvironment(
        environment_id="tsl",
        conda_env="env_tsl",
        python_executable=Path(sys.executable),
        source_roots=(),
        required_imports=("yaml",),
        resolution_source="current_interpreter",
    )
    # The temporary project has no Time-Series-Library directory. Its empty
    # source_roots are the whole environment-level source requirement.
    monkeypatch.setenv("PYTHONPATH", str(ROOT / "src"))
    result = preflight_environment(resolved, project_root=tmp_path, device="cpu")
    assert result["environment_id"] == "tsl"


def test_model_preflight_is_invoked_for_every_model_after_one_environment_preflight(monkeypatch, tmp_path: Path) -> None:
    model_paths = {}
    for name in ("model_a", "model_b", "model_c"):
        path = tmp_path / f"{name}.yaml"
        path.write_text("model:\n  width: 2\n", encoding="utf-8")
        model_paths[name] = path
    environment_calls: list[str] = []
    model_calls: list[str] = []

    def fake_environment(resolved, **kwargs):
        environment_calls.append(resolved.environment_id)
        return {"environment_id": resolved.environment_id, "conda_env": resolved.conda_env, "python_executable": str(resolved.python_executable), "python_version": "3.11.0"}

    def fake_model(resolved, *, model_name, **kwargs):
        model_calls.append(model_name)
        return {"model": model_name, "runtime_environment": resolved.environment_id, "parameter_count": 1}

    monkeypatch.setattr(orchestrator, "preflight_environment", fake_environment)
    monkeypatch.setattr(orchestrator, "preflight_model", fake_model)
    resolved, results = orchestrator._prepare_batch_environments(
        models=list(model_paths), model_configs=model_paths, project_root=ROOT, device="cpu"
    )
    assert list(resolved) == ["model_a", "model_b", "model_c"]
    assert environment_calls == ["tslib"]
    assert model_calls == ["model_a", "model_b", "model_c"]
    assert set(results["tslib"]["model_preflights"]) == set(model_paths)


def test_preflight_import_failure_names_model_and_environment() -> None:
    resolved = ResolvedEnvironment(
        environment_id="tsl",
        conda_env="env_tsl",
        python_executable=Path(sys.executable),
        source_roots=(),
        required_imports=("module_that_does_not_exist_phdpaper3",),
        resolution_source="environment_variable",
    )
    with pytest.raises(RuntimeError, match="model dcrnn requires runtime environment 'tsl'.*could not import"):
        preflight_environment(resolved, project_root=ROOT, device="cpu", model_name="dcrnn")


def test_batch_preflights_each_environment_once(monkeypatch, tmp_path: Path) -> None:
    model_paths = {}
    for name in ("model_a", "model_b", "model_c"):
        path = tmp_path / f"{name}.yaml"
        path.write_text("model:\n  width: 2\n", encoding="utf-8")
        model_paths[name] = path
    calls: list[str] = []

    def fake_preflight(resolved, **kwargs):
        calls.append(resolved.environment_id)
        return {
            "environment_id": resolved.environment_id,
            "conda_env": resolved.conda_env,
            "python_executable": str(resolved.python_executable),
            "python_version": "3.11.0",
        }

    monkeypatch.setattr(orchestrator, "preflight_environment", fake_preflight)
    monkeypatch.setattr(
        orchestrator,
        "preflight_model",
        lambda resolved, *, model_name, **kwargs: {"model": model_name, "runtime_environment": resolved.environment_id, "parameter_count": 1},
    )
    resolved, results = orchestrator._prepare_batch_environments(
        models=list(model_paths),
        model_configs=model_paths,
        project_root=ROOT,
        device="cpu",
    )
    assert list(resolved) == ["model_a", "model_b", "model_c"]
    assert calls == ["tslib"]
    assert set(results) == {"tslib"}

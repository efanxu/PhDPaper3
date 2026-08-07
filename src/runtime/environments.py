"""Conda environment discovery and isolated worker environment setup.

The project configuration contains portable environment identities only.  A
machine-specific Python executable is discovered at runtime and is never
written back to the YAML configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path, PureWindowsPath
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import yaml


class EnvironmentResolutionError(RuntimeError):
    """Raised when a configured Conda environment cannot be resolved safely."""


class EnvironmentPreflightError(RuntimeError):
    """Raised when a resolved worker environment is not usable."""


@dataclass(frozen=True)
class EnvironmentSpec:
    """Portable definition of one supported runtime environment."""

    environment_id: str
    conda_env: str
    python_override_env_var: str
    source_roots: tuple[str, ...]
    required_imports: tuple[str, ...]

    @property
    def id(self) -> str:
        return self.environment_id


@dataclass(frozen=True)
class EnvironmentConfig:
    """Validated contents of ``configs/environments.yaml``."""

    source: Path
    default_environment: str
    environments: dict[str, EnvironmentSpec]

    def get(self, environment_id: str) -> EnvironmentSpec:
        try:
            return self.environments[environment_id]
        except KeyError as exc:
            supported = ", ".join(sorted(self.environments))
            raise EnvironmentResolutionError(
                f"unknown runtime environment {environment_id!r}; supported: {supported}"
            ) from exc


@dataclass(frozen=True)
class ResolvedEnvironment:
    """A model environment plus its machine-specific Python executable."""

    environment_id: str
    conda_env: str
    python_executable: Path
    source_roots: tuple[Path, ...]
    required_imports: tuple[str, ...]
    resolution_source: str

    @property
    def id(self) -> str:
        return self.environment_id


def _project_root_for_config(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.parent.name.lower() == "configs":
        return resolved.parent.parent
    return Path.cwd().resolve()


def _is_absolute_portable_path(value: str) -> bool:
    return Path(value).is_absolute() or PureWindowsPath(value).is_absolute()


def _load_yaml(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"environment configuration does not exist: {path}")
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise EnvironmentResolutionError(f"invalid environment YAML: {path}: {exc}") from exc


def load_environment_config(path: str | Path = "configs/environments.yaml") -> EnvironmentConfig:
    """Load the portable environment registry."""

    source = Path(path).resolve()
    loaded = _load_yaml(source)
    if not isinstance(loaded, Mapping):
        raise EnvironmentResolutionError(f"environment configuration root must be a mapping: {source}")
    if not all(isinstance(key, str) for key in loaded):
        raise EnvironmentResolutionError("environment configuration keys must be strings")
    unknown = sorted(set(loaded) - {"default_environment", "environments"})
    if unknown:
        raise EnvironmentResolutionError(f"unknown field in environment configuration: {unknown[0]}")
    default = loaded.get("default_environment")
    environments = loaded.get("environments")
    if not isinstance(default, str) or not default:
        raise EnvironmentResolutionError("default_environment must be a non-empty string")
    if not isinstance(environments, Mapping) or not environments:
        raise EnvironmentResolutionError("environments must be a non-empty mapping")

    specs: dict[str, EnvironmentSpec] = {}
    allowed_fields = {
        "conda_env",
        "python_override_env_var",
        "source_roots",
        "required_imports",
    }
    for raw_id, raw_spec in environments.items():
        if not isinstance(raw_id, str) or not raw_id:
            raise EnvironmentResolutionError("environment names must be non-empty strings")
        if not isinstance(raw_spec, Mapping):
            raise EnvironmentResolutionError(f"environment {raw_id!r} must be a mapping")
        if not all(isinstance(key, str) for key in raw_spec):
            raise EnvironmentResolutionError(
                f"environment {raw_id!r} configuration keys must be strings"
            )
        unknown_fields = sorted(set(raw_spec) - allowed_fields)
        if unknown_fields:
            raise EnvironmentResolutionError(
                f"unknown field at environments.{raw_id}: {unknown_fields[0]}"
            )
        conda_env = raw_spec.get("conda_env")
        override_var = raw_spec.get("python_override_env_var")
        source_roots = raw_spec.get("source_roots")
        required_imports = raw_spec.get("required_imports")
        if not isinstance(conda_env, str) or not conda_env:
            raise EnvironmentResolutionError(f"environments.{raw_id}.conda_env must be a string")
        if not isinstance(override_var, str) or not override_var:
            raise EnvironmentResolutionError(
                f"environments.{raw_id}.python_override_env_var must be a string"
            )
        if not isinstance(source_roots, list) or not all(
            isinstance(value, str) and value for value in source_roots
        ):
            raise EnvironmentResolutionError(
                f"environments.{raw_id}.source_roots must be a list of strings"
            )
        for root in source_roots:
            if _is_absolute_portable_path(root):
                raise EnvironmentResolutionError(
                    f"environments.{raw_id}.source_roots must use relative paths: {root}"
                )
        if not isinstance(required_imports, list) or not all(
            isinstance(value, str) and value for value in required_imports
        ):
            raise EnvironmentResolutionError(
                f"environments.{raw_id}.required_imports must be a list of strings"
            )
        specs[raw_id] = EnvironmentSpec(
            environment_id=raw_id,
            conda_env=conda_env,
            python_override_env_var=override_var,
            source_roots=tuple(source_roots),
            required_imports=tuple(required_imports),
        )
    if default not in specs:
        raise EnvironmentResolutionError(
            f"default_environment {default!r} is not defined in environments"
        )
    return EnvironmentConfig(source=source, default_environment=default, environments=specs)


def _environment_spec(
    value: EnvironmentSpec | str,
    *,
    config: EnvironmentConfig | None,
) -> EnvironmentSpec:
    if isinstance(value, EnvironmentSpec):
        return value
    if not isinstance(value, str):
        raise TypeError("environment must be an EnvironmentSpec or environment id")
    environment_config = config or load_environment_config()
    return environment_config.get(value)


def _path_without_unwanted_resolution(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute() or path.exists():
        return path.resolve()
    # Preserve a portable-looking Windows override when tests or configuration
    # are inspected on a different host.  Preflight will fail closed if it is
    # not an executable path on the current machine.
    if PureWindowsPath(str(value)).is_absolute():
        return path
    return path.absolute()


def _path_basename(value: str | Path) -> str:
    text = str(value).replace("/", "\\")
    return PureWindowsPath(text).name or Path(value).name


def _belongs_to_conda_environment(
    executable: Path,
    spec: EnvironmentSpec,
    *,
    current_prefix: str | Path | None,
    environment: Mapping[str, str],
) -> bool:
    expected = spec.conda_env.casefold()
    candidates = {
        _path_basename(executable.parent),
        _path_basename(current_prefix) if current_prefix else "",
        _path_basename(environment.get("CONDA_PREFIX", "")) if environment.get("CONDA_PREFIX") else "",
    }
    return expected in {value.casefold() for value in candidates if value}


def _conda_command(environment: Mapping[str, str]) -> str:
    configured = environment.get("CONDA_EXE")
    if configured:
        return configured
    return shutil.which("conda") or "conda"


def _run_conda(command: Sequence[str], environment: Mapping[str, str]) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            check=False,
            env=dict(environment),
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _python_candidates(environment_directory: Path) -> list[Path]:
    return [
        environment_directory / "python.exe",
        environment_directory / "bin" / "python",
        environment_directory / "python",
        environment_directory / "Scripts" / "python.exe",
    ]


def _python_from_environment_directory(value: str | Path) -> Path | None:
    directory = _path_without_unwanted_resolution(value)
    if directory.is_file():
        return directory
    for candidate in _python_candidates(directory):
        if candidate.is_file():
            return candidate.resolve()
    return None


def _find_conda_env_list_python(spec: EnvironmentSpec, environment: Mapping[str, str]) -> Path | None:
    conda = _conda_command(environment)
    completed = _run_conda([conda, "env", "list", "--json"], environment)
    if completed is None or completed.returncode != 0:
        return None
    try:
        payload = json.loads(completed.stdout)
    except (TypeError, ValueError):
        return None
    entries = payload.get("envs") if isinstance(payload, Mapping) else None
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if not isinstance(entry, str) or _path_basename(entry).casefold() != spec.conda_env.casefold():
            continue
        python = _python_from_environment_directory(entry)
        if python is not None:
            return python
    return None


def _base_directories(spec: EnvironmentSpec, environment: Mapping[str, str]) -> list[Path]:
    bases: list[Path] = []
    conda_exe = environment.get("CONDA_EXE")
    if conda_exe:
        exe_path = _path_without_unwanted_resolution(conda_exe)
        if exe_path.parent.parent not in bases:
            bases.append(exe_path.parent.parent)
    conda = _conda_command(environment)
    completed = _run_conda([conda, "info", "--base"], environment)
    if completed is not None and completed.returncode == 0:
        for line in completed.stdout.splitlines():
            candidate = line.strip()
            if candidate:
                base = _path_without_unwanted_resolution(candidate)
                if base not in bases:
                    bases.append(base)
                break
    return bases


def resolve_python_executable(
    environment: EnvironmentSpec | str,
    *,
    environment_config: EnvironmentConfig | None = None,
    project_root: str | Path | None = None,
    current_executable: str | Path | None = None,
    current_prefix: str | Path | None = None,
    environment_variables: Mapping[str, str] | None = None,
) -> ResolvedEnvironment:
    """Resolve a Python executable in the documented fail-closed order."""

    spec = _environment_spec(environment, config=environment_config)
    environment_values = dict(os.environ if environment_variables is None else environment_variables)
    project = Path(project_root).resolve() if project_root is not None else Path.cwd().resolve()
    source_roots = tuple((project / root).resolve() for root in spec.source_roots)
    override = environment_values.get(spec.python_override_env_var)
    if override:
        return ResolvedEnvironment(
            environment_id=spec.environment_id,
            conda_env=spec.conda_env,
            python_executable=_path_without_unwanted_resolution(override),
            source_roots=source_roots,
            required_imports=spec.required_imports,
            resolution_source="environment_variable",
        )

    current = _path_without_unwanted_resolution(current_executable or sys.executable)
    prefix = current_prefix if current_prefix is not None else sys.prefix
    if _belongs_to_conda_environment(current, spec, current_prefix=prefix, environment=environment_values):
        return ResolvedEnvironment(
            environment_id=spec.environment_id,
            conda_env=spec.conda_env,
            python_executable=current,
            source_roots=source_roots,
            required_imports=spec.required_imports,
            resolution_source="current_interpreter",
        )

    from_env_list = _find_conda_env_list_python(spec, environment_values)
    if from_env_list is not None:
        return ResolvedEnvironment(
            environment_id=spec.environment_id,
            conda_env=spec.conda_env,
            python_executable=from_env_list,
            source_roots=source_roots,
            required_imports=spec.required_imports,
            resolution_source="conda_env_list",
        )

    for base in _base_directories(spec, environment_values):
        candidate = _python_from_environment_directory(base / "envs" / spec.conda_env)
        if candidate is None and _path_basename(base).casefold() == spec.conda_env.casefold():
            candidate = _python_from_environment_directory(base)
        if candidate is not None:
            return ResolvedEnvironment(
                environment_id=spec.environment_id,
                conda_env=spec.conda_env,
                python_executable=candidate,
                source_roots=source_roots,
                required_imports=spec.required_imports,
                resolution_source="conda_base",
            )

    raise EnvironmentResolutionError(
        f"could not resolve Python for runtime environment {spec.environment_id!r} "
        f"(Conda environment {spec.conda_env!r}); set {spec.python_override_env_var} "
        "or make the named Conda environment discoverable"
    )


def resolve_model_environment(
    model_config_path: str | Path,
    *,
    environment_config_path: str | Path | None = None,
    project_root: str | Path | None = None,
    current_executable: str | Path | None = None,
    current_prefix: str | Path | None = None,
    environment_variables: Mapping[str, str] | None = None,
) -> ResolvedEnvironment:
    """Resolve the environment declared by one model YAML file."""

    from .config import load_model_config_document

    model_path = Path(model_config_path).resolve()
    root = Path(project_root).resolve() if project_root is not None else _project_root_for_config(model_path)
    config_path = (
        Path(environment_config_path).resolve()
        if environment_config_path is not None
        else root / "configs" / "environments.yaml"
    )
    environment_config = load_environment_config(config_path)
    document = load_model_config_document(model_path)
    declared = document.get("runtime", {}).get("environment")
    environment_id = declared or environment_config.default_environment
    if not isinstance(environment_id, str):
        raise EnvironmentResolutionError(
            f"model config {model_path} has an invalid runtime.environment"
        )
    if environment_id not in environment_config.environments:
        raise EnvironmentResolutionError(
            f"model config {model_path} selects unknown runtime environment {environment_id!r}"
        )
    return resolve_python_executable(
        environment_config.get(environment_id),
        environment_config=environment_config,
        project_root=root,
        current_executable=current_executable,
        current_prefix=current_prefix,
        environment_variables=environment_variables,
    )


def build_worker_environment(
    resolved_environment: ResolvedEnvironment,
    *,
    project_root: str | Path,
    base_environment: Mapping[str, str] | None = None,
    resolved_seed: int | None = None,
) -> dict[str, str]:
    """Build a worker environment with resolved reproducibility variables."""

    if not isinstance(resolved_environment, ResolvedEnvironment):
        raise TypeError("build_worker_environment requires ResolvedEnvironment")
    root = Path(project_root).resolve()
    values = dict(os.environ if base_environment is None else base_environment)
    entries = [root / "src", *resolved_environment.source_roots]
    existing = values.get("PYTHONPATH", "")
    if existing:
        entries.extend(Path(item) for item in existing.split(os.pathsep) if item)
    unique: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        value = str(entry)
        key = value.casefold() if os.name == "nt" else value
        if key not in seen:
            seen.add(key)
            unique.append(value)
    values["PYTHONPATH"] = os.pathsep.join(unique)
    values["PHDPAPER3_RUNTIME_ENVIRONMENT"] = resolved_environment.environment_id
    values["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if resolved_seed is not None:
        if not isinstance(resolved_seed, int) or isinstance(resolved_seed, bool) or resolved_seed < 0:
            raise ValueError("resolved_seed must be a non-negative integer")
        values["PYTHONHASHSEED"] = str(resolved_seed)
    return values


_PREFLIGHT_SCRIPT = r'''
import importlib
import json
import sys


def fail(kind, message, module=None):
    print(json.dumps({"ok": False, "kind": kind, "message": message, "module": module}))
    raise SystemExit(1)


if sys.version_info < (3, 11):
    fail("python_version", f"Python {sys.version_info.major}.{sys.version_info.minor} is below 3.11")

required = json.loads(sys.argv[1])
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        fail("import", str(exc), name)

try:
    import torch
except Exception as exc:
    fail("import", str(exc), "torch")

try:
    import runtime.config
    import models.base
except Exception as exc:
    fail("project_import", str(exc))

device = sys.argv[2]
cuda_available = bool(torch.cuda.is_available())
if device == "cuda" and not cuda_available:
    fail("cuda", "CUDA was requested but torch.cuda.is_available() is false")

print(json.dumps({
    "ok": True,
    "python_version": ".".join(str(item) for item in sys.version_info[:3]),
    "python_executable": sys.executable,
    "cuda_available": cuda_available,
    "torch_version": getattr(torch, "__version__", "unknown"),
}))
'''


_MODEL_PREFLIGHT_SCRIPT = r'''
import json
import sys
from pathlib import Path


def fail(kind, message):
    print(json.dumps({"ok": False, "kind": kind, "message": message}))
    raise SystemExit(1)


project_root = Path(sys.argv[1]).resolve()
model_name = sys.argv[2]
config_path = Path(sys.argv[3]).resolve()
model_config_path = Path(sys.argv[4]).resolve()
try:
    from models.base import DataInfoView
    from models.loader import build_model, load_model_module
    from engine.model_execution import build_execution_plan
    from runtime.config import load_experiment_config, load_model_config

    config = load_experiment_config(config_path)
    model_config = load_model_config(model_config_path)
    load_model_module(model_name)
    features = tuple(config.data["feature_columns"])
    input_power = str(config.data["input_power_column"])
    info = DataInfoView(
        num_nodes=int(config.data["num_nodes"]),
        num_features=len(features),
        lookback=int(config.data["lookback"]),
        max_pred_len=int(config.data["max_pred_len"]),
        feature_columns=features,
        input_power_column=input_power,
        input_power_index=features.index(input_power) if input_power in features else -1,
        node_ids=tuple(range(1, int(config.data["num_nodes"]) + 1)),
        graph_config=dict(config.resources["graph"]),
        project_root=project_root,
    )
    model = build_model(model_name, model_config, info)
    execution = build_execution_plan(
        model,
        total_nodes=int(info.num_nodes),
        node_shared_chunk_size=int(config.runtime["node_shared_chunk_size"]),
    ).as_dict()
except Exception as exc:
    fail("model", str(exc))

print(json.dumps({
    "ok": True,
    "model": model_name,
    "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    "execution": execution,
}))
'''


def _preflight_failure(
    *,
    resolved: ResolvedEnvironment,
    model_name: str | None,
    kind: str,
    message: str,
    module: str | None = None,
) -> EnvironmentPreflightError:
    if model_name and kind == "import" and module:
        return EnvironmentPreflightError(
            f"model {model_name} requires runtime environment '{resolved.environment_id}', "
            f"but {resolved.conda_env} could not import {module}: {message}"
        )
    prefix = (
        f"environment '{resolved.environment_id}' ({resolved.conda_env}) preflight failed"
    )
    return EnvironmentPreflightError(f"{prefix}: {message}")


def _last_json_line(output: str) -> dict[str, Any] | None:
    for line in reversed(output.splitlines()):
        try:
            value = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict):
            return value
    return None


def preflight_environment(
    resolved_environment: ResolvedEnvironment,
    *,
    project_root: str | Path,
    device: str = "auto",
    model_name: str | None = None,
    model_config_path: str | Path | None = None,
    resolved_seed: int | None = None,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    """Check one environment by launching its target Python exactly once."""

    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"unsupported device for environment preflight: {device}")
    root = Path(project_root).resolve()
    executable = resolved_environment.python_executable
    if not executable.is_file():
        raise _preflight_failure(
            resolved=resolved_environment,
            model_name=model_name,
            kind="python",
            message=f"target Python does not exist: {executable}",
        )
    missing_roots = [path for path in resolved_environment.source_roots if not path.is_dir()]
    if missing_roots:
        missing = missing_roots[0]
        raise _preflight_failure(
            resolved=resolved_environment,
            model_name=model_name,
            kind="source_root",
            message=f"required source root does not exist: {missing}",
        )
    # Model validation belongs to ``preflight_model``. Keeping this operation
    # environment-only means a pure TSL worker never depends on TSLib source.
    del model_config_path
    worker_environment = build_worker_environment(
        resolved_environment,
        project_root=root,
        resolved_seed=resolved_seed,
    )
    required = list(dict.fromkeys((*resolved_environment.required_imports, "torch")))
    try:
        completed = subprocess.run(
            [
                str(executable),
                "-c",
                _PREFLIGHT_SCRIPT,
                json.dumps(required),
                device,
            ],
            cwd=root,
            env=worker_environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise _preflight_failure(
            resolved=resolved_environment,
            model_name=model_name,
            kind="startup",
            message=str(exc),
        ) from exc
    payload = _last_json_line(completed.stdout)
    if completed.returncode != 0 or not payload or not payload.get("ok"):
        payload = payload or {}
        raise _preflight_failure(
            resolved=resolved_environment,
            model_name=model_name,
            kind=str(payload.get("kind", "startup")),
            module=payload.get("module"),
            message=str(payload.get("message") or completed.stderr.strip() or "target Python failed to start"),
        )
    return {
        "environment_id": resolved_environment.environment_id,
        "conda_env": resolved_environment.conda_env,
        "python_executable": str(resolved_environment.python_executable),
        "python_version": payload.get("python_version"),
        "source_roots": [str(path) for path in resolved_environment.source_roots],
        "required_imports": list(resolved_environment.required_imports),
        "resolution_source": resolved_environment.resolution_source,
        "environment_resolution_source": resolved_environment.resolution_source,
        "cuda_available": bool(payload.get("cuda_available", False)),
        "torch_version": payload.get("torch_version"),
    }


def preflight_model(
    resolved_environment: ResolvedEnvironment,
    *,
    project_root: str | Path,
    model_name: str,
    config_path: str | Path,
    model_config_path: str | Path,
    resolved_seed: int | None = None,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    """Validate one model module and small construction in its own runtime."""

    root = Path(project_root).resolve()
    executable = resolved_environment.python_executable
    if not executable.is_file():
        raise _preflight_failure(
            resolved=resolved_environment,
            model_name=model_name,
            kind="python",
            message=f"target Python does not exist: {executable}",
        )
    worker_environment = build_worker_environment(
        resolved_environment,
        project_root=root,
        resolved_seed=resolved_seed,
    )
    try:
        completed = subprocess.run(
            [
                str(executable),
                "-c",
                _MODEL_PREFLIGHT_SCRIPT,
                str(root),
                model_name,
                str(Path(config_path).resolve()),
                str(Path(model_config_path).resolve()),
            ],
            cwd=root,
            env=worker_environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise _preflight_failure(
            resolved=resolved_environment,
            model_name=model_name,
            kind="model",
            message=str(exc),
        ) from exc
    payload = _last_json_line(completed.stdout)
    if completed.returncode != 0 or not payload or not payload.get("ok"):
        payload = payload or {}
        raise _preflight_failure(
            resolved=resolved_environment,
            model_name=model_name,
            kind="model",
            message=str(payload.get("message") or completed.stderr.strip() or "model preflight failed"),
        )
    return {
        "model": model_name,
        "runtime_environment": resolved_environment.environment_id,
        "parameter_count": payload.get("parameter_count"),
        "execution": payload.get("execution"),
    }

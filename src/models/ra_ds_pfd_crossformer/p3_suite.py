"""Strict resolver for the P3-A foundation derived from frozen R2."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from runtime.config import _MODEL_FORBIDDEN_KEYS

from .model import _validate_config
from .p3_feature_bank import (
    P3_BASE_FEATURES,
    P3_CANDIDATE_TRANSFORMS,
    validate_p3_model_config,
)
from .r0_r7_suite import resolve_r0_r7_variants


DEFAULT_SUITE_PATH = Path("configs/experiments/ra_ds_pfd_p3.yaml")
BASE_VARIANT = "R2"
FROZEN_BASE_SUITE_PATH = "configs/experiments/ra_ds_pfd_r0_r7.yaml"
P3_SUITE_FIELDS = frozenset({"suite", "model", "base", "p3"})
P3_BASE_FIELDS = frozenset({"suite_file", "variant"})


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.Node,
    deep: bool = False,
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate YAML mapping key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"RA-DS-PFD P3 suite {field} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"RA-DS-PFD P3 suite {field} keys must be strings")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str] | frozenset[str],
    *,
    field: str,
) -> None:
    unexpected = sorted(set(value) - set(expected))
    missing = sorted(set(expected) - set(value))
    if unexpected:
        raise ValueError(f"RA-DS-PFD P3 suite {field} has unsupported field: {unexpected[0]}")
    if missing:
        raise ValueError(f"RA-DS-PFD P3 suite {field} is missing field: {missing[0]}")


def _relative_file(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"RA-DS-PFD P3 suite {field} must be a non-empty project-relative path")
    path = Path(value)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise ValueError(f"RA-DS-PFD P3 suite {field} must stay within the project root")
    return value


def _find_public_parameter(value: Any, path: str = "p3") -> tuple[str, str] | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str) and key in _MODEL_FORBIDDEN_KEYS:
                return key, f"{path}.{key}"
            found = _find_public_parameter(child, f"{path}.{key}")
            if found is not None:
                return found
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found = _find_public_parameter(child, f"{path}[{index}]")
            if found is not None:
                return found
    return None


def _validate_definition(value: Any) -> dict[str, Any]:
    suite = dict(_mapping(value, field="root"))
    _exact_keys(suite, P3_SUITE_FIELDS, field="root")
    if suite["suite"] != "ra_ds_pfd_p3":
        raise ValueError("RA-DS-PFD P3 suite has the wrong suite name")
    if suite["model"] != "ra_ds_pfd_crossformer":
        raise ValueError("RA-DS-PFD P3 suite must use ra_ds_pfd_crossformer")

    base = _mapping(suite["base"], field="base")
    _exact_keys(base, P3_BASE_FIELDS, field="base")
    suite_file = _relative_file(base["suite_file"], field="base.suite_file")
    if suite_file.replace("\\", "/") != FROZEN_BASE_SUITE_PATH:
        raise ValueError(
            "RA-DS-PFD P3 suite base.suite_file must be the frozen R0-R7 suite"
        )
    if base["variant"] != BASE_VARIANT:
        raise ValueError("RA-DS-PFD P3 suite base.variant must be R2")

    p3 = _mapping(suite["p3"], field="p3")
    public_parameter = _find_public_parameter(p3)
    if public_parameter is not None:
        name, location = public_parameter
        raise ValueError(
            f"RA-DS-PFD P3 suite cannot define public experiment parameter "
            f"{name!r} at {location}"
        )
    validate_p3_model_config(p3)
    return deepcopy(suite)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"RA-DS-PFD P3 suite file does not exist: {path}")
    try:
        loaded = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid RA-DS-PFD P3 suite YAML: {path}: {exc}") from exc
    return _validate_definition(loaded)


def load_p3_suite(path: str | Path = DEFAULT_SUITE_PATH) -> dict[str, Any]:
    """Load and validate one machine-readable P3-A suite definition."""

    return _load_yaml(Path(path).resolve())


def _suite_and_root(
    suite_or_path: Mapping[str, Any] | str | Path,
    project_root: str | Path | None,
) -> tuple[dict[str, Any], Path]:
    if isinstance(suite_or_path, (str, Path)):
        suite_path = Path(suite_or_path).resolve()
        suite = load_p3_suite(suite_path)
        if (
            project_root is None
            and suite_path.parent.name == "experiments"
            and suite_path.parent.parent.name == "configs"
        ):
            root = suite_path.parents[2]
        else:
            root = Path(project_root or Path.cwd()).resolve()
    else:
        suite = _validate_definition(suite_or_path)
        root = Path(project_root or Path.cwd()).resolve()
    return suite, root


def _resolve_project_file(value: str, *, project_root: Path, field: str) -> Path:
    relative = Path(_relative_file(value, field=field))
    resolved = (project_root / relative).resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(
            f"RA-DS-PFD P3 suite {field} resolves outside the project root"
        ) from exc
    return resolved


def resolve_p3_model_config(
    suite_or_path: Mapping[str, Any] | str | Path = DEFAULT_SUITE_PATH,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve P3 by deep-copying frozen R2 and replacing only P3 fields."""

    suite, root = _suite_and_root(suite_or_path, project_root)
    base = _mapping(suite["base"], field="base")
    base_path = _resolve_project_file(
        base["suite_file"],
        project_root=root,
        field="base.suite_file",
    )
    # This is intentionally the only architecture source used by P3.
    frozen_r2 = deepcopy(
        resolve_r0_r7_variants(base_path, project_root=root)[BASE_VARIANT]
    )
    resolved = deepcopy(frozen_r2)
    resolved["pfd_mode"] = "pfd3_global_topk"
    resolved["p3"] = deepcopy(dict(_mapping(suite["p3"], field="p3")))
    _validate_config(resolved)

    for field in set(frozen_r2) | set(resolved):
        if field in {"pfd_mode", "p3"}:
            continue
        if resolved.get(field) != frozen_r2.get(field):
            raise ValueError(
                f"RA-DS-PFD P3 resolver changed frozen R2 field: {field}"
            )
    if resolved["p3"]["top_k"] != 2:
        raise ValueError("RA-DS-PFD P3 resolver requires top_k=2")
    if len(resolved["p3"]["candidate_features"]) * len(
        resolved["p3"]["candidate_transforms"]
    ) != 26:
        raise ValueError("RA-DS-PFD P3 resolver requires candidate_count=26")
    return resolved


def resolve_p3_config(
    suite_or_path: Mapping[str, Any] | str | Path = DEFAULT_SUITE_PATH,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Alias for the model-config resolver used by callers and tests."""

    return resolve_p3_model_config(suite_or_path, project_root=project_root)


def validate_p3_suite(
    suite_or_path: Mapping[str, Any] | str | Path = DEFAULT_SUITE_PATH,
    *,
    project_root: str | Path | None = None,
) -> None:
    """Validate both the P3 definition and its frozen-R2 resolution."""

    resolve_p3_model_config(suite_or_path, project_root=project_root)


__all__ = [
    "BASE_VARIANT",
    "DEFAULT_SUITE_PATH",
    "FROZEN_BASE_SUITE_PATH",
    "P3_BASE_FEATURES",
    "P3_CANDIDATE_TRANSFORMS",
    "load_p3_suite",
    "resolve_p3_config",
    "resolve_p3_model_config",
    "validate_p3_suite",
]

"""Strict IA-1 suite resolver derived from frozen R2."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from runtime.config import _MODEL_FORBIDDEN_KEYS

from .model import _validate_config
from .p3_ia_propagation import (
    IA_SELECTION_MODE,
    validate_selected_candidates,
)
from .r0_r7_suite import resolve_r0_r7_variants


DEFAULT_SUITE_PATH = Path("configs/experiments/ra_ds_pfd_p3_ia1.yaml")
BASE_VARIANT = "R2"
FROZEN_BASE_SUITE_PATH = "configs/experiments/ra_ds_pfd_r0_r7.yaml"
VARIANT_IDS = ("IA1_R2_PAIR", "IA1_AUTO_K2_PAIR")
IA1_SELECTIONS = {
    "IA1_R2_PAIR": ("Wspd.level", "Wspd.diff1"),
    "IA1_AUTO_K2_PAIR": ("Wspd.level", "Patv_clean_for_input.diff1"),
}
IA1_SUITE_FIELDS = frozenset({"suite", "model", "base", "variants"})
IA1_BASE_FIELDS = frozenset({"suite_file", "variant"})


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
        raise ValueError(f"RA-DS-PFD IA-1 suite {field} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"RA-DS-PFD IA-1 suite {field} keys must be strings")
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
        raise ValueError(f"RA-DS-PFD IA-1 suite {field} has unsupported field: {unexpected[0]}")
    if missing:
        raise ValueError(f"RA-DS-PFD IA-1 suite {field} is missing field: {missing[0]}")


def _relative_file(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"RA-DS-PFD IA-1 suite {field} must be a non-empty project-relative path")
    path = Path(value)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise ValueError(f"RA-DS-PFD IA-1 suite {field} must stay within the project root")
    return value


def _find_public_parameter(value: Any, path: str = "variants") -> tuple[str, str] | None:
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
    _exact_keys(suite, IA1_SUITE_FIELDS, field="root")
    if suite["suite"] != "ra_ds_pfd_p3_ia1":
        raise ValueError("RA-DS-PFD IA-1 suite has the wrong suite name")
    if suite["model"] != "ra_ds_pfd_crossformer":
        raise ValueError("RA-DS-PFD IA-1 suite must use ra_ds_pfd_crossformer")

    base = _mapping(suite["base"], field="base")
    _exact_keys(base, IA1_BASE_FIELDS, field="base")
    suite_file = _relative_file(base["suite_file"], field="base.suite_file")
    if suite_file.replace("\\", "/") != FROZEN_BASE_SUITE_PATH:
        raise ValueError("RA-DS-PFD IA-1 suite base.suite_file must be the frozen R0-R7 suite")
    if base["variant"] != BASE_VARIANT:
        raise ValueError("RA-DS-PFD IA-1 suite base.variant must be R2")

    variants = _mapping(suite["variants"], field="variants")
    if set(variants) != set(VARIANT_IDS):
        missing = sorted(set(VARIANT_IDS) - set(variants))
        unexpected = sorted(set(variants) - set(VARIANT_IDS))
        if missing:
            raise ValueError(f"RA-DS-PFD IA-1 suite is missing variant: {missing[0]}")
        raise ValueError(f"RA-DS-PFD IA-1 suite has unsupported variant: {unexpected[0]}")

    public_parameter = _find_public_parameter(variants)
    if public_parameter is not None:
        name, location = public_parameter
        raise ValueError(
            f"RA-DS-PFD IA-1 suite cannot define public experiment parameter "
            f"{name!r} at {location}"
        )

    for variant_id in VARIANT_IDS:
        variant = _mapping(variants[variant_id], field=f"variants.{variant_id}")
        _exact_keys(variant, {"selected_candidates"}, field=f"variants.{variant_id}")
        selected = validate_selected_candidates(variant["selected_candidates"])
        if len(selected) != 2:
            raise ValueError("RA-DS-PFD IA-1 suite requires exactly two fixed arms with K=2")
        if selected != IA1_SELECTIONS[variant_id]:
            raise ValueError(
                f"RA-DS-PFD IA-1 {variant_id} must use its canonical fixed candidate pair"
            )
    return deepcopy(suite)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"RA-DS-PFD IA-1 suite file does not exist: {path}")
    try:
        loaded = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid RA-DS-PFD IA-1 suite YAML: {path}: {exc}") from exc
    return _validate_definition(loaded)


def load_p3_ia1_suite(path: str | Path = DEFAULT_SUITE_PATH) -> dict[str, Any]:
    """Load and validate the IA-1 two-arm suite definition."""

    return _load_yaml(Path(path).resolve())


def _suite_and_root(
    suite_or_path: Mapping[str, Any] | str | Path,
    project_root: str | Path | None,
) -> tuple[dict[str, Any], Path]:
    if isinstance(suite_or_path, (str, Path)):
        suite_path = Path(suite_or_path).resolve()
        suite = load_p3_ia1_suite(suite_path)
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
        raise ValueError(f"RA-DS-PFD IA-1 suite {field} resolves outside the project root") from exc
    return resolved


def resolve_p3_ia1_variants(
    suite_or_path: Mapping[str, Any] | str | Path = DEFAULT_SUITE_PATH,
    *,
    project_root: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Resolve IA-1 by deep-copying frozen R2 and replacing only IA-1 fields."""

    suite, root = _suite_and_root(suite_or_path, project_root)
    base = _mapping(suite["base"], field="base")
    base_path = _resolve_project_file(
        base["suite_file"],
        project_root=root,
        field="base.suite_file",
    )
    frozen_r2 = deepcopy(resolve_r0_r7_variants(base_path, project_root=root)[BASE_VARIANT])
    variants = _mapping(suite["variants"], field="variants")
    resolved: dict[str, dict[str, Any]] = {}
    for variant_id in VARIANT_IDS:
        selected = list(validate_selected_candidates(variants[variant_id]["selected_candidates"]))
        config = deepcopy(frozen_r2)
        config["pfd_mode"] = "pfd3_ia_fixed"
        config["p3_ia"] = {
            "selection_mode": IA_SELECTION_MODE,
            "selected_candidates": selected,
        }
        _validate_config(config)
        resolved[variant_id] = config

    for field in set(frozen_r2) | set(resolved[VARIANT_IDS[0]]):
        if field in {"pfd_mode", "p3_ia"}:
            continue
        for variant_id in VARIANT_IDS:
            if resolved[variant_id].get(field) != frozen_r2.get(field):
                raise ValueError(f"RA-DS-PFD IA-1 resolver changed frozen R2 field: {field}")

    left = resolved[VARIANT_IDS[0]]["p3_ia"]
    right = resolved[VARIANT_IDS[1]]["p3_ia"]
    if (
        left["selection_mode"] != IA_SELECTION_MODE
        or right["selection_mode"] != IA_SELECTION_MODE
    ):
        raise ValueError("RA-DS-PFD IA-1 arms must use selection_mode=fixed")
    if left["selected_candidates"] == right["selected_candidates"]:
        raise ValueError("RA-DS-PFD IA-1 arms must differ only by selected candidate set")
    if tuple(left["selected_candidates"]) != IA1_SELECTIONS[VARIANT_IDS[0]]:
        raise ValueError("RA-DS-PFD IA-1 R2 pair drifted from its canonical candidate set")
    if tuple(right["selected_candidates"]) != IA1_SELECTIONS[VARIANT_IDS[1]]:
        raise ValueError("RA-DS-PFD IA-1 AUTO K2 pair drifted from its canonical candidate set")
    return resolved


def resolve_p3_ia1_variant(
    suite_or_path: Mapping[str, Any] | str | Path,
    variant_id: str,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    if variant_id not in VARIANT_IDS:
        raise ValueError(f"unsupported RA-DS-PFD IA-1 variant: {variant_id}")
    return deepcopy(
        resolve_p3_ia1_variants(suite_or_path, project_root=project_root)[variant_id]
    )


def validate_p3_ia1_suite(
    suite_or_path: Mapping[str, Any] | str | Path = DEFAULT_SUITE_PATH,
    *,
    project_root: str | Path | None = None,
) -> None:
    """Fail closed unless the IA-1 suite and its frozen-R2 resolution are valid."""

    resolve_p3_ia1_variants(suite_or_path, project_root=project_root)


__all__ = [
    "BASE_VARIANT",
    "DEFAULT_SUITE_PATH",
    "FROZEN_BASE_SUITE_PATH",
    "IA1_SELECTIONS",
    "VARIANT_IDS",
    "load_p3_ia1_suite",
    "resolve_p3_ia1_variant",
    "resolve_p3_ia1_variants",
    "validate_p3_ia1_suite",
]

"""Strict P3-B1 resolver derived from the canonical P3 suite.

P3-B1 deliberately has one experiment degree of freedom: the temporal
operator basis used by the candidate bank.  All selector settings and all
frozen R2/P3 architecture fields are inherited from the canonical P3 suite.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .p3_feature_bank import P3_CANDIDATE_TRANSFORMS, validate_p3_model_config
from .p3_suite import (
    DEFAULT_SUITE_PATH as CANONICAL_P3_SUITE_PATH,
    resolve_p3_model_config,
)
from .model import _validate_config


DEFAULT_SUITE_PATH = Path("configs/experiments/ra_ds_pfd_p3_b1.yaml")
FROZEN_BASE_SUITE_PATH = "configs/experiments/ra_ds_pfd_p3.yaml"
MODEL_NAME = "ra_ds_pfd_crossformer"
VARIANT_IDS = ("B1_LD", "B1_L")
B1_VARIANT_IDS = VARIANT_IDS
B1_BASE_VARIANT = "canonical_p3"

P3_B1_SUITE_FIELDS = frozenset({"suite", "model", "base", "variants"})
P3_B1_BASE_FIELDS = frozenset({"suite_file"})
P3_B1_VARIANT_FIELDS = frozenset({"candidate_transforms"})
_EXPECTED_TRANSFORMS = {
    "B1_LD": ("level", "diff1"),
    "B1_L": ("level",),
}


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
        raise ValueError(f"RA-DS-PFD P3-B1 suite {field} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"RA-DS-PFD P3-B1 suite {field} keys must be strings")
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
        raise ValueError(
            f"RA-DS-PFD P3-B1 suite {field} has unsupported field: {unexpected[0]}"
        )
    if missing:
        raise ValueError(
            f"RA-DS-PFD P3-B1 suite {field} is missing field: {missing[0]}"
        )


def _relative_file(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"RA-DS-PFD P3-B1 suite {field} must be a non-empty project-relative path"
        )
    path = Path(value)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise ValueError(
            f"RA-DS-PFD P3-B1 suite {field} must stay within the project root"
        )
    return value


def _validate_transforms(value: Any, *, variant_id: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(
            f"RA-DS-PFD P3-B1 {variant_id}.candidate_transforms must be an ordered list"
        )
    transforms = tuple(value)
    if not transforms or any(not isinstance(item, str) or not item for item in transforms):
        raise ValueError(
            f"RA-DS-PFD P3-B1 {variant_id}.candidate_transforms must contain strings"
        )
    if len(set(transforms)) != len(transforms):
        raise ValueError(
            f"RA-DS-PFD P3-B1 {variant_id}.candidate_transforms contains a duplicate operator"
        )
    unknown = sorted(set(transforms) - set(P3_CANDIDATE_TRANSFORMS))
    if unknown:
        raise ValueError(
            f"RA-DS-PFD P3-B1 {variant_id}.candidate_transforms contains unknown operator: {unknown[0]}"
        )
    expected = _EXPECTED_TRANSFORMS[variant_id]
    if transforms != expected:
        raise ValueError(
            f"RA-DS-PFD P3-B1 {variant_id} must use the frozen operator basis "
            f"{list(expected)!r}"
        )
    return transforms


def _validate_definition(value: Any) -> dict[str, Any]:
    suite = dict(_mapping(value, field="root"))
    _exact_keys(suite, P3_B1_SUITE_FIELDS, field="root")
    if suite["suite"] != "ra_ds_pfd_p3_b1":
        raise ValueError("RA-DS-PFD P3-B1 suite has the wrong suite name")
    if suite["model"] != MODEL_NAME:
        raise ValueError(
            "RA-DS-PFD P3-B1 suite must use ra_ds_pfd_crossformer"
        )

    base = _mapping(suite["base"], field="base")
    _exact_keys(base, P3_B1_BASE_FIELDS, field="base")
    suite_file = _relative_file(base["suite_file"], field="base.suite_file")
    if suite_file.replace("\\", "/") != FROZEN_BASE_SUITE_PATH:
        raise ValueError(
            "RA-DS-PFD P3-B1 suite base.suite_file must be the canonical P3 suite"
        )

    variants = _mapping(suite["variants"], field="variants")
    if set(variants) != set(VARIANT_IDS):
        missing = sorted(set(VARIANT_IDS) - set(variants))
        unexpected = sorted(set(variants) - set(VARIANT_IDS))
        if missing:
            raise ValueError(f"RA-DS-PFD P3-B1 suite is missing variant: {missing[0]}")
        raise ValueError(
            f"RA-DS-PFD P3-B1 suite has unsupported variant: {unexpected[0]}"
        )
    for variant_id in VARIANT_IDS:
        variant = _mapping(variants[variant_id], field=f"variants.{variant_id}")
        _exact_keys(
            variant,
            P3_B1_VARIANT_FIELDS,
            field=f"variants.{variant_id}",
        )
        _validate_transforms(
            variant["candidate_transforms"],
            variant_id=variant_id,
        )
    return deepcopy(suite)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"RA-DS-PFD P3-B1 suite file does not exist: {path}")
    try:
        loaded = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid RA-DS-PFD P3-B1 suite YAML: {path}: {exc}") from exc
    return _validate_definition(loaded)


def load_p3_b1_suite(path: str | Path = DEFAULT_SUITE_PATH) -> dict[str, Any]:
    """Load and validate the machine-readable P3-B1 arm definition."""

    return _load_yaml(Path(path).resolve())


def _suite_and_root(
    suite_or_path: Mapping[str, Any] | str | Path,
    project_root: str | Path | None,
) -> tuple[dict[str, Any], Path]:
    if isinstance(suite_or_path, (str, Path)):
        suite_path = Path(suite_or_path).resolve()
        suite = load_p3_b1_suite(suite_path)
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
            f"RA-DS-PFD P3-B1 suite {field} resolves outside the project root"
        ) from exc
    return resolved


def _difference_paths(
    left: Any,
    right: Any,
    prefix: tuple[str, ...] = (),
) -> set[tuple[str, ...]]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        differences: set[tuple[str, ...]] = set()
        for key in set(left) | set(right):
            if key not in left or key not in right:
                differences.add((*prefix, str(key)))
            else:
                differences.update(_difference_paths(left[key], right[key], (*prefix, str(key))))
        return differences
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return {prefix}
        differences = set()
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            differences.update(_difference_paths(left_item, right_item, (*prefix, str(index))))
        return differences
    return set() if left == right else {prefix}


def _validate_isolation(
    canonical: Mapping[str, Any],
    resolved: Mapping[str, Any],
    *,
    variant_id: str,
) -> None:
    differences = _difference_paths(canonical, resolved)
    allowed = (
        {("p3", "candidate_transforms", str(index)) for index in range(2)}
        if variant_id == "B1_LD"
        else {("p3", "candidate_transforms", "0")}
    )
    # List-length changes are represented by the list field itself; permit the
    # exact transform field for B1-L and no field at all for B1-LD.
    if variant_id == "B1_L":
        allowed.add(("p3", "candidate_transforms"))
    if variant_id == "B1_LD":
        allowed = set()
    unexpected = differences - allowed
    if unexpected:
        field = ".".join(sorted(unexpected)[0])
        raise ValueError(
            f"RA-DS-PFD P3-B1 {variant_id} changed a frozen field: {field}"
        )


def resolve_p3_b1_variants(
    suite_or_path: Mapping[str, Any] | str | Path = DEFAULT_SUITE_PATH,
    *,
    project_root: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Resolve both B1 arms from canonical P3 with strict field isolation."""

    suite, root = _suite_and_root(suite_or_path, project_root)
    base = _mapping(suite["base"], field="base")
    canonical_path = _resolve_project_file(
        base["suite_file"],
        project_root=root,
        field="base.suite_file",
    )
    canonical = deepcopy(
        resolve_p3_model_config(canonical_path, project_root=root)
    )
    variants = _mapping(suite["variants"], field="variants")
    resolved: dict[str, dict[str, Any]] = {}
    for variant_id in VARIANT_IDS:
        config = deepcopy(canonical)
        config["p3"]["candidate_transforms"] = list(
            _validate_transforms(
                variants[variant_id]["candidate_transforms"],
                variant_id=variant_id,
            )
        )
        validate_p3_model_config(config["p3"])
        _validate_config(config)
        _validate_isolation(canonical, config, variant_id=variant_id)
        resolved[variant_id] = config
    if resolved["B1_LD"] != canonical:
        raise ValueError("RA-DS-PFD P3-B1 B1_LD drifted from canonical P3")
    return resolved


def resolve_p3_b1_variant(
    suite_or_path: Mapping[str, Any] | str | Path,
    variant_id: str,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve one exact P3-B1 arm."""

    if variant_id not in VARIANT_IDS:
        raise ValueError(f"unsupported RA-DS-PFD P3-B1 variant: {variant_id}")
    return deepcopy(
        resolve_p3_b1_variants(suite_or_path, project_root=project_root)[variant_id]
    )


def validate_p3_b1_suite(
    suite_or_path: Mapping[str, Any] | str | Path = DEFAULT_SUITE_PATH,
    *,
    project_root: str | Path | None = None,
) -> None:
    """Fail closed unless both B1 arms preserve canonical P3 architecture."""

    resolve_p3_b1_variants(suite_or_path, project_root=project_root)


# Short aliases follow the existing P3/R0-R7 resolver vocabulary.
resolve_b1_variants = resolve_p3_b1_variants
resolve_b1_variant = resolve_p3_b1_variant
load_b1_suite = load_p3_b1_suite


__all__ = [
    "B1_BASE_VARIANT",
    "B1_VARIANT_IDS",
    "DEFAULT_SUITE_PATH",
    "FROZEN_BASE_SUITE_PATH",
    "MODEL_NAME",
    "VARIANT_IDS",
    "load_b1_suite",
    "load_p3_b1_suite",
    "resolve_b1_variant",
    "resolve_b1_variants",
    "resolve_p3_b1_variant",
    "resolve_p3_b1_variants",
    "validate_p3_b1_suite",
]

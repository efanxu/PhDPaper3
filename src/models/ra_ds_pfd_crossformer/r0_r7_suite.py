"""Strict resolver for the frozen RA-DS-PFD R0-R7 structure matrix."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from math import isfinite
from pathlib import Path
from typing import Any

import yaml

from runtime.config import load_model_config

from .model import (
    BIAS_SCALING_MODES,
    CANONICAL_CONFIG_FIELDS,
    PROPAGATION_ENCODER_MODES,
    SPATIAL_QUERY_MODES,
    TURBINE_EMBEDDING_MODES,
    _validate_config,
)


DEFAULT_SUITE_PATH = Path("configs/experiments/ra_ds_pfd_r0_r7.yaml")
VARIANT_IDS = ("R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7")
_AXIS_FIELDS = (
    "spatial_query_mode",
    "propagation_encoder_mode",
    "turbine_embedding_mode",
    "bias_scaling_mode",
)
_P2_COMMON_FIELDS = frozenset(
    {
        "pfd_mode",
        "spatial_heads",
        "spatial_d_ff",
        "relation_dim",
        "base_turbine_dim",
        "spatial_dropout",
        "gamma_init",
        "relation_resource",
        "spatial_edge_chunk_size",
    }
)
_SUITE_FIELDS = frozenset(
    {"suite", "model", "base_model_config", "p2_common", "variants"}
)


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that does not silently overwrite duplicate keys."""


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
        raise ValueError(f"RA-DS-PFD R0-R7 suite {field} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"RA-DS-PFD R0-R7 suite {field} keys must be strings")
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
            f"RA-DS-PFD R0-R7 suite {field} has unsupported field: {unexpected[0]}"
        )
    if missing:
        raise ValueError(
            f"RA-DS-PFD R0-R7 suite {field} is missing field: {missing[0]}"
        )


def _relative_file(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"RA-DS-PFD R0-R7 suite {field} must be a non-empty project-relative path"
        )
    path = Path(value)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise ValueError(
            f"RA-DS-PFD R0-R7 suite {field} must stay within the project root"
        )
    return value


def _validate_p2_common(value: Any) -> None:
    common = _mapping(value, field="p2_common")
    _exact_keys(common, _P2_COMMON_FIELDS, field="p2_common")
    if common["pfd_mode"] != "pfd0":
        raise ValueError("RA-DS-PFD R0-R7 suite requires pfd_mode=pfd0")
    for field in ("spatial_heads", "spatial_d_ff", "relation_dim", "base_turbine_dim"):
        number = common[field]
        if not isinstance(number, int) or isinstance(number, bool) or number < 1:
            raise ValueError(
                f"RA-DS-PFD R0-R7 suite {field} must be a positive integer"
            )
    spatial_dropout = common["spatial_dropout"]
    if (
        isinstance(spatial_dropout, bool)
        or not isinstance(spatial_dropout, (int, float))
        or not isfinite(float(spatial_dropout))
        or not 0.0 <= float(spatial_dropout) < 1.0
    ):
        raise ValueError(
            "RA-DS-PFD R0-R7 suite spatial_dropout must be finite and in [0, 1)"
        )
    gamma_init = common["gamma_init"]
    if (
        isinstance(gamma_init, bool)
        or not isinstance(gamma_init, (int, float))
        or not isfinite(float(gamma_init))
        or abs(float(gamma_init) - 0.1) > 1e-12
    ):
        raise ValueError("RA-DS-PFD R0-R7 suite requires gamma_init=0.1")
    edge_chunk = common["spatial_edge_chunk_size"]
    if edge_chunk is not None and (
        not isinstance(edge_chunk, int) or isinstance(edge_chunk, bool) or edge_chunk < 1
    ):
        raise ValueError(
            "RA-DS-PFD R0-R7 suite spatial_edge_chunk_size must be positive or null"
        )
    resource = _mapping(common["relation_resource"], field="p2_common.relation_resource")
    _exact_keys(resource, {"file"}, field="p2_common.relation_resource")
    _relative_file(resource["file"], field="p2_common.relation_resource.file")


def _validate_definition(value: Any) -> dict[str, Any]:
    suite = dict(_mapping(value, field="root"))
    _exact_keys(suite, _SUITE_FIELDS, field="root")
    if suite["suite"] != "ra_ds_pfd_r0_r7":
        raise ValueError("RA-DS-PFD R0-R7 suite has the wrong suite name")
    if suite["model"] != "ra_ds_pfd_crossformer":
        raise ValueError("RA-DS-PFD R0-R7 suite must use ra_ds_pfd_crossformer")

    base_model_config = _mapping(suite["base_model_config"], field="base_model_config")
    _exact_keys(base_model_config, {"file"}, field="base_model_config")
    _relative_file(base_model_config["file"], field="base_model_config.file")
    _validate_p2_common(suite["p2_common"])

    variants = _mapping(suite["variants"], field="variants")
    if set(variants) != set(VARIANT_IDS):
        missing = sorted(set(VARIANT_IDS) - set(variants))
        unexpected = sorted(set(variants) - set(VARIANT_IDS))
        if missing:
            raise ValueError(f"RA-DS-PFD R0-R7 suite is missing variant: {missing[0]}")
        raise ValueError(f"RA-DS-PFD R0-R7 suite has unsupported variant: {unexpected[0]}")

    for variant_id in VARIANT_IDS:
        variant = _mapping(variants[variant_id], field=f"variants.{variant_id}")
        if variant_id == "R0":
            _exact_keys(variant, {"spatial_disabled"}, field=f"variants.{variant_id}")
        else:
            _exact_keys(
                variant,
                {"spatial_disabled", *_AXIS_FIELDS},
                field=f"variants.{variant_id}",
            )
        if not isinstance(variant["spatial_disabled"], bool):
            raise ValueError(
                f"RA-DS-PFD R0-R7 suite {variant_id}.spatial_disabled must be boolean"
            )
        if variant_id == "R0":
            if variant["spatial_disabled"] is not True:
                raise ValueError(
                    "RA-DS-PFD R0-R7 suite R0.spatial_disabled must be true"
                )
            continue
        if variant["spatial_disabled"] is not False:
            raise ValueError(
                f"RA-DS-PFD R0-R7 suite {variant_id}.spatial_disabled must be false"
            )
        for field, allowed in (
            ("spatial_query_mode", SPATIAL_QUERY_MODES),
            ("propagation_encoder_mode", PROPAGATION_ENCODER_MODES),
            ("turbine_embedding_mode", TURBINE_EMBEDDING_MODES),
            ("bias_scaling_mode", BIAS_SCALING_MODES),
        ):
            if not isinstance(variant[field], str) or variant[field] not in allowed:
                raise ValueError(
                    f"RA-DS-PFD R0-R7 suite {variant_id}.{field} has an unsupported value"
                )
    return deepcopy(suite)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"RA-DS-PFD R0-R7 suite file does not exist: {path}")
    try:
        loaded = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid RA-DS-PFD R0-R7 suite YAML: {path}: {exc}") from exc
    return _validate_definition(loaded)


def load_r0_r7_suite(path: str | Path = DEFAULT_SUITE_PATH) -> dict[str, Any]:
    """Load and validate one machine-readable R0-R7 suite definition."""

    return _load_yaml(Path(path).resolve())


def _project_root_for(
    suite_or_path: Mapping[str, Any] | str | Path,
    project_root: str | Path | None,
) -> tuple[dict[str, Any], Path]:
    if isinstance(suite_or_path, (str, Path)):
        suite_path = Path(suite_or_path).resolve()
        suite = load_r0_r7_suite(suite_path)
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
            f"RA-DS-PFD R0-R7 suite {field} resolves outside the project root"
        ) from exc
    return resolved


def _load_canonical_model_config(
    suite: Mapping[str, Any],
    *,
    project_root: Path,
) -> dict[str, Any]:
    base_document = _mapping(suite["base_model_config"], field="base_model_config")
    path = _resolve_project_file(
        base_document["file"],
        project_root=project_root,
        field="base_model_config.file",
    )
    base = load_model_config(path)
    if set(base) != set(CANONICAL_CONFIG_FIELDS):
        raise ValueError(
            "RA-DS-PFD R0-R7 R0 requires the base model config to contain only canonical P1 fields"
        )
    if base.get("spatial_disabled") is not True:
        raise ValueError("RA-DS-PFD R0-R7 base model config must have spatial_disabled=true")
    _validate_config(base)
    return deepcopy(base)


def _different_fields(left: Mapping[str, Any], right: Mapping[str, Any]) -> set[str]:
    return {
        field
        for field in set(left) | set(right)
        if left.get(field) != right.get(field)
    }


def _validate_resolved_matrix(
    suite: Mapping[str, Any],
    resolved: Mapping[str, Mapping[str, Any]],
    canonical: Mapping[str, Any],
) -> None:
    if set(resolved) != set(VARIANT_IDS):
        raise ValueError("RA-DS-PFD R0-R7 resolver did not produce the exact variant set")
    if resolved["R0"] != canonical or set(resolved["R0"]) != set(CANONICAL_CONFIG_FIELDS):
        raise ValueError("RA-DS-PFD R0 resolved config is not the canonical P1 identity")
    if resolved["R0"].get("spatial_disabled") is not True:
        raise ValueError("RA-DS-PFD R0 resolved config must have spatial_disabled=true")

    p2_common = _mapping(suite["p2_common"], field="p2_common")
    for variant_id in VARIANT_IDS[1:]:
        config = resolved[variant_id]
        if config.get("spatial_disabled") is not False:
            raise ValueError(
                f"RA-DS-PFD {variant_id} resolved config must have spatial_disabled=false"
            )
        for field, value in p2_common.items():
            if config.get(field) != value:
                raise ValueError(
                    f"RA-DS-PFD {variant_id} does not preserve shared P2 field: {field}"
                )

    baseline = resolved["R1"]
    axis_domains = {
        "spatial_query_mode": SPATIAL_QUERY_MODES,
        "propagation_encoder_mode": PROPAGATION_ENCODER_MODES,
        "turbine_embedding_mode": TURBINE_EMBEDDING_MODES,
        "bias_scaling_mode": BIAS_SCALING_MODES,
    }
    for variant_id in VARIANT_IDS[1:]:
        config = resolved[variant_id]
        for field, allowed in axis_domains.items():
            if baseline[field] not in allowed or config[field] not in allowed:
                raise ValueError(
                    f"RA-DS-PFD {variant_id}.{field} is outside the formal architecture domain"
                )

    expected_differences = {
        "R2": set(_AXIS_FIELDS),
        "R3": {"spatial_query_mode"},
        "R4": {"propagation_encoder_mode"},
        "R5": {"turbine_embedding_mode"},
        "R6": {"bias_scaling_mode"},
        "R7": {"spatial_query_mode", "propagation_encoder_mode"},
    }
    for variant_id, expected in expected_differences.items():
        actual = _different_fields(resolved[variant_id], baseline)
        if actual != expected:
            raise ValueError(
                f"RA-DS-PFD {variant_id} bridge drifted relative to R1: "
                f"expected {sorted(expected)}, got {sorted(actual)}"
            )

    if resolved["R7"]["spatial_query_mode"] != resolved["R3"]["spatial_query_mode"]:
        raise ValueError("RA-DS-PFD R7 must use the R3 spatial query endpoint")
    if (
        resolved["R7"]["propagation_encoder_mode"]
        != resolved["R4"]["propagation_encoder_mode"]
    ):
        raise ValueError("RA-DS-PFD R7 must use the R4 propagation endpoint")
    for field in ("turbine_embedding_mode", "bias_scaling_mode"):
        if resolved["R7"][field] != baseline[field]:
            raise ValueError(f"RA-DS-PFD R7 must preserve the R1 {field} endpoint")


def resolve_r0_r7_variants(
    suite_or_path: Mapping[str, Any] | str | Path = DEFAULT_SUITE_PATH,
    *,
    project_root: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Resolve every R0-R7 variant to a model-only config mapping."""

    suite, root = _project_root_for(suite_or_path, project_root)
    canonical = _load_canonical_model_config(suite, project_root=root)
    common = dict(_mapping(suite["p2_common"], field="p2_common"))
    variants = _mapping(suite["variants"], field="variants")
    resolved: dict[str, dict[str, Any]] = {}
    for variant_id in VARIANT_IDS:
        if variant_id == "R0":
            config = deepcopy(canonical)
        else:
            config = {
                **deepcopy(canonical),
                **deepcopy(common),
                **deepcopy(dict(variants[variant_id])),
            }
        _validate_config(config)
        resolved[variant_id] = config
    _validate_resolved_matrix(suite, resolved, canonical)
    return resolved


def resolve_r0_r7_variant(
    suite_or_path: Mapping[str, Any] | str | Path,
    variant_id: str,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve one exact R0-R7 variant to a model-only config mapping."""

    if variant_id not in VARIANT_IDS:
        raise ValueError(f"unsupported RA-DS-PFD R0-R7 variant: {variant_id}")
    return deepcopy(resolve_r0_r7_variants(suite_or_path, project_root=project_root)[variant_id])


def validate_r0_r7_suite(
    suite_or_path: Mapping[str, Any] | str | Path = DEFAULT_SUITE_PATH,
    *,
    project_root: str | Path | None = None,
) -> None:
    """Fail closed unless the suite and resolved R0-R7 matrix are frozen."""

    resolve_r0_r7_variants(suite_or_path, project_root=project_root)


__all__ = [
    "DEFAULT_SUITE_PATH",
    "VARIANT_IDS",
    "load_r0_r7_suite",
    "resolve_r0_r7_variant",
    "resolve_r0_r7_variants",
    "validate_r0_r7_suite",
]

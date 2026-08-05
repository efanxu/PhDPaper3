"""Read-only, prebuilt TrueUnion relation resources for RA-DS-PFD.

The model never constructs a relation graph from data.  A relation resource is
an immutable ``.npz`` artifact containing only node/edge metadata and the
stable static edge features required by the spatial attention module.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


RELATION_RESOURCE_SCHEMA_VERSION = 1
STATIC_EDGE_FEATURE_NAMES: tuple[str, ...] = (
    "semantic_similarity",
    "semantic_overlap_ratio",
    "distance_kernel_weight",
    "normalized_distance",
    "relative_x",
    "relative_y",
    "delta_elevation",
    "terrain_slope",
    "terrain_slope_angle",
    "is_semantic_edge",
    "is_distance_edge",
    "is_both_edge",
    "has_elevation",
)

_ARTIFACT_KEYS = frozenset(
    {
        "schema_version",
        "node_ids",
        "edge_index",
        "edge_static_features",
        "edge_feature_names",
    }
)
_CONFIG_KEYS = frozenset({"file"})


@dataclass(frozen=True)
class RelationResource:
    """Validated relation tensors and their source artifact path."""

    schema_version: int
    node_ids: tuple[int, ...]
    edge_index: torch.Tensor
    edge_static_features: torch.Tensor
    edge_feature_names: tuple[str, ...]
    path: Path

    @property
    def static_edge_features(self) -> torch.Tensor:
        """Compatibility alias using the shorter name used by attention code."""

        return self.edge_static_features

    @property
    def edge_count(self) -> int:
        return int(self.edge_index.shape[1])


def _scalar_int(value: Any, *, name: str) -> int:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "iu":
        raise ValueError(f"relation artifact {name} must be one scalar integer")
    return int(array.item())


def _validate_resource_config(resource_config: Mapping[str, Any]) -> str:
    if not isinstance(resource_config, Mapping):
        raise ValueError("relation_resource must be a mapping")
    unknown = sorted(set(resource_config) - _CONFIG_KEYS)
    missing = sorted(_CONFIG_KEYS - set(resource_config))
    if unknown:
        raise ValueError(f"relation_resource has unknown field: {unknown[0]}")
    if missing:
        raise ValueError(f"relation_resource is missing field: {missing[0]}")
    configured_file = resource_config["file"]
    if not isinstance(configured_file, str) or not configured_file.strip():
        raise ValueError("relation_resource.file must be a non-empty project-relative path")
    relative = Path(configured_file)
    if relative.is_absolute() or any(part == ".." for part in relative.parts):
        raise ValueError("relation_resource.file must be a project-relative path within the project root")
    return configured_file


def _resolve_resource_path(configured_file: str, project_root: str | Path) -> Path:
    root = Path(project_root).resolve()
    relative = Path(configured_file)
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("relation_resource.file resolves outside the project root") from exc
    return resolved


def _validate_artifact_arrays(
    arrays: Mapping[str, Any],
    *,
    expected_node_ids: tuple[int, ...],
) -> tuple[int, tuple[int, ...], np.ndarray, np.ndarray, tuple[str, ...]]:
    actual_keys = set(arrays)
    if actual_keys != _ARTIFACT_KEYS:
        unexpected = sorted(actual_keys - _ARTIFACT_KEYS)
        missing = sorted(_ARTIFACT_KEYS - actual_keys)
        if unexpected:
            raise ValueError(f"relation artifact contains unsupported field: {unexpected[0]}")
        raise ValueError(f"relation artifact is missing field: {missing[0]}")

    schema_version = _scalar_int(arrays["schema_version"], name="schema_version")
    if schema_version != RELATION_RESOURCE_SCHEMA_VERSION:
        raise ValueError(
            "relation artifact schema_version does not match the supported version: "
            f"{schema_version}"
        )

    node_ids_array = np.asarray(arrays["node_ids"])
    if node_ids_array.ndim != 1 or node_ids_array.dtype != np.int64:
        raise ValueError("relation artifact node_ids must have shape [N] and dtype int64")
    if len(np.unique(node_ids_array)) != len(node_ids_array):
        raise ValueError("relation artifact node_ids contain duplicates")
    node_ids = tuple(int(value) for value in node_ids_array.tolist())
    if node_ids != expected_node_ids:
        raise ValueError("relation artifact node_ids do not match the public data node order")

    edge_index = np.asarray(arrays["edge_index"])
    if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.dtype != np.int64:
        raise ValueError("relation artifact edge_index must have shape [2,E] and dtype int64")
    if edge_index.shape[1] == 0:
        raise ValueError("relation artifact must contain at least one edge")
    source, target = edge_index
    num_nodes = len(node_ids)
    if int(source.min()) < 0 or int(source.max()) >= num_nodes:
        raise ValueError("relation artifact edge_index source is outside the node range")
    if int(target.min()) < 0 or int(target.max()) >= num_nodes:
        raise ValueError("relation artifact edge_index target is outside the node range")
    if np.any(source == target):
        raise ValueError("relation artifact contains a self-loop")
    pairs = list(zip(target.tolist(), source.tolist()))
    if len(set(pairs)) != len(pairs):
        raise ValueError("relation artifact contains duplicate edges")
    if pairs != sorted(pairs):
        raise ValueError("relation artifact edges must be sorted by (target, source)")
    indegree = np.bincount(target, minlength=num_nodes)
    if np.any(indegree == 0):
        missing_target = int(np.flatnonzero(indegree == 0)[0])
        raise ValueError(f"relation artifact target {missing_target} has zero indegree")

    edge_static_features = np.asarray(arrays["edge_static_features"])
    if (
        edge_static_features.ndim != 2
        or edge_static_features.shape != (edge_index.shape[1], len(STATIC_EDGE_FEATURE_NAMES))
        or edge_static_features.dtype != np.float32
    ):
        raise ValueError("relation artifact edge_static_features must have shape [E,13] and dtype float32")
    if not np.isfinite(edge_static_features).all():
        raise ValueError("relation artifact edge_static_features contain non-finite values")

    names_array = np.asarray(arrays["edge_feature_names"])
    if names_array.ndim != 1:
        raise ValueError("relation artifact edge_feature_names must be a one-dimensional string array")
    if names_array.dtype.kind not in "SU":
        raise ValueError("relation artifact edge_feature_names must be a string array")
    edge_feature_names = tuple(str(value) for value in names_array.tolist())
    if edge_feature_names != STATIC_EDGE_FEATURE_NAMES:
        raise ValueError("relation artifact edge_feature_names do not match the stable 13-name schema")

    return schema_version, node_ids, edge_index, edge_static_features, edge_feature_names


def load_relation_resource(
    resource_config: Mapping[str, Any],
    *,
    project_root: str | Path,
    node_ids: Iterable[int],
) -> RelationResource:
    """Load and structurally validate a project-relative relation artifact."""

    configured_file = _validate_resource_config(resource_config)
    expected_node_ids = tuple(int(value) for value in node_ids)
    if not expected_node_ids:
        raise ValueError("relation resource validation requires public data node_ids")
    path = _resolve_resource_path(configured_file, project_root)
    if not path.is_file():
        raise FileNotFoundError(f"relation resource file does not exist: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
    except ValueError as exc:
        raise ValueError(f"relation resource is not a valid allow_pickle=False NPZ: {path}") from exc
    schema_version, actual_node_ids, edge_index, edge_static_features, names = _validate_artifact_arrays(
        arrays,
        expected_node_ids=expected_node_ids,
    )
    return RelationResource(
        schema_version=schema_version,
        node_ids=actual_node_ids,
        edge_index=torch.from_numpy(edge_index.copy()).long(),
        edge_static_features=torch.from_numpy(edge_static_features.copy()).float(),
        edge_feature_names=names,
        path=path,
    )


__all__ = [
    "RELATION_RESOURCE_SCHEMA_VERSION",
    "STATIC_EDGE_FEATURE_NAMES",
    "RelationResource",
    "load_relation_resource",
]

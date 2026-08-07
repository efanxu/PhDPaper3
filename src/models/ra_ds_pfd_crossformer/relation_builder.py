"""Offline construction and migration of the RA-DS-PFD TrueUnion resource.

This module is deliberately separate from the model loader.  It owns the
train-only Wspd semantic graph, the target-directed distance KNN graph, their
13-field union, and the small migration path for the old prototype artifact.
The model consumes only the five arrays written by :func:`write_relation_artifact`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from runtime.config import load_experiment_config
from data.split import chronological_split


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


@dataclass(frozen=True)
class TrueUnionGraph:
    """In-memory representation of the loader-compatible resource."""

    node_ids: np.ndarray
    edge_index: np.ndarray
    edge_static_features: np.ndarray
    edge_feature_names: tuple[str, ...] = STATIC_EDGE_FEATURE_NAMES
    semantic_edge_count: int = 0
    distance_edge_count: int = 0
    both_edge_count: int = 0
    raw_edge_features: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def edge_count(self) -> int:
        return int(self.edge_index.shape[1])


class RelationBuildError(ValueError):
    """Raised when a graph cannot be built or safely migrated."""


class RelationConsistencyError(RelationBuildError):
    """Raised when a rebuilt graph does not match an old graph exactly enough."""

    def __init__(self, report: Mapping[str, Any]) -> None:
        self.report = dict(report)
        super().__init__(json.dumps(self.report, ensure_ascii=False, default=str, indent=2))


def _as_int64_node_ids(node_ids: Iterable[int], *, name: str = "node_ids") -> np.ndarray:
    values = np.asarray(list(node_ids), dtype=np.int64)
    if values.ndim != 1 or values.size == 0:
        raise RelationBuildError(f"{name} must be a non-empty one-dimensional sequence")
    if np.unique(values).size != values.size:
        raise RelationBuildError(f"{name} contains duplicate turbine IDs")
    return values


def _as_datetime_index(timestamps: Iterable[Any]) -> pd.DatetimeIndex:
    result = pd.DatetimeIndex(pd.to_datetime(list(timestamps)))
    if result.empty:
        raise RelationBuildError("timestamps must be non-empty")
    if result.has_duplicates or not result.is_monotonic_increasing:
        raise RelationBuildError("timestamps must be strictly increasing")
    if len(result) > 1 and not np.all(np.diff(result.asi8) > 0):
        raise RelationBuildError("timestamps must be strictly increasing")
    return result


def _validate_10_minute_axis(timestamps: pd.DatetimeIndex) -> None:
    if len(timestamps) < 2:
        raise RelationBuildError("the train graph needs at least two timestamps")
    expected = pd.date_range(timestamps[0], timestamps[-1], freq="10min")
    if not expected.equals(timestamps):
        raise RelationBuildError("train timestamps are not an exact unified 10-minute time axis")


def build_train_wspd_matrix(
    raw: pd.DataFrame,
    node_ids: Iterable[int],
    train_timestamps: Iterable[Any],
    *,
    timestamp_col: str = "Tmstamp",
    turbine_id_col: str = "TurbID",
    wspd_col: str = "Wspd",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Create ``[T_train, N]`` Wspd without filling or interpolating values.

    Duplicate timestamp/turbine rows are reduced using the mean of finite
    values only.  A group with no finite values remains NaN.
    """

    required = (timestamp_col, turbine_id_col, wspd_col)
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise RelationBuildError(f"raw data is missing semantic graph column: {missing[0]}")
    ids = _as_int64_node_ids(node_ids)
    timestamps = _as_datetime_index(train_timestamps)
    _validate_10_minute_axis(timestamps)

    frame = raw.loc[:, list(required)].copy()
    frame[timestamp_col] = pd.to_datetime(frame[timestamp_col])
    frame[turbine_id_col] = pd.to_numeric(frame[turbine_id_col], errors="raise").astype(np.int64)
    frame[wspd_col] = pd.to_numeric(frame[wspd_col], errors="coerce")
    frame = frame[frame[timestamp_col].isin(timestamps)]
    duplicate_mask = frame.duplicated([timestamp_col, turbine_id_col], keep=False)
    duplicate_row_count = int(duplicate_mask.sum())

    # Treat +/-inf as missing before the finite-only duplicate reduction.
    frame.loc[~np.isfinite(frame[wspd_col].to_numpy(dtype=np.float64)), wspd_col] = np.nan
    grouped = (
        frame.groupby([timestamp_col, turbine_id_col], sort=True, observed=True)[wspd_col]
        .mean()
        .unstack(turbine_id_col)
    )
    wide = grouped.reindex(index=timestamps, columns=ids)
    matrix = wide.to_numpy(dtype=np.float64)
    return matrix, {
        "duplicate_row_count": duplicate_row_count,
        "duplicate_policy": "finite_mean_per_timestamp_turbine",
        "missing_values_preserved": int(np.isnan(matrix).sum()),
        "forward_fill": False,
        "backward_fill": False,
        "interpolation": False,
        "sampling_interval_minutes": 10,
    }


def raw_delta_wspd(wspd: np.ndarray, timestamps: Iterable[Any]) -> np.ndarray:
    """Difference only adjacent finite Wspd values on a 10-minute axis."""

    values = np.asarray(wspd, dtype=np.float64)
    if values.ndim != 2:
        raise RelationBuildError("Wspd matrix must have shape [T, N]")
    index = _as_datetime_index(timestamps)
    if values.shape[0] != len(index):
        raise RelationBuildError("Wspd matrix and timestamp axis have different lengths")
    output = np.full_like(values, np.nan, dtype=np.float64)
    if len(index) < 2:
        return output
    consecutive = np.diff(index.asi8) == pd.Timedelta(minutes=10).value
    valid = np.isfinite(values[1:]) & np.isfinite(values[:-1]) & consecutive[:, None]
    difference = values[1:] - values[:-1]
    output[1:] = np.where(valid, difference, np.nan)
    return output


def _standardize_delta(delta: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(delta, dtype=np.float64)
    mask = np.isfinite(values)
    count = mask.sum(axis=0).astype(np.float64)
    safe_count = np.maximum(count, 1.0)
    safe_values = np.where(mask, values, 0.0)
    mean = safe_values.sum(axis=0) / safe_count
    centered = np.where(mask, values - mean, 0.0)
    std = np.sqrt(np.square(centered).sum(axis=0) / safe_count)
    std = np.where(std < eps, 1.0, std)
    standardized = np.where(mask, (values - mean) / std, 0.0)
    return standardized, mask


def pairwise_overlap_cosine(
    delta: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the old pairwise-overlap cosine formula in float64 on CPU."""

    standardized, mask = _standardize_delta(delta, eps=eps)
    finite = mask.astype(np.float64)
    numerator = standardized.T @ standardized
    overlap = finite.T @ finite
    left_norm = np.square(standardized).T @ finite
    right_norm = finite.T @ np.square(standardized)
    denominator = np.sqrt(np.maximum(left_norm * right_norm, eps))
    similarity = numerator / denominator
    valid_count = finite.sum(axis=0)
    minimum = np.minimum(valid_count[:, None], valid_count[None, :])
    overlap_ratio = overlap / np.maximum(minimum, 1.0)
    return similarity, overlap, overlap_ratio


def pairwise_overlap_cosine_torch(
    delta: np.ndarray,
    *,
    eps: float = 1e-8,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Torch reference/accelerated implementation used by CPU/CUDA tests."""

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for relation construction but is unavailable")
    values = torch.as_tensor(np.asarray(delta, dtype=np.float64), dtype=dtype, device=resolved)
    if values.ndim != 2:
        raise RelationBuildError("delta must have shape [T, N]")
    mask = torch.isfinite(values)
    finite = mask.to(dtype)
    safe = torch.where(mask, values, torch.zeros_like(values))
    count = finite.sum(dim=0).clamp_min(1.0)
    mean = safe.sum(dim=0) / count
    centered = torch.where(mask, values - mean, torch.zeros_like(values))
    std = torch.sqrt(centered.square().sum(dim=0) / count).clamp_min(eps)
    standardized = torch.where(mask, (values - mean) / std, torch.zeros_like(values))
    numerator = standardized.transpose(0, 1) @ standardized
    overlap = finite.transpose(0, 1) @ finite
    left_norm = standardized.square().transpose(0, 1) @ finite
    right_norm = finite.transpose(0, 1) @ standardized.square()
    denominator = torch.sqrt(torch.clamp(left_norm * right_norm, min=eps))
    similarity = numerator / denominator
    valid_count = finite.sum(dim=0)
    minimum = torch.minimum(valid_count[:, None], valid_count[None, :])
    overlap_ratio = overlap / minimum.clamp_min(1.0)
    return tuple(
        value.detach().cpu().numpy().astype(np.float64, copy=False)
        for value in (similarity, overlap, overlap_ratio)
    )


def select_semantic_edges(
    similarity: np.ndarray,
    overlap: np.ndarray,
    overlap_ratio: np.ndarray,
    *,
    top_k: int = 10,
    min_overlap_ratio: float = 0.5,
    min_overlap_count: int = 500,
    min_similarity: float = 0.0,
) -> np.ndarray:
    """Select up to ``top_k`` source nodes for every target node."""

    similarity = np.asarray(similarity, dtype=np.float64)
    overlap = np.asarray(overlap, dtype=np.float64)
    overlap_ratio = np.asarray(overlap_ratio, dtype=np.float64)
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise RelationBuildError("semantic similarity must be square")
    n = similarity.shape[0]
    if overlap.shape != similarity.shape or overlap_ratio.shape != similarity.shape:
        raise RelationBuildError("semantic overlap matrices must match similarity shape")
    if int(top_k) < 1:
        raise RelationBuildError("semantic top_k must be positive")

    edges: list[tuple[int, int]] = []
    for target in range(n):
        candidates = np.flatnonzero(
            np.isfinite(similarity[target])
            & (similarity[target] > float(min_similarity))
            & (overlap[target] >= int(min_overlap_count))
            & (overlap_ratio[target] >= float(min_overlap_ratio))
            & (np.arange(n) != target)
        )
        if candidates.size:
            # This mirrors the old prototype's descending argsort selection.
            order = candidates[np.argsort(similarity[target, candidates])[::-1][: int(top_k)]]
        else:
            finite = np.flatnonzero(np.isfinite(similarity[target]) & (np.arange(n) != target))
            if not finite.size:
                raise RelationBuildError(
                    f"semantic graph target {target} has no finite non-self similarity fallback"
                )
            order = np.asarray([finite[np.argmax(similarity[target, finite])]], dtype=np.int64)
        edges.extend((int(source), int(target)) for source in order)
    return np.asarray(edges, dtype=np.int64).T


def _location_columns(location: pd.DataFrame) -> tuple[str, str, str, str | None]:
    columns = {str(column).lower(): str(column) for column in location.columns}
    required = ["turbid", "x", "y"]
    if any(name not in columns for name in required):
        raise RelationBuildError("location file must contain TurbID, x and y columns")
    elevation = columns.get("ele")
    return columns["turbid"], columns["x"], columns["y"], elevation


def build_distance_graph(
    location: pd.DataFrame,
    node_ids: Iterable[int],
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    """Build target-directed Euclidean KNN and all geometry feature matrices."""

    ids = _as_int64_node_ids(node_ids)
    if int(top_k) < 1 or int(top_k) >= len(ids):
        raise RelationBuildError("distance top_k must be in [1, node_count)")
    id_col, x_col, y_col, elevation_col = _location_columns(location)
    frame = location.copy()
    frame[id_col] = pd.to_numeric(frame[id_col], errors="raise").astype(np.int64)
    if frame[id_col].duplicated().any():
        raise RelationBuildError("location file contains duplicate TurbID rows")
    indexed = frame.set_index(id_col).reindex(ids)
    if indexed[[x_col, y_col]].isna().any().any():
        raise RelationBuildError("location file does not cover every formal turbine ID")
    x = indexed[x_col].to_numpy(dtype=np.float64)
    y = indexed[y_col].to_numpy(dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise RelationBuildError("location x/y values must be finite")
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    distance = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(distance, np.inf)
    edges: list[tuple[int, int]] = []
    for target in range(len(ids)):
        sources = np.argsort(distance[target])[: int(top_k)]
        edges.extend((int(source), int(target)) for source in sources)
    edge_index = np.asarray(edges, dtype=np.int64).T
    finite_distance = distance[np.isfinite(distance)]
    distance_scale = float(np.median(finite_distance)) if finite_distance.size else 1.0
    distance_scale = max(distance_scale, 1e-8)

    has_elevation = False
    elevation = np.zeros(len(ids), dtype=np.float64)
    if elevation_col is not None:
        candidate = indexed[elevation_col].to_numpy(dtype=np.float64)
        if np.isfinite(candidate).all():
            elevation = candidate
            has_elevation = True

    normalized_distance = distance / distance_scale
    distance_kernel_weight = np.exp(-np.square(normalized_distance))
    return {
        "edge_index": edge_index,
        "distance": distance,
        "normalized_distance": normalized_distance,
        "distance_kernel_weight": distance_kernel_weight,
        "relative_x": dx,
        "relative_y": dy,
        "elevation": elevation,
        "has_elevation": has_elevation,
        "distance_scale": distance_scale,
    }


def build_trueunion_graph(
    raw: pd.DataFrame,
    location: pd.DataFrame,
    node_ids: Iterable[int],
    train_timestamps: Iterable[Any],
    *,
    semantic_top_k: int = 10,
    distance_top_k: int = 5,
    semantic_min_overlap_ratio: float = 0.5,
    semantic_min_overlap_count: int = 500,
    semantic_min_similarity: float = 0.0,
    self_loops: bool = False,
    timestamp_col: str = "Tmstamp",
    turbine_id_col: str = "TurbID",
    wspd_col: str = "Wspd",
    device: str | torch.device = "cpu",
    metadata: Mapping[str, Any] | None = None,
) -> TrueUnionGraph:
    """Build the train-only semantic∪distance TrueUnion resource."""

    if self_loops:
        raise RelationBuildError("TrueUnion resource requires self_loops=false")
    ids = _as_int64_node_ids(node_ids)
    timestamps = _as_datetime_index(train_timestamps)
    matrix, alignment = build_train_wspd_matrix(
        raw,
        ids,
        timestamps,
        timestamp_col=timestamp_col,
        turbine_id_col=turbine_id_col,
        wspd_col=wspd_col,
    )
    delta = raw_delta_wspd(matrix, timestamps)
    resolved_device = torch.device(device)
    if resolved_device.type == "cpu":
        similarity, overlap, overlap_ratio = pairwise_overlap_cosine(delta)
        compute_dtype = "float64"
    else:
        similarity, overlap, overlap_ratio = pairwise_overlap_cosine_torch(
            delta, device=resolved_device, dtype=torch.float64
        )
        compute_dtype = "float64"
    semantic_edge_index = select_semantic_edges(
        similarity,
        overlap,
        overlap_ratio,
        top_k=semantic_top_k,
        min_overlap_ratio=semantic_min_overlap_ratio,
        min_overlap_count=semantic_min_overlap_count,
        min_similarity=semantic_min_similarity,
    )
    geometry = build_distance_graph(location, ids, top_k=distance_top_k)
    semantic_edges = set(map(tuple, semantic_edge_index.T.tolist()))
    distance_edges = set(map(tuple, geometry["edge_index"].T.tolist()))
    all_edges = sorted(semantic_edges | distance_edges, key=lambda pair: (pair[1], pair[0]))
    if not all_edges:
        raise RelationBuildError("TrueUnion contains no edges")

    raw_features = np.zeros((len(all_edges), 13), dtype=np.float64)
    for row, (source, target) in enumerate(all_edges):
        if (source, target) in semantic_edges:
            raw_features[row, 0] = similarity[target, source]
            raw_features[row, 1] = overlap_ratio[target, source]
        raw_features[row, 2] = geometry["distance_kernel_weight"][target, source]
        raw_features[row, 3] = geometry["normalized_distance"][target, source]
        raw_features[row, 4] = geometry["relative_x"][target, source]
        raw_features[row, 5] = geometry["relative_y"][target, source]
        if geometry["has_elevation"]:
            delta_elevation = geometry["elevation"][target] - geometry["elevation"][source]
            horizontal = max(float(geometry["distance"][target, source]), 1e-8)
            raw_features[row, 6] = delta_elevation
            raw_features[row, 7] = delta_elevation / horizontal
            raw_features[row, 8] = np.arctan2(delta_elevation, horizontal)
        raw_features[row, 9] = float((source, target) in semantic_edges)
        raw_features[row, 10] = float((source, target) in distance_edges)
        raw_features[row, 11] = raw_features[row, 9] * raw_features[row, 10]
        raw_features[row, 12] = float(geometry["has_elevation"])

    continuous = raw_features[:, :9]
    mean = np.nanmean(continuous, axis=0)
    std = np.nanstd(continuous, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    normalized = raw_features.copy()
    normalized[:, :9] = (continuous - mean) / std
    normalized = np.where(np.isfinite(normalized), normalized, 0.0).astype(np.float32)
    edge_index = np.asarray(all_edges, dtype=np.int64).T
    if np.any(edge_index[0] == edge_index[1]):
        raise RelationBuildError("TrueUnion contains a self-loop")
    if list(zip(edge_index[1].tolist(), edge_index[0].tolist())) != sorted(
        zip(edge_index[1].tolist(), edge_index[0].tolist())
    ):
        raise RelationBuildError("TrueUnion edges are not sorted by (target, source)")
    counts = {
        "semantic_edge_count": int(normalized[:, 9].sum()),
        "distance_edge_count": int(normalized[:, 10].sum()),
        "both_edge_count": int(normalized[:, 11].sum()),
    }
    return TrueUnionGraph(
        node_ids=ids.copy(),
        edge_index=edge_index,
        edge_static_features=normalized,
        edge_feature_names=STATIC_EDGE_FEATURE_NAMES,
        raw_edge_features=raw_features.astype(np.float32),
        metadata={
            **dict(metadata or {}),
            "alignment": alignment,
            "semantic_compute_device": str(resolved_device),
            "semantic_compute_dtype": compute_dtype,
            "train_timestamp_count": len(timestamps),
            "semantic_top_k": int(semantic_top_k),
            "distance_top_k": int(distance_top_k),
            "semantic_min_overlap_ratio": float(semantic_min_overlap_ratio),
            "semantic_min_overlap_count": int(semantic_min_overlap_count),
            "semantic_min_similarity": float(semantic_min_similarity),
            **counts,
        },
        **counts,
    )


def _validate_artifact_for_write(graph: TrueUnionGraph) -> None:
    node_ids = _as_int64_node_ids(graph.node_ids)
    if not np.array_equal(node_ids, np.asarray(graph.node_ids, dtype=np.int64)):
        raise RelationBuildError("node_ids must be int64")
    edge_index = np.asarray(graph.edge_index)
    if edge_index.dtype != np.int64 or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise RelationBuildError("edge_index must have dtype int64 and shape [2,E]")
    if edge_index.shape[1] == 0:
        raise RelationBuildError("relation artifact must contain at least one edge")
    if np.any(edge_index[0] == edge_index[1]):
        raise RelationBuildError("relation artifact contains a self-loop")
    pairs = list(zip(edge_index[1].tolist(), edge_index[0].tolist()))
    if len(set(pairs)) != len(pairs) or pairs != sorted(pairs):
        raise RelationBuildError("relation artifact edges must be unique and sorted by (target, source)")
    features = np.asarray(graph.edge_static_features)
    if features.dtype != np.float32 or features.shape != (edge_index.shape[1], 13):
        raise RelationBuildError("edge_static_features must have dtype float32 and shape [E,13]")
    if not np.isfinite(features).all():
        raise RelationBuildError("edge_static_features contain non-finite values")
    if tuple(graph.edge_feature_names) != STATIC_EDGE_FEATURE_NAMES:
        raise RelationBuildError("edge_feature_names do not match the stable 13-name schema")
    if int(edge_index.max()) >= len(node_ids) or int(edge_index.min()) < 0:
        raise RelationBuildError("edge_index is outside the node range")
    if np.any(np.bincount(edge_index[1], minlength=len(node_ids)) == 0):
        raise RelationBuildError("relation artifact contains a target with zero indegree")


def write_relation_artifact(graph: TrueUnionGraph, path: str | Path) -> Path:
    """Write exactly the five arrays understood by ``relation_resource``."""

    _validate_artifact_for_write(graph)
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        schema_version=np.asarray(1, dtype=np.int64),
        node_ids=np.asarray(graph.node_ids, dtype=np.int64),
        edge_index=np.asarray(graph.edge_index, dtype=np.int64),
        edge_static_features=np.asarray(graph.edge_static_features, dtype=np.float32),
        edge_feature_names=np.asarray(graph.edge_feature_names),
    )
    return destination


def _node_order_hash(node_ids: Iterable[int]) -> str:
    payload = {"turbine_ids": [int(value) for value in node_ids]}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


def _load_old_graph(old_directory: str | Path, expected_node_ids: Iterable[int] | None = None) -> TrueUnionGraph:
    directory = Path(old_directory).resolve()
    required = ("edge_index.npy", "edge_features.npz", "graph_config.json", "graph_manifest.json")
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise RelationBuildError(f"old graph directory is missing {missing[0]}")
    edge_index = np.load(directory / "edge_index.npy", allow_pickle=False)
    with np.load(directory / "edge_features.npz", allow_pickle=False) as archive:
        if "features" not in archive.files:
            raise RelationBuildError("old edge_features.npz is missing the features array")
        if "names" not in archive.files:
            raise RelationBuildError("old edge_features.npz is missing the names array")
        # Deliberately select features, never raw: the current loader consumes
        # the already z-scored static edge features.
        features = archive["features"].copy()
        names = tuple(str(value) for value in archive["names"].tolist())
    manifest = json.loads((directory / "graph_manifest.json").read_text(encoding="utf-8"))
    json.loads((directory / "graph_config.json").read_text(encoding="utf-8"))
    if "node_ids" in manifest:
        node_ids = _as_int64_node_ids(manifest["node_ids"], name="old graph node_ids")
        if expected_node_ids is not None and not np.array_equal(
            node_ids, _as_int64_node_ids(expected_node_ids, name="expected node_ids")
        ):
            raise RelationBuildError(
                "old graph node_ids do not match the current formal node order"
            )
    else:
        if expected_node_ids is None:
            raise RelationBuildError(
                "old graph node order cannot be proved: provide current formal node_ids"
            )
        node_ids = _as_int64_node_ids(expected_node_ids)
        recorded_hash = manifest.get("node_order_hash")
        if recorded_hash is None or str(recorded_hash) != _node_order_hash(node_ids):
            raise RelationBuildError(
                "old graph node order cannot be proved from graph_manifest.json"
            )
    if int(manifest.get("node_count", len(node_ids))) != len(node_ids):
        raise RelationBuildError("old graph node_count does not match the proved node order")
    edge_index = np.asarray(edge_index)
    features = np.asarray(features)
    if edge_index.dtype != np.int64 or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise RelationBuildError("old edge_index.npy must have dtype int64 and shape [2,E]")
    if features.dtype != np.float32 or features.shape != (edge_index.shape[1], 13):
        raise RelationBuildError("old features must have dtype float32 and shape [E,13]")
    if names != STATIC_EDGE_FEATURE_NAMES:
        raise RelationBuildError("old graph edge feature names do not match the current 13-name schema")
    pairs = list(zip(edge_index[1].tolist(), edge_index[0].tolist()))
    if pairs != sorted(pairs) or len(set(pairs)) != len(pairs):
        raise RelationBuildError("old graph edges are not unique and sorted by (target, source)")
    if np.any(edge_index[0] == edge_index[1]):
        raise RelationBuildError("old graph contains a self-loop")
    counts = {
        "semantic_edge_count": int(np.rint(features[:, 9]).sum()),
        "distance_edge_count": int(np.rint(features[:, 10]).sum()),
        "both_edge_count": int(np.rint(features[:, 11]).sum()),
    }
    return TrueUnionGraph(
        node_ids=node_ids,
        edge_index=edge_index.copy(),
        edge_static_features=features.copy(),
        edge_feature_names=names,
        metadata={"old_directory": str(directory), "manifest": manifest},
        **counts,
    )


def convert_old_graph(
    old_directory: str | Path,
    output_path: str | Path,
    *,
    expected_node_ids: Iterable[int] | None = None,
) -> TrueUnionGraph:
    """Convert an old graph directory to the current five-array NPZ contract."""

    graph = _load_old_graph(old_directory, expected_node_ids=expected_node_ids)
    write_relation_artifact(graph, output_path)
    return graph


def compare_against_old_graph(
    graph: TrueUnionGraph,
    old_directory: str | Path,
    *,
    expected_node_ids: Iterable[int] | None = None,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict[str, Any]:
    """Compare the rebuilt arrays and report the first useful mismatch."""

    old = _load_old_graph(old_directory, expected_node_ids=expected_node_ids)
    report: dict[str, Any] = {
        "old_directory": str(Path(old_directory).resolve()),
        "new_metadata": dict(graph.metadata),
        "old_counts": {
            "node_count": len(old.node_ids),
            "edge_count": old.edge_count,
            "semantic_edge_count": old.semantic_edge_count,
            "distance_edge_count": old.distance_edge_count,
            "both_edge_count": old.both_edge_count,
        },
        "new_counts": {
            "node_count": len(graph.node_ids),
            "edge_count": graph.edge_count,
            "semantic_edge_count": graph.semantic_edge_count,
            "distance_edge_count": graph.distance_edge_count,
            "both_edge_count": graph.both_edge_count,
        },
        "node_ids_exact": bool(np.array_equal(graph.node_ids, old.node_ids)),
        "edge_index_exact": bool(np.array_equal(graph.edge_index, old.edge_index)),
        "edge_feature_names_exact": graph.edge_feature_names == old.edge_feature_names,
        "edge_static_features_allclose": False,
    }
    if graph.edge_static_features.shape == old.edge_static_features.shape:
        report["edge_static_features_allclose"] = bool(
            np.allclose(graph.edge_static_features, old.edge_static_features, atol=atol, rtol=rtol)
        )
    report["counts_match"] = report["old_counts"] == report["new_counts"]
    checks = (
        "node_ids_exact",
        "edge_index_exact",
        "edge_feature_names_exact",
        "edge_static_features_allclose",
        "counts_match",
    )
    if not all(report[name] for name in checks):
        if not report["edge_index_exact"]:
            old_edge = old.edge_index
            new_edge = graph.edge_index
            first = None
            for index in range(min(old_edge.shape[1], new_edge.shape[1])):
                if not np.array_equal(old_edge[:, index], new_edge[:, index]):
                    first = index
                    break
            if first is None and old_edge.shape[1] != new_edge.shape[1]:
                first = min(old_edge.shape[1], new_edge.shape[1])
            if first is not None:
                report["first_edge_mismatch"] = {
                    "edge_position": int(first),
                    "old_source_target": old_edge[:, first].tolist() if first < old_edge.shape[1] else None,
                    "new_source_target": new_edge[:, first].tolist() if first < new_edge.shape[1] else None,
                }
        if not report["edge_static_features_allclose"]:
            if old.edge_static_features.shape == graph.edge_static_features.shape:
                difference = np.abs(old.edge_static_features - graph.edge_static_features)
                position = np.unravel_index(int(np.argmax(difference)), difference.shape)
                edge_position = int(position[0])
                report["first_feature_mismatch"] = {
                    "edge_position": edge_position,
                    "target": int(graph.edge_index[1, edge_position]),
                    "source": int(graph.edge_index[0, edge_position]),
                    "feature_index": int(position[1]),
                    "old": old.edge_static_features[position].item(),
                    "new": graph.edge_static_features[position].item(),
                }
            else:
                report["feature_shape_mismatch"] = {
                    "old": list(old.edge_static_features.shape),
                    "new": list(graph.edge_static_features.shape),
                }
        raise RelationConsistencyError(report)
    report["passed"] = True
    return report


def _resource_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RelationBuildError("TrueUnion resource config must be a mapping")
    required = {"schema_version", "semantic_source", "semantic", "distance", "union", "output"}
    unknown = sorted(set(value) - required)
    missing = sorted(required - set(value))
    if unknown:
        raise RelationBuildError(f"TrueUnion resource config has unknown field: {unknown[0]}")
    if missing:
        raise RelationBuildError(f"TrueUnion resource config is missing field: {missing[0]}")
    if value["schema_version"] != 1:
        raise RelationBuildError("TrueUnion resource config schema_version must be 1")
    source_section = value["semantic_source"]
    semantic = value["semantic"]
    distance = value["distance"]
    union = value["union"]
    output = value["output"]
    if not isinstance(source_section, Mapping) or set(source_section) != {"file", "wspd_column"}:
        raise RelationBuildError("semantic_source must contain file and wspd_column")
    if not isinstance(semantic, Mapping) or set(semantic) != {
        "top_k", "min_overlap_ratio", "min_overlap_count", "min_similarity"
    }:
        raise RelationBuildError("semantic config fields do not match the TrueUnion protocol")
    if not isinstance(distance, Mapping) or set(distance) != {"top_k"}:
        raise RelationBuildError("distance config must contain only top_k")
    if not isinstance(union, Mapping) or set(union) != {"self_loops"}:
        raise RelationBuildError("union config must contain only self_loops")
    if not isinstance(output, Mapping) or set(output) != {"file"}:
        raise RelationBuildError("output config must contain only file")
    if not isinstance(source_section["file"], str) or not source_section["file"]:
        raise RelationBuildError("semantic_source.file must be non-empty")
    if source_section["wspd_column"] != "Wspd":
        raise RelationBuildError("TrueUnion semantic source must be Wspd")
    if not isinstance(output["file"], str) or not output["file"]:
        raise RelationBuildError("output.file must be non-empty")
    return dict(value)


def _resolve_relative(root: Path, value: str, *, label: str) -> Path:
    path = Path(value)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise RelationBuildError(f"{label} must be a project-relative path")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise RelationBuildError(f"{label} resolves outside the project root") from exc
    return resolved


def _read_public_grid(path: Path, timestamp_col: str, turbine_id_col: str, num_nodes: int) -> tuple[pd.DatetimeIndex, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"public model-input data file does not exist: {path}")
    frame = pd.read_parquet(path, columns=[timestamp_col, turbine_id_col])
    frame[timestamp_col] = pd.to_datetime(frame[timestamp_col])
    frame[turbine_id_col] = pd.to_numeric(frame[turbine_id_col], errors="raise").astype(np.int64)
    frame = frame.sort_values([timestamp_col, turbine_id_col], kind="mergesort")
    if frame.duplicated([timestamp_col, turbine_id_col]).any():
        raise RelationBuildError("public model-input grid contains duplicate timestamp/turbine rows")
    timestamps = pd.DatetimeIndex(frame[timestamp_col].drop_duplicates().to_numpy())
    if len(frame) != len(timestamps) * int(num_nodes):
        raise RelationBuildError("public model-input rows do not form the configured node grid")
    first_nodes: np.ndarray | None = None
    for timestamp, group in frame.groupby(timestamp_col, sort=True, observed=True):
        values = group[turbine_id_col].to_numpy(dtype=np.int64)
        if len(values) != num_nodes:
            raise RelationBuildError(f"timestamp {timestamp} does not contain all formal nodes")
        if first_nodes is None:
            first_nodes = values
        elif not np.array_equal(first_nodes, values):
            raise RelationBuildError("public node order changes between timestamps")
    assert first_nodes is not None
    return timestamps, first_nodes


def build_trueunion_from_project(
    project_root: str | Path,
    resource_config_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    output_path: str | Path | None = None,
) -> tuple[TrueUnionGraph, Path]:
    """Build the formal resource using experiment.yaml and raw aligned Wspd."""

    root = Path(project_root).resolve()
    resource_path = Path(resource_config_path)
    if not resource_path.is_absolute():
        resource_path = root / resource_path
    resource = _resource_config(resource_path)
    experiment = load_experiment_config(root / "configs" / "experiment.yaml")
    data = experiment.data
    data_root_value = Path(str(data["data_root"]))
    data_root = (
        data_root_value.resolve()
        if data_root_value.is_absolute()
        else _resolve_relative(root, str(data_root_value), label="data.data_root")
    )
    raw_path = _resolve_relative(data_root, str(resource["semantic_source"]["file"]), label="semantic_source.file")
    if not raw_path.is_file():
        raise FileNotFoundError(
            "build-from-data requires the exact raw semantic source "
            f"{raw_path}; no model-input Wspd substitution is permitted"
        )
    location_path = _resolve_relative(data_root, str(data["resources_location_file"]), label="location file") if "resources_location_file" in data else _resolve_relative(
        data_root, str(experiment.resources["graph"]["location_file"]), label="resources.graph.location_file"
    )
    model_input_path = _resolve_relative(data_root, str(data["model_input_file"]), label="data.model_input_file")
    timestamps, node_ids = _read_public_grid(
        model_input_path,
        str(data["timestamp_column"]),
        str(data["turbine_id_column"]),
        int(data["num_nodes"]),
    )
    split = chronological_split(len(timestamps), experiment)
    train_timestamps = timestamps[split.train.start : split.train.end]
    raw = pd.read_parquet(
        raw_path,
        columns=[str(data["timestamp_column"]), str(data["turbine_id_column"]), str(resource["semantic_source"]["wspd_column"])],
    )
    location = pd.read_csv(location_path)
    semantic = resource["semantic"]
    graph = build_trueunion_graph(
        raw,
        location,
        node_ids,
        train_timestamps,
        semantic_top_k=int(semantic["top_k"]),
        distance_top_k=int(resource["distance"]["top_k"]),
        semantic_min_overlap_ratio=float(semantic["min_overlap_ratio"]),
        semantic_min_overlap_count=int(semantic["min_overlap_count"]),
        semantic_min_similarity=float(semantic["min_similarity"]),
        self_loops=bool(resource["union"]["self_loops"]),
        timestamp_col=str(data["timestamp_column"]),
        turbine_id_col=str(data["turbine_id_column"]),
        wspd_col=str(resource["semantic_source"]["wspd_column"]),
        device=device,
        metadata={
            "semantic_source_file": str(raw_path),
            "location_file": str(location_path),
            "train_boundary": {"start": split.train.start, "end": split.train.end},
            "node_ids": node_ids.tolist(),
        },
    )
    configured_output = _resolve_relative(root, str(resource["output"]["file"]), label="output.file")
    destination = Path(output_path) if output_path is not None else configured_output
    if not destination.is_absolute():
        destination = root / destination
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_relation_artifact(graph, destination)
    return graph, destination


def public_node_ids_from_project(project_root: str | Path) -> np.ndarray:
    """Resolve current formal node order for old-resource conversion."""

    root = Path(project_root).resolve()
    experiment = load_experiment_config(root / "configs" / "experiment.yaml")
    data = experiment.data
    data_root_value = Path(str(data["data_root"]))
    data_root = (
        data_root_value.resolve()
        if data_root_value.is_absolute()
        else _resolve_relative(root, str(data_root_value), label="data.data_root")
    )
    path = _resolve_relative(data_root, str(data["model_input_file"]), label="data.model_input_file")
    _timestamps, node_ids = _read_public_grid(
        path,
        str(data["timestamp_column"]),
        str(data["turbine_id_column"]),
        int(data["num_nodes"]),
    )
    return node_ids


__all__ = [
    "RelationBuildError",
    "RelationConsistencyError",
    "STATIC_EDGE_FEATURE_NAMES",
    "TrueUnionGraph",
    "build_distance_graph",
    "build_train_wspd_matrix",
    "build_trueunion_from_project",
    "build_trueunion_graph",
    "compare_against_old_graph",
    "convert_old_graph",
    "pairwise_overlap_cosine",
    "pairwise_overlap_cosine_torch",
    "public_node_ids_from_project",
    "raw_delta_wspd",
    "select_semantic_edges",
    "write_relation_artifact",
]

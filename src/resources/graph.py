"""Shared deterministic graph resources for spatial forecasting models."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch


@dataclass(frozen=True)
class GraphResource:
    """One immutable dense graph plus the sparse tensors consumed by models."""

    adjacency: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor


def validate_adjacency(adjacency: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Validate one finite dense ``(nodes, nodes)`` adjacency matrix."""

    if not isinstance(adjacency, torch.Tensor):
        raise TypeError("adjacency must be a torch.Tensor")
    if tuple(adjacency.shape) != (num_nodes, num_nodes):
        raise ValueError(f"adjacency must have shape ({num_nodes}, {num_nodes})")
    if not torch.isfinite(adjacency).all():
        raise ValueError("adjacency contains NaN or Inf")
    return adjacency


def dense_adjacency_to_edges(
    adjacency: torch.Tensor,
    *,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a dense graph to deterministic COO edges and their weights."""

    dense = validate_adjacency(adjacency, num_nodes)
    edge_index = torch.nonzero(dense != 0, as_tuple=False).T.contiguous().long()
    if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.shape[1] == 0:
        raise ValueError("graph must contain at least one non-zero edge")
    edge_weight = dense[edge_index[0], edge_index[1]].contiguous()
    validate_edge_tensors(edge_index, edge_weight, num_nodes=num_nodes)
    return edge_index, edge_weight


def validate_edge_tensors(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    *,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate sparse graph tensors before they are registered as buffers."""

    if not isinstance(edge_index, torch.Tensor) or not isinstance(edge_weight, torch.Tensor):
        raise TypeError("edge_index and edge_weight must be torch.Tensor values")
    if edge_index.dtype != torch.long:
        raise ValueError("edge_index must use torch.long")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, edges)")
    if edge_index.shape[1] == 0:
        raise ValueError("graph must contain at least one edge")
    if edge_weight.ndim != 1 or edge_weight.shape[0] != edge_index.shape[1]:
        raise ValueError("edge_weight length must equal edge_index edge count")
    if not torch.isfinite(edge_weight).all():
        raise ValueError("edge_weight contains NaN or Inf")
    if int(edge_index.min()) < 0 or int(edge_index.max()) >= int(num_nodes):
        raise ValueError("edge_index contains a node outside the configured range")
    return edge_index, edge_weight


def _require_columns(fieldnames: Iterable[str] | None) -> None:
    actual = set(fieldnames or ())
    required = {"TurbID", "x", "y"}
    missing = sorted(required - actual)
    if missing:
        raise ValueError(
            "location CSV must contain the actual SDWPF columns TurbID, x and y; "
            f"missing: {missing[0]}"
        )


def _read_locations(path: Path, *, node_ids: tuple[int, ...]) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"graph location file does not exist: {path}")
    by_turbine: dict[int, tuple[float, float]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require_columns(reader.fieldnames)
        for row in reader:
            try:
                turbine_id = int(str(row["TurbID"]))
                point = (float(str(row["x"])), float(str(row["y"])))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid TurbID/x/y row in graph location file: {path}") from exc
            if not np.isfinite(point).all():
                raise ValueError(f"graph location contains non-finite coordinates for TurbID {turbine_id}")
            if turbine_id in by_turbine:
                raise ValueError(f"graph location contains duplicate TurbID: {turbine_id}")
            by_turbine[turbine_id] = point
    expected = set(node_ids)
    if len(expected) != len(node_ids):
        raise ValueError("public data node order contains duplicate TurbID values")
    missing = sorted(expected - set(by_turbine))
    extra = sorted(set(by_turbine) - expected)
    if missing:
        raise ValueError(f"graph location is missing public data TurbID: {missing[0]}")
    if extra:
        raise ValueError(f"graph location contains TurbID absent from public data: {extra[0]}")
    return np.asarray([by_turbine[turbine_id] for turbine_id in node_ids], dtype=np.float64)


def build_physical_knn_adjacency(
    *,
    location_file: str | Path,
    node_ids: Iterable[int],
    num_nodes: int,
    k: int,
    symmetrize: bool,
    self_loops: bool,
    weighting: str,
) -> torch.Tensor:
    """Build a deterministic binary physical kNN graph aligned to data nodes."""

    ordered_ids = tuple(int(value) for value in node_ids)
    if len(ordered_ids) != int(num_nodes):
        raise ValueError("graph node ids must have exactly data.num_nodes entries")
    if not 1 <= int(k) < int(num_nodes):
        raise ValueError("physical kNN requires 1 <= k < num_nodes")
    if weighting != "binary":
        raise ValueError(f"unsupported graph weighting: {weighting!r}")
    coordinates = _read_locations(Path(location_file), node_ids=ordered_ids)
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for source in range(num_nodes):
        distances = np.linalg.norm(coordinates - coordinates[source], axis=1)
        candidates = sorted(
            (float(distances[target]), target)
            for target in range(num_nodes)
            if target != source
        )
        for _, target in candidates[:k]:
            adjacency[source, target] = 1.0
    if symmetrize:
        adjacency = np.maximum(adjacency, adjacency.T)
    if self_loops:
        np.fill_diagonal(adjacency, 1.0)
    else:
        np.fill_diagonal(adjacency, 0.0)
    return validate_adjacency(torch.from_numpy(adjacency), num_nodes)


def build_graph_resource(
    graph_config: Mapping[str, Any] | None,
    *,
    node_ids: Iterable[int],
    num_nodes: int,
    project_root: str | Path,
) -> GraphResource:
    """Build one configured shared graph resource for a graph-capable adapter."""

    if not isinstance(graph_config, Mapping):
        raise ValueError("graph model requires public resources.graph configuration")
    expected = {"type", "location_file", "k", "symmetrize", "self_loops", "weighting"}
    unknown = sorted(set(graph_config) - expected)
    missing = sorted(expected - set(graph_config))
    if unknown:
        raise ValueError(f"graph configuration has unknown field: {unknown[0]}")
    if missing:
        raise ValueError(f"graph configuration is missing field: {missing[0]}")
    if graph_config["type"] != "physical_knn":
        raise ValueError(f"unsupported graph type: {graph_config['type']!r}")
    root = Path(project_root).resolve()
    location_file = root / "dataset" / str(graph_config["location_file"])
    adjacency = build_physical_knn_adjacency(
        location_file=location_file,
        node_ids=node_ids,
        num_nodes=int(num_nodes),
        k=int(graph_config["k"]),
        symmetrize=bool(graph_config["symmetrize"]),
        self_loops=bool(graph_config["self_loops"]),
        weighting=str(graph_config["weighting"]),
    )
    edge_index, edge_weight = dense_adjacency_to_edges(adjacency, num_nodes=int(num_nodes))
    return GraphResource(adjacency=adjacency, edge_index=edge_index, edge_weight=edge_weight)
